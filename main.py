"""Volatility-of-Volatility Mean-Reversion Engine (VVIX/VIX).

Pipeline: Data Layer -> Signal Core -> Execution Engine
          -> Performance Analytics -> Export/Visualization

Usage:
    python main.py --window 20 --entry 2.0 --cost-bps 5.0
    python main.py --end 2026-10-22              # pin the data end date (reproducible)
    python main.py --sweep                       # robustness grid in one run
    python main.py --split-date 2023-01-01       # pick params in-sample, test out-of-sample
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; we only save files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

# yfinance 1.7 still uses pandas calls that pandas 3.1 deprecates; harmless noise.
warnings.filterwarnings("ignore", category=Warning, module=r"yfinance.*")

START_DATE = "2018-01-01"  # FR-2
TRADING_DAYS = 252
EPS = 1e-8  # zero-division containment floor
SWEEP_COLS = [
    "Ann. Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
    "Cumulative Return", "Trades", "Time in Market",
]


# --------------------------------------------------------------------------- #
# CLI (FR-1)
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VVIX/VIX mean-reversion backtest")
    p.add_argument("--window", type=int, default=20, help="rolling lookback (days)")
    p.add_argument("--entry", type=float, default=2.0, help="|Z| entry threshold")
    p.add_argument("--cost-bps", type=float, default=5.0, help="cost per leg in bps")
    p.add_argument("--end", type=str, default=None,
                   help="exclusive data end date YYYY-MM-DD (default: today, which "
                        "drops today's unfinished bar)")
    p.add_argument("--out", type=str, default="outputs", help="relative output dir")
    p.add_argument("--sweep", action="store_true", help="run the robustness grid")
    p.add_argument("--sweep-windows", type=int, nargs="+", default=[10, 20, 40, 60])
    p.add_argument("--sweep-entries", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    p.add_argument("--sweep-costs", type=float, nargs="+", default=[5.0, 25.0, 50.0])
    p.add_argument("--split-date", type=str, default=None,
                   help="YYYY-MM-DD: choose params on data before this date, "
                        "evaluate them on data from this date onward")
    args = p.parse_args(argv)
    if args.split_date is not None:
        try:
            pd.Timestamp(args.split_date)
        except ValueError:
            p.error("--split-date must look like YYYY-MM-DD")
    if args.window < 2 or min(args.sweep_windows) < 2:
        p.error("windows must be >= 2")
    if args.entry <= 0 or min(args.sweep_entries) <= 0:
        p.error("entry thresholds must be > 0")
    if args.cost_bps < 0 or min(args.sweep_costs) < 0:
        p.error("cost-bps must be >= 0")
    return args


# --------------------------------------------------------------------------- #
# Data Layer (FR-2)
# --------------------------------------------------------------------------- #
def fetch_data(start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    raw = yf.download(
        ["^VIX", "^VVIX"], start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance. Check your connection.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={"^VIX": "VIX", "^VVIX": "VVIX"})
    missing = {"VIX", "VVIX"} - set(close.columns)
    if missing:
        raise RuntimeError(f"Missing columns after download: {sorted(missing)}")

    df = close[["VIX", "VVIX"]].sort_index().ffill().dropna()
    df = df[(df > 0).all(axis=1)]  # guard the ratio denominator
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


# --------------------------------------------------------------------------- #
# Signal Core (vectorized)
# --------------------------------------------------------------------------- #
def compute_signal(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["ratio"] = out["VVIX"] / out["VIX"]
    roll = out["ratio"].rolling(window, min_periods=window)
    mu = roll.mean()
    sigma = roll.std().clip(lower=EPS)  # no zero / near-zero denominators
    z = (out["ratio"] - mu) / sigma
    out["z"] = z.replace([np.inf, -np.inf], np.nan)
    return out


# --------------------------------------------------------------------------- #
# Execution Engine (boolean state machine, no row loops)
# --------------------------------------------------------------------------- #
def run_backtest(sig: pd.DataFrame, entry: float, cost_bps: float) -> pd.DataFrame:
    z = sig["z"]

    entry_short = z > entry    # ratio rich vs. history -> short
    entry_long = z < -entry    # ratio cheap vs. history -> long
    crossed_mean = (np.sign(z) * np.sign(z.shift(1))) <= 0  # Z crossed 0 (NaN -> False)

    # Event series: NaN = "no event, hold regime"; 0 = exit; +/-1 = entry.
    # Later masks override earlier ones, so entries beat exits on the same bar.
    state = pd.Series(np.nan, index=sig.index)
    state = state.mask(crossed_mean, 0.0).mask(entry_short, -1.0).mask(entry_long, 1.0)
    signal = state.ffill().fillna(0.0)

    position = signal.shift(1).fillna(0.0)  # anti-lookahead: T signal -> T+1 return

    asset_ret = np.log(sig["ratio"]).diff().fillna(0.0)  # log return of the ratio
    gross = position * asset_ret

    trade = position.diff().fillna(position).ne(0.0)  # every position modification
    cost = trade.astype(float) * (2.0 * cost_bps * 1e-4)  # FR-4
    net = gross - cost

    res = sig.copy()
    res["signal"] = signal
    res["position"] = position
    res["asset_ret"] = asset_ret
    res["gross_ret"] = gross
    res["cost"] = cost
    res["net_ret"] = net
    res["equity"] = np.exp(net.cumsum())
    res["bh_equity"] = np.exp(asset_ret.cumsum())
    return res


# --------------------------------------------------------------------------- #
# Performance Analytics
# --------------------------------------------------------------------------- #
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else float("nan")


def compute_metrics(res: pd.DataFrame) -> pd.Series:
    r = res["net_ret"]
    bh = res["asset_ret"]
    n = len(r)

    ann_ret = res["equity"].iloc[-1] ** (TRADING_DAYS / n) - 1.0
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = _safe_div(r.mean(), r.std()) * np.sqrt(TRADING_DAYS)

    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    sortino = _safe_div(r.mean(), downside) * np.sqrt(TRADING_DAYS)

    active = r - bh
    info_ratio = _safe_div(active.mean(), active.std()) * np.sqrt(TRADING_DAYS)

    drawdown = res["equity"] / res["equity"].cummax() - 1.0
    bh_dd = res["bh_equity"] / res["bh_equity"].cummax() - 1.0

    strat_cum = res["equity"].iloc[-1] - 1.0
    bh_cum = res["bh_equity"].iloc[-1] - 1.0

    metrics = {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Information Ratio": info_ratio,
        "Max Drawdown": drawdown.min(),
        "Buy&Hold Max Drawdown": bh_dd.min(),
        "Cumulative Return": strat_cum,
        "Buy&Hold Cumulative Return": bh_cum,
        "Cumulative Alpha vs B&H": strat_cum - bh_cum,
        "Trades": float(res["cost"].gt(0).sum()),
        "Time in Market": float(res["position"].ne(0).mean()),
    }
    return pd.Series(metrics)


def yearly_returns(res: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year simple returns: shows whether profit is concentrated."""
    years = res.index.year
    return pd.DataFrame({
        "Strategy": np.expm1(res["net_ret"].groupby(years).sum()),
        "Buy&Hold": np.expm1(res["asset_ret"].groupby(years).sum()),
    })


def evaluate(df: pd.DataFrame, window: int, entry: float, cost_bps: float):
    res = run_backtest(compute_signal(df, window), entry, cost_bps)
    return res, compute_metrics(res)


# --------------------------------------------------------------------------- #
# Sweep (robustness grid)
# --------------------------------------------------------------------------- #
def run_sweep(df, windows, entries, costs, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"window": w, "entry": e, "cost_bps": c, **evaluate(df, w, e, c)[1][SWEEP_COLS].to_dict()}
        for w, e, c in itertools.product(windows, entries, costs)
    ]
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out_dir / "sweep_results.csv", index=False)

    base_cost = costs[0]
    pivot = table[table["cost_bps"] == base_cost].pivot(
        index="window", columns="entry", values="Sharpe Ratio"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{i}" for i in pivot.index])
    ax.set_xlabel("entry threshold (|Z|)")
    ax.set_ylabel("window (days)")
    ax.set_title(f"Sharpe ratio, cost {base_cost:g} bps")
    [ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")
     for i, j in np.ndindex(pivot.shape)]
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_sharpe_heatmap.png", dpi=200)
    plt.close(fig)
    return table


# --------------------------------------------------------------------------- #
# Out-of-sample split
# --------------------------------------------------------------------------- #
def slice_period(res: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Rows with start <= date < end, with equity curves rebased to 1."""
    mask = np.ones(len(res), dtype=bool)
    if start is not None:
        mask &= np.asarray(res.index >= start)
    if end is not None:
        mask &= np.asarray(res.index < end)
    part = res.loc[mask].copy()
    part["equity"] = np.exp(part["net_ret"].cumsum())
    part["bh_equity"] = np.exp(part["asset_ret"].cumsum())
    return part


def run_split(df, split, windows, entries, cost_bps, out_dir: Path):
    """Select (window, entry) by in-sample Sharpe only, then report out-of-sample.

    Signals use only past data, so one full-sample backtest sliced by date is
    identical to running each period separately (no look-ahead across the split).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ts = pd.Timestamp(split)
    n_is = int((df.index < split_ts).sum())
    n_oos = len(df) - n_is
    need = 2 * max(windows)
    if n_is < need or n_oos < need:
        raise ValueError(
            f"Split leaves too little data (in-sample {n_is}, out-of-sample {n_oos} "
            f"days; need >= {need} each). Choose a different --split-date."
        )

    runs = {(w, e): run_backtest(compute_signal(df, w), e, cost_bps)
            for w, e in itertools.product(windows, entries)}
    periods = {k: (compute_metrics(slice_period(r, end=split_ts)),
                   compute_metrics(slice_period(r, start=split_ts)))
               for k, r in runs.items()}

    table = pd.DataFrame([
        {"window": w, "entry": e,
         "IS Sharpe": i["Sharpe Ratio"], "OOS Sharpe": o["Sharpe Ratio"],
         "IS Ann. Return": i["Ann. Return"], "OOS Ann. Return": o["Ann. Return"],
         "OOS Max Drawdown": o["Max Drawdown"], "OOS Trades": o["Trades"]}
        for (w, e), (i, o) in periods.items()
    ]).sort_values("IS Sharpe", ascending=False).reset_index(drop=True)
    table.round(4).to_csv(out_dir / "split_results.csv", index=False)

    best = table.iloc[0]
    key = (int(best["window"]), float(best["entry"]))
    is_m, oos_m = periods[key]
    show = ["Ann. Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
            "Cumulative Return", "Buy&Hold Cumulative Return", "Trades", "Time in Market"]
    cmp = pd.DataFrame({"In-sample": is_m, "Out-of-sample": oos_m}).loc[show]

    res = runs[key]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res.index, res["equity"], label="Strategy (net)", color="tab:blue")
    ax.plot(res.index, res["bh_equity"], label="Buy & Hold ratio", color="gray", lw=1)
    ax.axvline(split_ts, color="black", ls="--", lw=1)
    ax.text(split_ts, 0.97, "  out-of-sample ->", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title(f"Selected in-sample: window={key[0]}, entry={key[1]:g}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "split_equity.png", dpi=200)
    plt.close(fig)
    return table, key, cmp


# --------------------------------------------------------------------------- #
# Export / Visualization
# --------------------------------------------------------------------------- #
def plot_and_export(res: pd.DataFrame, metrics: pd.Series, entry: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    ax1.plot(res.index, res["equity"], label="Strategy (net)", color="tab:blue", lw=1.5)
    ax1.plot(res.index, res["bh_equity"], label="Buy & Hold ratio", color="gray", lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log scale)")
    ax1.set_title("VVIX/VIX Mean-Reversion Engine")

    trans = ax1.get_xaxis_transform()
    ax1.fill_between(res.index, 0, 1, where=(res["position"] > 0), transform=trans,
                     color="green", alpha=0.12, label="Long ratio")
    ax1.fill_between(res.index, 0, 1, where=(res["position"] < 0), transform=trans,
                     color="red", alpha=0.12, label="Short ratio")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.text(
        0.99, 0.03,
        f"Sharpe {metrics['Sharpe Ratio']:.2f} | Sortino {metrics['Sortino Ratio']:.2f} | "
        f"MaxDD {metrics['Max Drawdown']:.1%}",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.plot(res.index, res["z"], color="tab:purple", lw=0.9, label="Z-score of VVIX/VIX")
    ax2.axhline(entry, color="red", ls="--", lw=0.8)
    ax2.axhline(-entry, color="green", ls="--", lw=0.8)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    chart_path = out_dir / "vvix_vix_backtest.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    metrics.round(4).to_csv(out_dir / "metrics.csv", header=["value"])
    yearly_returns(res).round(4).to_csv(out_dir / "yearly_returns.csv")
    return chart_path


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)

    end = args.end or date.today().isoformat()  # end is exclusive -> drops today's bar
    df = fetch_data(end=end)
    print(f"\nData: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days), "
          f"end<{end}")

    if args.split_date:
        table, key, cmp = run_split(
            df, args.split_date, args.sweep_windows, args.sweep_entries,
            args.cost_bps, out_dir,
        )
        print(f"\nSplit at {args.split_date}: parameters ranked on in-sample Sharpe only "
              f"(cost {args.cost_bps:g} bps)\n")
        print(table.round(4).to_string(index=False))
        print(f"\nSelected in-sample: window={key[0]} entry={key[1]:g}\n")
        print(cmp.map(lambda v: f"{v:.4f}").to_string())
        pos = (table["OOS Sharpe"] > 0).mean()
        rho = table["IS Sharpe"].corr(table["OOS Sharpe"], method="spearman")
        print(f"\nGrid-wide out-of-sample: median Sharpe {table['OOS Sharpe'].median():.4f}, "
              f"positive in {pos:.0%} of cells, IS-vs-OOS rank correlation {rho:.2f}")
        print(f"\nSaved: {out_dir / 'split_results.csv'}  and  {out_dir / 'split_equity.png'}")
        return

    if args.sweep:
        table = run_sweep(df, args.sweep_windows, args.sweep_entries, args.sweep_costs, out_dir)
        print("\nSweep results (all grid points):\n")
        print(table.round(4).to_string(index=False))
        print(f"\nSaved: {out_dir / 'sweep_results.csv'}  and  "
              f"{out_dir / 'sweep_sharpe_heatmap.png'}")
        return

    res, metrics = evaluate(df, args.window, args.entry, args.cost_bps)
    chart = plot_and_export(res, metrics, args.entry, out_dir)

    print(f"\nVVIX/VIX Mean-Reversion | window={args.window} entry={args.entry} "
          f"cost_bps={args.cost_bps}\n")
    print(metrics.map(lambda v: f"{v:.4f}").to_string())
    print("\nCalendar-year returns:\n")
    print(yearly_returns(res).map(lambda v: f"{v:.2%}").to_string())
    print(f"\nSaved: {chart}, {out_dir / 'metrics.csv'}, {out_dir / 'yearly_returns.csv'}")


if __name__ == "__main__":
    main()
    