#!/usr/bin/env python
"""Run a factor or composite backtest and generate a tearsheet.

Usage:
    python scripts/run_backtest.py                                  # top5 composite
    python scripts/run_backtest.py --factor vol_21                  # single factor
    python scripts/run_backtest.py --rebalance MS --side long       # long-only
"""

from __future__ import annotations

import argparse
import sys

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.engine import run_backtest
from alphaforge.factors import FactorEvaluator
from alphaforge.factors.library import FactorLibrary
from alphaforge.reporting.tearsheet import tearsheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor", default=None, help="single factor name")
    ap.add_argument("--composite", choices=["all", "top5"], default="top5")
    ap.add_argument("--rebalance", default="W-FRI")
    ap.add_argument("--top-n", dest="top_n", type=int, default=15)
    ap.add_argument("--side", choices=["long", "short", "long_short"], default="long_short")
    ap.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    ap.add_argument("--vol-target", dest="vol_target", type=float, default=None)
    ap.add_argument(
        "--horizon", type=int, default=5, help="IC horizon for composite factor selection"
    )
    args = ap.parse_args()

    panel = DataRepository().load_panel()
    print(f"loaded panel: {panel!r}")
    lib = FactorLibrary(panel)

    if args.factor:
        signal = lib.compute(args.factor)
        name = args.factor
        print(f"signal: single factor {name}")
    else:
        mats = lib.compute_all()
        stats = FactorEvaluator(panel, horizon=args.horizon).evaluate_all(mats)
        if args.composite == "top5":
            ranked = stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
            top = ranked["name"].head(5).tolist()
            direction = {
                n: (1.0 if s.ic_mean > 0 else -1.0)
                for n, s in ranked.set_index("name").loc[top].iterrows()
            }
            signal = lib.composite(names=top, direction=direction)
            name = "composite_top5"
        else:
            signal = lib.composite()
            name = "composite_all"
        print(f"signal: {name} ({signal.shape[0]} dates x {signal.shape[1]} tickers)")

    result = run_backtest(
        panel,
        signal,
        signal_name=name,
        rebalance=args.rebalance,
        top_n=args.top_n,
        side=args.side,
        cost_bps=args.cost_bps,
        vol_target=args.vol_target,
    )

    out_dir = get_config().results_dir / f"backtest_{name}"
    metrics = tearsheet(result, out_dir, name=name)
    s = metrics["performance"]
    print("\n=== summary ===")
    print(
        f"period      {result.net_returns.index[0]:%Y-%m-%d} .. "
        f"{result.net_returns.index[-1]:%Y-%m-%d}"
    )
    print(f"net CAGR    {s['cagr']:+.2%}   (benchmark {s['benchmark_cagr']:+.2%})")
    print(f"sharpe      {s['sharpe']:.2f}    (benchmark {s['benchmark_sharpe']:.2f})")
    print(f"max DD      {s['max_drawdown']:.1%}")
    print(f"info ratio  {s['information_ratio']:+.2f}")
    print(
        f"avg TO      {result.turnover.mean():.2f} per rebalance, "
        f"cost drag {metrics['turnover']['cost_drag_ann']:.2%}/yr"
    )
    print(f"\nreport: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
