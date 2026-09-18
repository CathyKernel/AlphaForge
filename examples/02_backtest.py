"""Example 2: backtest a strategy end-to-end.

Builds the top-5 composite (by |ICIR|, direction-corrected) and runs it
through the vectorised engine with realistic assumptions: T+1 execution,
drifting weights, 10 bps per-side costs, liquidity filters and SPY as
benchmark.  Produces a tearsheet with equity curves, drawdowns, monthly
heatmap and a full metrics JSON.

Run:  python examples/02_backtest.py
"""

from pathlib import Path

from alphaforge.data import PricePanel
from alphaforge.engine import run_backtest
from alphaforge.factors import FactorEvaluator
from alphaforge.factors.library import FactorLibrary
from alphaforge.reporting.tearsheet import tearsheet

panel = PricePanel.from_parquet("data/cache/prices.parquet")
library = FactorLibrary(panel)

# rank factors by |IC IR| and keep the top five
stats = FactorEvaluator(panel, horizon=5).evaluate_all(library.compute_all())
ranked = stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
top = ranked["name"].head(5).tolist()

# direction-correct each leg: flip factors whose IC is negative
direction = {
    n: (1.0 if s.ic_mean > 0 else -1.0) for n, s in ranked.set_index("name").loc[top].iterrows()
}
signal = library.composite(names=top, direction=direction)
print(f"composite of {top}")

# monthly rebalance, long-only top-15, SPY benchmark, realistic costs
result = run_backtest(
    panel, signal, signal_name="composite_top5", rebalance="MS", top_n=15, side="long"
)

summary = tearsheet(result, Path("results") / "example_backtest", name="example")
s = summary["performance"]
print(
    f"net CAGR {s['cagr']:+.2%} (SPY {s['benchmark_cagr']:+.2%}), "
    f"Sharpe {s['sharpe']:.2f}, MaxDD {s['max_drawdown']:.1%}, "
    f"IR {s['information_ratio']:+.2f}"
)
