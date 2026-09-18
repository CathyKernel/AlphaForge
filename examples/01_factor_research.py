"""Example 1: quick factor research in ~20 lines.

Evaluates every factor in the library on the bundled dataset and prints
a ranked table of information coefficients.  This is the typical first
session with AlphaForge: load data, ask "which signals actually work?",
and look at the answer in seconds.

Run:  python examples/01_factor_research.py
"""

from alphaforge.data import PricePanel
from alphaforge.factors import FactorEvaluator
from alphaforge.factors.library import FactorLibrary

# 1. load the bundled 10-year S&P 100 panel (clone-and-run, no download)
panel = PricePanel.from_parquet("data/cache/prices.parquet")
print(f"panel: {panel!r}")

# 2. compute all 24 registered factors (cached per session)
library = FactorLibrary(panel)
factors = library.compute_all()

# 3. evaluate: rank IC, ICIR, Newey-West t-stats, quantile spreads
evaluator = FactorEvaluator(panel, horizon=5, quantiles=5, cost_bps=10.0)
stats = evaluator.evaluate_all(factors)

cols = [
    "name",
    "ic_mean",
    "ic_ir",
    "ic_tstat",
    "ic_positive_rate",
    "quantile_spread_ann",
    "monotonicity",
]
print(stats[cols].round(4).to_string(index=False))

# 4. look at signal persistence for the best factor
best = stats.iloc[0]["name"]
decay = evaluator.ic_decay(library.compute(best))
print(f"\nIC decay for {best}:")
print(decay.round(4).to_string())
