"""Example 4: train ML alpha models with leakage-safe validation.

Walks six purged folds over ten years of data: LightGBM and a PyTorch
LSTM are retrained per fold and predict strictly out-of-sample blocks.
The output is an OOS rank-IC table per model plus a rank-average
ensemble -- the honest number to report for any ML alpha.

Run:  python examples/04_ml_pipeline.py            # LightGBM only (fast)
      python examples/04_ml_pipeline.py --lstm     # add the LSTM (slow)
"""

import argparse

from alphaforge.data import PricePanel
from alphaforge.factors.library import FactorLibrary
from alphaforge.ml import (
    AlphaTrainingPipeline,
    FeatureConfig,
    FeaturePanelBuilder,
    LightGBMAlphaModel,
    PurgedWalkForwardCV,
)

ap = argparse.ArgumentParser()
ap.add_argument("--lstm", action="store_true", help="also train the LSTM")
ap.add_argument("--splits", type=int, default=6)
args = ap.parse_args()

panel = PricePanel.from_parquet("data/cache/prices.parquet")
factors = FactorLibrary(panel).compute_all()

models = [LightGBMAlphaModel()]
if args.lstm:
    from alphaforge.ml import LSTMAlphaModel

    models.append(LSTMAlphaModel(max_epochs=12, patience=4))

cv = PurgedWalkForwardCV(n_splits=args.splits, purge=10, embargo=5)
features = FeaturePanelBuilder(panel, config=FeatureConfig(label_horizon=5))

pipeline = AlphaTrainingPipeline(panel, models, cv=cv, feature_builder=features)
result = pipeline.run(factor_values=factors)

print("\n=== out-of-sample rank IC ===")
print(result.ic_summary.round(4).to_string())

# the OOS ensemble signal is immediately backtestable
if "ensemble" in result.predictions.columns:
    from alphaforge.engine import run_backtest

    signal = result.signal("ensemble")
    res = run_backtest(
        panel, signal, signal_name="ml_ensemble_oos", rebalance="W-FRI", top_n=15, side="long_short"
    )
    s = res.summary()
    print(
        f"\nOOS ensemble L/S backtest: CAGR {s['cagr']:+.2%}, "
        f"Sharpe {s['sharpe']:.2f}, IR {s['information_ratio']:+.2f}"
    )
