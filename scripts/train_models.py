#!/usr/bin/env python
"""Train ML alpha models with purged walk-forward validation.

Usage:
    python scripts/train_models.py                       # LightGBM only
    python scripts/train_models.py --models lightgbm,lstm
    python scripts/train_models.py --splits 4 --horizon 10
"""

from __future__ import annotations

import argparse
import sys
import time

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.factors.library import FactorLibrary
from alphaforge.ml import (
    AlphaTrainingPipeline,
    FeatureConfig,
    FeaturePanelBuilder,
    LightGBMAlphaModel,
    PurgedWalkForwardCV,
    lstm_available,
)
from alphaforge.reporting import plots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="lightgbm", help="comma-separated: lightgbm,lstm")
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--lstm-epochs", dest="lstm_epochs", type=int, default=25)
    args = ap.parse_args()

    panel = DataRepository().load_panel()
    print(f"loaded panel: {panel!r}")

    lib = FactorLibrary(panel)
    print("computing 24 factors ...")
    mats = lib.compute_all()

    models = []
    names = [m.strip().lower() for m in args.models.split(",")]
    if "lightgbm" in names:
        models.append(LightGBMAlphaModel())
    if "lstm" in names:
        if not lstm_available():
            print("PyTorch not installed; skipping LSTM (pip install alphaforge[dl])")
        else:
            from alphaforge.ml import LSTMAlphaModel

            models.append(LSTMAlphaModel(max_epochs=args.lstm_epochs))
    if not models:
        print("no models selected")
        return 1

    cv = PurgedWalkForwardCV(n_splits=args.splits, purge=args.horizon + 5, embargo=5)
    fb = FeaturePanelBuilder(panel, config=FeatureConfig(label_horizon=args.horizon))

    print(f"training {len(models)} model(s) on {args.splits} walk-forward folds")
    t0 = time.time()
    pipeline = AlphaTrainingPipeline(panel, models, cv=cv, feature_builder=fb)
    result = pipeline.run(factor_values=mats)
    elapsed = time.time() - t0
    print(f"\nfinished in {elapsed / 60:.1f} min")

    out_dir = get_config().results_dir / "ml_run"
    result.save(out_dir)

    print("\n=== out-of-sample rank IC ===")
    print(result.ic_summary.round(4).to_string())

    # charts
    ens = "ensemble" if "ensemble" in result.ic_by_date.columns else result.ic_by_date.columns[0]
    fig = plots.plot_ic_series(result.ic_by_date[ens], title=f"OOS Daily Rank IC - {ens}")
    plots.save(fig, out_dir / f"ic_daily_{ens}.png")
    fig = plots.plot_ic_series(result.ic_by_date[ens], cum=True)
    plots.save(fig, out_dir / f"ic_cumulative_{ens}.png")
    for mname, imp in result.importances.items():
        fig = plots.plot_feature_importance(imp, title=f"Feature importance - {mname}")
        plots.save(fig, out_dir / f"importance_{mname}.png")
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
