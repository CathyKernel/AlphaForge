#!/usr/bin/env python
"""Train the S&P 500 ML alpha models: purged walk-forward LightGBM + Ridge.

Strictly out-of-sample by construction: the 24 price/volume factors are the
ONLY features — the three value factors (ep/bp/sp) are excluded on purpose
because their fundamentals snapshot is static (look-ahead); they live in the
factor table and composite experiments only, where the leakage is disclosed.

Outputs under ``results/ml_run_sp500/``: OOS predictions, per-model IC
tables, run config, feature importances.  Post-processing (plots, PIT
backtest, PSI drift) lives in ``run_sp500_ml_post.py``.

Usage:
    python scripts/run_sp500_ml.py
"""

from __future__ import annotations

import time

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.factors import FactorLibrary
from alphaforge.factors.base import REGISTRY
from alphaforge.ml.features import FeatureConfig, FeaturePanelBuilder
from alphaforge.ml.models import LightGBMAlphaModel, RidgeAlphaModel
from alphaforge.ml.pipeline import AlphaTrainingPipeline
from alphaforge.ml.validation import PurgedWalkForwardCV


def main() -> int:
    out_dir = get_config().results_dir / "ml_run_sp500"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # 1. Features: the 24 price/volume factors (value excluded on purpose) #
    price_volume_factors = sorted(n for n, f in REGISTRY.items() if f.category != "value")
    log(f"{len(price_volume_factors)} price/volume features (value excluded)")

    panel = DataRepository().load_panel("prices_sp500")
    lib = FactorLibrary(panel)
    factor_values = {n: lib.compute(n) for n in price_volume_factors}
    log("factors computed")

    # 2. Purged walk-forward training (LightGBM + Ridge) ------------------ #
    models = [
        LightGBMAlphaModel(n_estimators=400, early_stopping_rounds=40),
        RidgeAlphaModel(),
    ]
    fb = FeaturePanelBuilder(
        panel, factor_names=price_volume_factors, config=FeatureConfig(label_horizon=5)
    )
    pipe = AlphaTrainingPipeline(
        panel,
        models=models,
        cv=PurgedWalkForwardCV(n_splits=6),
        feature_builder=fb,
    )
    result = pipe.run(factor_values=factor_values, verbose=True)
    out = result.save(out_dir)
    log(f"ML run saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
