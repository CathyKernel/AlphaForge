#!/usr/bin/env python
"""Incremental LSTM training with per-fold checkpointing.

Sequence models are the slowest part of the pipeline; this utility trains
the LSTM one walk-forward fold at a time, checkpointing OOS predictions
after each fold, so interrupted runs resume without losing work.  After
the final fold it merges the LSTM predictions with any existing model run
(e.g. `results/ml_run` from the LightGBM pipeline) into a combined
result set including the rank-average ensemble.

Usage:
    python scripts/train_lstm_incremental.py [--splits 6] [--epochs 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from alphaforge.config import get_config
from alphaforge.data import PricePanel
from alphaforge.factors.library import FactorLibrary
from alphaforge.ml import FeatureConfig, FeaturePanelBuilder, LSTMAlphaModel, PurgedWalkForwardCV
from alphaforge.ml.pipeline import AlphaTrainingPipeline, _clone_model

RESULTS = get_config().results_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=40)
    ap.add_argument(
        "--merge-dir", default=str(RESULTS / "ml_run"), help="existing run to merge with"
    )
    args = ap.parse_args()

    out_dir = RESULTS / "ml_run_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = PricePanel.from_parquet(get_config().cache_dir / "prices.parquet")
    print(f"panel: {panel!r}", flush=True)
    factors = FactorLibrary(panel).compute_all()
    fb = FeaturePanelBuilder(panel, config=FeatureConfig(label_horizon=5))
    X, y, _ = fb.build(factors)
    dates = pd.DatetimeIndex(sorted(set(y.index.get_level_values(0))))
    folds = PurgedWalkForwardCV(n_splits=args.splits, purge=10, embargo=5).split(dates)

    model_tmpl = LSTMAlphaModel(
        lookback=args.lookback,
        hidden_size=48,
        num_layers=2,
        max_epochs=args.epochs,
        patience=3,
        batch_size=2048,
    )
    lstm_pred_chunks = []
    for k, fold in enumerate(folds):
        ckpt = out_dir / f"preds_fold_{k + 1}.parquet"
        if ckpt.exists():
            try:
                chunk = pd.read_parquet(ckpt)["lstm"]
                lstm_pred_chunks.append(chunk)
                print(f"fold {k + 1}: checkpoint found, skipping", flush=True)
                continue
            except Exception:
                print(f"fold {k + 1}: corrupt checkpoint, retraining", flush=True)
                ckpt.unlink()

        train_dates = dates[fold.train]
        test_dates = dates[fold.test]
        n_val = max(int(len(train_dates) * 0.15), 20)
        val_dates, fit_dates = train_dates[-n_val:], train_dates[:-n_val]

        X_fit = X[X.index.get_level_values(0).isin(fit_dates)]
        y_fit = y[y.index.get_level_values(0).isin(fit_dates)]
        X_val = X[X.index.get_level_values(0).isin(val_dates)]
        y_val = y[y.index.get_level_values(0).isin(val_dates)]

        hist_start = test_dates[0] - pd.tseries.offsets.BDay(model_tmpl.lookback + 15)
        X_pred = X.loc[
            (X.index.get_level_values(0) >= hist_start)
            & (X.index.get_level_values(0) <= test_dates[-1])
        ]
        in_test = X_pred.index.get_level_values(0).isin(test_dates)

        print(f"fold {k + 1}/{len(folds)}: {fold.description}", flush=True)
        t0 = time.time()
        fresh = _clone_model(model_tmpl)
        fresh.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
        preds = fresh.predict(X_pred)
        preds = preds[in_test]
        tmp = ckpt.with_suffix(".tmp.parquet")
        preds.rename("lstm").to_frame().to_parquet(tmp)
        tmp.rename(ckpt)  # atomic: never leave a half-written checkpoint
        lstm_pred_chunks.append(preds.rename("lstm"))
        print(
            f"fold {k + 1} done in {time.time() - t0:.0f}s ({len(preds)} predictions)", flush=True
        )

    # ---------------- merge with the existing run ---------------------- #
    print("\nmerging with existing run ...", flush=True)
    lstm = pd.concat(lstm_pred_chunks).sort_index().rename("lstm")
    merge_path = Path(args.merge_dir) / "oos_predictions.parquet"
    if merge_path.exists():
        existing = pd.read_parquet(merge_path)
        existing = existing.drop(columns=["lstm", "ensemble"], errors="ignore")
        combined = existing.join(lstm, how="outer")
    else:
        combined = lstm.to_frame()

    label = y.reindex(combined.index).rename("label")
    # rank-average ensemble over available models, per date
    model_cols = [c for c in combined.columns if c != "label"]
    ranks = pd.concat([combined[m].groupby(level=0).rank(pct=True) for m in model_cols], axis=1)
    combined["ensemble"] = ranks.mean(axis=1)

    final_dir = RESULTS / "ml_run"
    final_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(final_dir / "oos_predictions.parquet")

    # recompute IC diagnostics for every model + ensemble
    pipe_summary = AlphaTrainingPipeline._ic_by_date(combined, label)
    pipe_summary.to_csv(final_dir / "ic_by_date.csv")
    ic_summary = AlphaTrainingPipeline._ic_summary(pipe_summary)
    ic_summary.to_csv(final_dir / "ic_summary.csv")
    print("\n=== merged out-of-sample rank IC ===")
    print(ic_summary.round(4).to_string(), flush=True)

    cfg = {
        "models": model_cols + ["ensemble"],
        "n_splits": args.splits,
        "lstm": {
            "lookback": args.lookback,
            "max_epochs": args.epochs,
            "hidden_size": 48,
            "num_layers": 2,
        },
        "merged_from": str(args.merge_dir),
    }
    (final_dir / "run_config.json").write_text(pd.Series(cfg).to_json(indent=2))
    print(f"\nfinal artifacts: {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
