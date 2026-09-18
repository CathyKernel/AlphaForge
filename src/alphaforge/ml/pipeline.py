"""End-to-end alpha model training with walk-forward validation.

The pipeline:

1. builds the (features, labels) panel from the factor library;
2. walks the purged CV folds chronologically;
3. in each fold, trains every model on the (purged) training window with
   a chronological validation tail for early stopping;
4. predicts the out-of-sample test block and stores the predictions;
5. aggregates diagnostics: per-model OOS rank IC series, ICIR, Newey-West
   t-stats, feature importances (averaged over folds) and a rank-average
   ensemble.

Every reported number is strictly out-of-sample: a model trained on data
through ``t`` predicts only dates it has never seen, with purged labels
and embargoed features.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import cs_winsorize, cs_zscore
from alphaforge.ml.features import FeatureConfig, FeaturePanelBuilder
from alphaforge.ml.models import AlphaModel, causal_ensemble
from alphaforge.ml.validation import PurgedWalkForwardCV

log = logging.getLogger(__name__)


@dataclass
class MLPipelineResult:
    """Artifacts of a walk-forward training run."""

    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    label: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_by_date: pd.DataFrame = field(default_factory=pd.DataFrame)
    ic_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    importances: dict[str, pd.Series] = field(default_factory=dict)
    fold_log: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def signal(self, model: str = "ensemble") -> pd.DataFrame:
        """OOS predictions as a wide date x ticker signal for the backtester."""
        if model not in self.predictions.columns:
            raise KeyError(f"model {model!r} not in {list(self.predictions.columns)}")
        wide = self.predictions[model].unstack("ticker")
        return cs_zscore(cs_winsorize(wide))

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(out / "oos_predictions.parquet")
        self.ic_summary.to_csv(out / "ic_summary.csv")
        self.ic_by_date.to_csv(out / "ic_by_date.csv")
        for name, imp in self.importances.items():
            imp.rename("importance").to_csv(out / f"importance_{name}.csv")
        (out / "run_config.json").write_text(
            json.dumps({"config": self.config, "folds": self.fold_log}, indent=2, default=str)
        )
        return out


class AlphaTrainingPipeline:
    """Orchestrates feature building, CV, training and OOS evaluation.

    Parameters
    ----------
    panel:
        Price panel of the universe.
    models:
        List of unfitted :class:`AlphaModel` instances (fresh copies are
        trained per fold, so the passed instances act as templates).
    cv:
        Purged walk-forward splitter.
    feature_builder:
        Feature panel builder (default: all registered factors).
    lstm_history:
        Extra calendar days of feature history passed to ``predict`` so
        sequence models can build their lookback windows.
    """

    def __init__(
        self,
        panel: PricePanel,
        models: list[AlphaModel],
        cv: PurgedWalkForwardCV | None = None,
        feature_builder: FeaturePanelBuilder | None = None,
        lstm_history: int = 75,
    ) -> None:
        self.panel = panel
        self.models = models
        self.cv = cv or PurgedWalkForwardCV(n_splits=6)
        self.fb = feature_builder or FeaturePanelBuilder(
            panel, config=FeatureConfig(label_horizon=5)
        )
        self.lstm_history = int(lstm_history)

    # ------------------------------------------------------------------ #
    def run(
        self, factor_values: dict[str, pd.DataFrame] | None = None, verbose: bool = True
    ) -> MLPipelineResult:
        X, y, _ = self.fb.build(factor_values)
        dates = pd.DatetimeIndex(sorted(set(y.index.get_level_values(0))))
        folds = self.cv.split(dates)

        preds: dict[str, list[pd.Series]] = {}
        importances: dict[str, list[pd.Series]] = {m.name: [] for m in self.models}
        fold_log: list[str] = []
        fold_ranges: list[tuple] = []

        for k, fold in enumerate(folds):
            train_dates = dates[fold.train]
            test_dates = dates[fold.test]
            # chronological validation tail (15%) for early stopping
            n_val = max(int(len(train_dates) * 0.15), 20)
            val_dates = train_dates[-n_val:]
            fit_dates = train_dates[:-n_val]

            X_fit = X.loc[X.index.get_level_values(0).isin(fit_dates)]
            y_fit = y.loc[y.index.get_level_values(0).isin(fit_dates)]
            X_val = X.loc[X.index.get_level_values(0).isin(val_dates)]
            y_val = y.loc[y.index.get_level_values(0).isin(val_dates)]

            # history-extended slice for prediction (sequence models)
            hist_start = test_dates[0] - pd.tseries.offsets.BDay(self.lstm_history)
            X_pred = X.loc[
                (X.index.get_level_values(0) >= hist_start)
                & (X.index.get_level_values(0) <= test_dates[-1])
            ]
            in_test = X_pred.index.get_level_values(0).isin(test_dates)

            for model in self.models:
                fresh = _clone_model(model)
                fresh.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
                p = fresh.predict(X_pred)
                p = p[in_test] if isinstance(p, pd.Series) else p
                preds.setdefault(model.name, []).append(p)
                imp = fresh.feature_importance()
                if imp is not None:
                    importances[model.name].append(imp)
            fold_log.append(f"fold {k + 1}/{len(folds)}: {fold.description}")
            fold_ranges.append((test_dates[0], test_dates[-1]))
            if verbose:
                log.info("fold %d/%d done", k + 1, len(folds))
                print(f"  {fold_log[-1]}", flush=True)

        # concatenate OOS predictions per model
        pred_cols = {name: pd.concat(chunks).sort_index() for name, chunks in preds.items()}
        pred_frame = pd.DataFrame(pred_cols)
        label = y.reindex(pred_frame.index).rename("label")

        # rank-average ensemble across models (per date)
        model_names = list(pred_cols)
        if len(model_names) > 1:
            ranks = pd.concat(
                [pred_frame[m].groupby(level=0).rank(pct=True) for m in model_names], axis=1
            )
            pred_frame["ensemble"] = ranks.mean(axis=1)
            # strictly-causal IC-weighted variant: weights for fold k use
            # only OOS performance from folds < k
            pred_frame["ensemble_causal"] = causal_ensemble(pred_frame, label, fold_ranges)

        result = MLPipelineResult(
            predictions=pred_frame,
            label=label,
            importances={
                n: pd.concat(imps, axis=1).mean(axis=1) for n, imps in importances.items() if imps
            },
            fold_log=fold_log,
            config={
                "n_splits": self.cv.n_splits,
                "mode": self.cv.mode,
                "purge": self.cv.purge,
                "embargo": self.cv.embargo,
                "label_horizon": self.fb.config.label_horizon,
                "models": model_names,
                "fold_ranges": [
                    [str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date())]
                    for a, b in fold_ranges
                ],
            },
        )
        result.ic_by_date = self._ic_by_date(pred_frame, label)
        result.ic_summary = self._ic_summary(result.ic_by_date)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ic_by_date(pred_frame: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
        """Daily rank IC of each model's OOS predictions vs the label."""
        out = {}
        for m in pred_frame.columns:
            wide = pred_frame[m].unstack("ticker")
            lab = label.unstack("ticker").reindex(index=wide.index, columns=wide.columns)
            from alphaforge.factors.evaluation import row_spearman

            min_obs = max(3, min(10, wide.shape[1] // 2 + 1))
            ic = row_spearman(wide, lab, min_obs=min_obs)
            out[m] = ic
        return pd.DataFrame(out).sort_index()

    @staticmethod
    def _ic_summary(ic_by_date: pd.DataFrame) -> pd.DataFrame:
        from alphaforge.factors.evaluation import newey_west_tstat

        rows = []
        for m in ic_by_date.columns:
            ic = ic_by_date[m].dropna()
            rows.append(
                {
                    "model": m,
                    "ic_mean": float(ic.mean()),
                    "ic_std": float(ic.std()),
                    "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
                    "ic_tstat": newey_west_tstat(ic),
                    "ic_positive_rate": float((ic > 0).mean()),
                    "n_days": int(len(ic)),
                }
            )
        return (
            pd.DataFrame(rows)
            .set_index("model")
            .sort_values("ic_ir", key=lambda s: s.abs(), ascending=False)
        )


def _clone_model(model: AlphaModel) -> AlphaModel:
    """Create an unfitted copy of a template model."""
    import copy

    clone = copy.deepcopy(model)
    # reset fitted state
    for attr in ("_booster", "_net", "_best_iter"):
        if hasattr(clone, attr):
            setattr(clone, attr, None)
    return clone
