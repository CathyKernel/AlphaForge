"""Alpha models: scikit-learn-style estimators returning cross-sectional scores.

All models share one interface so the walk-forward pipeline can treat
LightGBM, the PyTorch LSTM and their ensemble interchangeably:

    fit(X, y, X_val=None, y_val=None)  -> self
    predict(X)                         -> pd.Series aligned with X.index
    feature_importance()               -> pd.Series | None
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AlphaModel(ABC):
    """Base class for alpha models."""

    name: str = "model"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> AlphaModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    def feature_importance(self) -> pd.Series | None:  # noqa: RUF100
        return None


class LightGBMAlphaModel(AlphaModel):
    """Gradient-boosted trees on cross-sectional features.

    Regression on forward excess returns (the practitioner-standard
    formulation for cross-sectional alpha).  Hyper-parameters are chosen
    for regularisation on panels with ~1e5-1e6 rows: shallow trees,
    row/column subsampling and early stopping on a chronological
    validation split.
    """

    name = "lightgbm"

    def __init__(
        self,
        params: dict | None = None,
        early_stopping_rounds: int = 50,
        n_estimators: int = 800,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
        if params:
            # accept the sklearn-style alias for convenience
            n_estimators = int(params.get("n_estimators", n_estimators))
            params = {k: v for k, v in params.items() if k != "n_estimators"}
            self.params.update(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._booster = None
        self._best_iter: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LightGBMAlphaModel:
        import lightgbm as lgb

        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None
        if X_val is not None and y_val is not None:
            valid_sets = [lgb.Dataset(X_val, label=y_val)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self._booster = lgb.train(
            {**self.params, "seed": self.random_state},
            lgb.Dataset(X, label=y),
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self._best_iter = self._booster.best_iteration
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._booster.predict(X)
        return pd.Series(preds, index=X.index, name=self.name)

    def feature_importance(self) -> pd.Series:
        imp = pd.Series(
            self._booster.feature_importance("gain"), index=self._booster.feature_name()
        )
        total = imp.sum()
        return (imp / total).sort_values(ascending=False) if total > 0 else imp


class RidgeAlphaModel(AlphaModel):
    """Linear ridge baseline on cross-sectional features.

    The classical Fama-French-style linear workhorse: fast, exactly
    reproducible, and a sanity bound — if a boosted tree cannot beat this,
    the signal is mostly linear (or the tree is overfitting).
    """

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self._model = None
        self._features: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> RidgeAlphaModel:
        from sklearn.linear_model import Ridge

        self._features = list(X.columns)
        self._model = Ridge(alpha=self.alpha).fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X.to_numpy(dtype=float))
        return pd.Series(preds, index=X.index, name=self.name)

    def feature_importance(self) -> pd.Series:
        """|standardised coefficient| — the linear model's importance."""
        import numpy as np

        coefs = np.abs(self._model.coef_)
        return (
            pd.Series(coefs / coefs.sum(), index=self._features).sort_values(ascending=False)
            if coefs.sum() > 0
            else pd.Series(coefs, index=self._features)
        )


def causal_ensemble(
    predictions: pd.DataFrame,
    label: pd.Series,
    fold_ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.Series:
    """IC-weighted ensemble whose weights only use *past* OOS folds.

    For fold ``k``, each model's weight is proportional to its mean rank IC
    over folds ``< k`` (clipped at zero; equal weights for the first fold).
    The blend is applied per fold on within-date percentile ranks.  Unlike
    a full-sample rank-average — which silently benefits from knowing all
    models' whole-history performance — this construction is strictly
    causal and could have been run live.

    Parameters
    ----------
    predictions:
        MultiIndex (date, ticker) frame with one column per model.
    label:
        The forward-return label aligned to ``predictions``.
    fold_ranges:
        ``(test_start, test_end)`` per walk-forward fold, in order.
    """
    models = [c for c in predictions.columns if c not in ("ensemble", "ensemble_causal", "label")]
    if len(models) < 2:
        raise ValueError("causal_ensemble needs >= 2 model columns")

    df = predictions.copy()
    df["_label"] = label.reindex(df.index)

    def _mean_ic(frame: pd.DataFrame) -> pd.Series:
        """Mean within-date rank IC per model column."""
        label_ranks = frame["_label"].groupby(level=0).rank(pct=True)
        out = {}
        for m in models:
            ranks = frame[m].groupby(level=0).rank(pct=True)
            per_date = (
                pd.concat([ranks.rename("r"), label_ranks.rename("l")], axis=1)
                .groupby(level=0)
                .apply(lambda g: g["r"].corr(g["l"]) if len(g) > 2 else np.nan)
            )
            out[m] = float(per_date.mean())
        return pd.Series(out)

    parts: list[pd.Series] = []
    for k, (start, end) in enumerate(fold_ranges):
        dates_lvl = df.index.get_level_values(0)
        frame = df.loc[(dates_lvl >= start) & (dates_lvl <= end)]
        if frame.empty:
            continue
        if k == 0:
            weights = pd.Series(1.0 / len(models), index=models)
        else:
            past = df.loc[dates_lvl < start]
            weights = _mean_ic(past).clip(lower=0.0)
            if weights.isna().any() or weights.sum() <= 0:
                weights = pd.Series(1.0 / len(models), index=models)
            else:
                weights = weights / weights.sum()
        ranks = frame[models].groupby(level=0).rank(pct=True)
        blend = ranks.mul(weights, axis=1).sum(axis=1)
        parts.append(pd.Series(blend.to_numpy(), index=frame.index, name="ensemble_causal"))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


class EnsembleAlphaModel(AlphaModel):
    """Rank-average ensemble of sub-models.

    Averaging *ranks* (rather than raw scores) makes the blend robust to
    arbitrary scale differences between model outputs -- standard practice
    for combining heterogeneous alpha signals.
    """

    name = "ensemble"

    def __init__(self, models: list[AlphaModel]) -> None:
        if not models:
            raise ValueError("Ensemble requires at least one sub-model")
        self.models = models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> EnsembleAlphaModel:
        for m in self.models:
            m.fit(X, y, X_val=X_val, y_val=y_val)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = [m.predict(X) for m in self.models]
        ranks = [p.groupby(level=0).rank(pct=True) for p in preds]
        avg = sum(ranks) / len(ranks)
        return pd.Series(avg.to_numpy(), index=X.index, name=self.name)

    def feature_importance(self) -> pd.Series | None:
        """Mean of sub-model importances when every sub-model provides one."""
        imps = [m.feature_importance() for m in self.models]
        imps = [i for i in imps if i is not None]
        if not imps:
            return None
        df = pd.concat(imps, axis=1)
        return df.mean(axis=1).sort_values(ascending=False)
