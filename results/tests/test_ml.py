"""Tests for the ML layer: features, purged CV, models, pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.ml import (
    AlphaTrainingPipeline,
    FeatureConfig,
    FeaturePanelBuilder,
    LightGBMAlphaModel,
    PurgedWalkForwardCV,
)


# ---------------- feature panel ------------------------------------------ #
def test_label_is_forward_return(planted_panel) -> None:
    fb = FeaturePanelBuilder(
        planted_panel, factor_names=["mom_21"], config=FeatureConfig(label_horizon=5)
    )
    X, y, raw = fb.build()
    h = 5
    close = planted_panel.close
    manual = (close.shift(-(h + 1)) / close.shift(-1) - 1).stack(future_stack=True)
    common = y.index
    # raw forward return matches the manual formula exactly
    np.testing.assert_allclose(
        raw.reindex(common).to_numpy(), manual.reindex(common).to_numpy(), equal_nan=True
    )
    # excess label = raw minus the per-date cross-sectional mean
    demeaned = raw.reindex(common) - raw.reindex(common).groupby(level=0).transform("mean")
    np.testing.assert_allclose(y.reindex(common).to_numpy(), demeaned.to_numpy(), atol=1e-12)


def test_features_standardised(planted_panel) -> None:
    fb = FeaturePanelBuilder(planted_panel, factor_names=["mom_21", "vol_21"])
    X, _, _ = fb.build()
    for _date, group in X.groupby(level=0):
        for col in X.columns:
            assert abs(group[col].mean()) < 1e-9
            assert abs(group[col].std(ddof=0) - 1.0) < 1e-6 or group[col].std(ddof=0) < 1e-9


# ---------------- purged walk-forward CV --------------------------------- #
def _dates(n: int = 1200) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def test_walk_forward_train_before_test() -> None:
    cv = PurgedWalkForwardCV(n_splits=5, purge=10, embargo=5, min_train_days=400)
    folds = cv.split(_dates())
    assert len(folds) == 5
    for f in folds:
        assert f.train.max() < f.test.min()
        # purge gap: no training sample within `purge` days before test
        assert (f.test.min() - f.train.max()) >= 10


def test_purged_kfold_excludes_test_block() -> None:
    cv = PurgedWalkForwardCV(
        n_splits=4, purge=10, embargo=7, min_train_days=400, mode="purged_kfold"
    )
    folds = cv.split(_dates())
    for f in folds:
        inside = (f.train >= f.test.min()) & (f.train <= f.test.max())
        assert not inside.any()
        embargoed = (f.train > f.test.max()) & (f.train <= f.test.max() + 7)
        assert not embargoed.any()


def test_folds_cover_all_dates_exactly_once() -> None:
    cv = PurgedWalkForwardCV(n_splits=5, min_train_days=300)
    folds = cv.split(_dates(1000))
    covered = np.concatenate([f.test for f in folds])
    assert len(covered) == len(set(covered)) + 300 - 300  # unique
    assert covered.max() == 999


# ---------------- models -------------------------------------------------- #
def test_lightgbm_learns_planted_signal(planted_panel) -> None:
    """A real signal must beat noise: OOS IC of a planted drift must be high."""
    fb = FeaturePanelBuilder(
        planted_panel,
        factor_names=["mom_21", "mom_63", "vol_21"],
        config=FeatureConfig(label_horizon=5),
    )
    X, y, _ = fb.build()
    dates = pd.DatetimeIndex(sorted(set(y.index.get_level_values(0))))
    cv = PurgedWalkForwardCV(n_splits=3, purge=10, min_train_days=90)
    folds = cv.split(dates)
    f = folds[0]
    fit_mask = y.index.get_level_values(0).isin(dates[f.train])
    test_mask = y.index.get_level_values(0).isin(dates[f.test])
    model = LightGBMAlphaModel(
        params={"num_leaves": 8, "min_child_samples": 20, "n_estimators": 60}
    ).fit(X[fit_mask], y[fit_mask])
    pred = model.predict(X[test_mask])
    ic = pred.groupby(level=0).corr(y[test_mask], method="spearman")
    assert ic.mean() > 0.25  # strong planted signal


def test_lightgbm_ignores_pure_noise() -> None:
    """On pure noise features, in-sample correlation stays near zero."""
    rng = np.random.default_rng(5)
    n_dates = 500
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = list("ABCDEFGH")
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(index)
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=index, columns=[f"f{i}" for i in range(3)])
    y = pd.Series(rng.normal(size=n), index=index)
    model = LightGBMAlphaModel(
        params={"num_leaves": 4, "min_child_samples": 50, "n_estimators": 30}
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert abs(np.corrcoef(pred.to_numpy(), y.to_numpy())[0, 1]) < 0.3


@pytest.mark.slow
def test_lstm_runs_and_predicts(planted_panel) -> None:
    pytest.importorskip("torch")
    from alphaforge.ml import LSTMAlphaModel

    fb = FeaturePanelBuilder(
        planted_panel, factor_names=["mom_21", "mom_63"], config=FeatureConfig(label_horizon=5)
    )
    X, y, _ = fb.build()
    dates = pd.DatetimeIndex(sorted(set(y.index.get_level_values(0))))
    cv = PurgedWalkForwardCV(n_splits=2, purge=10, min_train_days=120)
    f = cv.split(dates)[0]
    fit_mask = y.index.get_level_values(0).isin(dates[f.train])
    test_mask = y.index.get_level_values(0).isin(dates[f.test])
    model = LSTMAlphaModel(lookback=20, hidden_size=16, num_layers=1, max_epochs=2, batch_size=256)
    model.fit(X[fit_mask], y[fit_mask])
    # sequence models need lookback history before the first test date
    hist_start = dates[f.test][0] - pd.tseries.offsets.BDay(30)
    X_ext = X.loc[(X.index.get_level_values(0) >= hist_start)]
    pred = model.predict(X_ext)
    pred = pred[pred.index.get_level_values(0).isin(dates[f.test])]
    assert len(pred) == int(test_mask.sum())
    assert pred.notna().mean() > 0.9


# ---------------- pipeline ------------------------------------------------ #
def test_pipeline_end_to_end(planted_panel) -> None:
    fb = FeaturePanelBuilder(
        planted_panel,
        factor_names=["mom_21", "mom_63", "vol_21"],
        config=FeatureConfig(label_horizon=5),
    )
    cv = PurgedWalkForwardCV(n_splits=3, purge=10, min_train_days=80)
    models = [
        LightGBMAlphaModel(params={"num_leaves": 8, "min_child_samples": 20, "n_estimators": 40})
    ]
    pipe = AlphaTrainingPipeline(planted_panel, models, cv=cv, feature_builder=fb)
    res = pipe.run(verbose=False)
    # OOS predictions exist for every test date
    dates_pred = res.predictions.index.get_level_values(0).unique()
    assert len(dates_pred) > 100
    assert res.ic_summary.loc["lightgbm", "n_days"] > 50
    assert "ic_mean" in res.ic_summary.columns
