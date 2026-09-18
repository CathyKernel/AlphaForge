"""PSI / KS feature drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.ml.monitoring import (
    drift_status,
    feature_drift_table,
    ks_statistic,
    psi,
    rolling_feature_psi,
)


def _feature_frame(
    n_days: int = 700,
    n_names: int = 20,
    seed: int = 2,
    shift_after: int | None = None,
    shift: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product(
        [dates, [f"S{i}" for i in range(n_names)]], names=["date", "ticker"]
    )
    x = rng.normal(0, 1, len(idx))
    if shift_after is not None:
        pos = dates.get_indexer(dates[dates > dates[shift_after]])
        row_sel = np.isin(idx.get_level_values("date"), dates[pos])
        x = x + np.where(row_sel, shift, 0.0)
    return pd.DataFrame({"f1": x, "f2": rng.normal(5, 2, len(idx))}, index=idx)


def test_psi_zero_for_identical_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50_000)
    a = rng.normal(0, 1, 50_000)
    assert psi(x, a, n_bins=10) < 0.01


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20_000)
    a = rng.normal(1.0, 1, 20_000)  # 1-SD shift -> massive drift
    assert psi(x, a, n_bins=10) > 0.25


def test_psi_zero_for_constant_feature() -> None:
    assert psi(np.ones(100), np.ones(50), n_bins=10) == 0.0


def test_psi_nan_on_insufficient_sample() -> None:
    assert np.isnan(psi([1.0, 2.0], [1.0, 2.0, 3.0], n_bins=10))


def test_drift_status_thresholds() -> None:
    assert drift_status(0.02) == "stable"
    assert drift_status(0.15) == "moderate"
    assert drift_status(0.40) == "significant"
    assert drift_status(np.nan) == "unknown"


def test_ks_statistic_bounds_and_separation() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 5_000)
    same = rng.normal(0, 1, 5_000)
    far = rng.normal(4, 1, 5_000)
    assert 0.0 <= ks_statistic(x, same) <= 1.0
    assert ks_statistic(x, same) < 0.05
    assert ks_statistic(x, far) > 0.9


def test_feature_drift_table_structure() -> None:
    ref = _feature_frame(n_days=500, seed=3)
    cur = _feature_frame(n_days=500, seed=4, shift_after=250, shift=0.8)
    tab = feature_drift_table(ref, cur)
    assert list(tab.columns) == ["psi", "status", "ks", "mean_ref", "mean_cur", "mean_shift_sd"]
    assert tab["psi"].is_monotonic_decreasing
    # f1 was shifted by 0.8 SD -> must be flagged; f2 untouched -> stable
    assert tab.loc["f1", "status"] in ("moderate", "significant")
    assert tab.loc["f2", "status"] == "stable"


def test_rolling_feature_psi_shape_and_causality() -> None:
    X = _feature_frame(n_days=600, n_names=10, seed=5, shift_after=450, shift=1.5)
    roll = rolling_feature_psi(X, window=200, step=100)
    # 600 days, window 200: evals at i=200,300,400,500 -> up to 4 rows
    assert 2 <= len(roll) <= 5
    assert set(roll.columns) == {"f1", "f2"}
    # the drift regime only exists in the LAST window (cur = [i, i+200)
    # overlaps the shifted tail only for i >= 450-ish); early rows stable
    assert roll["f1"].iloc[0] < 0.1
    # the last evaluation's current window [500, 700) is heavily shifted
    # (shift starts at 450) -> PSI must be significant
    assert roll["f1"].iloc[-1] > 0.10


def test_rolling_psi_needs_enough_history() -> None:
    X = _feature_frame(n_days=100, seed=6)
    with pytest.raises(ValueError):
        rolling_feature_psi(X, window=200, step=50)
