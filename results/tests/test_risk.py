"""Tests for risk analytics and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.engine.metrics import (
    cvar_historical,
    performance_summary,
    var_cornish_fisher,
    var_historical,
)
from alphaforge.risk.analytics import drawdown_episodes, risk_summary, rolling_risk, stress_test


def _rets(seed: int = 3, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=pd.bdate_range("2019-01-01", periods=n))


def _big_rets(seed: int = 1, n: int = 200_000) -> pd.Series:
    """Large sample without a datetime index (avoids 2262 overflow)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_var_historical_approx_normal() -> None:
    rets = _big_rets(seed=1)
    v95 = var_historical(rets, 0.95)
    assert 0.014 < v95 < 0.019  # ~1.645 * 0.01


def test_cvar_ge_var() -> None:
    rets = _rets()
    assert cvar_historical(rets, 0.95) >= var_historical(rets, 0.95)


def test_cornish_fisher_close_to_historical_for_normal() -> None:
    rets = _big_rets(seed=2, n=100_000)
    cf = var_cornish_fisher(rets, 0.95)
    hist = var_historical(rets, 0.95)
    assert abs(cf - hist) / hist < 0.10  # moments add noise but not much


def test_drawdown_episodes_structure() -> None:
    rets = pd.Series(
        [0.01, -0.10, 0.02, -0.05, 0.20, 0.01, -0.15, 0.05, 0.30],
        index=pd.bdate_range("2020-01-01", periods=9),
    )
    eps = drawdown_episodes(rets, top=5, min_depth=0.05)
    assert len(eps) >= 1
    assert eps[0].depth <= -0.05 or len(eps) == 0
    for e in eps:
        assert e.peak_date <= e.trough_date


def test_stress_test_windows_present() -> None:
    idx = pd.bdate_range("2018-01-01", "2025-12-31")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    df = stress_test(rets)
    assert "COVID crash (2020-02..03)" in df.index
    assert (df["strategy_return"].abs() < 1).all()


def test_rolling_risk_columns() -> None:
    rets = _rets(n=400)
    rr = rolling_risk(rets, window=63)
    assert {"ann_vol", "ann_return", "sharpe", "var95_daily"} <= set(rr.columns)
    assert rr["ann_vol"].iloc[-1] > 0


def test_risk_summary_keys() -> None:
    rets = _rets(n=400)
    s = risk_summary(rets)
    for key in ("var95_historical", "cvar95", "max_drawdown", "worst_day"):
        assert key in s


def test_performance_summary_benchmark_block() -> None:
    r = _rets(n=500)
    b = _rets(seed=9, n=500)
    s = performance_summary(r, b)
    for key in ("information_ratio", "tracking_error", "alpha_ann", "beta"):
        assert key in s
    assert s["n_days"] == 500
