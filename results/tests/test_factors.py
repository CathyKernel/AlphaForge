"""Tests for the factor library: correctness, NaN handling, no-look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors._utils import cs_zscore, safe_div
from alphaforge.factors.base import REGISTRY, get_factor, list_factors
from alphaforge.factors.evaluation import row_spearman
from alphaforge.factors.library import FactorLibrary
from conftest import TICKERS, make_panel


def test_27_factors_registered() -> None:
    df = list_factors()
    assert len(df) >= 27
    categories = set(df["category"])
    assert categories == {
        "momentum",
        "reversal",
        "volatility",
        "liquidity",
        "technical",
        "higher_moments",
        "value",
    }


def test_every_factor_produces_values(panel) -> None:
    lib = FactorLibrary(panel, min_history=0)
    for name in sorted(REGISTRY):
        mat = lib.compute(name)
        assert isinstance(mat, pd.DataFrame)
        assert mat.shape == panel.close.shape
        if REGISTRY[name].category == "value":
            # value factors need the fundamentals snapshot; without it they
            # degrade to all-NaN (disclosed behaviour), so only check shape
            continue
        # some values must be finite after warm-up
        assert np.isfinite(mat.iloc[150:].to_numpy()).sum() > 0


def test_value_factors_compute_with_snapshot(panel, tmp_path, monkeypatch) -> None:
    """With a fundamentals snapshot on disk the ep/bp/sp factors produce values."""

    snap = pd.DataFrame(
        {
            "net_income_ttm": [1e10, 5e9, 2e9, -1e9, 3e9, 8e9],
            "revenue_ttm": [5e10, 2e10, 1e10, 8e9, 2.5e10, 3e10],
            "book_equity": [3e10, 1.5e10, 8e9, 2e9, 1.2e10, 2e10],
            "shares": [1e10, 2e9, 5e8, 3e8, 1.5e9, 4e9],
        },
        index=list(TICKERS),
    )
    snap.index.name = "ticker"
    snap_path = tmp_path / "fundamentals.parquet"
    snap.to_parquet(snap_path)

    import alphaforge.data.fundamentals as fund_mod

    # value.py imports FundamentalsRepository lazily at call time, so the
    # patch must live on the *source* module
    monkeypatch.setattr(fund_mod, "FundamentalsRepository", lambda: _SnapRepo(snap_path))
    lib = FactorLibrary(panel, min_history=0)
    for name in ("ep", "sp", "bp"):
        mat = lib.compute(name)
        assert mat.shape == panel.close.shape
        assert np.isfinite(mat.to_numpy()).sum() > 0


class _SnapRepo:
    """Minimal stand-in for FundamentalsRepository bound to a temp path."""

    def __init__(self, path) -> None:
        self._path = path

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self._path)


def test_momentum_matches_manual(panel) -> None:
    lib = FactorLibrary(panel, min_history=0)
    m21 = lib.compute("mom_21")
    manual = panel.close / panel.close.shift(21) - 1
    np.testing.assert_allclose(m21.to_numpy(), manual.to_numpy(), equal_nan=True)


def test_mom_252_skips_last_month(panel) -> None:
    lib = FactorLibrary(panel, min_history=0)
    m = lib.compute("mom_252")
    manual = panel.close.shift(21) / panel.close.shift(252) - 1
    np.testing.assert_allclose(m.to_numpy(), manual.to_numpy(), equal_nan=True)


def test_no_lookahead_property() -> None:
    """Truncating the panel must not change factor values on shared dates."""
    longer = make_panel(seed=21, n_days=260)
    short = longer.slice_dates(end=longer.dates[199])
    lib_s = FactorLibrary(short, min_history=0)
    lib_l = FactorLibrary(longer, min_history=0)
    for name in ("mom_21", "vol_21", "rsi_14", "amihud", "bb_pos"):
        a = lib_s.compute(name).iloc[:190]
        b = lib_l.compute(name).iloc[:190]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, err_msg=name)


def test_safe_div_zero_denominator() -> None:
    num = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    den = pd.DataFrame({"a": [0.0, 2.0], "b": [np.nan, 2.0]})
    out = safe_div(num, den)
    assert np.isnan(out["a"].iloc[0])
    assert np.isnan(out["b"].iloc[0])
    assert out["a"].iloc[1] == 1.0


def test_safe_div_series_axis_broadcast() -> None:
    num = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    den = pd.Series([1.0, 2.0])  # index 0,1 -> rows
    out = safe_div(num, den, axis=0)
    np.testing.assert_allclose(out["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(out["b"].to_numpy(), [3.0, 2.0])


def test_cs_zscore_rowwise() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
    z = cs_zscore(df)
    assert abs(z.iloc[0].mean()) < 1e-12
    assert np.allclose(z.iloc[0].std(ddof=0), 1.0)


def test_row_spearman_matches_scipy() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(30, 8)), index=pd.bdate_range("2020-01-01", periods=30))
    r = pd.DataFrame(rng.normal(size=(30, 8)), index=f.index)
    ours = row_spearman(f, r, min_obs=5)
    for date in f.index[:5]:
        ref = spearmanr(f.loc[date], r.loc[date]).statistic
        assert abs(ours.loc[date] - ref) < 1e-10


def test_composite_shape_and_finite(panel) -> None:
    lib = FactorLibrary(panel, min_history=20)
    comp = lib.composite(names=["mom_21", "vol_21", "rsi_14"])
    assert comp.shape == panel.close.iloc[20:].shape
    assert comp.notna().any(axis=1).iloc[50:].all()


def test_unknown_factor_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        get_factor("not_a_factor")
