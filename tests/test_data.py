"""Tests for the data layer: panels, persistence, universe, quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.data.quality import DataQualityChecker
from alphaforge.data.universe import SP100_TICKERS, Universe
from conftest import make_panel


def test_panel_roundtrip_long() -> None:
    panel = make_panel(seed=3)
    long = panel.to_long()
    back = PricePanel.from_long(long)
    pd.testing.assert_frame_equal(
        panel.close.rename_axis(index=None, columns=None),
        back.close.rename_axis(index=None, columns=None),
        check_freq=False,
    )


def test_panel_parquet_roundtrip(tmp_path) -> None:
    panel = make_panel(seed=4)
    path = panel.to_parquet(tmp_path / "p.parquet")
    loaded = PricePanel.from_parquet(path)
    pd.testing.assert_frame_equal(
        panel.close.rename_axis(index=None, columns=None),
        loaded.close.rename_axis(index=None, columns=None),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        panel.volume.rename_axis(index=None, columns=None),
        loaded.volume.rename_axis(index=None, columns=None),
        check_freq=False,
    )


def test_panel_alignment_validation() -> None:
    panel = make_panel(seed=5)
    bad = {"close": panel.close, "open": panel.open.iloc[:-1]}
    import pytest

    with pytest.raises(ValueError):
        PricePanel(bad)


def test_returns_first_row_nan() -> None:
    panel = make_panel(seed=6)
    rets = panel.returns
    assert rets.iloc[0].isna().all()
    np.testing.assert_allclose(
        rets.iloc[1].to_numpy(),
        (panel.close.iloc[1] / panel.close.iloc[0] - 1).to_numpy(),
    )


def test_coverage_report() -> None:
    panel = make_panel(seed=8)
    cov = panel.coverage()
    assert set(cov.columns) >= {"n_days", "pct_available"}
    assert (cov["n_days"] <= len(panel)).all()


def test_quality_catches_injected_errors() -> None:
    panel = make_panel(seed=9)
    # duplicate rows: duplicate the last date in every field frame
    fields = {f: pd.concat([df, df.iloc[[-1]]]).sort_index() for f, df in panel._fields.items()}
    dup = PricePanel(fields)
    qr = DataQualityChecker().run(dup)
    d = {c.name: c for c in qr.checks}
    assert not d["duplicates"].passed

    close = panel.close.copy()
    close.iloc[10, 0] = -5.0
    neg = PricePanel({**panel._fields, "close": close})
    qr2 = DataQualityChecker().run(neg)
    d2 = {c.name: c for c in qr2.checks}
    assert not d2["non_positive_prices"].passed


def test_quality_passes_clean_panel() -> None:
    panel = make_panel(seed=10)
    qr = DataQualityChecker().run(panel)
    errors = [c for c in qr.checks if c.severity == "error"]
    assert all(c.passed for c in errors)


def test_universe_bundled_snapshot() -> None:
    u = Universe.sp500()
    assert len(u) > 400
    assert u.sector_counts().shape[0] == 11
    sp100 = Universe.sp100()
    assert set(sp100.tickers) == set(SP100_TICKERS)
    assert len(sp100) >= 95
    # every SP100 ticker must have a sector
    assert sp100.sectors.notna().all()


def test_universe_restrict_and_dummies() -> None:
    u = Universe.sp100()
    small = u.restrict(["AAPL", "JPM", "NOT-A-TICKER"])
    assert list(small.tickers) == ["AAPL", "JPM"]
    dummies = small.sector_dummies()
    assert dummies.shape == (2, 2)
    assert dummies.sum(axis=1).eq(1).all()
