"""Shared fixtures: deterministic synthetic panels for fast tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.panel import PricePanel

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N_DAYS = 260


def make_panel(seed: int = 7, n_days: int = N_DAYS, tickers: list[str] | None = None) -> PricePanel:
    """Deterministic synthetic OHLCV panel with 1% drift, 1.5% vol."""
    rng = np.random.default_rng(seed)
    tickers = tickers or TICKERS
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    closes = {}
    for t in tickers:
        rets = rng.normal(0.0004, 0.015, n_days)
        closes[t] = 100 * np.exp(np.cumsum(rets))
    close = pd.DataFrame(closes, index=idx)
    open_ = close * (1 + rng.normal(0, 0.002, close.shape))
    high = pd.DataFrame(np.maximum(open_.values, close.values), index=idx, columns=tickers) * 1.005
    low = pd.DataFrame(np.minimum(open_.values, close.values), index=idx, columns=tickers) * 0.995
    volume = pd.DataFrame(rng.uniform(5e7, 5e8, close.shape), index=idx, columns=tickers)
    return PricePanel(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture(scope="session")
def panel() -> PricePanel:
    return make_panel()


@pytest.fixture(scope="session")
def planted_panel() -> PricePanel:
    """Panel with a strong, learnable cross-sectional drift ranking."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=N_DAYS)
    drift = {
        "AAA": 0.0040,
        "BBB": 0.0024,
        "CCC": 0.0008,
        "DDD": 0.0,
        "EEE": -0.0024,
        "FFF": -0.0040,
    }
    closes = {}
    for t, d in drift.items():
        closes[t] = 100 * np.exp(np.cumsum(rng.normal(d, 0.008, N_DAYS)))
    close = pd.DataFrame(closes, index=idx)
    open_ = close.shift(1).fillna(100.0)
    high = (
        pd.DataFrame(np.maximum(open_.values, close.values), index=idx, columns=list(drift)) * 1.004
    )
    low = (
        pd.DataFrame(np.minimum(open_.values, close.values), index=idx, columns=list(drift)) * 0.996
    )
    volume = pd.DataFrame(1e8, index=idx, columns=list(drift))
    return PricePanel({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
