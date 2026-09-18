"""Value factors from the fundamentals snapshot (look-ahead disclosed).

The three classical value ratios — earnings yield, sales yield and book
yield — measured as *current snapshot fundamentals over current market
price*.  High yield = cheap name.

**Look-ahead disclosure:** the fundamentals are a static snapshot (see
:mod:`alphaforge.data.fundamentals`), so these factors are NOT
point-in-time.  They are included to complete the factor zoo and to
demonstrate the integration; the factor table and README label them
accordingly.  With a PIT fundamentals source the same code becomes
leakage-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors.base import Factor, register


def _fundamentals() -> pd.DataFrame:
    """Load the snapshot or return an empty frame (factor becomes NaN)."""
    from alphaforge.data.fundamentals import FundamentalsRepository

    try:
        return FundamentalsRepository().load()
    except FileNotFoundError:
        return pd.DataFrame()


def _market_cap(panel: PricePanel) -> pd.DataFrame:
    fund = _fundamentals()
    if fund.empty or "shares" not in fund.columns:
        return pd.DataFrame(np.nan, index=panel.dates, columns=panel.tickers)
    shares = fund["shares"].reindex(panel.tickers)
    return panel.close.mul(shares, axis=1)


@register
class EarningsYield(Factor):
    """TTM net income over market cap — the classic E/P."""

    name = "ep"
    category = "value"
    description = "earnings yield (TTM E/P; static-snapshot fundamentals, look-ahead disclosed)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        fund = _fundamentals()
        mcap = _market_cap(panel)
        if fund.empty or "net_income_ttm" not in fund.columns:
            return mcap * np.nan
        ni = fund["net_income_ttm"].reindex(panel.tickers)
        return mcap.rdiv(ni, axis=1)


@register
class SalesYield(Factor):
    """TTM revenue over market cap — the S/P of sales-based value."""

    name = "sp"
    category = "value"
    description = "sales yield (TTM S/P; static-snapshot fundamentals, look-ahead disclosed)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        fund = _fundamentals()
        mcap = _market_cap(panel)
        if fund.empty or "revenue_ttm" not in fund.columns:
            return mcap * np.nan
        rev = fund["revenue_ttm"].reindex(panel.tickers)
        return mcap.rdiv(rev, axis=1)


@register
class BookYield(Factor):
    """Latest book equity over market cap — B/P (Fama-French HML leg)."""

    name = "bp"
    category = "value"
    description = "book yield (B/P; static-snapshot fundamentals, look-ahead disclosed)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        fund = _fundamentals()
        mcap = _market_cap(panel)
        if fund.empty or "book_equity" not in fund.columns:
            return mcap * np.nan
        eq = fund["book_equity"].reindex(panel.tickers)
        return mcap.rdiv(eq, axis=1)
