"""Reversal factors: short-horizon overreaction signals.

Short-term reversal (1-5 day) is one of the most robust cross-sectional
anomalies in US equities (Jegadeesh 1990; Nagel 2012 attributes much of it
to liquidity provision).  Signals are *sign-flipped at portfolio time* by
the engine (low factor value -> long) -- factors document their expected
direction in :attr:`Factor.description`.
"""

from __future__ import annotations

import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors.base import Factor, register


@register
class Reversal5(Factor):
    """5-day cumulative return; expect NEGATIVE predictive correlation
    (losers bounce), so the engine ranks ascending (low value = long)."""

    name = "rev_5"
    category = "reversal"
    description = "5-day return (short-term reversal; losers bounce)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        close = panel.close
        return close.shift(1) / close.shift(6) - 1


@register
class Reversal1(Factor):
    """Previous day's return (1-day reversal)."""

    name = "rev_1"
    category = "reversal"
    description = "1-day return (overnight microstructure reversal)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        return panel.close.shift(1) / panel.close.shift(2) - 1


@register
class OvernightReversal(Factor):
    """Mean overnight gap over the last 5 sessions.

    Gap = open_t / close_{t-1} - 1.  Stocks that gapped up repeatedly tend
    to mean-revert intraday over the following days.
    """

    name = "rev_gap"
    category = "reversal"
    description = "mean overnight gap over 5 days (gap-up fade)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        gap = panel.open / panel.close.shift(1) - 1
        return gap.rolling(5).mean().shift(1)
