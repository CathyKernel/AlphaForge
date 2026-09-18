"""Momentum factors: cross-sectional price trend signals.

The classic academic momentum factor (Jegadeesh & Titman 1993) buys winners
and sells losers over 3-12 month horizons.  The 12-1 variant skips the most
recent month to sidestep the short-term reversal effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import rolling_tstat
from alphaforge.factors.base import Factor, register


class _PriceMomentum(Factor):
    category = "momentum"
    window: int = 63
    skip: int = 0

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        close = panel.close
        if self.skip:
            # return over [T - window, T - skip]: classic 12-1 momentum
            # skips the most recent `skip` days
            return close.shift(self.skip) / close.shift(self.window) - 1
        return close / close.shift(self.window) - 1


@register
class Momentum21(_PriceMomentum):
    name = "mom_21"
    window = 21
    skip = 0
    description = "1-month price momentum: close / close[-21] - 1"


@register
class Momentum63(_PriceMomentum):
    name = "mom_63"
    window = 63
    skip = 0
    description = "3-month price momentum: close / close[-63] - 1"


@register
class Momentum126(_PriceMomentum):
    name = "mom_126"
    window = 126
    skip = 0
    description = "6-month price momentum: close / close[-126] - 1"


@register
class Momentum12_1(_PriceMomentum):
    """12-1 momentum: skip the most recent month (classic academic spec)."""

    name = "mom_252"
    window = 252  # 12-month formation window
    skip = 21  # skip the most recent month
    category = "momentum"
    description = "12-1 momentum: close[-21] / close[-252] - 1 (skips last month)"


@register
class MomentumSlope(Factor):
    """t-stat of the log-price regression slope over ~6 months.

    A smoothed momentum measure: statistically significant uptrends score
    high, choppy drift scores near zero.  More robust to a single price pop
    than a raw ratio.
    """

    name = "mom_slope"
    category = "momentum"
    description = "t-stat of log-price slope over 126d (trend significance)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        log_close = np.log(panel.close.where(panel.close > 0))
        return rolling_tstat(log_close, window=126)
