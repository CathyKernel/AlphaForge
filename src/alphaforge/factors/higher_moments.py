"""Higher-moment / "quality of returns" factors.

Return distribution shape carries cross-sectional information: consistent
compounders (high rolling Sharpe) outperform lottery-like names (high
skew / deep own-drawdowns) -- see Bali, Cakici & Whitelaw (2011) on the
MAX effect and Boyer, Mitton & Vorkink (2010) on expected skewness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import rolling_skew
from alphaforge.factors.base import Factor, register


@register
class RollingSharpe63(Factor):
    """63-day mean/std of daily returns (annualised).

    Cross-sectionally this proxies for consistency of the return stream;
    high values mark steady compounders rather than one-day wonders.
    """

    name = "sharpe_63"
    category = "higher_moments"
    description = "rolling 63d Sharpe of the stock's own returns (consistency)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        rets = panel.returns
        mean = rets.rolling(63).mean() * 252
        std = rets.rolling(63).std() * np.sqrt(252)
        return mean / std


@register
class ReturnSkew63(Factor):
    """63-day rolling skewness of daily returns.

    Lottery-preference theory predicts NEGATIVE cross-sectional relation
    with future returns: positively skewed "lottery stocks" are
    over-priced.
    """

    name = "skew_63"
    category = "higher_moments"
    description = "rolling 63d return skewness (lottery preference, negative sign)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        return rolling_skew(panel.returns, window=63)


@register
class OwnDrawdown126(Factor):
    """Depth of the stock's own max drawdown over ~6 months.

    Value = current drawdown recovery status: 0 means at the 126d peak,
    -0.5 means 50% below it.  Sign-flipped by construction so that "less
    damaged" names score higher.
    """

    name = "max_dd_126"
    category = "higher_moments"
    description = "own drawdown from 126d peak (recovery quality, positive = near high)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        close = panel.close
        roll_max = close.rolling(126).max()
        return close / roll_max - 1
