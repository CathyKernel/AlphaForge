"""Volatility factors: low-vol anomaly and risk-shape signals.

The low-volatility anomaly (Baker, Bradley & Wurgler 2011) is one of the
most persistent puzzles in equity markets: low-beta / low-vol stocks
outperform their high-vol peers on a risk-adjusted basis over long
samples.  ``idio_vol_21`` isolates stock-specific risk by removing the
cap-weight-free (equal-weight) market return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import safe_div
from alphaforge.factors.base import Factor, register


@register
class RealizedVol21(Factor):
    name = "vol_21"
    category = "volatility"
    description = "annualised 21-day realised volatility (low-vol anomaly)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        return panel.returns.rolling(21).std() * np.sqrt(252)


@register
class RealizedVol63(Factor):
    name = "vol_63"
    category = "volatility"
    description = "annualised 63-day realised volatility"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        return panel.returns.rolling(63).std() * np.sqrt(252)


@register
class VolRatio(Factor):
    """Short-term vol / long-term vol.

    > 1 means risk is expanding (often pre-earnings or stress);
    < 1 means calm regime relative to the stock's own history.
    """

    name = "vol_ratio"
    category = "volatility"
    description = "vol_21 / vol_126: short-term risk expansion ratio"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        rets = panel.returns
        short = rets.rolling(21).std()
        long = rets.rolling(126).std()
        return safe_div(short, long)


@register
class IdioVol21(Factor):
    """21-day idiosyncratic (residual) volatility vs. equal-weight market.

    Residual returns r_i - beta_i * r_m are computed with a rolling 126d
    beta; the market return is the equal-weight mean of the panel (a
    cap-weight-free benchmark).  High idio-vol names are penalised by
    lottery-demand theory (Bali et al. 2011, MAX effect).
    """

    name = "idio_vol_21"
    category = "volatility"
    description = "21d idiosyncratic vol vs equal-weight market (lottery effect)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        rets = panel.returns
        market = rets.mean(axis=1)
        cov = rets.rolling(126).cov(market)  # DataFrame: date x ticker
        var_m = market.rolling(126).var()
        beta = safe_div(cov, var_m, axis=0)
        resid = rets.sub(beta.mul(market, axis=0))
        return resid.rolling(21).std() * np.sqrt(252)
