"""Liquidity & volume factors.

Includes the Amihud (2002) illiquidity measure -- |return| per dollar
traded -- which monetises liquidity provision: illiquid names pay a
premium to those willing to hold them.  Dollar volume is computed as
``close * volume``; with yfinance's split-adjusted prices AND volumes this
stays comparable across corporate actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import safe_div
from alphaforge.factors.base import Factor, register


@register
class DollarVolume21(Factor):
    name = "dol_vol_21"
    category = "liquidity"
    description = "log mean dollar volume over 21d (size/liquidity proxy)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        return np.log(panel.dollar_volume.rolling(21).mean())


@register
class VolumeTrend(Factor):
    """5-day volume vs. 63-day average: attention / crowding proxy."""

    name = "vol_trend"
    category = "liquidity"
    description = "volume_5 / volume_63 - 1: abnormal attention"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        vol = panel.volume
        short = vol.rolling(5).mean()
        long = vol.rolling(63).mean()
        return safe_div(short, long) - 1


@register
class AmihudIlliquidity(Factor):
    """Amihud |r| / dollar-volume, averaged over 21 days (x 1e6 scaling).

    Higher = less liquid.  Historically, high illiquidity earns a premium,
    but the effect concentrates in small caps; used as a risk control in
    AlphaForge's composite and as a liquidity filter input.
    """

    name = "amihud"
    category = "liquidity"
    description = "Amihud illiquidity: mean |ret| / dollar volume over 21d"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        dol = panel.dollar_volume.where(panel.dollar_volume > 0)
        illiq = panel.returns.abs() / dol * 1e6
        return illiq.rolling(21).mean()


@register
class VolumeSpike(Factor):
    """Today's volume relative to its own 21-day average."""

    name = "vol_spike"
    category = "liquidity"
    description = "volume / MA21(volume): single-day volume surprise"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        ma = panel.volume.rolling(21).mean()
        return safe_div(panel.volume, ma)
