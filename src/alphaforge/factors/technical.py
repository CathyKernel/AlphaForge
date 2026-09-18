"""Technical factors computed from OHLCV.

Classical technical indicators, re-expressed as cross-sectional signals:
RSI, Bollinger band position, distance from the 52-week high (the
"proximity to high" effect of George & Hwang 2004), moving-average ratios
(trend), and normalised average true range (risk per unit of price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import safe_div
from alphaforge.factors.base import Factor, register


@register
class RSI14(Factor):
    """Relative Strength Index (Wilder), 14 days.

    Low RSI = oversold; combined with cross-sectional ranking this behaves
    like a mean-reversion signal.  Implemented with Wilder's smoothing.
    """

    name = "rsi_14"
    category = "technical"
    description = "14d RSI (oversold names bounce cross-sectionally)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        delta = panel.close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = safe_div(avg_gain, avg_loss)
        rsi = 100 - 100 / (1 + rs)
        # perfectly flat series -> avg_loss = 0 and avg_gain = 0 -> NaN -> RSI 50
        return rsi.where((avg_gain + avg_loss) > 0, 50.0)


@register
class BollingerPosition(Factor):
    """%B position inside 20-day Bollinger bands (±2 sigma)."""

    name = "bb_pos"
    category = "technical"
    description = "Bollinger %B: (close - SMA20) / (2 * std20)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        close = panel.close
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        return safe_div(close - sma, 2 * std)


@register
class Distance52wHigh(Factor):
    """Close / 52-week high - 1 (George & Hwang 2004 proximity effect).

    Nearness to the 52-week high predicts continuation -- anchoring bias
    makes investors under-react when a stock approaches its historical
    peak.
    """

    name = "dist_52w_high"
    category = "technical"
    description = "close / 252d rolling max - 1 (proximity-to-high effect)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        roll_max = panel.close.rolling(252, min_periods=126).max()
        return safe_div(panel.close, roll_max) - 1


@register
class SMARatio50(Factor):
    """Close / 50-day SMA - 1: medium-term trend filter."""

    name = "sma_ratio_50"
    category = "technical"
    description = "close / SMA50 - 1 (medium-term trend)"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        sma = panel.close.rolling(50).mean()
        return safe_div(panel.close, sma) - 1


@register
class ATR21(Factor):
    """21-day average true range normalised by price.

    Captures intraday risk per unit of price -- a volatility measure that
    uses the full OHLC bar rather than close-to-close only.
    """

    name = "atr_21"
    category = "technical"
    description = "ATR(21) / close: range-based risk per unit price"

    def compute(self, panel: PricePanel) -> pd.DataFrame:
        close, high, low_ = panel.close, panel.high, panel.low
        prev_close = close.shift(1)
        # True range = max(high-low, |high-prev_close|, |low-prev_close|),
        # computed with a single vectorised element-wise maximum.
        tr = np.maximum.reduce(
            [
                (high - low_).to_numpy(),
                (high - prev_close).abs().to_numpy(),
                (low_ - prev_close).abs().to_numpy(),
            ]
        )
        tr_df = pd.DataFrame(tr, index=close.index, columns=close.columns)
        atr = tr_df.rolling(21).mean()
        return safe_div(atr, close)
