"""Transaction cost models.

Two models are provided:

* :class:`TransactionCostModel` — linear per-side costs in basis points,
  applied to total traded notional.  The default of 10 bps/side is
  deliberately conservative for liquid large caps: it covers commissions
  (~1 bps), half the quoted spread (2-5 bps for S&P 100 names) and a
  realistic market-impact allowance.

* :class:`SquareRootImpact` — the institutional standard square-root law:
  impact per name grows with the *square root of participation* and with
  the name's daily volatility::

      impact_bps_i = fixed_bps + k * sigma_i * sqrt(participation_i)

  where ``participation_i = (|dw_i| * book_value) / ADV_i`` and ADV is the
  trailing 21-day average dollar volume.  Small trades in liquid names cost
  ~the fixed floor; a 5% ADV clip in a volatile name costs several times
  more.  ``k`` around 1.0 matches the empirical academic estimates
  (Almgren et al. 2005; Grinold-Kahn scaling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """Per-side linear transaction costs in basis points.

    Parameters
    ----------
    bps_per_side:
        Commission + slippage + half-spread per unit of traded notional.
    min_trade_pct:
        Weight changes below this threshold are ignored (noise trades),
        which slightly reduces reported turnover.
    """

    bps_per_side: float = 10.0
    min_trade_pct: float = 0.0

    def cost_return(self, turnover: float) -> float:
        """Portfolio return deducted for a given one-way turnover.

        ``turnover`` here is total traded notional: sum(|dw|) across both
        buy and sell legs, so cost = turnover * bps / 1e4.
        """
        if turnover <= self.min_trade_pct:
            return 0.0
        return turnover * self.bps_per_side / 1e4

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TransactionCostModel({self.bps_per_side} bps/side)"


@dataclass(frozen=True)
class SquareRootImpact:
    """Per-name square-root market impact + fixed cost (institutional style).

    Parameters
    ----------
    book_value:
        Notional dollars of the portfolio being traded — turns weight
        changes into dollar trades so participation in ADV is meaningful.
    fixed_bps:
        Commission + non-impact slippage floor, per side.
    impact_coef:
        ``k`` in ``k * sigma * sqrt(participation)`` (return units).
    adv:
        Optional pre-computed trailing average dollar volume (date x
        ticker); when omitted it is computed from the panel at run time
        by the engine (21d mean, known at the signal date — no lookahead).
    per_name:
        Marker used by the engine to select the per-name cost path.
    """

    book_value: float = 10_000_000.0
    fixed_bps: float = 5.0
    impact_coef: float = 1.0
    adv: pd.DataFrame | None = field(default=None, repr=False)
    per_name: bool = True

    def cost_return_per_name(
        self, traded: pd.Series, adv: pd.Series, daily_vol: pd.Series
    ) -> float:
        """Total portfolio return deducted for one rebalance.

        Parameters
        ----------
        traded:
            |dw| per ticker (weight units, one side).
        adv:
            Average daily dollar volume per ticker **as known at the
            signal date** (trailing window).
        daily_vol:
            Realised daily vol per ticker (trailing window).
        """
        if traded.empty:
            return 0.0
        adv = adv.reindex(traded.index).fillna(0.0)
        vol = daily_vol.reindex(traded.index).fillna(0.0)
        dollars = (traded.abs() * self.book_value).clip(lower=0.0)
        participation = (dollars / adv.clip(lower=1.0)).clip(upper=1.0)
        # impact as a return fraction per name, applied to that name's
        # traded weight: cost = sum_i |dw_i| * (fixed + k sigma_i sqrt(q_i))
        impact_ret = self.impact_coef * vol * np.sqrt(participation)
        return float((impact_ret * traded.abs()).sum() + self.fixed_bps / 1e4 * traded.abs().sum())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SquareRootImpact(book=${self.book_value / 1e6:.0f}M, "
            f"fixed={self.fixed_bps}bps, k={self.impact_coef})"
        )
