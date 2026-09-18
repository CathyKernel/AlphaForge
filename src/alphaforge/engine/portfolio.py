"""Portfolio construction from cross-sectional signals.

A constructor maps a signal row (ticker -> value, higher = more bullish
after any sign flips applied upstream) into a weight row that sums to
``gross_exposure`` (1.0 for a fully invested long-only book, 2.0 for a
dollar-neutral long-short book).

All implementations are pure functions of the current signal snapshot:
no future data can leak in by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PortfolioConstructor(ABC):
    """Base class: turn one signal row into one weight row.

    ``date`` is the signal date (the close the row was computed at); it is
    optional so simple constructors can ignore it, but constructors that
    estimate covariance or track state (e.g. the mean-variance optimizer)
    need it to stay leakage-safe.
    """

    #: total absolute weight (1.0 long-only, 2.0 dollar-neutral L/S)
    gross_exposure: float = 1.0
    #: 'long' | 'short' | 'long_short'
    side: str = "long"

    @abstractmethod
    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        """Return target weights (sums to ``gross_exposure`` in abs terms)."""

    # ------------------------------------------------------------------ #
    def _apply_weighting(self, ranks: pd.Series, n: int) -> pd.Series:
        """Weight the selected names: equal / rank-linear / z-score."""
        sel = ranks.dropna()
        if len(sel) == 0:
            return pd.Series(dtype=float)
        if self.weighting == "equal":
            w = pd.Series(1.0 / len(sel), index=sel.index)
        elif self.weighting == "rank":
            # linear in cross-sectional rank, normalised to sum 1
            r = sel.rank()
            w = r / r.sum()
        elif self.weighting == "zscore":
            z = (
                (sel - sel.mean()) / sel.std(ddof=0)
                if sel.std(ddof=0) > 0
                else pd.Series(0.0, index=sel.index)
            )
            w = z / z.abs().sum()
            w = w.abs()  # z-score weighting: magnitude, sign handled by side
        else:
            raise ValueError(f"unknown weighting {self.weighting!r}")
        # cap any single position and renormalise
        if self.max_weight is not None and self.max_weight < float("inf"):
            w = w.clip(upper=self.max_weight)
            w = w / w.sum()
        return w

    weighting: str = "equal"
    max_weight: float | None = None


class TopN(PortfolioConstructor):
    """Long (and optionally short) the top/bottom-N names by signal.

    Parameters
    ----------
    n:
        Names per leg.
    side:
        ``'long'``, ``'short'`` or ``'long_short'``.
    weighting:
        ``'equal'`` | ``'rank'`` | ``'zscore'``.
    max_weight:
        Optional cap on any single position's weight.
    """

    def __init__(
        self,
        n: int = 20,
        side: str = "long_short",
        weighting: str = "equal",
        max_weight: float | None = 0.10,
    ) -> None:
        self.n = int(n)
        self.side = side
        self.weighting = weighting
        self.max_weight = max_weight
        self.gross_exposure = 2.0 if side == "long_short" else 1.0

    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        sig = signal.dropna().astype(float)
        if len(sig) < 2 * self.n:
            # not enough breadth: degenerate to whatever is available
            return pd.Series(dtype=float)
        ranked = sig.sort_values(ascending=False)
        long_names = ranked.index[: self.n]
        short_names = ranked.index[-self.n :]
        w_long = self._apply_weighting(sig[long_names], self.n)
        w_short = self._apply_weighting(sig[short_names], self.n)
        out = pd.Series(0.0, index=sig.index)
        if self.side in ("long", "long_short"):
            out[w_long.index] += w_long
        if self.side in ("short", "long_short"):
            out[w_short.index] -= w_short
        return out


class Quantile(PortfolioConstructor):
    """Long (and optionally short) the top/bottom quantile by signal."""

    def __init__(
        self,
        q: float = 0.2,
        side: str = "long_short",
        weighting: str = "equal",
        max_weight: float | None = None,
    ) -> None:
        if not 0 < q <= 0.5:
            raise ValueError("q must be in (0, 0.5]")
        self.q = float(q)
        self.side = side
        self.weighting = weighting
        self.max_weight = max_weight
        self.gross_exposure = 2.0 if side == "long_short" else 1.0

    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        sig = signal.dropna().astype(float)
        if len(sig) < 5:
            return pd.Series(dtype=float)
        ranks = sig.rank(pct=True)
        long_mask = ranks >= 1 - self.q
        short_mask = ranks <= self.q
        w_long = self._apply_weighting(sig[long_mask], long_mask.sum())
        w_short = self._apply_weighting(sig[short_mask], short_mask.sum())
        out = pd.Series(0.0, index=sig.index)
        if self.side in ("long", "long_short") and len(w_long):
            out[w_long.index] += w_long
        if self.side in ("short", "long_short") and len(w_short):
            out[w_short.index] -= w_short
        return out


class ZScoreBlend(PortfolioConstructor):
    """Continuous weights proportional to (capped) z-scores.

    Every name gets a position; dollar-neutral by construction.  Useful
    when the signal is already a composite z-score.
    """

    def __init__(self, max_weight: float = 0.05, side: str = "long_short") -> None:
        self.max_weight = max_weight
        self.side = side
        self.weighting = "zscore"
        self.gross_exposure = 2.0 if side == "long_short" else 1.0

    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        sig = signal.dropna().astype(float)
        if len(sig) < 3:
            return pd.Series(dtype=float)
        std = sig.std(ddof=0)
        z = (sig - sig.mean()) / std if std > 0 else pd.Series(0.0, index=sig.index)
        w = z / 2.0  # z sums to 0; gross exposure = sum|w| ≈ n * avg|z| / 2
        w = w.clip(lower=-self.max_weight, upper=self.max_weight)
        scale = self.gross_exposure / w.abs().sum()
        return w * scale
