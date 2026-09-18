"""FactorLibrary: batch computation, caching and composite signals."""

from __future__ import annotations

import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors import (  # noqa: F401
    higher_moments,
    liquidity,
    momentum,
    reversal,
    technical,
    value,
    volatility,
)
from alphaforge.factors._utils import cs_winsorize, cs_zscore
from alphaforge.factors.base import REGISTRY, list_factors

#: Modules whose @register decorators populate the global registry.
_FACTOR_MODULES = (
    momentum,
    reversal,
    volatility,
    liquidity,
    technical,
    higher_moments,
    value,
)


class FactorLibrary:
    """Compute and cache every registered factor for a given panel.

    Parameters
    ----------
    panel:
        Price panel the factors are computed on.
    min_history:
        Rows before this index are dropped from factor output so that
        indicators that need a year of history do not emit half-baked
        values from the warm-up region.
    """

    def __init__(self, panel: PricePanel, min_history: int = 260) -> None:
        self.panel = panel
        self.min_history = min_history
        self._cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def available() -> pd.DataFrame:
        """Registry listing (name, category, description)."""
        return list_factors()

    def compute(self, name: str) -> pd.DataFrame:
        """Compute (or fetch cached) a single factor matrix."""
        if name not in self._cache:
            from alphaforge.factors.base import get_factor

            values = get_factor(name).compute(self.panel)
            if self.min_history and len(values) > self.min_history:
                values = values.iloc[self.min_history :]
            self._cache[name] = values
        return self._cache[name]

    def compute_all(self, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Compute a set of factors (default: the entire registry)."""
        names = names or sorted(REGISTRY)
        return {name: self.compute(name) for name in names}

    # ------------------------------------------------------------------ #
    # Composite construction
    # ------------------------------------------------------------------ #
    def composite(
        self,
        names: list[str] | None = None,
        weights: dict[str, float] | None = None,
        direction: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Average of winsorised, z-scored factors (NaN-aware).

        Parameters
        ----------
        names:
            Factors to combine (default: all registered).
        weights:
            Optional ``{factor_name: weight}`` (renormalised, NaN-aware).
        direction:
            Optional ``{factor_name: +1 | -1}`` sign flips for factors whose
            empirical IC is negative (e.g. reversal factors), so the
            composite is oriented "high = long".
        """
        names = names or sorted(REGISTRY)
        if not names:
            raise ValueError("No factors selected for the composite")
        zdfs = []
        for n in names:
            z = cs_zscore(cs_winsorize(self.compute(n)))
            if direction and n in direction:
                z = z * float(direction[n])
            zdfs.append(z)
        big = pd.concat(zdfs, axis=1, keys=names)  # (date) x (factor, ticker)
        long = big.stack(level=1, future_stack=True)  # (date, ticker) x factor
        if weights:
            w = pd.Series(weights, dtype=float).reindex(long.columns).fillna(0.0)
            w = w / w.sum()
            wsum = long.notna().mul(w, axis=1).sum(axis=1)
            num = long.fillna(0.0).mul(w, axis=1).sum(axis=1)
            score = num / wsum.where(wsum > 0)
        else:
            score = long.mean(axis=1)  # NaN-aware equal weight
        return score.unstack(level=-1).reindex(columns=big.columns.levels[1])
