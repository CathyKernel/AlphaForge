"""Factor base class and global registry.

A *factor* maps a :class:`~alphaforge.data.panel.PricePanel` to a signal
matrix: one value per (date, ticker) computed **using information available
up to and including that date's close**.  The backtesting engine, not the
factor, owns the T+1 execution lag -- this separation keeps look-ahead bias
out of factor code by construction and makes it unit-testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd

from alphaforge.data.panel import PricePanel

REGISTRY: dict[str, Factor] = {}


class Factor(ABC):
    """Abstract base class for cross-sectional factors."""

    #: short snake_case identifier, e.g. ``mom_63``
    name: str = ""
    #: category, e.g. ``momentum``
    category: str = ""
    #: one-line intuition for the signal
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Factor {self.name} [{self.category}]>"

    @abstractmethod
    def compute(self, panel: PricePanel) -> pd.DataFrame:
        """Return a date x ticker DataFrame of raw factor values.

        Values must depend only on data up to and including each date.
        Missing values are ``NaN``; no cross-sectional standardisation is
        applied here (that is the evaluator's job).
        """


def register(cls: type[Factor]) -> type[Factor]:
    """Class decorator that adds a factor to the global registry."""
    instance = cls()
    if not instance.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    if instance.name in REGISTRY:
        raise ValueError(f"Duplicate factor name: {instance.name}")
    REGISTRY[instance.name] = instance
    return cls


def get_factor(name: str) -> Factor:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown factor {name!r}. Available: {sorted(REGISTRY)}") from exc


def list_factors() -> pd.DataFrame:
    """Return a DataFrame of all registered factors (name, category, description)."""
    rows = [
        {"name": f.name, "category": f.category, "description": f.description}
        for f in REGISTRY.values()
    ]
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)


# --------------------------------------------------------------------- #
# Convenience: wrap a plain function as a Factor
# --------------------------------------------------------------------- #
def function_factor(
    name: str, category: str, description: str
) -> Callable[[Callable[[PricePanel], pd.DataFrame]], type[Factor]]:
    """Decorator turning ``f(panel) -> DataFrame`` into a registered Factor."""

    def deco(fn: Callable[[PricePanel], pd.DataFrame]) -> type[Factor]:
        @register
        class _FnFactor(Factor):
            _name = name
            _category = category
            _description = description

            def compute(self, panel: PricePanel) -> pd.DataFrame:
                return fn(panel)

        # expose class-level metadata required by the base contract
        _FnFactor.name = name  # type: ignore[attr-defined]
        _FnFactor.category = category  # type: ignore[attr-defined]
        _FnFactor.description = description  # type: ignore[attr-defined]
        return _FnFactor

    return deco
