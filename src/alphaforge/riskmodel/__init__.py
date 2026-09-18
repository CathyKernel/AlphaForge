"""Barra-lite style factor risk model (see :mod:`alphaforge.riskmodel.style`)."""

from alphaforge.riskmodel.attribution import AttributionResult, attribute_returns
from alphaforge.riskmodel.style import STYLE_FACTORS, RiskDecomposition, StyleRiskModel

__all__ = [
    "STYLE_FACTORS",
    "AttributionResult",
    "RiskDecomposition",
    "StyleRiskModel",
    "attribute_returns",
]
