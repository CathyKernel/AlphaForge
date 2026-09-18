"""Factor library: 27 cross-sectional alpha factors across 7 categories."""

from alphaforge.factors.base import Factor, get_factor, list_factors, register
from alphaforge.factors.evaluation import FactorEvaluator, FactorStats, newey_west_tstat
from alphaforge.factors.library import FactorLibrary
from alphaforge.factors.neutralize import (
    Neutralizer,
    beta_residual,
    sector_demean,
    trailing_beta,
    vol_scale,
    winsorize_zscore,
)

__all__ = [
    "Factor",
    "register",
    "get_factor",
    "list_factors",
    "FactorLibrary",
    "FactorEvaluator",
    "FactorStats",
    "newey_west_tstat",
    "Neutralizer",
    "sector_demean",
    "beta_residual",
    "vol_scale",
    "winsorize_zscore",
    "trailing_beta",
]
