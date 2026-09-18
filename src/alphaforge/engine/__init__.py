"""Backtesting engine: costs, portfolio construction, vectorised simulation."""

from alphaforge.engine.costs import TransactionCostModel
from alphaforge.engine.metrics import (
    alpha_beta,
    cagr,
    calmar,
    information_ratio,
    max_drawdown,
    monthly_returns,
    performance_summary,
    rolling_sharpe,
    sharpe,
    sortino,
    tracking_error,
)
from alphaforge.engine.portfolio import PortfolioConstructor, Quantile, TopN, ZScoreBlend
from alphaforge.engine.vectorized import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    run_backtest,
)

__all__ = [
    "TransactionCostModel",
    "PortfolioConstructor",
    "TopN",
    "Quantile",
    "ZScoreBlend",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "run_backtest",
    "performance_summary",
    "monthly_returns",
    "rolling_sharpe",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "cagr",
    "alpha_beta",
    "tracking_error",
    "information_ratio",
]
