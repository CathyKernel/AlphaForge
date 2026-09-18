"""AlphaForge: an end-to-end quantitative research platform for US equities.

AlphaForge covers the full research workflow for systematic equity strategies:

- **Data** -- bulk downloading of real US equity prices (Yahoo Finance), a
  Parquet-backed local data repository, universe management (S&P 500 / S&P 100)
  and data-quality validation.
- **Factors** -- a library of 27 cross-sectional alpha factors across seven
  categories (momentum, reversal, volatility, liquidity, technical,
  higher-moments) with a common interface and an evaluation toolkit
  (rank IC, IC decay, quantile spreads, turnover).
- **Backtesting** -- a vectorised, long/short backtesting engine with realistic
  assumptions: T+1 execution lag, drifting weights between rebalances,
  per-side transaction costs, liquidity filters and benchmark-relative
  analytics.
- **Machine learning** -- a feature panel builder, LightGBM and PyTorch LSTM
  alpha models, and a purged / embargoed walk-forward validation framework
  that eliminates look-ahead and leakage from overlapping labels.
- **Risk & reporting** -- VaR / CVaR analytics, drawdown forensics, stress
  tests, matplotlib tearsheets and an interactive web dashboard.

The library is deliberately vectorised (NumPy / pandas / LightGBM) so that a
ten-year, one-hundred-stock research backtest runs in seconds, not minutes.
"""

from alphaforge.config import Config, get_config
from alphaforge.data.cache import DataRepository
from alphaforge.data.panel import PricePanel
from alphaforge.factors.library import FactorLibrary, list_factors

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Config",
    "get_config",
    "PricePanel",
    "DataRepository",
    "FactorLibrary",
    "list_factors",
]
