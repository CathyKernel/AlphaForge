"""Feature panel construction for machine-learning alpha models.

The panel is a long table indexed by ``(date, ticker)``:

* **Features** -- raw factor values, cross-sectionally winsorised and
  z-scored per date, NaN-filled with 0 (a "no opinion" value after
  standardisation).
* **Label** -- forward return from close T+1 to close T+1+h (identical
  convention to the factor evaluator and backtest engine), optionally
  demeaned cross-sectionally so the model predicts *relative* strength --
  the quantity a market-neutral book actually monetises.

Because features at date T use only data through T's close, and labels
are realised strictly after T, the panel itself is look-ahead-clean;
leakage between *train and test periods* is handled separately by the
purged walk-forward validator.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.factors._utils import cs_winsorize, cs_zscore
from alphaforge.factors.base import REGISTRY


@dataclass
class FeatureConfig:
    """Parameters of the feature/label panel."""

    label_horizon: int = 5
    label_type: str = "excess"  # 'raw' | 'excess'
    winsor_pct: tuple[float, float] = (0.01, 0.99)
    fill_value: float = 0.0  # post-standardisation NaN fill


class FeaturePanelBuilder:
    """Build the (features, labels) training table from a price panel.

    Parameters
    ----------
    panel:
        Price panel with the investment universe.
    factor_names:
        Factors used as features (default: every registered factor).
    config:
        See :class:`FeatureConfig`.
    """

    def __init__(
        self,
        panel: PricePanel,
        factor_names: list[str] | None = None,
        config: FeatureConfig | None = None,
    ) -> None:
        self.panel = panel
        self.factor_names = factor_names or sorted(REGISTRY)
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------ #
    def build(
        self, factor_values: dict[str, pd.DataFrame] | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Return ``(X, y, fwd_raw)``.

        ``X`` is indexed by (date, ticker) with one column per factor;
        ``y`` is the (excess) forward return; ``fwd_raw`` is the raw
        forward return before demeaning (useful for diagnostics).
        """
        from alphaforge.factors.library import FactorLibrary

        if factor_values is None:
            lib = FactorLibrary(self.panel)
            factor_values = {n: lib.compute(n) for n in self.factor_names}

        cfg = self.config
        processed: dict[str, pd.Series] = {}
        for name in self.factor_names:
            mat = factor_values[name]
            mat = cs_winsorize(mat, cfg.winsor_pct)
            z = cs_zscore(mat)
            processed[name] = z.stack(future_stack=True)

        X = pd.DataFrame(processed)
        X = X.fillna(cfg.fill_value)

        close = self.panel.close
        h = cfg.label_horizon
        fwd = close.shift(-(h + 1)) / close.shift(-1) - 1
        fwd_raw = fwd.stack(future_stack=True)
        if cfg.label_type == "excess":
            y = fwd.sub(fwd.mean(axis=1), axis=0).stack(future_stack=True)
        else:
            y = fwd_raw.copy()
        y.name = "fwd_ret"

        common = X.index.intersection(y.dropna().index)
        X, y = X.loc[common], y.loc[common]
        fwd_raw = fwd_raw.reindex(common)
        X.index.names = ["date", "ticker"]
        y.index.names = ["date", "ticker"]
        return X, y, fwd_raw

    # ------------------------------------------------------------------ #
    def build_prediction_panel(self, factor_values: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Feature rows for every (date, ticker) with any factor value.

        Used at inference time when labels are not yet known.
        """
        cfg = self.config
        processed = {}
        for name in self.factor_names:
            mat = factor_values[name]
            processed[name] = cs_zscore(cs_winsorize(mat, cfg.winsor_pct)).stack(future_stack=True)
        X = pd.DataFrame(processed).fillna(cfg.fill_value)
        X.index.names = ["date", "ticker"]
        return X
