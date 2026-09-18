"""Vectorised cross-sectional backtesting engine.

Timing convention (look-ahead-proof by construction)
----------------------------------------------------
1. The signal is computed from data **through the close of day T**.
2. The engine trades at the **close of day T + ``execution_lag``** (default
   1 trading day).  Nothing computed on day T can plausibly be executed
   at T's own close, so this one-day lag is the honest default.
3. New weights therefore earn returns starting on day ``T + lag + 1``.

Between rebalances, weights **drift** with realised returns exactly as a
real account would: a position's dollar value compounds by ``1 + r_i``
each day, and turnover at the next rebalance is measured against the
*drifted* weights -- not the stale targets.  This inflates neither costs
nor returns and matches how prime-broker PnL actually accrues.

Costs are charged per side on total traded notional at the rebalance
date: ``cost = sum(|dw|) * bps / 1e4``.

The simulation is vectorised per rebalance segment (a few hundred tiny
matrix products for a decade of weekly rebalancing) and runs in well
under a second on the bundled dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.engine.costs import SquareRootImpact, TransactionCostModel
from alphaforge.engine.metrics import performance_summary
from alphaforge.engine.portfolio import PortfolioConstructor, TopN


@dataclass
class BacktestConfig:
    """Parameters of one backtest run."""

    rebalance: str = "W-FRI"  # pandas offset alias or 'D'
    execution_lag: int = 1  # trade at close of T + lag
    top_n: int = 15  # names per leg
    side: str = "long_short"  # 'long' | 'short' | 'long_short'
    weighting: str = "equal"  # 'equal' | 'rank' | 'zscore'
    max_weight: float | None = 0.10
    min_price: float = 5.0  # skip penny/micro-cap regimes
    min_dollar_vol_musd: float = 20.0  # 21d avg $ volume filter
    benchmark_ticker: str = "SPY"
    fallback_benchmark: str = "equal_weight"
    vol_target: float | None = None  # e.g. 0.10 = 10% ann. vol target
    max_leverage: float = 3.0
    exclude_tickers: tuple[str, ...] = ("SPY",)

    def describe(self) -> dict:
        return {
            "rebalance": self.rebalance,
            "execution_lag": self.execution_lag,
            "top_n": self.top_n,
            "side": self.side,
            "weighting": self.weighting,
            "min_price": self.min_price,
            "min_dollar_vol_musd": self.min_dollar_vol_musd,
            "benchmark": self.benchmark_ticker,
            "vol_target": self.vol_target,
        }


@dataclass
class BacktestResult:
    """Output container of one engine run."""

    config: BacktestConfig
    cost_model: TransactionCostModel | SquareRootImpact
    signal_name: str = "signal"
    net_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    n_holdings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> dict:
        return performance_summary(self.net_returns, self.benchmark_returns)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns.fillna(0.0)).cumprod()

    @property
    def benchmark_equity(self) -> pd.Series:
        return (1.0 + self.benchmark_returns.fillna(0.0)).cumprod()

    @property
    def active_returns(self) -> pd.Series:
        return self.net_returns - self.benchmark_returns

    def holdings_snapshot(self, date: pd.Timestamp | None = None) -> pd.Series:
        """Target weights at the last rebalance <= ``date``."""
        if self.weights.empty:
            return pd.Series(dtype=float)
        if date is None:
            return self.weights.iloc[-1]
        sub = self.weights.loc[:date]
        return sub.iloc[-1] if len(sub) else pd.Series(dtype=float)


class BacktestEngine:
    """Vectorised long/short backtester for cross-sectional signals.

    Parameters
    ----------
    panel:
        Price panel (OHLCV), including the benchmark ticker if available.
    constructor:
        Portfolio construction rule (default: :class:`TopN`).
    cost_model:
        Linear per-side transaction costs or per-name square-root impact.
    config:
        Backtest parameters; see :class:`BacktestConfig`.
    universe_mask:
        Optional date x ticker boolean membership matrix (e.g. from
        :class:`~alphaforge.data.pit.PointInTimeUniverse`).  A signal value
        is *tradable* only where the mask is True, so names that were not
        yet index members (or had left) are excluded exactly as a
        point-in-time strategy would have been able to trade them.
    """

    def __init__(
        self,
        panel: PricePanel,
        constructor: PortfolioConstructor | None = None,
        cost_model: TransactionCostModel | SquareRootImpact | None = None,
        config: BacktestConfig | None = None,
        universe_mask: pd.DataFrame | None = None,
    ) -> None:
        self.panel = panel
        self.constructor = constructor or TopN(
            n=(config.top_n if config else 15),
            side=(config.side if config else "long_short"),
            weighting=(config.weighting if config else "equal"),
            max_weight=(config.max_weight if config else 0.10),
        )
        self.cost_model = cost_model or TransactionCostModel()
        self.config = config or BacktestConfig()
        self.universe_mask = universe_mask
        self._adv: pd.DataFrame | None = None  # lazily: trailing 21d mean dvol
        self._dvol21: pd.DataFrame | None = None  # lazily: trailing 21d vol

    # ------------------------------------------------------------------ #
    def _liquidity_stats(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trailing ADV and daily vol (both 21d, shift-safe: include date t).

        Used by the per-name square-root impact cost model.  Both series at
        date t only use data through t's close — the same information set as
        the signal, so no lookahead.
        """
        if self._adv is None:
            self._adv = self.panel.dollar_volume.rolling(21, min_periods=5).mean()
            self._dvol21 = self.panel.returns.rolling(21, min_periods=10).std()
        return self._adv, self._dvol21

    # ------------------------------------------------------------------ #
    def _charge(self, target: pd.Series, prev_drifted, sig_date) -> tuple[float, float]:
        """Return ``(cost_return, turnover)`` for one rebalance."""
        per_name = getattr(self.cost_model, "per_name", False)
        if prev_drifted is not None:
            traded = (target - prev_drifted).abs()
            turnover = float(traded.sum())
        else:
            traded = target.abs()  # initial buy-in
            turnover = float(traded.sum())
        if per_name:
            adv, vol = self._liquidity_stats()
            return self.cost_model.cost_return_per_name(
                traded, adv.loc[sig_date], vol.loc[sig_date]
            ), turnover
        return self.cost_model.cost_return(turnover), turnover

    # ------------------------------------------------------------------ #
    def _tradable_mask(self) -> pd.DataFrame:
        """Date x ticker boolean mask of liquid, non-excluded names."""
        cfg = self.config
        close = self.panel.close
        dol = self.panel.dollar_volume.rolling(21).mean() / 1e6
        mask = (close >= cfg.min_price) & (dol >= cfg.min_dollar_vol_musd)
        for t in cfg.exclude_tickers:
            if t in mask.columns:
                mask[t] = False
        return mask

    def _returns_matrix(self) -> pd.DataFrame:
        """Daily returns with gaps forward-filled (halts hold value)."""
        close = self.panel.close.ffill()
        # leading NaNs are fine; internal gaps -> 0 via ffill above
        return close.pct_change(fill_method=None)

    def _benchmark(self, rets: pd.DataFrame) -> pd.Series:
        cfg = self.config
        if cfg.benchmark_ticker and cfg.benchmark_ticker in self.panel.tickers:
            return self.panel.close[cfg.benchmark_ticker].pct_change(fill_method=None)
        # equal-weight of the tradable universe as a robust fallback
        mask = self._tradable_mask()
        return rets.where(mask).mean(axis=1)

    # ------------------------------------------------------------------ #
    def run(self, signal: pd.DataFrame, signal_name: str = "signal") -> BacktestResult:
        """Backtest a date x ticker signal matrix.

        The signal must already be direction-corrected (high = long).
        """
        cfg = self.config
        rets = self._returns_matrix()
        mask = self._tradable_mask()

        # align signal with the panel
        idx = signal.index.intersection(rets.index)
        cols = signal.columns.intersection(rets.columns)
        sig = signal.loc[idx, cols]
        rets_a = rets.loc[idx, cols]
        mask_a = mask.loc[idx, cols]
        if self.universe_mask is not None:
            umask = self.universe_mask.reindex(index=idx, columns=cols).eq(True)
            mask_a = mask_a & umask
        sig_masked = sig.where(mask_a)

        # rebalance dates: the last trading day with a valid signal in each
        # period.  Using the actual last valid date (rather than the period
        # label, which may fall on a holiday) keeps holiday weeks trading.
        valid = sig_masked.notna().any(axis=1)
        if cfg.rebalance == "D":
            rebal_dates = sig_masked.index[valid]
        else:
            pos_series = pd.Series(np.arange(len(sig_masked)), index=sig_masked.index)
            last_pos = pos_series[valid].resample(cfg.rebalance).max().dropna()
            if len(last_pos) == 0:
                raise ValueError(
                    "No valid signal dates found for the requested rebalance frequency."
                )
            rebal_dates = sig_masked.index[last_pos.round().astype(int).to_numpy()]
        rebal_dates = pd.DatetimeIndex(rebal_dates)
        if len(rebal_dates) < 2:
            raise ValueError(
                "Fewer than two rebalance dates -- check the signal index and rebalance frequency."
            )

        lag = max(int(cfg.execution_lag), 0)
        positions = rets_a.index.get_indexer(rebal_dates) + lag
        positions = positions[positions < len(rets_a)]
        rebal_dates = rebal_dates[: len(positions)]

        net, gross, costs = [], [], []
        weights_rows, turnover_rows, n_hold = [], [], []
        prev_drifted = None  # drifted dollar weights just before a rebalance
        vol_scaler = 1.0

        strat_hist: list[float] = []  # realised strategy returns so far

        for k, (sig_date, pos) in enumerate(zip(rebal_dates, positions, strict=True)):
            target = self.constructor.weights_from_signal(sig_masked.loc[sig_date], date=sig_date)
            if target.empty:
                target = pd.Series(0.0, index=rets_a.columns)
            target = target.reindex(rets_a.columns).fillna(0.0)

            # ---- position sizing: optional volatility targeting --------
            if cfg.vol_target is not None and k > 0 and len(strat_hist) >= 21:
                realised = float(np.std(strat_hist[-63:], ddof=1) * np.sqrt(252))
                if realised > 0:
                    vol_scaler = float(np.clip(cfg.vol_target / realised, 0.0, cfg.max_leverage))
            target = target * vol_scaler

            # ---- turnover vs drifted book -------------------------------
            cost_ret, turnover = self._charge(target, prev_drifted, sig_date)

            # ---- simulate until the next effective rebalance ------------
            start = pos + 1  # first day earning new weights
            end = positions[k + 1] if k + 1 < len(positions) else len(rets_a) - 1
            if end < start:
                # next rebalance happens before this one is effective:
                # overlap -> keep latest target (rare, only with lag > freq)
                weights_rows.append((sig_date, target))
                prev_drifted = target
                continue

            seg = rets_a.iloc[start : end + 1]
            growth = (1.0 + seg).cumprod()
            # dollar weights at the START of each day s: w_k * cum(->s-1)
            prior_growth = pd.concat(
                [
                    pd.DataFrame(1.0, index=[rets_a.index[start - 1]], columns=rets_a.columns),
                    growth.iloc[:-1],
                ]
            )
            prior_growth.index = seg.index
            dollar = prior_growth.mul(target, axis=1)  # d_i(s-1)
            day_gross = (dollar * seg).sum(axis=1)  # r_p(s) = sum d_i(s-1) r_i(s)
            # charge cost on the trade day (first day of the segment)
            day_net = day_gross.copy()
            if len(day_net):
                day_net.iloc[0] -= cost_ret

            gross.append(day_gross)
            net.append(day_net)
            costs.append(pd.Series({rets_a.index[start]: cost_ret}))
            weights_rows.append((sig_date, target))
            turnover_rows.append((sig_date, turnover))
            n_hold.append((sig_date, float((target != 0).sum())))
            strat_hist.extend(day_net.tolist())

            prev_drifted = dollar.iloc[-1] if len(dollar) else target
            # dollar weights at end of last day include that day's return:
            end_growth = growth.iloc[-1] if len(growth) else pd.Series(1.0, index=rets_a.columns)
            prev_drifted = target * end_growth

        result = BacktestResult(
            config=cfg,
            cost_model=self.cost_model,
            signal_name=signal_name,
            net_returns=pd.concat(net).sort_index(),
            gross_returns=pd.concat(gross).sort_index(),
            costs=pd.concat(costs).sort_index() if costs else pd.Series(dtype=float),
            benchmark_returns=self._benchmark(rets).reindex(pd.concat(net).sort_index().index),
            turnover=pd.Series(dict(turnover_rows)).sort_index(),
            weights=pd.DataFrame(dict(weights_rows)).T.sort_index(),
            n_holdings=pd.Series(dict(n_hold), dtype=float).sort_index(),
        )
        result.weights.index.name = "signal_date"
        result.weights.columns.name = "ticker"
        return result


def run_backtest(
    panel: PricePanel,
    signal: pd.DataFrame,
    signal_name: str = "signal",
    cost_bps: float = 10.0,
    universe_mask: pd.DataFrame | None = None,
    **config_kwargs,
) -> BacktestResult:
    """Convenience one-liner: default engine + config overrides."""
    cfg = BacktestConfig(**config_kwargs)
    engine = BacktestEngine(
        panel,
        cost_model=TransactionCostModel(bps_per_side=cost_bps),
        config=cfg,
        universe_mask=universe_mask,
    )
    return engine.run(signal, signal_name=signal_name)
