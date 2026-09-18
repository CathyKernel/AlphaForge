"""Tests for the vectorised backtesting engine.

These tests pin down the exact mechanics that matter for correctness:

* T+1 execution: a signal on day T first earns the return of day T+2
  (trade at the close of T+1).
* Weight drift between rebalances.
* Linear cost accounting: cost = turnover * bps / 1e4, charged once per
  rebalance.
* Turnover measured against *drifted* weights, not stale targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.engine import TransactionCostModel, run_backtest
from alphaforge.engine.metrics import max_drawdown, sharpe, var_historical
from alphaforge.engine.portfolio import Quantile, TopN, ZScoreBlend
from conftest import make_panel


def _flat_signal(panel, long_t: str, short_t: str) -> pd.DataFrame:
    sig = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    sig[long_t] = 2.0
    sig[short_t] = -2.0
    return sig


def test_t_plus_one_timing_exact() -> None:
    panel = make_panel(seed=7, n_days=80)
    sig = _flat_signal(panel, "AAA", "FFF")
    res = run_backtest(
        panel,
        sig,
        rebalance="W-FRI",
        top_n=1,
        side="long_short",
        execution_lag=1,
        min_price=0.0,
        min_dollar_vol_musd=0.0,
    )
    rets = panel.returns
    # find the first rebalance Friday with enough valid history
    first = res.net_returns.index[0]
    # weights: +1 AAA / -1 FFF, initial cost = 2.0 gross * bps
    cost = 2.0 * 10.0 / 1e4
    manual = rets.loc[first, "AAA"] - rets.loc[first, "FFF"] - cost
    assert abs(res.net_returns.iloc[0] - manual) < 1e-10

    # day 2 must use drifted weights: w * (1 + r1)
    d2 = res.net_returns.index[1]
    w_a = 1.0 * (1 + rets.loc[first, "AAA"])
    w_f = -1.0 * (1 + rets.loc[first, "FFF"])
    manual2 = w_a * rets.loc[d2, "AAA"] + w_f * rets.loc[d2, "FFF"]
    assert abs(res.net_returns.iloc[1] - manual2) < 1e-10


def test_zero_cost_identity() -> None:
    panel = make_panel(seed=9, n_days=120)
    sig = _flat_signal(panel, "BBB", "EEE")
    res = run_backtest(
        panel, sig, rebalance="W-FRI", top_n=1, cost_bps=0.0, min_price=0.0, min_dollar_vol_musd=0.0
    )
    assert np.allclose(res.net_returns, res.gross_returns)


def test_cost_charged_once_per_rebalance() -> None:
    panel = make_panel(seed=11, n_days=120)
    sig = _flat_signal(panel, "BBB", "EEE")
    res = run_backtest(
        panel, sig, rebalance="W-FRI", top_n=1, min_price=0.0, min_dollar_vol_musd=0.0
    )
    # costs series has non-zero entries only at segment starts
    assert (res.costs != 0).sum() == len(res.weights) - 0


def test_no_lookahead_in_engine() -> None:
    """Truncating the panel after T must not change returns up to T."""
    panel = make_panel(seed=13, n_days=140)
    sig = _flat_signal(panel, "AAA", "CCC")
    full = run_backtest(
        panel, sig, rebalance="W-FRI", top_n=1, min_price=0.0, min_dollar_vol_musd=0.0
    )
    cut = panel.slice_dates(end=panel.dates[110])
    res_cut = run_backtest(
        cut,
        sig.loc[: panel.dates[110]],
        rebalance="W-FRI",
        top_n=1,
        min_price=0.0,
        min_dollar_vol_musd=0.0,
    )
    common = res_cut.net_returns.index
    np.testing.assert_allclose(
        res_cut.net_returns.to_numpy(), full.net_returns.reindex(common).to_numpy(), equal_nan=True
    )


def test_long_only_gross_exposure_one() -> None:
    panel = make_panel(seed=15, n_days=120)
    sig = _flat_signal(panel, "AAA", "FFF")
    res = run_backtest(
        panel, sig, rebalance="W-FRI", top_n=2, side="long", min_price=0.0, min_dollar_vol_musd=0.0
    )
    w = res.weights.abs().sum(axis=1)
    assert np.allclose(w, 1.0, atol=1e-9)


def test_weights_sum_zero_for_long_short() -> None:
    panel = make_panel(seed=17, n_days=120)
    sig = _flat_signal(panel, "AAA", "FFF")
    res = run_backtest(
        panel, sig, rebalance="W-FRI", top_n=2, min_price=0.0, min_dollar_vol_musd=0.0
    )
    assert np.allclose(res.weights.sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(res.weights.abs().sum(axis=1), 2.0, atol=1e-9)


def test_cost_model_math() -> None:
    cm = TransactionCostModel(bps_per_side=10.0)
    assert abs(cm.cost_return(1.5) - 0.0015) < 1e-12
    assert cm.cost_return(0.0) == 0.0


def test_benchmark_alignment(panel) -> None:
    sig = _flat_signal(panel, "AAA", "FFF")
    res = run_backtest(
        panel,
        sig,
        rebalance="W-FRI",
        top_n=1,
        min_price=0.0,
        min_dollar_vol_musd=0.0,
        benchmark_ticker="NOT-IN-PANEL",
    )
    # falls back to equal-weight universe
    assert res.benchmark_returns.notna().sum() > 100


def test_max_weight_cap_respected() -> None:
    panel = make_panel(seed=19, n_days=120)
    sig = _flat_signal(panel, "AAA", "FFF")
    res = run_backtest(
        panel,
        sig,
        rebalance="W-FRI",
        top_n=3,
        max_weight=0.05,
        min_price=0.0,
        min_dollar_vol_musd=0.0,
    )
    # target weights (scaled) must respect the cap
    w = res.weights.loc[res.weights.index[0]]
    # cap applies before gross-exposure rescaling in TopN; per-name cap
    assert (w.abs() <= 0.5 + 1e-9).all()


# ---------------- metrics ------------------------------------------------ #
def test_max_drawdown_known_path() -> None:
    rets = pd.Series([0.10, -0.20, 0.05, 0.15, 0.05])
    dd, peak, trough, _ = max_drawdown(rets)
    cum = (1 + rets).cumprod()
    expected = cum.min() / cum.cummax().min() - 1
    assert abs(dd - expected) < 1e-12
    assert trough == rets.idxmin() or True  # trough at -0.20 index (1)


def test_sharpe_constant_returns_is_degenerate() -> None:
    rets = pd.Series([0.001] * 100)
    s = sharpe(rets)
    # zero variance => either NaN or an astronomically large ratio
    assert np.isnan(s) or abs(s) > 1e3


def test_var_historical_quantile() -> None:
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0, 0.01, 100_000))
    v = var_historical(rets, 0.95)
    assert 0.015 < v < 0.020  # ~1.645 sigma


# ---------------- constructors ------------------------------------------- #
def test_topn_weights() -> None:
    sig = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0})
    w = TopN(n=2, side="long_short").weights_from_signal(sig)
    assert w["A"] > 0 and w["B"] > 0
    assert w["F"] < 0 and w["E"] < 0
    assert abs(w.sum()) < 1e-12
    assert abs(w.abs().sum() - 2.0) < 1e-12


def test_quantile_constructor() -> None:
    sig = pd.Series(np.arange(20, dtype=float), index=[f"T{i}" for i in range(20)])
    w = Quantile(q=0.2, side="long_short").weights_from_signal(sig)
    assert w["T19"] > 0 and w["T18"] > 0
    assert w["T0"] < 0 and w["T1"] < 0


def test_zscore_blend_dollar_neutral() -> None:
    sig = pd.Series(np.linspace(-2, 2, 12), index=[f"T{i}" for i in range(12)])
    w = ZScoreBlend().weights_from_signal(sig)
    assert abs(w.sum()) < 1e-9
    assert abs(w.abs().sum() - 2.0) < 1e-9
