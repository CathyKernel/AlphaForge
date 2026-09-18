"""Example 3: plug in your own factor.

AlphaForge factors are tiny classes: subclass :class:`Factor`, implement
``compute(panel) -> DataFrame`` and register it.  Two contracts matter:

1. values at date T must use information available at T's close
   (the engine owns the T+1 execution lag -- never shift inside a factor);
2. return raw values -- cross-sectional standardisation happens
   downstream.

This example adds a market-neutral residual volatility factor, evaluates
it, and runs a quick backtest with it.

Run:  python examples/03_custom_factor.py
"""

import numpy as np

from alphaforge.data import PricePanel
from alphaforge.engine import run_backtest
from alphaforge.factors import Factor, FactorEvaluator, register
from alphaforge.factors.library import FactorLibrary


@register
class LowVolResidual(Factor):
    """Residual (market-neutral) 21-day volatility.

    Regresses each stock's returns on the equal-weight market with a
    rolling 63d beta, then takes the volatility of what is left over.
    """

    name = "example_idio_vol"
    category = "volatility"
    description = "rolling residual vol vs equal-weight market (example)"

    def compute(self, panel: PricePanel) -> np.ndarray:
        rets = panel.returns
        market = rets.mean(axis=1)
        cov = rets.rolling(63).cov(market)
        beta = cov.div(market.rolling(63).var(), axis=0)
        resid = rets.sub(beta.mul(market, axis=0))
        return resid.rolling(21).std() * np.sqrt(252)


def main() -> None:
    panel = PricePanel.from_parquet("data/cache/prices.parquet")
    lib = FactorLibrary(panel)

    # the custom factor is now part of the registry
    ev = FactorEvaluator(panel, horizon=5)
    stats = ev.evaluate(lib.compute("example_idio_vol"))
    print(f"custom factor IC: {stats.ic_mean:+.4f}, ICIR: {stats.ic_ir:+.3f}")

    # and directly backtestable
    signal = -lib.compute("example_idio_vol")  # long LOW residual vol
    res = run_backtest(
        panel, signal, signal_name="example_idio_vol", rebalance="MS", top_n=15, side="long"
    )
    s = res.summary()
    print(f"backtest: CAGR {s['cagr']:+.2%}, Sharpe {s['sharpe']:.2f}")


if __name__ == "__main__":
    main()
