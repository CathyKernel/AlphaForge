#!/usr/bin/env python
"""Full S&P 500 research pipeline: factors + the universe-bias experiments.

Reproduces the README's headline tables on the bundled ``prices_sp500``
panel: 27-factor evaluation, point-in-time universe statistics, and the
A-E experiment matrix that quantifies the survivorship / look-ahead /
neutralisation / construction / impact "bias tax".

Usage:
    python scripts/run_sp500_pipeline.py
"""

from __future__ import annotations

import json
import time

import pandas as pd

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.data.pit import PointInTimeUniverse
from alphaforge.data.universe import Universe
from alphaforge.engine import BacktestConfig, BacktestEngine, run_backtest
from alphaforge.engine.costs import SquareRootImpact, TransactionCostModel
from alphaforge.engine.portfolio import TopN
from alphaforge.factors import FactorEvaluator, FactorLibrary, Neutralizer
from alphaforge.optimize import MeanVariance
from alphaforge.riskmodel import StyleRiskModel

BT = {
    "rebalance": "MS",
    "side": "long",
    "top_n": 15,
    "cost_bps": 10.0,
    "max_weight": 0.08,
}


def metrics(res) -> dict:
    s = res.summary()
    return {
        "net_cagr": round(s.get("cagr", float("nan")), 4),
        "sharpe": round(s.get("sharpe", float("nan")), 2),
        "max_dd": round(s.get("max_drawdown", float("nan")), 3),
        "info_ratio": round(s.get("information_ratio", float("nan")), 2),
        "turnover": round(float(res.turnover.mean()), 2) if len(res.turnover) else None,
        "cost_drag": round(float(res.costs.mean() * 12), 4) if len(res.costs) else None,
    }


def main() -> int:
    results = get_config().results_dir
    results.mkdir(exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # 1. Data + point-in-time membership ------------------------------- #
    panel = DataRepository().load_panel("prices_sp500")
    uni = Universe.sp500()
    log(f"panel loaded: {panel!r}")

    pit = PointInTimeUniverse.from_snapshot(panel, uni)
    log(f"PIT universe: {pit!r}")
    (results / "pit_stats.json").write_text(json.dumps(pit.summary(), indent=2))

    # 2. Factor library (27 factors incl. value from fundamentals) ----- #
    lib = FactorLibrary(panel)
    log("computing 27 factors on 1.48M rows ...")
    factors = lib.compute_all()
    log(f"factors computed: {len(factors)}")

    ev = FactorEvaluator(panel, horizon=5)
    stats = ev.evaluate_all(factors)
    stats.to_csv(results / "factor_evaluation.csv", index=False)
    log(f"factor evaluation: {len(stats)} rows -> factor_evaluation.csv")
    print(
        stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
        .head(12)
        .to_string(index=False)
    )

    # 3. Composite signal (top-5 |ICIR|, in-sample selection disclosed) - #
    ranked = stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
    top = ranked["name"].head(5).tolist()
    direction = {
        n: (1.0 if s > 0 else -1.0) for n, s in ranked.set_index("name").loc[top, "ic_mean"].items()
    }
    signal = lib.composite(names=top, direction=direction)
    log(f"composite signal from {top}")

    # 4. Experiment matrix ---------------------------------------------- #
    experiments: dict[str, dict] = {}

    # A. static universe (today's 503 names -- the survivorship-biased view)
    res_a = run_backtest(panel, signal, signal_name="composite_static", **BT)
    experiments["A_static_universe"] = metrics(res_a)
    log(f"A static: {experiments['A_static_universe']}")

    # B. PIT universe (only names that were index members at the time)
    res_b = run_backtest(
        panel, signal, signal_name="composite_pit", universe_mask=pit.members, **BT
    )
    experiments["B_pit_universe"] = metrics(res_b)
    log(f"B PIT: {experiments['B_pit_universe']}")

    # C. PIT + sector/beta-neutralised signal
    neu = Neutralizer(universe=uni, panel=panel, sector=True, beta=True, winsorize=True)
    sig_neu = neu.transform(signal)
    res_c = run_backtest(
        panel, sig_neu, signal_name="composite_pit_neutral", universe_mask=pit.members, **BT
    )
    experiments["C_pit_sector_beta_neutral"] = metrics(res_c)
    log(f"C neutral: {experiments['C_pit_sector_beta_neutral']}")

    # D. PIT + neutral + mean-variance construction (style-risk aware)
    risk_model = StyleRiskModel(panel)
    log("risk model fitted")
    mv = MeanVariance(
        risk_model,
        universe=uni,
        risk_aversion=8.0,
        max_weight=0.05,
        sector_cap=0.30,
        turnover_penalty=0.05,
        side="long",
        fallback_n=15,
    )
    cfg = BacktestConfig(rebalance="MS", side="long", top_n=15, max_weight=0.08)
    engine = BacktestEngine(
        panel,
        constructor=mv,
        cost_model=TransactionCostModel(bps_per_side=10.0),
        config=cfg,
        universe_mask=pit.members,
    )
    res_d = engine.run(sig_neu, signal_name="mv_pit")
    experiments["D_pit_mean_variance"] = metrics(res_d)
    log(f"D MV: {experiments['D_pit_mean_variance']}")

    # E. PIT + neutral + square-root market impact (institutional costs)
    engine = BacktestEngine(
        panel,
        constructor=TopN(n=15, side="long", max_weight=0.08),
        cost_model=SquareRootImpact(book_value=50_000_000.0, fixed_bps=5.0, impact_coef=1.0),
        config=cfg,
        universe_mask=pit.members,
    )
    res_e = engine.run(sig_neu, signal_name="sqrt_impact_pit")
    experiments["E_pit_sqrt_impact"] = metrics(res_e)
    log(f"E impact: {experiments['E_pit_sqrt_impact']}")

    # benchmark
    spy = panel.close["SPY"].dropna()
    bench = {
        "SPY": {
            "net_cagr": round(float((spy.iloc[-1] / spy.iloc[0]) ** (252 / len(spy)) - 1), 4),
            "sharpe": None,
            "note": "buy-and-hold SPY, gross of costs",
        }
    }

    out = {
        "universe": "S&P 500 (503 tickers + SPY), 2015-01-02..2026-09-17",
        "signal": "composite top-5 |ICIR| (in-sample selection disclosed)",
        "backtest_config": BT,
        "pit_summary": pit.summary(),
        "experiments": experiments,
        "benchmark": bench,
    }
    (results / "universe_experiments.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(experiments).T.to_csv(results / "universe_experiments.csv")

    log("=== comparison table ===")
    print(pd.DataFrame(experiments).T.to_string())
    a, b = experiments["A_static_universe"], experiments["B_pit_universe"]
    print("\nbias tax (A - B): static-universe flattery")
    print(
        f"  Sharpe {a['sharpe']} -> {b['sharpe']}  |  CAGR {a['net_cagr']:.1%} -> {b['net_cagr']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
