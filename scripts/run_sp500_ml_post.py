#!/usr/bin/env python
"""Post-process the S&P 500 ML run: plots, PIT backtest, PSI drift.

Reads the artifacts written by ``run_sp500_ml.py`` under
``results/ml_run_sp500/`` and adds:
* ``ic_cumulative_sp500.png`` — cumulative OOS rank IC per model
* ``ml_equity_sp500.png`` + ``backtest_metrics.json`` — OOS ensemble
  backtest on the point-in-time universe
* ``feature_psi_rolling.csv`` + ``psi_summary.json`` — rolling PSI
  feature-drift (causal; backtests a production retrain trigger)

Usage:
    python scripts/run_sp500_ml_post.py
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.data.pit import PointInTimeUniverse
from alphaforge.data.universe import Universe
from alphaforge.engine import run_backtest
from alphaforge.factors.base import REGISTRY
from alphaforge.ml.features import FeatureConfig, FeaturePanelBuilder
from alphaforge.ml.monitoring import rolling_feature_psi


def main() -> int:
    out_dir = get_config().results_dir / "ml_run_sp500"
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # 1. cumulative OOS IC curve (from the pipeline's per-date IC table) -- #
    ic = pd.read_csv(out_dir / "ic_by_date.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for col, color in [("lightgbm", "#1a6faf"), ("ensemble_causal", "#d95f02"), ("ridge", "#666")]:
        if col in ic.columns:
            ax.plot(ic.index, ic[col].cumsum(), lw=1.5, color=color, label=col)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.legend(loc="upper left")
    ax.set_title("Cumulative OOS rank IC — S&P 500, purged walk-forward (6 folds)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative rank IC")
    fig.savefig(out_dir / "ic_cumulative_sp500.png", dpi=120)
    plt.close(fig)
    log("IC curve saved")

    # 2. OOS ensemble backtest on the PIT universe ------------------------ #
    panel = DataRepository().load_panel("prices_sp500")
    uni = Universe.sp500()
    preds = pd.read_parquet(out_dir / "oos_predictions.parquet")
    sig = (
        preds[["ensemble_causal"]]
        .reset_index()
        .pivot(index="date", columns="ticker", values="ensemble_causal")
    )
    log(f"signal panel: {sig.shape}")

    pit = PointInTimeUniverse.from_snapshot(panel, uni)
    log(f"PIT: {pit!r}")

    res = run_backtest(
        panel,
        sig,
        signal_name="ml_ensemble_pit",
        rebalance="MS",
        side="long",
        top_n=15,
        cost_bps=10.0,
        max_weight=0.08,
        universe_mask=pit.members,
    )
    s = res.summary()
    bt = {
        "net_cagr": round(s["cagr"], 4),
        "sharpe": round(s["sharpe"], 2),
        "max_dd": round(s["max_drawdown"], 3),
        "info_ratio": round(s["information_ratio"], 2),
        "turnover_per_rebalance": round(float(res.turnover.mean()), 2),
    }
    (out_dir / "backtest_metrics.json").write_text(json.dumps(bt, indent=2))
    log(f"ML OOS ensemble PIT backtest: {bt}")

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.plot(res.equity.index, res.equity, lw=1.6, color="#1a6faf", label="ML OOS ensemble (net)")
    ax.plot(res.benchmark_equity.index, res.benchmark_equity, lw=1.2, color="#888", label="SPY")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.set_title("ML OOS ensemble vs SPY — S&P 500 point-in-time universe, 10 bps/side")
    fig.savefig(out_dir / "ml_equity_sp500.png", dpi=120)
    plt.close(fig)
    log("equity curve saved")

    # 3. rolling PSI feature drift (causal, backtests the retrain trigger) #
    names = sorted(n for n, f in REGISTRY.items() if f.category != "value")
    fb = FeaturePanelBuilder(panel, factor_names=names, config=FeatureConfig(label_horizon=5))
    log("building feature matrix for PSI ...")
    X, _, _ = fb.build()
    log(f"X: {X.shape}")
    psi_roll = rolling_feature_psi(X, window=252, step=126)
    psi_roll.round(4).to_csv(out_dir / "feature_psi_rolling.csv")
    mean_psi = psi_roll.mean().sort_values(ascending=False)
    log("mean PSI per feature (top 8):")
    print(mean_psi.head(8).to_string())
    (out_dir / "psi_summary.json").write_text(
        json.dumps(
            {
                "mean_psi": mean_psi.round(4).to_dict(),
                "n_eval_dates": int(len(psi_roll)),
                "significant_at_some_point": {
                    c: int((psi_roll[c] > 0.25).sum())
                    for c in psi_roll.columns
                    if (psi_roll[c] > 0.25).any()
                },
            },
            indent=2,
        )
    )
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
