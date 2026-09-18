"""Tearsheet generation: a full institutional-style report in one call.

Produces, for a given backtest result:

* ``<name>_tearsheet.png``   -- multi-panel performance chart page
* ``<name>_monthly.png``     -- monthly return heatmap
* ``<name>_metrics.json``    -- machine-readable metric dictionary
* ``<name>_summary.md``      -- human-readable summary block (README-ready)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alphaforge.engine import BacktestResult
from alphaforge.reporting import plots
from alphaforge.risk.analytics import drawdown_episodes, risk_summary


def _fmt(v: float, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if pct:
        return f"{v * 100:+.2f}%"
    return f"{v:+.{digits}f}"


def build_metrics_block(result: BacktestResult) -> dict:
    s = result.summary()
    risk = risk_summary(result.net_returns, result.benchmark_returns)
    dd_eps = drawdown_episodes(result.net_returns, top=3)
    return {
        "signal": result.signal_name,
        "config": result.config.describe(),
        "costs_bps_per_side": result.cost_model.bps_per_side,
        "performance": {k: v for k, v in s.items() if not isinstance(v, (str,))},
        "risk": risk,
        "drawdown_episodes": [e.as_dict() for e in dd_eps],
        "turnover": {
            "avg_turnover_per_rebalance": float(result.turnover.mean()),
            "avg_n_holdings": float(result.n_holdings.mean()),
            "cost_drag_ann": float(-result.costs.sum() / max(len(result.net_returns), 1) * 252),
        },
    }


def tearsheet(result: BacktestResult, out_dir: str | Path, name: str = "backtest") -> dict:
    """Render the full report; returns the metrics dictionary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics_block(result)

    fig = plots.plot_equity_curves(
        result.net_returns,
        result.benchmark_returns,
        title=f"AlphaForge backtest - {result.signal_name}",
    )
    plots.save(fig, out / f"{name}_equity.png")

    fig = plots.plot_rolling_sharpe(result.net_returns)
    plots.save(fig, out / f"{name}_rolling_sharpe.png")

    fig = plots.plot_monthly_heatmap(result.net_returns)
    plots.save(fig, out / f"{name}_monthly.png")

    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    s = metrics["performance"]
    lines = [
        f"## {result.signal_name} - backtest summary",
        "",
        f"- Period: {result.net_returns.index[0]:%Y-%m-%d} .. "
        f"{result.net_returns.index[-1]:%Y-%m-%d} "
        f"({metrics['performance'].get('n_days', 0)} trading days)",
        f"- Rebalance: {result.config.describe()['rebalance']}, "
        f"{result.config.side}, top {result.config.top_n} per leg, "
        f"{result.cost_model.bps_per_side:.0f} bps per side",
        f"- Net CAGR: **{_fmt(s.get('cagr'), pct=True)}** "
        f"(benchmark {_fmt(s.get('benchmark_cagr'), pct=True)})",
        f"- Sharpe: **{_fmt(s.get('sharpe'))}** (benchmark {_fmt(s.get('benchmark_sharpe'))})",
        f"- Max drawdown: {_fmt(s.get('max_drawdown'), pct=True)}",
        f"- Information ratio vs benchmark: {_fmt(s.get('information_ratio'))}",
        f"- Turnover per rebalance: "
        f"{metrics['turnover']['avg_turnover_per_rebalance']:.2f}, "
        f"cost drag {_fmt(metrics['turnover']['cost_drag_ann'], pct=True)}/yr",
        "",
    ]
    (out / f"{name}_summary.md").write_text("\n".join(lines))
    return metrics
