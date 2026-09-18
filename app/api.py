"""AlphaForge dashboard — REST API layer (FastAPI).

All heavy artefacts (price panel, factor library, factor statistics,
flagship backtests) are computed lazily exactly once and cached in
process memory behind a lock, so the first request pays the cost and
every later request is instant.

The API is deliberately JSON-only and framework-light: the frontend is
a hand-written single-page app (``app/static/``) with no build step and
no CDN dependency, so the whole dashboard works offline on a fresh
clone.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from alphaforge.config import get_config
from alphaforge.data.cache import DataRepository
from alphaforge.data.quality import DataQualityChecker
from alphaforge.engine import run_backtest
from alphaforge.engine.metrics import drawdown_series, monthly_returns, rolling_sharpe
from alphaforge.factors import FactorEvaluator
from alphaforge.factors.base import REGISTRY
from alphaforge.factors.library import FactorLibrary

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------- #
# Lazy, thread-safe shared state
# --------------------------------------------------------------------- #
class DashboardState:
    """Caches every expensive artefact the API serves.

    ``create_app()`` constructs one instance; tests can inject a state
    primed with synthetic data instead of the bundled panel.
    """

    def __init__(self) -> None:
        # RLock (reentrant): flagship() holds the lock while calling
        # factor_stats(), which re-acquires it — a plain Lock would
        # deadlock the first /api/overview request.
        self._lock = threading.RLock()
        self._panel: Any = None
        self._library: FactorLibrary | None = None
        self._factor_stats: dict[int, pd.DataFrame] = {}
        self._flagship: dict[str, dict[str, Any]] = {}
        self._pit: Any = None

    # -- lazy properties ------------------------------------------------ #
    @property
    def panel_name(self) -> str:
        """The panel the dashboard serves: full S&P 500 when bundled,
        otherwise the S&P 100 demo panel (e.g. in trimmed checkouts)."""
        repo = DataRepository()
        return "prices_sp500" if repo.has_panel("prices_sp500") else "prices"

    @property
    def panel(self):
        with self._lock:
            if self._panel is None:
                self._panel = DataRepository().load_panel(self.panel_name)
            return self._panel

    @property
    def pit(self):
        """Point-in-time universe (lazy; cached under the same lock)."""
        with self._lock:
            if self._pit is None:
                from alphaforge.data.pit import PointInTimeUniverse
                from alphaforge.data.universe import Universe

                self._pit = PointInTimeUniverse.from_snapshot(self.panel, Universe.sp500())
            return self._pit

    @property
    def registry(self) -> dict[str, Any]:
        """Global factor registry (name -> Factor instance)."""
        return REGISTRY

    @property
    def library(self) -> FactorLibrary:
        with self._lock:
            if self._library is None:
                # self.panel (property) — NOT self._panel: the panel must
                # be lazily loaded if no endpoint touched it yet.
                lib = FactorLibrary(self.panel)
                lib.compute_all()
                self._library = lib
            return self._library

    def factor_stats(self, horizon: int) -> pd.DataFrame:
        with self._lock:
            if horizon not in self._factor_stats:
                ev = FactorEvaluator(self.panel, horizon=horizon)
                self._factor_stats[horizon] = ev.evaluate_all(self.library.compute_all())
            return self._factor_stats[horizon]

    def flagship(self, key: str) -> dict[str, Any]:
        """Re-run the two headline strategies once and cache the bundle.

        ``composite`` — top-5 factors by |ICIR|, equal-weight composite,
        long-only, monthly rebalance (the README's factor result).
        ``ml_ensemble`` — strictly OOS ensemble predictions from
        ``results/ml_run/oos_predictions.parquet``.
        """
        with self._lock:
            if key in self._flagship:
                return self._flagship[key]

            if key == "composite":
                stats = self.factor_stats(5).sort_values(
                    "ic_ir", key=lambda c: c.abs(), ascending=False
                )
                top = stats["name"].head(5).tolist()
                direction = {
                    n: (1.0 if s > 0 else -1.0)
                    for n, s in stats.set_index("name").loc[top, "ic_mean"].items()
                }
                signal = self.library.composite(names=top, direction=direction)
                name = "composite_top5"
                kwargs: dict[str, Any] = {
                    "rebalance": "MS",
                    "side": "long",
                    # top-15 names: matches the README's headline composite
                    # (CAGR 27.6% / Sharpe 1.04 / IR 0.80) — top-5 names is
                    # a different, more concentrated strategy.
                    "top_n": 15,
                    "cost_bps": 10.0,
                    # point-in-time membership: only names that were index
                    # members on each date are tradable
                    "universe_mask": self.pit.members,
                }
            elif key == "ml_ensemble":
                path = get_config().results_dir / "ml_run_sp500" / "oos_predictions.parquet"
                if not path.exists():
                    path = get_config().results_dir / "ml_run" / "oos_predictions.parquet"
                if not path.exists():
                    raise FileNotFoundError("results/ml_run*/oos_predictions.parquet")
                preds = pd.read_parquet(path)
                col = "ensemble_causal" if "ensemble_causal" in preds.columns else "ensemble"
                signal = preds[col].unstack("ticker")
                name = "ml_ensemble_oos"
                kwargs = {
                    "rebalance": "MS",
                    "side": "long",
                    "top_n": 15,
                    "cost_bps": 10.0,
                    "universe_mask": self.pit.members,
                }
            else:
                raise KeyError(key)

            result = run_backtest(self.panel, signal, signal_name=name, **kwargs)
            # the mask is a DataFrame (not JSON-serialisable) — describe it
            json_params = {
                k: (v if k != "universe_mask" else "point-in-time membership matrix")
                for k, v in kwargs.items()
            }
            bundle = _backtest_bundle(result, name=name, params=json_params)
            self._flagship[key] = bundle
            return bundle


STATE = DashboardState()


# --------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------- #
def _series(s: pd.Series, ndigits: int | None = None) -> list[float]:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if ndigits is not None:
        s = s.round(ndigits)
    return [None if not np.isfinite(v) else float(v) for v in s.to_numpy()]


def _dates(idx: pd.Index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _risk_block(returns: pd.Series) -> dict[str, Any]:
    """Tail-risk statistics for one return stream."""
    from alphaforge.risk import analytics as risk

    return risk.risk_summary(returns.dropna())


def _backtest_bundle(result, name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Everything the frontend needs to draw one backtest view."""
    net = result.net_returns
    cum = result.equity
    bench = result.benchmark_equity
    dd = drawdown_series(cum)
    rs = rolling_sharpe(net, window=126)
    mon = monthly_returns(net)
    month_nums = {
        m: i
        for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
        )
    }
    mon = mon[[c for c in mon.columns if c in month_nums]]  # calendar order
    holdings = result.holdings_snapshot()
    holdings = holdings[holdings.abs() > 1e-9].sort_values(ascending=False)

    summary = result.summary()
    keep = [
        "total_return",
        "cagr",
        "benchmark_cagr",
        "ann_vol",
        "sharpe",
        "benchmark_sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "information_ratio",
        "tracking_error",
        "alpha",
        "beta",
        "up_capture",
        "down_capture",
        "var95_historical",
        "var95_cornish_fisher",
        "cvar95",
        "skew",
        "excess_kurtosis",
        "pct_positive_days",
    ]
    metrics = {k: round(float(summary[k]), 4) for k in keep if k in summary}

    return {
        "name": name,
        "params": params,
        "dates": _dates(net.index),
        "equity": _series(cum, 4),
        "benchmark_equity": _series(bench, 4),
        "drawdown": _series(dd, 4),
        "rolling_sharpe": _series(rs, 2),
        "monthly": {
            "years": [int(y) for y in mon.index],
            "months": [month_nums[str(m)] for m in mon.columns],
            "values": [
                [None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in mon.to_numpy()
            ],
        },
        "metrics": metrics,
        "turnover_mean": round(float(result.turnover.mean()), 3),
        "n_trading_days": int(len(net)),
        "holdings": [{"ticker": str(t), "weight": round(float(w), 4)} for t, w in holdings.items()],
        "risk": _risk_block(net),
    }


# --------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------- #
@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "alphaforge-dashboard"}


@router.get("/overview")
def overview() -> dict[str, Any]:
    panel = STATE.panel
    quality = DataQualityChecker().run(panel).summary()
    cfg = get_config()

    ml_summary: list[dict[str, Any]] = []
    ml_dir = "ml_run_sp500" if (cfg.results_dir / "ml_run_sp500").exists() else "ml_run"
    ic_path = cfg.results_dir / ml_dir / "ic_summary.csv"
    if ic_path.exists():
        df = pd.read_csv(ic_path)
        ml_summary = df.to_dict(orient="records")

    flagship: dict[str, Any] = {}
    for key, label in (("composite", "factor composite"), ("ml_ensemble", "ML ensemble OOS")):
        try:
            bundle = STATE.flagship(key)
            flagship[key] = {
                "label": label,
                "metrics": bundle["metrics"],
                "dates": bundle["dates"],
                "equity": bundle["equity"],
                "benchmark_equity": bundle["benchmark_equity"],
            }
        except (FileNotFoundError, KeyError):
            flagship[key] = None

    return {
        "data": {
            "tickers": int(len(panel.tickers)),
            "trading_days": int(len(panel)),
            "first_date": str(panel.dates.min().date()),
            "last_date": str(panel.dates.max().date()),
            "source": "Yahoo Finance (yfinance), split-adjusted OHLCV",
        },
        "factors": {
            "n_factors": len(STATE.library.compute_all()),
            "categories": sorted({f.category for f in STATE.registry.values()}),
        },
        "quality": [
            {
                "name": str(r["name"]),
                "passed": bool(r["passed"]),
                "severity": str(r["severity"]),
                "n_flagged": int(r["n_flagged"]),
            }
            for _, r in quality.iterrows()
        ],
        "ml": ml_summary,
        "flagship": flagship,
    }


@router.get("/factors")
def factors(
    horizon: int = Query(default=5, ge=1, le=63, description="IC horizon in trading days"),
) -> dict[str, Any]:
    stats = STATE.factor_stats(horizon).copy()
    registry = STATE.registry
    stats["category"] = stats["name"].map(lambda n: registry[n].category)
    stats["description"] = stats["name"].map(lambda n: registry[n].description)
    cols = [
        "name",
        "category",
        "description",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "ic_tstat",
        "ic_positive_rate",
        "turnover_daily",
        "quantile_spread_ann",
        "quantile_spread_sharpe",
        "monotonicity",
    ]
    rows = []
    for _, r in stats[cols].iterrows():
        rows.append(
            {
                "name": r["name"],
                "category": r["category"],
                "description": r["description"],
                "ic_mean": round(float(r["ic_mean"]), 4),
                "ic_std": round(float(r["ic_std"]), 4),
                "ic_ir": round(float(r["ic_ir"]), 3),
                "ic_tstat": round(float(r["ic_tstat"]), 2),
                "ic_positive_rate": round(float(r["ic_positive_rate"]), 3),
                "turnover_daily": round(float(r["turnover_daily"]), 3),
                "quantile_spread_ann": round(float(r["quantile_spread_ann"]), 4),
                "quantile_spread_sharpe": round(float(r["quantile_spread_sharpe"]), 3),
                "monotonicity": round(float(r["monotonicity"]), 2),
            }
        )
    return {"horizon": horizon, "factors": rows}


@router.get("/factors/decay")
def factor_decay(
    name: str = Query(description="factor name"),
    horizon: int = Query(default=5, ge=1, le=63),
) -> dict[str, Any]:
    lib = STATE.library
    if name not in STATE.registry:
        raise HTTPException(status_code=404, detail=f"unknown factor: {name}")
    ev = FactorEvaluator(STATE.panel, horizon=horizon)
    decay = ev.ic_decay(lib.compute(name))
    return {
        "name": name,
        "horizons": [int(h) for h in decay.index],
        "ic": _series(decay, 5),
    }


class BacktestRequest(BaseModel):
    factor: str = Field(description="factor name, or 'composite_top5'")
    rebalance: str = Field(default="W-FRI")
    side: str = Field(default="long_short")
    top_n: int = Field(default=15, ge=1, le=100)
    cost_bps: float = Field(default=10.0, ge=0.0, le=50.0)
    horizon: int = Field(default=5, ge=1, le=63)


@router.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    lib = STATE.library
    panel = STATE.panel

    if req.rebalance not in ("D", "W-FRI", "W-MON", "MS", "ME"):
        raise HTTPException(status_code=422, detail=f"bad rebalance: {req.rebalance}")
    if req.side not in ("long", "short", "long_short"):
        raise HTTPException(status_code=422, detail=f"bad side: {req.side}")

    if req.factor == "composite_top5":
        stats = STATE.factor_stats(req.horizon).sort_values(
            "ic_ir", key=lambda c: c.abs(), ascending=False
        )
        top = stats["name"].head(5).tolist()
        direction = {
            n: (1.0 if s > 0 else -1.0)
            for n, s in stats.set_index("name").loc[top, "ic_mean"].items()
        }
        signal = lib.composite(names=top, direction=direction)
        name = "composite_top5"
    elif req.factor in STATE.registry:
        stats = STATE.factor_stats(req.horizon).set_index("name")
        signal = lib.compute(req.factor)
        if stats.loc[req.factor, "ic_mean"] < 0:
            signal = -signal  # direction-correct: engine longs high values
        name = req.factor
    else:
        raise HTTPException(status_code=404, detail=f"unknown factor: {req.factor}")

    result = run_backtest(
        panel,
        signal,
        signal_name=name,
        rebalance=req.rebalance,
        top_n=req.top_n,
        side=req.side,
        cost_bps=req.cost_bps,
    )
    return _backtest_bundle(result, name=name, params=req.model_dump())


@router.get("/ml")
def ml() -> dict[str, Any]:
    cfg = get_config()
    path = cfg.results_dir / "ml_run"
    if not (path / "ic_by_date.csv").exists():
        raise HTTPException(
            status_code=404,
            detail="no ML artifacts; run `python scripts/train_models.py` first",
        )
    ic_by_date = pd.read_csv(path / "ic_by_date.csv", index_col=0, parse_dates=True)
    ic_summary = pd.read_csv(path / "ic_summary.csv")
    run_config: dict[str, Any] = {}
    importance: list[dict[str, Any]] = []
    if (path / "run_config.json").exists():
        import json

        run_config = json.loads((path / "run_config.json").read_text())
    if (path / "importance_lightgbm.csv").exists():
        imp = pd.read_csv(path / "importance_lightgbm.csv", index_col=0)
        imp = imp.sort_values("importance", ascending=False)
        importance = [
            {"feature": str(i), "importance": round(float(v), 4)}
            for i, v in imp["importance"].items()
        ]
    return {
        "ic_summary": ic_summary.to_dict(orient="records"),
        "ic_dates": _dates(ic_by_date.index),
        "ic_series": {col: _series(ic_by_date[col], 5) for col in ic_by_date.columns},
        "importance": importance,
        "run_config": run_config,
    }


@router.get("/risk")
def risk() -> dict[str, Any]:
    from alphaforge.risk import analytics as risk_analytics

    try:
        bundle = STATE.flagship("ml_ensemble")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="no ML artifacts; run `python scripts/train_models.py` first",
        ) from None
    # reconstruct daily returns from the cached equity curve (no re-run)
    eq = pd.Series(bundle["equity"], index=pd.to_datetime(bundle["dates"]))
    rets = eq.pct_change().dropna()
    dd_episodes = risk_analytics.drawdown_episodes(rets)
    stress = risk_analytics.stress_test(rets)
    ep_rows = [ep.as_dict() for ep in dd_episodes]
    stress_rows = stress.reset_index().to_dict(orient="records")
    for row in stress_rows:
        for k, v in list(row.items()):
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = round(float(v), 4)
    return {
        "strategy": bundle["name"],
        "metrics": bundle["metrics"],
        "risk_summary": {k: round(float(v), 4) for k, v in bundle["risk"].items()},
        "drawdown_episodes": ep_rows,
        "stress_test": stress_rows,
    }


@router.get("/universe")
def universe() -> dict[str, Any]:
    panel = STATE.panel
    cov = panel.coverage()
    cov_rows = []
    for ticker, r in cov.iterrows():
        cov_rows.append(
            {
                "ticker": str(ticker),
                "n_days": int(r["n_days"]),
                "first_date": str(pd.Timestamp(r["first_date"]).date()),
                "last_date": str(pd.Timestamp(r["last_date"]).date()),
                "avg_dollar_volume_musd": round(float(r["avg_dollar_volume_musd"]), 1),
                "pct_available": round(float(r["pct_available"]), 3),
            }
        )
    sectors: dict[str, int] = {}
    uni_path = get_config().universe_dir / "sp500_snapshot.csv"
    if uni_path.exists():
        uni = pd.read_csv(uni_path)
        sectors = uni["sector"].value_counts().to_dict()
    return {
        "n_tickers": int(len(panel.tickers)),
        "coverage": cov_rows,
        "sectors": sectors,
    }


@router.get("/universe/pit")
def universe_pit() -> dict[str, Any]:
    """Point-in-time universe view: membership curve, additions, experiments.

    This endpoint quantifies the *universe bias tax*: the same composite
    signal backtested on today's static S&P 500 list vs the point-in-time
    membership (names join on their official addition date, leave when
    their price data ends).  Also serves the ML OOS numbers and the
    per-feature PSI drift summary when those runs are on disk.
    """
    pit = STATE.pit
    n_members = pit.n_members
    # downsample the daily curve to weekly for the payload
    weekly = n_members.resample("W-FRI").last().dropna()
    additions = pit.additions_by_year()

    out: dict[str, Any] = {
        "stats": pit.summary(),
        "panel": STATE.panel_name,
        "members_curve": {
            "dates": _dates(weekly.index),
            "n_members": _series(weekly, 0),
        },
        "additions_by_year": {str(k): int(v) for k, v in additions.items()},
        "snapshot_note": (
            "Additions from the official S&P `date_added` column; removals "
            "approximated by end-of-price-data (free sources publish no "
            "removal history). Dual-class listings make 503 tickers out of "
            "500 companies."
        ),
    }

    cfg = get_config()
    exp_path = cfg.results_dir / "universe_experiments.json"
    if exp_path.exists():
        exp = json.loads(exp_path.read_text())
        out["experiments"] = exp.get("experiments")
        out["benchmark"] = exp.get("benchmark")
    ml_bt_path = cfg.results_dir / "ml_run_sp500" / "backtest_metrics.json"
    if ml_bt_path.exists():
        out["ml_oos_pit"] = json.loads(ml_bt_path.read_text())
    psi_path = cfg.results_dir / "ml_run_sp500" / "psi_summary.json"
    if psi_path.exists():
        psi = json.loads(psi_path.read_text())
        top8 = dict(list(psi.get("mean_psi", {}).items())[:8])
        out["feature_psi"] = {
            "mean_psi_top8": top8,
            "significant_at_some_point": psi.get("significant_at_some_point", {}),
            "thresholds": "PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant",
        }
    return out
