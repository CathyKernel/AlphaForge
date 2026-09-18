"""Tests for the FastAPI dashboard (app.server / app.api).

Uses the deterministic synthetic panel from ``conftest`` behind a fake
``DashboardState`` so the endpoints are exercised offline, without the
bundled 300k-row cache.
"""

from __future__ import annotations

import pandas as pd
import pytest
from app import api
from fastapi.testclient import TestClient

from conftest import make_panel


def _fake_universe(panel):
    """A tiny index universe covering every synthetic panel ticker.

    Members are dated 2000-01-01 (long before the panel starts), so the
    point-in-time membership matrix is all-True — the backtests behave
    exactly as they did before the PIT mask was introduced.
    """
    from alphaforge.data.universe import Universe

    tickers = list(panel.tickers)
    return Universe(
        tuple(tickers),
        pd.Series("Information Technology", index=tickers),
        "test snapshot",
        "test",
        pd.Series(pd.Timestamp("2000-01-01"), index=tickers),
    )


class FakeState:
    """Minimal stand-in for ``api.DashboardState`` (real compute, tiny data)."""

    def __init__(self, panel) -> None:
        from alphaforge.factors import FactorEvaluator
        from alphaforge.factors.base import REGISTRY
        from alphaforge.factors.library import FactorLibrary

        self._panel = panel
        self.library = FactorLibrary(panel)
        self.library.compute_all()
        self.registry = REGISTRY
        self._stats = FactorEvaluator(panel, horizon=5).evaluate_all(self.library.compute_all())
        self._pit = None

    @property
    def panel(self):
        return self._panel

    @property
    def panel_name(self) -> str:
        return "prices"

    @property
    def pit(self):
        """PIT membership for the synthetic panel (all six names, full history)."""
        if self._pit is None:
            from alphaforge.data.pit import PointInTimeUniverse

            self._pit = PointInTimeUniverse.from_snapshot(self._panel, _fake_universe(self._panel))
        return self._pit

    def factor_stats(self, horizon: int) -> pd.DataFrame:
        return self._stats

    def flagship(self, key: str) -> dict:
        raise FileNotFoundError("no flagship runs in test fixture")


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    tmp = tmp_path_factory.mktemp("results")
    (tmp / "results").mkdir(exist_ok=True)
    # point the app at an empty results dir: /api/ml and flagship must
    # degrade gracefully instead of 500-ing
    import alphaforge.config as config

    cfg = config.get_config()
    old_results, old_data = cfg.results_dir, cfg.data_dir
    object.__setattr__(cfg, "results_dir", tmp / "results")
    object.__setattr__(cfg, "data_dir", tmp / "data")
    (tmp / "data" / "universe").mkdir(parents=True, exist_ok=True)

    api.STATE = FakeState(make_panel())
    from app.server import create_app

    app = create_app()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        api.STATE = api.DashboardState()
        object.__setattr__(cfg, "results_dir", old_results)
        object.__setattr__(cfg, "data_dir", old_data)


def test_serves_index_page(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "AlphaForge" in r.text
    assert "/static/js/app.js" in r.text


def test_static_assets_available(client: TestClient) -> None:
    for asset in ("css/style.css", "js/charts.js", "js/app.js"):
        r = client.get(f"/static/{asset}")
        assert r.status_code == 200, asset


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_overview_shape(client: TestClient) -> None:
    body = client.get("/api/overview").json()
    assert body["data"]["tickers"] == 6
    assert body["factors"]["n_factors"] >= 20
    assert isinstance(body["quality"], list) and body["quality"]
    # flagship views degrade to null when no artifacts are present
    assert body["flagship"]["ml_ensemble"] is None


def test_factors_table(client: TestClient) -> None:
    body = client.get("/api/factors?horizon=5").json()
    names = [f["name"] for f in body["factors"]]
    assert "mom_63" in names
    row = next(f for f in body["factors"] if f["name"] == "mom_63")
    assert row["category"] == "momentum"
    # synthetic 260-day panel: some stats can be NaN -> serialised as null
    assert row["ic_tstat"] is None or isinstance(row["ic_tstat"], float)


def test_factor_decay(client: TestClient) -> None:
    body = client.get("/api/factors/decay?name=vol_21&horizon=5").json()
    assert body["name"] == "vol_21"
    assert len(body["horizons"]) == len(body["ic"])
    r = client.get("/api/factors/decay?name=nope&horizon=5")
    assert r.status_code == 404


def test_live_backtest_endpoint(client: TestClient) -> None:
    r = client.post(
        "/api/backtest",
        json={"factor": "mom_63", "rebalance": "W-FRI", "side": "long_short", "top_n": 3},
    )
    assert r.status_code == 200
    b = r.json()
    assert len(b["dates"]) == len(b["equity"])
    assert "cagr" in b["metrics"] and "sharpe" in b["metrics"]
    assert b["name"] == "mom_63"
    assert b["holdings"] is not None

    # invalid factor -> 404, invalid rebalance -> 422
    assert client.post("/api/backtest", json={"factor": "nope"}).status_code == 404
    assert (
        client.post("/api/backtest", json={"factor": "mom_63", "rebalance": "bad"}).status_code
        == 422
    )


def test_ml_and_risk_missing_artifacts(client: TestClient) -> None:
    """No ml_run artifacts in the fixture: endpoints must fail cleanly."""
    assert client.get("/api/ml").status_code == 404
    # /api/risk needs the flagship ml_ensemble run; without artifacts it 500s
    # with a controlled FileNotFoundError -> FastAPI turns it into a 500
    r = client.get("/api/risk")
    assert r.status_code in (404, 500)


def test_universe_coverage(client: TestClient) -> None:
    body = client.get("/api/universe").json()
    assert body["n_tickers"] == 6
    assert len(body["coverage"]) == 6
    cov = body["coverage"][0]
    assert cov["n_days"] > 0 and "pct_available" in cov
    assert body["sectors"] == {}  # no universe snapshot in tmp dir


def test_universe_pit_endpoint(client: TestClient) -> None:
    """The point-in-time universe view: stats, member curve, additions."""
    body = client.get("/api/universe/pit").json()
    assert body["panel"] == "prices"
    stats = body["stats"]
    assert stats["n_tickers_ever"] == 6
    assert stats["members_first"] == 6
    curve = body["members_curve"]
    assert len(curve["dates"]) == len(curve["n_members"]) > 0
    assert set(curve["n_members"]) == {6.0}
    assert isinstance(body["additions_by_year"], dict)
    # empty results dir: optional blocks must be absent, not 500
    assert "experiments" not in body
    assert "ml_oos_pit" not in body
    assert "feature_psi" not in body


def test_real_dashboard_state_flagship_no_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression test: the real DashboardState (not FakeState) must compute
    ``flagship("composite")`` without deadlocking.

    ``flagship()`` holds the state lock while calling ``factor_stats()``,
    which re-acquires it — with a plain ``threading.Lock`` the first real
    ``/api/overview`` request hangs forever (observed on the bundled panel).
    The lock must therefore be reentrant.
    """
    from types import SimpleNamespace

    from alphaforge.data.universe import Universe

    panel = make_panel()

    # hermetic environment: fake index universe + empty results dir so
    # the ml_ensemble flagship raises FileNotFoundError instead of reading
    # the real repo's artifacts
    fake_universe = _fake_universe(panel)
    monkeypatch.setattr(Universe, "sp500", classmethod(lambda cls, *a, **kw: fake_universe))
    monkeypatch.setattr(
        api, "get_config", lambda: SimpleNamespace(results_dir=tmp_path / "results")
    )

    class FakeRepo:  # injected instead of the bundled Parquet cache
        def has_panel(self, name: str = "prices") -> bool:
            return False

        def load_panel(self, name: str = "prices"):
            return panel

    monkeypatch.setattr(api, "DataRepository", FakeRepo)
    state = api.DashboardState()
    bundle = state.flagship("composite")  # would deadlock with Lock()
    assert bundle["name"] == "composite_top5"
    assert "sharpe" in bundle["metrics"]
    # the PIT mask is part of the served params (described, not serialised)
    assert bundle["params"]["universe_mask"] == "point-in-time membership matrix"
    # second call is served from cache and must be instant + identical
    again = state.flagship("composite")
    assert again is bundle
    # unknown key raises KeyError, missing artifact raises FileNotFoundError
    with pytest.raises(KeyError):
        state.flagship("nope")
    with pytest.raises(FileNotFoundError):
        state.flagship("ml_ensemble")
