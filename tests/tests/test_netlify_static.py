"""Consistency tests for the Netlify static snapshot bundle.

Netlify serves static files only, so ``scripts/make_netlify_snapshot.py``
pre-renders the read-only API responses under ``app/static/api/snap/`` and
``netlify.toml`` rewrites ``/api/*`` onto them. These tests are deliberately
filesystem-only (no FastAPI, no data): they fail whenever the redirects,
the marker file, the frontend fallback hooks or the snapshot payloads drift
apart, and they run in every CI job so a stale bundle never ships.

They do NOT verify the numbers themselves — the snapshots come from the
real endpoints through ``TestClient`` (same path as uvicorn), so their
content is validated by the live webapp tests in ``test_webapp.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "app" / "static"
SNAP = STATIC / "api" / "snap"

HORIZONS = [1, 3, 5, 10, 21]  # keep in sync with #f-horizon in index.html
# snapshot filename (no .json) -> live URL path it must be rewritten from
ONE_SHOT = {
    "health": "/api/health",
    "overview": "/api/overview",
    "backtest": "/api/backtest",
    "ml": "/api/ml",
    "risk": "/api/risk",
    "universe": "/api/universe",
    "universe_pit": "/api/universe/pit",
}


def _toml_redirects() -> list[tuple[str, str]]:
    try:
        import tomllib  # py311+
    except ImportError:  # py310: minimal line parsing, netlify.toml is ours
        text = (REPO / "netlify.toml").read_text(encoding="utf-8")
        return re.findall(r'from = "([^"]+)"\s*\n\s*to = "([^"]+)"', text)
    with (REPO / "netlify.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    return [(r["from"], r["to"]) for r in cfg.get("redirects", [])]


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_marker_and_publish_directory() -> None:
    marker = _load(STATIC / "netlify.json")
    assert isinstance(marker, dict) and marker.get("mode") == "static"
    text = (REPO / "netlify.toml").read_text(encoding="utf-8")
    assert re.search(r'^\s*publish\s*=\s*"app/static"', text, re.M), (
        "publish dir must be app/static"
    )
    assert re.search(r'command\s*=\s*""', text), "no build step expected"


def test_every_rewrite_target_exists() -> None:
    redirects = _toml_redirects()
    assert redirects, "netlify.toml must define redirects"
    for src, dst in redirects:
        assert src.startswith("/"), src
        if dst.startswith("/api/snap/") or dst == "/:splat":
            continue
        raise AssertionError(f"unexpected rewrite target: {src} -> {dst}")
    covered = {src for src, _ in redirects}
    for url in list(ONE_SHOT.values()) + ["/api/factors"]:
        assert url in covered, f"missing rewrite for {url}"


def test_one_shot_snapshots_present_and_parse() -> None:
    for ep in ONE_SHOT:
        payload = _load(SNAP / f"{ep}.json")
        assert isinstance(payload, dict), ep
    health = _load(SNAP / "health.json")
    assert health["status"] == "ok"
    overview = _load(SNAP / "overview.json")
    assert overview["data"]["tickers"] > 400  # S&P 500 panel (plus SPY)
    assert overview["flagship"]["composite"]["metrics"]["sharpe"] > 0


def test_factor_table_snapshots_for_every_horizon() -> None:
    names: list[str] = []
    for h in HORIZONS:
        payload = _load(SNAP / f"factors_h{h}.json")
        assert payload["horizon"] == h
        rows = payload["factors"]
        assert len(rows) >= 20, f"expected ~27 factors, got {len(rows)} at h={h}"
        if h == 5:
            names = [r["name"] for r in rows]
    # every factor name must be URL-path-safe: app.js builds snapshot URLs
    # without encodeURIComponent (see showDecay)
    for name in names:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), f"factor name not URL-safe: {name}"


def test_decay_snapshots_cover_every_factor_and_horizon() -> None:
    rows = _load(SNAP / "factors_h5.json")["factors"]
    for r in rows:
        for h in HORIZONS:
            p = SNAP / "decay" / f"{r['name']}_h{h}.json"
            assert p.is_file(), f"missing decay snapshot {p.name}"
            decay = _load(p)
            assert decay["name"] == r["name"]
            assert decay["horizons"] and len(decay["horizons"]) == len(decay["ic"])


def test_backtest_snapshot_matches_form_defaults() -> None:
    payload = _load(SNAP / "backtest.json")
    assert payload["name"].startswith("composite")
    assert payload["n_trading_days"] > 2000
    # keep in sync with BT_DEFAULT in app.js / BACKTEST_DEFAULTS in the script
    script = (REPO / "scripts" / "make_netlify_snapshot.py").read_text(encoding="utf-8")
    js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    for token in ("composite_top5", '"W-FRI"', '"long_short"'):
        assert token in script and token in js, f"default drifted from {token}"


def test_frontend_ships_static_fallback_hooks() -> None:
    js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    # detection marker, POST fallback, snapshot paths
    assert '"netlify.json"' in js
    assert "AF_STATIC" in js
    assert 'opts.method === "POST"' in js
    assert "/api/snap/factors_h" in js
    assert "/api/snap/decay/" in js


def test_fastapi_mode_is_untouched() -> None:
    """The live backend must still answer the original query-style URLs."""
    js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert "`/api/factors?horizon=${h}`" in js
    assert "`/api/factors/decay?name=" in js.replace("encodeURIComponent(name)", "name")
    assert 'fetch("netlify.json", { method: "HEAD" })' in js
