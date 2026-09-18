#!/usr/bin/env python
"""Pre-render dashboard API responses as static JSON snapshots for Netlify.

Netlify serves static files only — it cannot run the FastAPI backend. This
script exercises every read-only endpoint through FastAPI's TestClient
(exactly the same request/response path as uvicorn) and writes the JSON
payloads under ``app/static/api/snap/``. ``netlify.toml`` publishes that
directory and rewrites ``/api/*`` to the matching snapshot, so the hosted
demo shows the same numbers as the local server.

Regenerate after changing factors, data or the API:

    PYTHONPATH=src:. python scripts/make_netlify_snapshot.py

Interactions that need live compute (custom-parameter backtests) are not
snapshotted: the frontend detects static hosting and degrades gracefully
(see ``app/static/js/app.js``).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from app.server import app
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SNAP_DIR = REPO / "app" / "static" / "api" / "snap"
MARKER = REPO / "app" / "static" / "netlify.json"

HORIZONS = [1, 3, 5, 10, 21]  # keep in sync with #f-horizon in index.html
BACKTEST_DEFAULTS = {
    "factor": "composite_top5",
    "rebalance": "W-FRI",
    "side": "long_short",
    "top_n": 15,
    "cost_bps": 10.0,
}  # keep in sync with the form defaults in index.html

written: list[Path] = []


def save(rel: str, payload: Any) -> None:
    path = SNAP_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: matches starlette's JSONResponse encoding — any NaN
    # would already be a 500 on the live server, so fail loudly here instead.
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    written.append(path)


def main() -> int:
    t0 = time.perf_counter()
    if SNAP_DIR.exists():
        shutil.rmtree(SNAP_DIR)
    SNAP_DIR.mkdir(parents=True)

    client = TestClient(app)

    def timed(label: str, fn: Any) -> Any:
        t = time.perf_counter()
        out = fn()
        print(f"  {label:34s} {time.perf_counter() - t:7.1f}s")
        return out

    print("[1/4] one-shot endpoints")
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    save("health.json", r.json())

    r = timed("GET /api/overview", lambda: client.get("/api/overview"))
    assert r.status_code == 200, r.text
    ov = r.json()
    save("overview.json", ov)

    r = timed(
        "POST /api/backtest (defaults)",
        lambda: client.post("/api/backtest", json=BACKTEST_DEFAULTS),
    )
    assert r.status_code == 200, r.text
    save("backtest.json", r.json())

    for endpoint in ("ml", "risk", "universe"):
        r = timed(f"GET /api/{endpoint}", lambda e=endpoint: client.get(f"/api/{e}"))
        assert r.status_code == 200, r.text
        save(f"{endpoint}.json", r.json())

    r = timed("GET /api/universe/pit", lambda: client.get("/api/universe/pit"))
    assert r.status_code == 200, r.text
    save("universe_pit.json", r.json())

    print("[2/4] factor tables for every IC horizon")
    factor_names: list[str] = []
    for h in HORIZONS:
        r = timed(
            f"GET /api/factors?horizon={h}", lambda h=h: client.get(f"/api/factors?horizon={h}")
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        save(f"factors_h{h}.json", payload)
        if h == 5:
            factor_names = [f["name"] for f in payload["factors"]]
    assert factor_names, "no factors found"

    print(f"[3/4] IC decay for {len(factor_names)} factors x {len(HORIZONS)} horizons")
    for i, name in enumerate(factor_names, 1):
        for h in HORIZONS:
            r = client.get(f"/api/factors/decay?name={name}&horizon={h}")
            assert r.status_code == 200, (name, h, r.text)
            save(f"decay/{name}_h{h}.json", r.json())
        if i % 9 == 0 or i == len(factor_names):
            print(f"  decay {i}/{len(factor_names)} done")

    print("[4/4] marker + self-check")
    MARKER.write_text(
        json.dumps({"mode": "static", "generated": ov["data"]["last_date"]}, separators=(",", ":")),
        encoding="utf-8",
    )
    total_kb = sum(p.stat().st_size for p in written) / 1024
    largest = max(written, key=lambda p: p.stat().st_size)
    for p in written:  # every file must be valid JSON
        json.loads(p.read_text(encoding="utf-8"))
    print(
        f"\nwrote {len(written)} snapshots ({total_kb:.0f} KB) to {SNAP_DIR.relative_to(REPO)}; "
        f"largest: {largest.relative_to(REPO)} ({largest.stat().st_size / 1024:.0f} KB)"
    )
    print(f"marker: {MARKER.relative_to(REPO)}")
    print(f"total time: {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
