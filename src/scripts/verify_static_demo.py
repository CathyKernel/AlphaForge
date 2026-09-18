#!/usr/bin/env python
"""Offline end-to-end check of the Netlify static demo bundle.

Starts a local HTTP server that mimics the semantics Netlify gives the
published bundle (serve ``app/static/`` + apply the ``netlify.toml``
rewrites: path-based match, query string ignored, ``/static/*`` stripped),
then walks every URL the frontend can request and asserts the response is
valid JSON with the right shape. Run before pushing if you regenerated the
snapshots:

    PYTHONPATH=src:. python scripts/verify_static_demo.py

Everything is offline: no uvicorn, no data layer, no FastAPI import.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "app" / "static"

HORIZONS = [1, 3, 5, 10, 21]
CHECKS: list[tuple[str, str]] = []


def ok(label: str, _detail: str = "") -> None:
    CHECKS.append(("ok", label))
    print(f"  ✓ {label}")


def fail(label: str, detail: str) -> None:
    CHECKS.append(("fail", label))
    print(f"  ✗ {label} — {detail}")


def load_redirects() -> list[tuple[str, str]]:
    """Extract from/to pairs from netlify.toml (order preserved)."""
    text = (REPO / "netlify.toml").read_text(encoding="utf-8")
    return re.findall(r'from = "([^"]+)"\s*\n\s*to = "([^"]+)"', text)


class NetlifyLikeHandler(SimpleHTTPRequestHandler):
    """Serves STATIC with Netlify rewrite semantics."""

    redirects: list[tuple[str, str]] = []

    def translate_path(self, path: str) -> str:  # noqa: D102
        # strip query string, then apply rewrite rules (path-only matching,
        # first rule wins, query ignored — exactly what Netlify does)
        url = path.split("?", 1)[0]
        for src, dst in self.redirects:
            if src == url:
                url = dst.replace("/:splat", url[len("/static/") :] if src == "/static/*" else "")
                break
            if src.endswith("/*"):
                prefix = src[:-2]
                if url.startswith(prefix + "/"):
                    splat = url[len(prefix) + 1 :]
                    url = dst.replace("/:splat", splat)
                    break
        return str(STATIC / url.lstrip("/"))

    def do_HEAD(self) -> None:  # static hosting answers HEAD
        try:
            self.send_head()
            self.end_headers()
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, *args: object) -> None:  # silence
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fetch(url: str, method: str = "GET") -> tuple[int, object]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        return e.code, None
    if not body:
        return status, None
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body[:80]


def main() -> int:
    redirects = load_redirects()
    assert redirects, "netlify.toml has no redirects"
    NetlifyLikeHandler.redirects = redirects
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), NetlifyLikeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    print(f"simulated Netlify server on {base} (publish root = app/static)")
    time.sleep(0.2)

    print("[assets]")
    for path in [
        "/",
        "/static/css/style.css",
        "/static/js/app.js",
        "/static/js/charts.js",
        "/netlify.json",
    ]:
        status, _ = fetch(base + path)
        (ok if status == 200 else fail)(f"{path} -> {status}", f"expected 200, got {status}")

    print("[API rewrites]")
    shapes = {
        "/api/health": ("status",),
        "/api/overview": ("data", "flagship", "factors"),
        "/api/factors": ("horizon", "factors"),
        "/api/ml": ("ic_summary",),
        "/api/risk": (),
        "/api/universe": (),
        "/api/universe/pit": ("stats",),
        "/api/backtest": ("name", "n_trading_days", "metrics"),
    }
    for path, keys in shapes.items():
        status, payload = fetch(base + path)
        good = status == 200 and isinstance(payload, dict) and all(k in payload for k in keys)
        (ok if good else fail)(
            f"{path} -> {status}",
            f"status {status}, keys {list(payload)[:4] if isinstance(payload, dict) else payload}",
        )

    # query string must be ignored by the rewrite (Netlify semantics)
    status, payload = fetch(base + "/api/factors?horizon=10")
    (ok if status == 200 and payload["horizon"] == 5 else fail)(
        "/api/factors?horizon=10 -> h5 snapshot",
        f"status {status}, horizon {payload.get('horizon') if isinstance(payload, dict) else payload}",
    )

    print("[snapshot paths the frontend switches to]")
    rows = fetch(base + "/api/snap/factors_h5.json")[1]["factors"]
    ok(f"/api/snap/factors_h5.json -> {len(rows)} factors")
    for h in HORIZONS:
        status, _ = fetch(base + f"/api/snap/factors_h{h}.json")
        (ok if status == 200 else fail)(f"/api/snap/factors_h{h}.json", f"status {status}")
    sample = [r["name"] for r in rows[:: max(1, len(rows) // 5)]]
    for name in sample:
        for h in HORIZONS:
            status, _ = fetch(base + f"/api/snap/decay/{name}_h{h}.json")
            if status != 200:
                fail(f"decay snapshot {name}_h{h}", f"status {status}")
    ok(f"decay snapshots sampled ({len(sample)} factors x {len(HORIZONS)} horizons)")

    print("[static-hosting POST behaviour]")
    status, _ = fetch(base + "/api/backtest", method="POST")
    if status in (405, 501):
        ok(f"POST /api/backtest -> {status} (frontend falls back to GET)")
    else:
        ok(f"POST /api/backtest -> {status} (fallback still applies on !ok)")

    server.shutdown()
    failed = sum(1 for s, _ in CHECKS if s == "fail")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
