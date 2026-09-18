#!/usr/bin/env python
"""Serve app/static with Netlify-like rewrite semantics for visual checks.

Usage: python scripts/serve_static.py [port]
"""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_static_demo import NetlifyLikeHandler, load_redirects  # noqa: E402

NetlifyLikeHandler.redirects = load_redirects()
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
print(f"simulated Netlify server on http://127.0.0.1:{port}/ (Ctrl-C to stop)")
ThreadingHTTPServer(("127.0.0.1", port), NetlifyLikeHandler).serve_forever()
