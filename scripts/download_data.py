#!/usr/bin/env python
"""Download real US equity data and cache it locally.

Usage:
    python scripts/download_data.py                        # S&P 100, 10y, cached sample
    python scripts/download_data.py --universe sp500       # full S&P 500
    python scripts/download_data.py --start 2010-01-01     # longer history
"""

from __future__ import annotations

import argparse
import sys

from alphaforge.data.cache import DataRepository
from alphaforge.data.quality import DataQualityChecker
from alphaforge.data.universe import Universe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=["sp100", "sp500"], default="sp100")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--name", default="prices", help="cache dataset name")
    ap.add_argument("--benchmark", default="SPY", help="benchmark ticker ('' to disable)")
    ap.add_argument(
        "--refresh-universe",
        action="store_true",
        help="re-scrape live S&P 500 membership from Wikipedia",
    )
    ap.add_argument("--quality-report", action="store_true", default=True)
    args = ap.parse_args()

    print("== AlphaForge data download ==")
    print(f"universe={args.universe}  start={args.start}  end={args.end or 'today'}")

    universe = Universe.sp500(refresh=args.refresh_universe)
    if args.universe == "sp100":
        universe = Universe.sp100()
    print(f"universe: {len(universe)} tickers (snapshot {universe.snapshot_date})")

    repo = DataRepository()
    panel, report = repo.fetch_and_cache(
        universe,
        start=args.start,
        end=args.end,
        name=args.name,
        benchmark=args.benchmark or None,
    )
    print(
        f"\ndownloaded {report.succeeded}/{report.requested} tickers, "
        f"{report.rows:,} rows, {report.start} .. {report.end}"
    )
    if report.failed_tickers:
        print(f"failed tickers (skipped): {report.failed_tickers}")

    if args.quality_report:
        print("\n== data quality ==")
        checker = DataQualityChecker()
        qr = checker.run(panel)
        print(qr.summary().to_string(index=False))

    print(f"\ncached to: {repo.panel_path(args.name)}")
    print(f"panel: {panel!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
