#!/usr/bin/env python
"""Fetch the S&P 500 quarterly fundamentals snapshot (value-factor inputs).

One-off network job (~2-5 min): pulls TTM earnings / book value / sales for
every S&P 500 constituent and stores ``data/cache/fundamentals.parquet``.
The value factors (ep / bp / sp) read this snapshot; refresh it quarterly.

Usage:
    python scripts/fetch_fundamentals.py
"""

from __future__ import annotations

import time

from alphaforge.data.fundamentals import FundamentalsRepository
from alphaforge.data.universe import Universe


def main() -> int:
    uni = Universe.sp500()
    tickers = list(uni.tickers)
    print(f"fetching fundamentals for {len(tickers)} tickers ...", flush=True)

    repo = FundamentalsRepository()
    t0 = time.time()
    df, path = repo.refresh(tickers, max_workers=12)
    print(f"got {len(df)} rows in {time.time() - t0:.0f}s -> {path}")
    print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
