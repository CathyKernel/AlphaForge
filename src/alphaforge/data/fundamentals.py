"""Fundamentals snapshot: quarterly financials from Yahoo Finance.

Fetches trailing-twelve-month net income, TTM revenue, latest book equity
and shares outstanding for every ticker in the universe, and caches the
result as a Parquet snapshot (``data/cache/fundamentals.parquet``).

**Point-in-time disclosure.**  Yahoo exposes only the *current* financial
statements — there is no free point-in-time history of as-reported
fundamentals.  The snapshot is therefore a *static* cross-section applied
to the whole backtest window: value factors built on it embed look-ahead
bias (we know 2026 statements in 2018).  AlphaForge keeps the module
because (a) the current-snapshot value signals are still informative
about the *cross-sectional* dispersion of valuations, (b) every consumer
(factor table, README, dashboard) discloses the bias explicitly, and
(c) swapping in a PIT source (Compustat via WRDS, Sharadar) only requires
re-implementing :meth:`FundamentalsFetcher.fetch` — the downstream factor
code is unchanged.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["net_income_ttm", "revenue_ttm", "book_equity", "shares"]


class FundamentalsFetcher:
    """Threaded fetcher for the quarterly-fundamentals snapshot."""

    def __init__(self, max_workers: int = 8, pause: float = 0.2) -> None:
        self.max_workers = int(max_workers)
        self.pause = float(pause)
        self.failed: list[str] = []

    def fetch(self, tickers: list[str]) -> pd.DataFrame:
        """Return a ticker-indexed frame with TTM financials."""
        rows: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._fetch_one, t): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    rec = fut.result()
                    if rec:
                        rows[t] = rec
                    else:
                        self.failed.append(t)
                except Exception as exc:  # noqa: BLE001
                    log.debug("fundamentals failed for %s: %s", t, exc)
                    self.failed.append(t)
        df = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]
        df.index.name = "ticker"
        return df.sort_index()

    def _fetch_one(self, ticker: str) -> dict | None:
        import yfinance as yf

        time.sleep(self.pause)  # be polite across threads
        try:
            tk = yf.Ticker(ticker)
            inc = tk.quarterly_income_stmt
            bal = tk.quarterly_balance_sheet

            def _ttm(stmt: pd.DataFrame | None, item: str) -> float | None:
                if stmt is None or stmt.empty or item not in stmt.index:
                    return None
                series = stmt.loc[item].dropna()
                if len(series) >= 4:
                    return float(series.iloc[:4].sum())  # last 4 quarters
                return float(series.sum()) if len(series) else None

            net_income = _ttm(inc, "Net Income")
            revenue = _ttm(inc, "Total Revenue")
            equity = None
            if bal is not None and not bal.empty:
                for item in (
                    "Total Equity Gross Minority Interest",
                    "Stockholders Equity",
                    "Common Stock Equity",
                ):
                    if item in bal.index and pd.notna(bal.loc[item].iloc[0]):
                        equity = float(bal.loc[item].iloc[0])
                        break
            shares = None
            try:
                shares = float(tk.fast_info["shares"])
            except Exception:  # noqa: BLE001
                if bal is not None and not bal.empty and "Ordinary Shares Number" in bal.index:
                    shares = float(bal.loc["Ordinary Shares Number"].dropna().iloc[0])
            if shares is None or (net_income is None and revenue is None and equity is None):
                return None
            return {
                "net_income_ttm": net_income,
                "revenue_ttm": revenue,
                "book_equity": equity,
                "shares": shares,
            }
        except Exception:  # noqa: BLE001
            return None


class FundamentalsRepository:
    """Load / persist the fundamentals snapshot next to the price cache."""

    def __init__(self, path: str | Path | None = None) -> None:
        from alphaforge.config import get_config

        self.path = Path(path) if path else get_config().data_dir / "cache" / "fundamentals.parquet"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                "no fundamentals snapshot; run `alphaforge fundamentals` with "
                "network access to fetch one (discloses look-ahead bias)"
            )
        df = pd.read_parquet(self.path)
        if "snapshot_date" in df.columns:
            df = df.drop(columns=["snapshot_date"])
        return df

    def save(self, df: pd.DataFrame, snapshot_date: str | None = None) -> Path:
        out = df.copy()
        out["snapshot_date"] = snapshot_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.path)
        return self.path

    def snapshot_date(self) -> str | None:
        if not self.path.exists():
            return None
        meta = pd.read_parquet(self.path)
        if "snapshot_date" in meta.columns and len(meta):
            return str(meta["snapshot_date"].iloc[0])
        return None

    def refresh(self, tickers: list[str], max_workers: int = 8) -> tuple[pd.DataFrame, Path]:
        fetcher = FundamentalsFetcher(max_workers=max_workers)
        df = fetcher.fetch(tickers)
        path = self.save(df)
        if fetcher.failed:
            log.warning("fundamentals unavailable for %d tickers", len(fetcher.failed))
        return df, path
