"""Bulk price downloader backed by Yahoo Finance (yfinance).

Design notes
------------
* Downloads are batched (default 50 tickers per request) with retry and
  backoff, which keeps Yahoo's rate limits happy and makes large-universe
  refreshes tractable.
* ``auto_adjust=True`` is used: prices are back-adjusted for splits **and**
  dividends, and -- as verified empirically with the NVDA 10:1 split on
  2024-06-10 -- yfinance >= 0.2.x also adjusts volume for splits, so
  ``close * volume`` is a split-invariant dollar-volume estimator.
* Failed tickers are collected and reported rather than aborting the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel

log = logging.getLogger(__name__)

_YF_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


@dataclass
class DownloadReport:
    """Summary of a bulk download."""

    requested: int = 0
    succeeded: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DownloadReport(requested={self.requested}, succeeded={self.succeeded}, "
            f"failed={self.failed_tickers}, rows={self.rows}, "
            f"range={self.start}..{self.end})"
        )


class PriceDownloader:
    """Batched, retrying Yahoo Finance downloader returning a :class:`PricePanel`.

    Parameters
    ----------
    batch_size:
        Number of tickers requested per yfinance call.
    max_retries:
        Retries per batch before giving up on the batch's tickers.
    pause:
        Seconds to sleep between batches (be polite to the API).
    auto_adjust:
        Yahoo split/dividend adjustment (recommended: True).
    """

    def __init__(
        self,
        batch_size: int = 50,
        max_retries: int = 3,
        pause: float = 1.0,
        auto_adjust: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.pause = float(pause)
        self.auto_adjust = bool(auto_adjust)

    def download(
        self,
        tickers: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        progress: bool = True,
    ) -> tuple[PricePanel, DownloadReport]:
        """Download daily OHLCV for ``tickers`` over ``[start, end]``.

        Returns
        -------
        (panel, report) :
            The panel contains only tickers that returned data; the report
            lists failures and basic coverage statistics.
        """
        import yfinance as yf

        end = end or pd.Timestamp.today()
        report = DownloadReport(requested=len(tickers))
        frames: list[pd.DataFrame] = []

        batches = [
            tickers[i : i + self.batch_size] for i in range(0, len(tickers), self.batch_size)
        ]
        for i, batch in enumerate(batches):
            data = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    data = yf.download(
                        batch,
                        start=str(pd.Timestamp(start).date()),
                        end=str(pd.Timestamp(end).date() + pd.Timedelta(days=1)),
                        auto_adjust=self.auto_adjust,
                        progress=False,
                        group_by="column",
                        threads=False,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = 2.0**attempt
                    log.warning(
                        "Batch %d/%d attempt %d failed: %s (retrying in %.0fs)",
                        i + 1,
                        len(batches),
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            if progress:
                print(
                    f"  batch {i + 1}/{len(batches)}: {len(batch)} tickers "
                    f"({'ok' if data is not None else 'FAILED'})",
                    flush=True,
                )
            if data is None or data.empty:
                report.failed_tickers.extend(batch)
                continue
            frames.append(self._flatten(data))
            if self.pause:
                time.sleep(self.pause)

        if not frames:
            raise RuntimeError(
                "Download failed for every batch. Check network access to "
                "query1.finance.yahoo.com and the ticker list."
            )

        long = pd.concat(frames, ignore_index=True)
        panel = PricePanel.from_long(long)

        report.succeeded = len(panel.tickers)
        report.failed_tickers = sorted(set(report.failed_tickers) - set(panel.tickers))
        report.rows = len(long)
        report.start = panel.dates.min()
        report.end = panel.dates.max()
        return panel, report

    @staticmethod
    def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise a yfinance MultiIndex frame into long (date, ticker) format."""
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance with group_by="column" returns (field, ticker) columns
            # such as ('Close', 'AAPL').  Stacking the *ticker* level yields
            # the desired (date, ticker) index with fields as columns.
            level = 1 if raw.columns.nlevels == 2 else 0
            long = raw.stack(level=level, future_stack=True)
            long.index.names = ["date", "ticker"]
            long = long.reset_index()
        else:  # very old yfinance / single-ticker plain columns
            long = raw.reset_index()
            long["ticker"] = "UNKNOWN"
            long = long.rename(columns={"Date": "date"})
        rename = dict(_YF_COLUMNS)
        long = long.rename(columns=rename)
        field_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        for c in field_cols:
            if c not in long.columns:
                long[c] = np.nan
        long = long[field_cols]
        long["date"] = pd.to_datetime(long["date"])
        long["ticker"] = long["ticker"].astype(str)
        return long.dropna(subset=["close"])
