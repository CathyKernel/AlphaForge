"""Incremental (watermark-based) panel updates with revision detection.

A production data pipeline does **not** re-download ten years of history
every night: it remembers how far it has seen (the *watermark*), fetches
only what is new, and — crucially for financial data — re-checks a short
**overlap window** before the watermark, because vendors revise recent
observations retroactively (dividend/split adjustments propagate backwards,
late corrections land days later).

:class:`IncrementalUpdater` implements exactly that CDC-style loop:

1. Load the cached panel and its watermark (last observed date).
2. Re-download a small overlap window ``[watermark - overlap, today]``.
3. **Revision detection**: compare overlap rows already cached vs the
   freshly downloaded ones; count and correct changed cells (adjustment
   revisions), so the stored history stays consistent.
4. Append strictly-new rows, re-pivot, persist, and bump the watermark in
   the manifest.
5. Emit an :class:`UpdateReport` — an audit record of everything that
   changed (new rows, revised cells, new tickers, silently dropped
   tickers), which is what a data quality dashboard would consume.

The update is *atomic*: if anything fails mid-way, the existing parquet
file is left untouched (writes go to a temp file then rename).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alphaforge.data.cache import DataRepository, Manifest
from alphaforge.data.downloader import PriceDownloader

log = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class UpdateReport:
    """Audit record of one incremental update run."""

    dataset: str = ""
    watermark_before: str | None = None
    watermark_after: str | None = None
    n_rows_before: int = 0
    n_rows_after: int = 0
    new_rows: int = 0
    revised_rows: int = 0
    new_tickers: list[str] = field(default_factory=list)
    dropped_tickers: list[str] = field(default_factory=list)
    up_to_date: bool = False
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"UpdateReport({self.dataset}: {self.watermark_before} -> "
            f"{self.watermark_after}, +{self.new_rows} new, "
            f"{self.revised_rows} revised, up_to_date={self.up_to_date})"
        )


class IncrementalUpdater:
    """Watermark-based incremental refresh of a cached price panel.

    Parameters
    ----------
    repo:
        Data repository holding the cached parquet panels.
    overlap_days:
        Calendar days before the watermark that are re-downloaded and
        diffed for vendor revisions.  10 is a safe default (covers most
        back-filled dividends and late corrections).
    rtol:
        Relative tolerance when comparing cached vs fresh values to decide
        whether a cell was "revised" (float round-trips through parquet
        are not always bit-exact).
    """

    def __init__(
        self,
        repo: DataRepository | None = None,
        overlap_days: int = 10,
        rtol: float = 1e-6,
        downloader: PriceDownloader | None = None,
    ) -> None:
        self.repo = repo or DataRepository()
        self.overlap_days = int(overlap_days)
        self.rtol = float(rtol)
        self.downloader = downloader or PriceDownloader(batch_size=100)

    # ------------------------------------------------------------------ #
    def update(
        self,
        name: str = "prices",
        tickers: list[str] | None = None,
        end: str | pd.Timestamp | None = None,
        progress: bool = True,
    ) -> tuple[pd.DataFrame, UpdateReport]:
        """Incrementally refresh dataset ``name``; returns (long frame, report).

        Parameters
        ----------
        tickers:
            The tickers the panel *should* cover (e.g. current universe +
            benchmark).  When given, newly required tickers are downloaded
            from the panel's own start date; when omitted, only the cached
            tickers are refreshed.
        """
        path = self.repo.panel_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached panel '{name}' at {path} — run a full download first."
            )

        cached_long = pd.read_parquet(path, engine="pyarrow")
        cached_long["date"] = pd.to_datetime(cached_long["date"])
        watermark = cached_long["date"].max()
        report = UpdateReport(dataset=name, watermark_before=str(watermark.date()))
        report.n_rows_before = len(cached_long)

        overlap_start = watermark - pd.Timedelta(days=self.overlap_days)
        fetch_start = min(overlap_start, watermark)  # trivially overlap_start

        # tickers to refresh: cached ones (+ requested new ones)
        cached_tickers = sorted(cached_long["ticker"].unique())
        wanted = list(tickers) if tickers is not None else cached_tickers
        new_tickers = [t for t in wanted if t not in set(cached_tickers)]
        report.new_tickers = new_tickers

        # ---- fetch the overlap + tail for the cached tickers -------------
        fresh_overlap, _ = self.downloader.download(
            cached_tickers, start=fetch_start, end=end, progress=progress
        )
        fresh_long = fresh_overlap.to_long()
        fresh_long["date"] = pd.to_datetime(fresh_long["date"])

        # ---- revision detection on the overlap window --------------------
        cached_overlap = cached_long[
            (cached_long["date"] >= fetch_start) & (cached_long["date"] <= watermark)
        ].set_index(["date", "ticker"])
        # rows with no usable fresh close carry no information -> drop them
        # (a vendor gap is NOT a revision; the cached values stand)
        fresh_window = fresh_long[
            (fresh_long["date"] <= watermark) & fresh_long["close"].notna()
        ].set_index(["date", "ticker"])
        common = cached_overlap.index.intersection(fresh_window.index)
        revised_mask = pd.Series(False, index=common)

        if len(common):
            old = cached_overlap.loc[common, list(FIELDS)].astype(float)
            new = fresh_window.loc[common, list(FIELDS)].astype(float)
            # cell-wise "revised": not close under rtol.  NaN == NaN counts
            # as equal, and a fresh NaN (missing cell) is never a revision.
            fresh_missing = new.isna()
            close = pd.DataFrame(
                np.isclose(
                    old.to_numpy(dtype=float),
                    new.reindex(columns=list(FIELDS)).to_numpy(dtype=float),
                    rtol=self.rtol,
                    atol=0.0,
                    equal_nan=True,
                ),
                index=common,
                columns=list(FIELDS),
            )
            diff = (~(close | fresh_missing)).any(axis=1)
            revised_mask = diff.fillna(False)
            report.revised_rows = int(revised_mask.sum())
        # drop cached overlap rows that were revised (or are not confirmed
        # by the fresh download at all -> stale), keep confirmed-identical
        stale = revised_mask.reindex(cached_overlap.index, fill_value=False)
        keep_mask = ~stale
        # cached rows *after* the watermark (should not exist, defensive)
        keep_mask &= cached_overlap.index.get_level_values("date") <= watermark
        kept_overlap = cached_overlap[keep_mask]

        # ---- strictly-new rows -------------------------------------------
        new_rows = fresh_long[(fresh_long["date"] > watermark) & fresh_long["close"].notna()]
        report.new_rows = len(new_rows)

        # ---- brand-new tickers: full history download --------------------
        extra_frames: list[pd.DataFrame] = []
        if new_tickers:
            start_date = str(cached_long["date"].min().date())
            new_panel, _ = self.downloader.download(
                new_tickers, start=start_date, end=end, progress=progress
            )
            if len(new_panel.tickers):
                extra = new_panel.to_long()
                extra["date"] = pd.to_datetime(extra["date"])
                extra_frames.append(extra)
                report.new_tickers = sorted(set(report.new_tickers) & set(extra["ticker"].unique()))
            else:
                report.new_tickers = []

        # ---- assemble ------------------------------------------------------
        parts = [cached_long[cached_long["date"] < fetch_start]]
        parts.append(kept_overlap.reset_index())
        parts.append(fresh_window.reset_index())  # includes revised overlap
        parts.append(new_rows)
        parts.extend(extra_frames)
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "ticker"], keep="last")
        merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
        report.n_rows_after = len(merged)
        report.watermark_after = str(merged["date"].max().date())
        report.up_to_date = report.watermark_after == report.watermark_before

        # dropped tickers: cached names missing from the fresh overlap tail
        tail = fresh_long[fresh_long["date"] >= watermark - pd.Timedelta(days=14)]
        if len(tail):
            seen_recent = set(tail["ticker"].unique())
            report.dropped_tickers = (
                sorted(set(cached_tickers) - seen_recent - set(new_tickers))
                if report.watermark_after == report.watermark_before
                else []
            )

        # ---- atomic persist -------------------------------------------------
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, engine="pyarrow", compression="snappy")
        tmp.replace(path)

        manifest = self.repo.load_manifest(name) or Manifest()
        manifest.end = report.watermark_after
        manifest.n_rows = len(merged)
        manifest.n_tickers = int(merged["ticker"].nunique())
        manifest.notes = f"incremental update {report.updated_at}"
        self.repo.manifest_path(name).write_text(json.dumps(manifest.to_dict(), indent=2))

        log.info("Incremental update of %s: %r", name, report)
        return merged, report
