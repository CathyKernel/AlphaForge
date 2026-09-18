"""Parquet-backed local data repository.

The repository stores one Parquet file per dataset ("prices" by default)
plus a JSON manifest with download metadata.  Parquet + snappy gives ~10x
compression relative to CSV and columnar reads that are fast enough for
interactive research on a laptop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from alphaforge.data.downloader import DownloadReport, PriceDownloader
from alphaforge.data.panel import PricePanel
from alphaforge.data.universe import Universe

log = logging.getLogger(__name__)


@dataclass
class Manifest:
    """Metadata persisted alongside each cached dataset."""

    universe: str = "sp100"
    source: str = "yahoo"
    start: str | None = None
    end: str | None = None
    n_tickers: int = 0
    n_rows: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DataRepository:
    """Load / persist price panels and universe snapshots on local disk."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        from alphaforge.config import get_config

        self.cache_dir = Path(cache_dir) if cache_dir else get_config().cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def panel_path(self, name: str = "prices") -> Path:
        return self.cache_dir / f"{name}.parquet"

    def manifest_path(self, name: str = "prices") -> Path:
        return self.cache_dir / f"{name}_manifest.json"

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    def save_panel(
        self, panel: PricePanel, name: str = "prices", manifest: Manifest | None = None
    ) -> Path:
        path = panel.to_parquet(self.panel_path(name))
        if manifest is None:
            manifest = Manifest(
                start=str(panel.dates.min().date()),
                end=str(panel.dates.max().date()),
                n_tickers=len(panel.tickers),
                n_rows=len(panel.to_long()),
            )
        self.manifest_path(name).write_text(json.dumps(manifest.to_dict(), indent=2))
        log.info(
            "Saved panel %s (%d tickers, %s..%s)",
            path.name,
            manifest.n_tickers,
            manifest.start,
            manifest.end,
        )
        return path

    def load_panel(self, name: str = "prices") -> PricePanel:
        path = self.panel_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached dataset '{name}' at {path}. "
                "Run `python scripts/download_data.py` (or alphaforge download) first."
            )
        return PricePanel.from_parquet(path)

    def load_manifest(self, name: str = "prices") -> Manifest | None:
        path = self.manifest_path(name)
        if not path.exists():
            return None
        return Manifest.from_dict(json.loads(path.read_text()))

    def has_panel(self, name: str = "prices") -> bool:
        return self.panel_path(name).exists()

    # ------------------------------------------------------------------ #
    # One-shot helpers
    # ------------------------------------------------------------------ #
    def fetch_and_cache(
        self,
        universe: Universe,
        start: str,
        end: str | None = None,
        name: str = "prices",
        benchmark: str | None = "SPY",
        progress: bool = True,
    ) -> tuple[PricePanel, DownloadReport]:
        """Download a universe (optionally + benchmark ticker) and cache it."""
        tickers = list(universe.tickers)
        if benchmark and benchmark not in tickers:
            tickers.append(benchmark)
        downloader = PriceDownloader()
        panel, report = downloader.download(tickers, start, end, progress=progress)
        if benchmark is not None and benchmark in panel.tickers:
            manifest = Manifest(
                universe=f"{universe.snapshot_date} ({len(universe)} tickers)",
                start=str(panel.dates.min().date()),
                end=str(panel.dates.max().date()),
                n_tickers=len(panel.tickers),
                n_rows=len(panel.to_long()),
                notes=f"includes benchmark {benchmark}; source={universe.source}",
            )
        else:
            manifest = Manifest(
                universe=f"{universe.snapshot_date} ({len(universe)} tickers)",
                start=str(panel.dates.min().date()),
                end=str(panel.dates.max().date()),
                n_tickers=len(panel.tickers),
                n_rows=len(panel.to_long()),
            )
        self.save_panel(panel, name=name, manifest=manifest)
        return panel, report
