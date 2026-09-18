"""IncrementalUpdater: watermark advance, revision detection, atomicity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.cache import DataRepository
from alphaforge.data.incremental import IncrementalUpdater
from alphaforge.data.panel import PricePanel


def _long_panel(start="2020-01-01", n_days=40, tickers=("A", "B")) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        for i, d in enumerate(idx):
            close = 100.0 + i
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                }
            )
    return pd.DataFrame(rows)


class FakeDownloader:
    """Downloader stub returning a canned long frame; records requests."""

    def __init__(self, frames_by_call: list[pd.DataFrame]) -> None:
        self.calls: list[dict] = []
        self._frames = list(frames_by_call)

    def download(self, tickers, start, end=None, progress=True):
        self.calls.append({"tickers": list(tickers), "start": str(start), "end": str(end)})
        frame = self._frames.pop(0) if self._frames else _long_panel(n_days=0)
        frame = frame[frame["ticker"].isin(tickers)]
        frame = frame[frame["date"] >= pd.Timestamp(start)]
        return PricePanel.from_long(frame), None


def _repo(tmp_path: Path, long: pd.DataFrame, name="prices") -> DataRepository:
    repo = DataRepository(cache_dir=tmp_path / "cache")
    p = repo.panel_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(p, engine="pyarrow")
    (tmp_path / "cache" / f"{name}_manifest.json").write_text("{}")
    return repo


def test_update_appends_new_rows_and_bumps_watermark(tmp_path) -> None:
    base = _long_panel(n_days=40)
    repo = _repo(tmp_path, base)
    # fresh download covers the overlap + 5 new days
    last = base["date"].max()
    extra = _long_panel(start=str((last + pd.Timedelta(days=1)).date()), n_days=5)
    overlap = base[base["date"] >= last - pd.Timedelta(days=10)]
    fresh = pd.concat([overlap, extra], ignore_index=True)

    upd = IncrementalUpdater(repo=repo, downloader=FakeDownloader([fresh]))
    merged, rep = upd.update()

    assert rep.new_rows == len(extra)
    assert rep.watermark_after == str(extra["date"].max().date())
    assert rep.watermark_before == str(base["date"].max().date())
    assert not rep.up_to_date
    assert rep.revised_rows == 0
    assert len(merged) == len(base) + len(extra)
    # panel file persisted with the new rows
    on_disk = pd.read_parquet(repo.panel_path("prices"))
    assert on_disk["date"].max() == extra["date"].max()


def test_revision_detection_corrects_stale_cells(tmp_path) -> None:
    base = _long_panel(n_days=40)
    repo = _repo(tmp_path, base)
    # fresh overlap revises one cached row of A (close +1)
    last = base["date"].max()
    fresh = base[base["date"] >= last - pd.Timedelta(days=10)].copy().reset_index(drop=True)
    revised_row = fresh.index[5]  # a mid-overlap row
    fresh.loc[revised_row, "close"] = fresh.loc[revised_row, "close"] + 1.0

    upd = IncrementalUpdater(repo=repo, downloader=FakeDownloader([fresh]))
    merged, rep = upd.update()

    assert rep.revised_rows == 1
    assert rep.up_to_date is True
    # merged carries the REVISED value, not the stale cached one
    tgt = merged[(merged["ticker"] == "A") & (merged["date"] == fresh.loc[revised_row, "date"])]
    assert len(tgt) == 1
    assert tgt["close"].iloc[0] == pytest.approx(fresh.loc[revised_row, "close"])


def test_no_new_data_is_reported_up_to_date(tmp_path) -> None:
    base = _long_panel(n_days=40)
    repo = _repo(tmp_path, base)
    upd = IncrementalUpdater(repo=repo, downloader=FakeDownloader([base.copy()]))
    _, rep = upd.update()
    assert rep.up_to_date is True
    assert rep.new_rows == 0
    assert rep.n_rows_after == len(base)


def test_new_ticker_gets_full_history(tmp_path) -> None:
    base = _long_panel(n_days=40, tickers=("A",))
    repo = _repo(tmp_path, base)
    # call 1: overlap refresh of A; call 2: full history of C
    hist_c = _long_panel(n_days=40, tickers=("C",))
    upd = IncrementalUpdater(repo=repo, downloader=FakeDownloader([base.iloc[-10:].copy(), hist_c]))
    merged, rep = upd.update(tickers=["A", "C"])
    assert rep.new_tickers == ["C"]
    assert set(merged["ticker"]) == {"A", "C"}
    assert merged[merged["ticker"] == "C"]["date"].min() == base["date"].min()


def test_failure_leaves_cache_untouched(tmp_path) -> None:
    base = _long_panel(n_days=40)
    repo = _repo(tmp_path, base)
    before = pd.read_parquet(repo.panel_path("prices"))

    class ExplodingDownloader:
        def download(self, *a, **k):
            raise RuntimeError("network down")

    upd = IncrementalUpdater(repo=repo, downloader=ExplodingDownloader())
    with pytest.raises(RuntimeError):
        upd.update()

    after = pd.read_parquet(repo.panel_path("prices"))
    assert before.equals(after)
    # no temp file left behind
    assert not list(repo.cache_dir.glob("*.tmp"))


def test_requires_existing_cache(tmp_path) -> None:
    repo = DataRepository(cache_dir=tmp_path / "cache")
    upd = IncrementalUpdater(repo=repo)
    with pytest.raises(FileNotFoundError):
        upd.update()
