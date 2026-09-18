"""PricePanel: the core OHLCV container used across AlphaForge.

A :class:`PricePanel` stores daily OHLCV data in *wide* format -- one
``date x ticker`` DataFrame per field -- which makes both cross-sectional
(ranking, z-scores) and time-series (rolling windows) operations efficient.

Panels round-trip to a single Parquet file in *long* format
(one row per ``date x ticker``), which is compact and robust to schema drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")
_PRICE_FIELDS = ("open", "high", "low", "close")


class PricePanel:
    """Container for adjusted daily OHLCV data of a stock universe.

    Parameters
    ----------
    fields:
        Mapping ``field -> DataFrame`` indexed by date, one column per
        ticker.  Must contain at least ``close``.
    """

    def __init__(self, fields: dict[str, pd.DataFrame]) -> None:
        missing = set(FIELDS) - set(fields)
        if "close" not in fields:
            raise ValueError("PricePanel requires a 'close' field")
        if missing:
            # Fill optional fields with NaN so downstream code can degrade gracefully.
            template = fields["close"]
            for f in missing:
                fields[f] = pd.DataFrame(np.nan, index=template.index, columns=template.columns)

        index = fields["close"].index
        columns = fields["close"].columns
        for name, df in fields.items():
            if not df.index.equals(index) or not df.columns.equals(columns):
                raise ValueError(
                    f"Field {name!r} is not aligned with the 'close' field "
                    "(identical index and columns required)."
                )
        self._fields = {f: fields[f].astype(float) for f in FIELDS}

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    @property
    def open(self) -> pd.DataFrame:  # noqa: A003
        return self._fields["open"]

    @property
    def high(self) -> pd.DataFrame:
        return self._fields["high"]

    @property
    def low(self) -> pd.DataFrame:
        return self._fields["low"]

    @property
    def close(self) -> pd.DataFrame:
        return self._fields["close"]

    @property
    def volume(self) -> pd.DataFrame:
        return self._fields["volume"]

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def returns(self) -> pd.DataFrame:
        """Simple daily returns of the *close* field (first row is NaN).

        Internal gaps are forward-filled first (a halted stock holds its
        last price), matching the engine's own returns matrix semantics.
        """
        return self.close.ffill().pct_change(fill_method=None)

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log(self.close).diff()

    @property
    def dollar_volume(self) -> pd.DataFrame:
        """Close x volume; split-invariant when both series are adjusted."""
        return self.close * self.volume

    # ------------------------------------------------------------------ #
    # Transformations
    # ------------------------------------------------------------------ #
    def slice_dates(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> PricePanel:
        """Return a new panel restricted to ``[start, end]``."""
        return PricePanel({f: df.loc[start:end] for f, df in self._fields.items()})

    def restrict_tickers(self, tickers: list[str]) -> PricePanel:
        """Return a new panel restricted to the given tickers."""
        cols = [t for t in tickers if t in self.tickers]
        return PricePanel({f: df[cols] for f, df in self._fields.items()})

    # ------------------------------------------------------------------ #
    # Long-format conversion and persistence
    # ------------------------------------------------------------------ #
    def to_long(self) -> pd.DataFrame:
        """Return a long DataFrame with one row per (date, ticker)."""
        frames = []
        for f, df in self._fields.items():
            stacked = df.stack(future_stack=True)
            stacked.name = f
            frames.append(stacked)
        long = pd.concat(frames, axis=1)
        long.index.names = ["date", "ticker"]
        return long.reset_index()

    @classmethod
    def from_long(cls, long: pd.DataFrame) -> PricePanel:
        """Build a panel from a long DataFrame (inverse of :meth:`to_long`)."""
        required = {"date", "ticker", "close"}
        if not required.issubset(long.columns):
            raise ValueError(f"long frame must contain columns {sorted(required)}")
        fields: dict[str, pd.DataFrame] = {}
        for f in FIELDS:
            if f in long.columns:
                fields[f] = long.pivot(index="date", columns="ticker", values=f)
        return cls(fields)

    def to_parquet(self, path: str | Path) -> Path:
        """Persist the panel to a single (snappy) Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_long().to_parquet(path, engine="pyarrow", compression="snappy")
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> PricePanel:
        long = pd.read_parquet(Path(path), engine="pyarrow")
        return cls.from_long(long)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def coverage(self) -> pd.DataFrame:
        """Per-ticker coverage summary (history length, missing days, etc.)."""
        close = self.close
        notna = close.notna()
        n_days = notna.sum()
        first = close.apply(lambda s: s.first_valid_index())
        last = close.apply(lambda s: s.last_valid_index())
        out = pd.DataFrame(
            {
                "n_days": n_days,
                "first_date": first,
                "last_date": last,
                "pct_available": n_days / max(len(close), 1),
                "avg_close": close.mean(),
                "avg_dollar_volume": self.dollar_volume.mean(),
            }
        )
        out["avg_dollar_volume_musd"] = out["avg_dollar_volume"] / 1e6
        return out.sort_values("avg_dollar_volume", ascending=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        start = self.dates.min().date() if len(self.dates) else "n/a"
        end = self.dates.max().date() if len(self.dates) else "n/a"
        return (
            f"PricePanel(dates={start}..{end}, n_dates={len(self)}, n_tickers={len(self.tickers)})"
        )
