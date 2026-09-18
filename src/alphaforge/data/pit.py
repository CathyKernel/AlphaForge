"""Point-in-time (PIT) universe construction.

Survivorship and look-ahead bias in the *universe* is the most under-appreciated
leak in retail backtests: selecting names by *today's* index membership means
the backtest already knows which companies survived and grew large enough to
be added later (e.g. TSLA joined the S&P 500 in Dec-2020; a 2016 backtest that
trades TSLA "because it is in the S&P 500" is looking ahead four years).

:class:`PointInTimeUniverse` reconstructs a daily membership matrix from the
information a free dataset *can* provide:

1. **Additions** (exact): the S&P snapshot's ``date_added`` column is the
   official index-addition date.  A ticker is a member only from that date.
   Dual-class listings (GOOG/GOOGL, FOXA/FOX, NWSA/NWS) share the addition
   date of the earlier class when the snapshot lacks per-class dates.
2. **Trading availability** (exact per data): a ticker is a member only on
   dates where the price panel has a valid observation (an IPO that starts
   trading in 2021 cannot be a 2015 member; a name whose data ends in 2022
   is not a member after that).
3. **Removals** (approximate): free sources do not publish removal dates.
   We approximate a removal by the *end of price data* when that end is
   strictly before the panel's last date, and document the residual bias
   (delisted-but-kept names still trading remain members).  For a 10-year
   S&P 500 study this affects single-digit names, not the aggregate.

The output is a **date x ticker boolean membership matrix** aligned with a
:class:`~alphaforge.data.panel.PricePanel`, directly consumable as a mask by
:class:`~alphaforge.engine.vectorized.BacktestEngine` (``universe_mask=``).

Example::

    pit = PointInTimeUniverse.from_snapshot(panel)
    pit.members.loc["2015-06-30"].sum()          # members on that date
    engine = BacktestEngine(panel, universe_mask=pit.members)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.data.universe import Universe


def _snapshot_additions(universe: Universe) -> pd.Series:
    """ticker -> addition date (NaT when the snapshot does not know)."""
    raw = getattr(universe, "additions", None)
    if raw is not None and len(raw):
        return raw

    # Universe snapshot file may carry date_added even when the dataclass
    # does not expose it; recover it from the original CSV columns.
    snap = getattr(universe, "raw_frame", None)
    if snap is not None and "date_added" in getattr(snap, "columns", []):
        s = pd.to_datetime(snap.set_index("ticker")["date_added"], errors="coerce")
        return s.reindex(list(universe.tickers))
    return pd.Series(pd.NaT, index=list(universe.tickers), dtype="datetime64[ns]")


@dataclass(frozen=True)
class PITStats:
    """Summary statistics of a PIT universe."""

    n_dates: int
    n_tickers_ever: int
    members_first: int
    members_last: int
    members_min: int
    members_max: int
    members_mean: float
    additions_after_start: int
    names_with_unknown_addition: int
    data_terminated: int


class PointInTimeUniverse:
    """Daily membership matrix of a stock index, as known *at the time*."""

    def __init__(self, members: pd.DataFrame, stats: PITStats | None = None) -> None:
        if not isinstance(members, pd.DataFrame) or members.empty:
            raise ValueError("members must be a non-empty date x ticker DataFrame")
        if not members.index.is_monotonic_increasing:
            members = members.sort_index()
        self.members: pd.DataFrame = members.astype(bool)
        self._stats = stats

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_snapshot(
        cls,
        panel: PricePanel,
        universe: Universe | None = None,
        min_history: int = 0,
    ) -> PointInTimeUniverse:
        """Build the PIT membership matrix for ``panel``.

        Parameters
        ----------
        panel:
            Price panel whose dates define the matrix rows.
        universe:
            Universe with (ideally) a ``date_added`` column.  When omitted,
            the S&P 500 snapshot is used.
        min_history:
            Require this many prior trading days of data before membership
            activates (freshly added names often have thin, jumpy data;
            index funds themselves phase in over days).
        """
        if universe is None:
            universe = Universe.sp500()

        dates = panel.dates
        tickers = [t for t in panel.tickers if t != "SPY"]
        close = panel.close.reindex(index=dates, columns=tickers)

        # 1) addition gate: True from the first panel date >= date_added
        added = _snapshot_additions(universe).reindex(tickers)
        first_trade = close.apply(lambda s: s.first_valid_index())
        last_trade = close.apply(lambda s: s.last_valid_index())

        panel_start = dates[0]
        panel_end = dates[-1]

        # effective listing = max(addition date, first trade date)
        eff_start = pd.DataFrame(
            {
                "added": added,
                "first": first_trade,
            }
        ).max(axis=1)  # NaN-safe per column: NaT ignored unless both NaT
        # rows where both are NaT stay NaT -> treat as always-member
        unknown = eff_start.isna()
        eff_start = eff_start.fillna(panel_start)

        # apply optional trading-history seasoning
        if min_history > 0:
            pos = {d: i for i, d in enumerate(dates)}
            seasoned = first_trade.map(
                lambda ft: (
                    dates[min(pos.get(ft, 0) + min_history, len(dates) - 1)]
                    if pd.notna(ft)
                    else pd.NaT
                )
            )
            eff_start = (
                pd.concat([eff_start.rename("a"), seasoned.rename("b")], axis=1)
                .max(axis=1)
                .fillna(panel_start)
            )

        starts = eff_start.to_dict()

        # 2) availability gate: member only while price data exists
        has_price = close.notna()
        # fill internal gaps (halts) as member=True between first/last trade
        avail = has_price.copy()
        for t in tickers:
            ft, lt = first_trade.get(t), last_trade.get(t)
            if pd.notna(ft) and pd.notna(lt):
                avail.loc[ft:lt, t] = True

        # 3) assemble: addition gate AND availability
        gate_added = pd.DataFrame(
            {
                t: pd.Series(np.asarray(dates >= starts.get(t, panel_start)), index=dates)
                for t in tickers
            }
        )
        members = gate_added & avail

        stats = PITStats(
            n_dates=len(dates),
            n_tickers_ever=int(members.any().sum()),
            members_first=int(members.iloc[0].sum()),
            members_last=int(members.iloc[-1].sum()),
            members_min=int(members.sum(axis=1).min()),
            members_max=int(members.sum(axis=1).max()),
            members_mean=float(members.sum(axis=1).mean()),
            additions_after_start=int((eff_start > panel_start).sum()),
            names_with_unknown_addition=int(unknown.sum()),
            data_terminated=int(
                (last_trade.notna() & (last_trade < panel_end - pd.Timedelta(days=5))).sum()
            ),
        )
        return cls(members, stats)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def n_members(self) -> pd.Series:
        """Daily member count."""
        return self.members.sum(axis=1)

    @property
    def stats(self) -> PITStats:
        if self._stats is None:
            m = self.members
            self._stats = PITStats(
                n_dates=len(m.index),
                n_tickers_ever=int(m.any().sum()),
                members_first=int(m.iloc[0].sum()),
                members_last=int(m.iloc[-1].sum()),
                members_min=int(m.sum(axis=1).min()),
                members_max=int(m.sum(axis=1).max()),
                members_mean=float(m.sum(axis=1).mean()),
                additions_after_start=0,
                names_with_unknown_addition=0,
                data_terminated=0,
            )
        return self._stats

    def additions_by_year(self) -> pd.Series:
        """Names newly activated per calendar year (member count deltas).

        The first panel row is not an addition event (it is the panel's
        starting state), so it is excluded.
        """
        prev = self.members.shift(1, fill_value=False)
        newly = self.members & ~prev
        newly = newly.iloc[1:]  # drop the panel's first row
        if not len(newly):
            return pd.Series(dtype=int)
        return newly.sum(axis=1).groupby(newly.index.year).sum().astype(int)

    def coverage_at(self, date: str | pd.Timestamp) -> pd.Series:
        """Boolean membership vector on the last date <= ``date``."""
        idx = self.members.index.asof(pd.Timestamp(date))
        if idx is None:
            raise KeyError(f"date {date} before first panel date")
        return self.members.loc[idx]

    def restrict_tickers(self, tickers: list[str]) -> PointInTimeUniverse:
        cols = [t for t in tickers if t in self.members.columns]
        return PointInTimeUniverse(self.members[cols], self._stats)

    def restrict_dates(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> PointInTimeUniverse:
        m = self.members.loc[start:end]
        return PointInTimeUniverse(m)

    def summary(self) -> dict:
        s = self.stats
        return {
            "n_dates": s.n_dates,
            "n_tickers_ever": s.n_tickers_ever,
            "members_first": s.members_first,
            "members_last": s.members_last,
            "members_min": s.members_min,
            "members_max": s.members_max,
            "members_mean": round(s.members_mean, 1),
            "additions_after_start": s.additions_after_start,
            "names_with_unknown_addition": s.names_with_unknown_addition,
            "data_terminated": s.data_terminated,
        }

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.stats
        return (
            "PointInTimeUniverse("
            f"dates={s.n_dates}, tickers_ever={s.n_tickers_ever}, "
            f"members {s.members_first} -> {s.members_last}, mean {s.members_mean:.0f})"
        )
