"""PointInTimeUniverse: membership timing, gating, and engine integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_panel

from alphaforge.data.pit import PointInTimeUniverse
from alphaforge.data.universe import Universe


def _uni(additions: dict[str, str], sectors: dict[str, str] | None = None) -> Universe:
    tickers = list(additions)
    sec = pd.Series(
        sectors or {t: "Tech" if i % 2 else "Health" for i, t in enumerate(tickers)},
        index=tickers,
    )
    return Universe(
        tuple(tickers),
        sec,
        "2026-01-01",
        "test",
        pd.to_datetime(pd.Series(additions)),
    )


def test_addition_gate_blocks_names_added_later() -> None:
    """A ticker added mid-sample is not a member before its addition date."""
    panel = make_panel(n_days=400, tickers=["OLD", "NEW", "BENCH"])
    additions = {
        "OLD": "1995-01-01",  # in index before the sample starts
        "NEW": "2021-06-01",  # added mid-sample
        "BENCH": "1995-01-01",
    }
    pit = PointInTimeUniverse.from_snapshot(panel, _uni(additions))

    before = pit.coverage_at("2021-05-28")
    after = pit.coverage_at("2021-06-30")
    assert bool(before["NEW"]) is False
    assert bool(after["NEW"]) is True
    assert bool(before["OLD"]) is True


def _blank(panel, ticker: str, lo: int, hi: int):
    """Return a copy of ``panel`` with ``ticker``'s close NaN on [lo, hi)."""
    fields = {f: df.copy() for f, df in panel._fields.items()}
    for f in fields:
        fields[f].iloc[lo:hi, fields[f].columns.get_loc(ticker)] = np.nan
    return panel.__class__(fields)


def test_late_listing_not_member_before_first_trade() -> None:
    """A name whose data starts mid-sample is not a member before listing."""
    panel = make_panel(n_days=300, tickers=["A", "B", "IPO"])
    panel = _blank(panel, "IPO", 0, 200)  # first trade around day 200
    uni = _uni({"A": "1995-01-01", "B": "1995-01-01", "IPO": "2019-01-01"})
    pit = PointInTimeUniverse.from_snapshot(panel, uni)

    early = pit.coverage_at(idx_date(panel, 100))
    late = pit.coverage_at(idx_date(panel, 250))
    assert bool(early["IPO"]) is False
    assert bool(late["IPO"]) is True


def idx_date(panel, i: int) -> pd.Timestamp:
    return panel.dates[i]


def test_data_terminated_name_leaves_universe() -> None:
    """A name whose price data ends mid-sample stops being a member."""
    panel = make_panel(n_days=300, tickers=["A", "B", "DEAD"])
    panel = _blank(panel, "DEAD", 150, 300)
    uni = _uni({"A": "1995-01-01", "B": "1995-01-01", "DEAD": "1995-01-01"})
    pit = PointInTimeUniverse.from_snapshot(panel, uni)

    before = pit.coverage_at(idx_date(panel, 140))
    after = pit.coverage_at(idx_date(panel, 200))
    assert bool(before["DEAD"]) is True
    assert bool(after["DEAD"]) is False


def test_benchmark_excluded_from_members() -> None:
    panel = make_panel(n_days=60, tickers=["A", "B", "SPY"])
    uni = _uni({"A": "1995-01-01", "B": "1995-01-01", "SPY": "1995-01-01"})
    pit = PointInTimeUniverse.from_snapshot(panel, uni)
    assert "SPY" not in pit.members.columns


def test_min_history_seasoning() -> None:
    """With min_history=n, membership activates n days after listing."""
    panel = make_panel(n_days=300, tickers=["A", "IPO"])
    panel = _blank(panel, "IPO", 0, 100)
    uni = _uni({"A": "1995-01-01", "IPO": "2019-01-01"})

    pit0 = PointInTimeUniverse.from_snapshot(panel, uni, min_history=0)
    pit20 = PointInTimeUniverse.from_snapshot(panel, uni, min_history=20)

    just_after = idx_date(panel, 102)
    assert bool(pit0.coverage_at(just_after)["IPO"]) is True
    assert bool(pit20.coverage_at(just_after)["IPO"]) is False
    later = idx_date(panel, 125)
    assert bool(pit20.coverage_at(later)["IPO"]) is True


def test_stats_and_additions_by_year() -> None:
    panel = make_panel(n_days=400, tickers=["A", "B", "C"])
    uni = _uni({"A": "1995-01-01", "B": "2020-05-01", "C": "2021-05-01"})
    pit = PointInTimeUniverse.from_snapshot(panel, uni)

    s = pit.summary()
    assert s["n_tickers_ever"] == 3
    assert s["members_first"] == 1  # only A at the start
    assert s["members_last"] == 3
    aby = pit.additions_by_year()
    assert int(aby.get(2020, 0)) == 1
    assert int(aby.get(2021, 0)) == 1


def test_unknown_addition_treated_as_always_member() -> None:
    """NaT date_added falls back to 'member from panel start'."""
    panel = make_panel(n_days=60, tickers=["A", "UNK"])
    uni = _uni({"A": "1995-01-01", "UNK": "2000-01-01"})
    uni = Universe(
        uni.tickers,
        uni.sectors,
        "t",
        "test",
        pd.Series([pd.Timestamp("1995-01-01"), pd.NaT], index=list(uni.tickers)),
    )
    pit = PointInTimeUniverse.from_snapshot(panel, uni)
    assert bool(pit.members.iloc[0]["UNK"]) is True


def test_invalid_members_rejected() -> None:
    with pytest.raises(ValueError):
        PointInTimeUniverse(pd.DataFrame())


def test_real_snapshot_smoke() -> None:
    """Bundled S&P 500 snapshot: TSLA enters 2021, AAPL is always in."""
    from alphaforge.data.universe import Universe as U

    uni = U.sp500()
    assert uni.additions is not None and uni.additions.notna().sum() > 400
    panel = make_panel(n_days=80, tickers=["AAPL", "TSLA", "KO"])
    # give TSLA full price history in the synthetic panel (addition gate
    # must be what keeps it out pre-2021, not missing data)
    pit = PointInTimeUniverse.from_snapshot(panel, uni)
    tsla_added = uni.additions.get("TSLA")
    pre = pit.coverage_at(min(pd.Timestamp("2020-12-01"), panel.dates[-1]))
    assert not bool(pre.get("TSLA", False))
    assert pd.Timestamp(tsla_added) >= pd.Timestamp("2020-12-01")


# ------------------------------------------------------------------ #
# Engine integration: the mask must actually constrain trading
# ------------------------------------------------------------------ #
def test_engine_universe_mask_blocks_ticker() -> None:
    from alphaforge.engine import run_backtest

    panel = make_panel(n_days=180, tickers=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    rng = np.random.default_rng(3)
    signal = pd.DataFrame(
        rng.normal(0, 1, (len(panel.dates), len(panel.tickers))),
        index=panel.dates,
        columns=panel.tickers,
    )
    mask = pd.DataFrame(True, index=panel.dates, columns=panel.tickers)
    mask["AAA"] = False  # banned for the whole period

    res = run_backtest(panel, signal, side="long", top_n=2, universe_mask=mask, rebalance="W-FRI")
    assert (res.weights["AAA"].abs() < 1e-12).all()

    res_free = run_backtest(panel, signal, side="long", top_n=2, rebalance="W-FRI")
    assert (res_free.weights["AAA"].abs() > 0).any()


def test_engine_universe_mask_time_varying() -> None:
    """Mask turning off mid-sample must zero the holding from that date."""
    from alphaforge.engine import run_backtest

    panel = make_panel(n_days=180, tickers=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    rng = np.random.default_rng(4)
    signal = -pd.DataFrame(
        rng.normal(0, 1, (len(panel.dates), len(panel.tickers))),
        index=panel.dates,
        columns=panel.tickers,
    )
    # make AAA the strongest signal every day so it is always selected
    signal["AAA"] = 10.0
    signal["BBB"] = 9.0

    mask = pd.DataFrame(True, index=panel.dates, columns=panel.tickers)
    mid = panel.dates[len(panel.dates) // 2]
    mask.loc[mid:, "AAA"] = False

    res = run_backtest(panel, signal, side="long", top_n=3, universe_mask=mask, rebalance="W-FRI")
    w_aaa = res.weights["AAA"]
    assert (w_aaa.loc[:mid].abs() > 0).any()
    assert (w_aaa.loc[mid:].abs() < 1e-12).all()
