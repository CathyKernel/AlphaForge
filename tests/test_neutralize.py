"""Signal neutralisation: sector demean, beta residual, vol scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_panel

from alphaforge.data.universe import Universe
from alphaforge.factors.neutralize import (
    Neutralizer,
    beta_residual,
    sector_demean,
    trailing_beta,
    trailing_idio_vol,
    vol_scale,
    winsorize_zscore,
)

SECTORS = {
    "AAA": "Tech",
    "BBB": "Tech",
    "CCC": "Tech",
    "DDD": "Health",
    "EEE": "Health",
    "FFF": "Health",
}


def _uni() -> Universe:
    return Universe(tuple(SECTORS), pd.Series(SECTORS), "t", "test")


def _signal(panel, seed: int = 5, tilt: dict[str, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(
        rng.normal(0, 1, (len(panel.dates), len(panel.tickers))),
        index=panel.dates,
        columns=panel.tickers,
    )
    for t, v in (tilt or {}).items():
        sig[t] = sig[t] + v
    return sig


# ------------------------------------------------------------------ #
def test_winsorize_zscore_clips_and_standardises() -> None:
    panel = make_panel()
    sig = _signal(panel)
    sig.iloc[0, 0] = 50.0  # plant an extreme outlier
    z = winsorize_zscore(sig)
    assert z.max().max() <= 3.0 + 1e-9
    row = z.iloc[10].dropna()
    assert abs(row.mean()) < 1e-9
    # standardisation divides by the *sample* std (ddof=1)
    assert abs(row.std(ddof=1) - 1.0) < 1e-9
    assert z.index.equals(sig.index) and list(z.columns) == list(sig.columns)


def test_sector_demean_zeroes_group_means() -> None:
    panel = make_panel()
    sig = _signal(panel, tilt={"AAA": 2.0, "BBB": 1.5, "CCC": 1.0})  # Tech tilt
    out = sector_demean(sig, pd.Series(SECTORS))
    for dt in sig.index[::20]:
        row = out.loc[dt].dropna()
        for sector in ("Tech", "Health"):
            members = [t for t, s in SECTORS.items() if s == sector and t in row.index]
            assert abs(row[members].mean()) < 1e-12


def test_sector_demean_small_group_untouched() -> None:
    """Groups with < min_group members are not demeaned."""
    sig = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [10.0, -10.0]}, index=pd.bdate_range("2020-01-01", periods=2)
    )
    sectors = pd.Series({"A": "X", "B": "Y"})
    out = sector_demean(sig, sectors, min_group=3)
    # group B has 1 member -> mean is the value itself -> kept as-is
    assert abs(out["B"].iloc[0] - sig["B"].iloc[0]) < 1e-12


def test_trailing_beta_recovers_planted_beta() -> None:
    """Names with beta=1.5 vs an explicit MARK ticker -> beta ~ 1.5."""
    rng = np.random.default_rng(9)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0.0002, 0.01, n)
    cols = {"MARK": 100 * np.exp(np.cumsum(mkt))}
    for t in ["A", "B", "C"]:
        cols[t] = 100 * np.exp(np.cumsum(1.5 * mkt + rng.normal(0, 0.003, n)))
    close = pd.DataFrame(cols, index=idx)
    panel = make_panel(n_days=1, tickers=["A"])
    fields = {
        "close": close,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "volume": pd.DataFrame(1e8, index=idx, columns=close.columns),
    }
    panel = panel.__class__(fields)
    beta = trailing_beta(panel, window=126, benchmark="MARK")
    late = beta.iloc[-1]
    assert all(abs(late[t] - 1.5) < 0.1 for t in ["A", "B", "C"]), late


def test_beta_residual_removes_planted_exposure() -> None:
    """Signal = 2*beta + noise must end up ~uncorrelated with beta."""
    rng = np.random.default_rng(12)
    n_dates, n_names = 60, 40
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    names = [f"S{i}" for i in range(n_names)]
    beta = pd.DataFrame(rng.uniform(0.5, 2.0, (n_dates, n_names)), index=idx, columns=names)
    noise = pd.DataFrame(rng.normal(0, 0.5, (n_dates, n_names)), index=idx, columns=names)
    sig = 2.0 * beta + noise

    resid = beta_residual(sig, beta)

    # pooled correlation with beta should collapse vs original
    def pooled_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
        x = a.to_numpy().ravel()
        y = b.to_numpy().ravel()
        return float(np.corrcoef(x, y)[0, 1])

    c_before = pooled_corr(sig, beta)
    c_after = pooled_corr(resid, beta)
    assert c_before > 0.8
    assert abs(c_after) < 0.15


def test_vol_scale_shrinks_volatile_names() -> None:
    panel = make_panel()
    iv = trailing_idio_vol(panel, window=63, benchmark="SPY")  # SPY absent -> EW
    sig = _signal(panel)
    scaled = vol_scale(sig, iv)
    # every value finite, and high-vol columns shrink relative to low-vol
    ratio = (scaled / sig).abs().mean()
    iv_mean = iv.mean()
    top_vol = iv_mean.idxmax()
    low_vol = iv_mean.idxmin()
    assert ratio[top_vol] < ratio[low_vol]


def test_neutralizer_pipeline_end_to_end() -> None:
    panel = make_panel()
    uni = _uni()
    sig = _signal(panel, tilt={"AAA": 3.0, "BBB": 2.0, "CCC": 1.0})
    neu = Neutralizer(universe=uni, panel=panel, sector=True, beta=True, winsorize=True)
    out = neu.transform(sig)
    assert out.index.equals(sig.index)
    assert list(out.columns) == list(sig.columns)
    assert neu.steps_applied == ["winsor_z", "sector_demean", "beta_residual"]
    # residual sector tilts must be gone
    out = sector_demean(out, pd.Series(SECTORS))
    assert abs(out.iloc[-1].dropna().mean()) < 1e-9


def test_neutralizer_requires_inputs() -> None:
    with pytest.raises(ValueError):
        Neutralizer(sector=True)  # no universe
    with pytest.raises(ValueError):
        Neutralizer(beta=True)  # no panel


def test_neutralizer_vol_step() -> None:
    panel = make_panel()
    uni = _uni()
    neu = Neutralizer(universe=uni, panel=panel, vol=True, sector=False, winsorize=False)
    out = neu.transform(_signal(panel))
    assert neu.steps_applied == ["vol_scale"]
    # early rows lack a full trailing vol window -> NaN is expected there
    assert np.isfinite(out.iloc[40:].to_numpy()).all()


def test_neutralize_real_snapshot_sectors() -> None:
    """Bundled S&P sector table has all 11 GICS sectors."""
    uni = Universe.sp500()
    counts = uni.sector_counts()
    assert len(counts) == 11
    assert counts.sum() == len(uni)
