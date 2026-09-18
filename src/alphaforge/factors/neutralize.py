"""Cross-sectional signal neutralisation (pre-backtest preprocessing).

A raw alpha signal usually carries *unwanted exposures*: value screens tilt
toward Financials, momentum toward high-beta growth, and low-vol toward
Utilities/Staples.  Running the raw signal therefore measures *sector bets*
as much as *stock-picking skill*.  This module strips those exposures
cross-sectionally, day by day, using only information available at each
date — so it can sit between factor computation and the backtest engine
without introducing lookahead:

* :func:`sector_demean` — subtract each GICS sector's mean signal, so every
  sector is dollar-neutral within itself.  The classic Barra-style
  preprocessing.
* :func:`beta_residual` — regress the day's signal on trailing market beta
  and keep the residual (kills the "high-beta names just went up" effect).
* :func:`vol_scale` — divide by trailing idiosyncratic volatility, so the
  position risk of each alpha unit is comparable (a crude but robust
  covariance-free risk normalisation).
* :func:`winsorize_zscore` — clip cross-sectional outliers then standardise.

:class:`Neutralizer` composes these into one reusable pipeline::

    neu = Neutralizer(universe=uni, panel=panel, sector=True, beta=True)
    clean_signal = neu.transform(raw_signal)
    run_backtest(panel, clean_signal, ...)

Each step is a pure vectorised cross-sectional operation (one groupby /
one lstsq per day, matrix-wide), and every trailing statistic is computed
with data **through day t only**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel
from alphaforge.data.universe import Universe


# --------------------------------------------------------------------- #
# Building blocks (each: date-aligned, vectorised, no lookahead)
# --------------------------------------------------------------------- #
def winsorize_zscore(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Row-wise winsorise at ``clip`` MAD-ish z units, then z-score."""
    med = signal.median(axis=1)
    mad = (signal.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
    z = signal.sub(med, axis=0).div(1.4826 * mad, axis=0)
    z = z.clip(-clip, clip)
    mu = z.mean(axis=1)
    sd = z.std(axis=1).replace(0.0, np.nan)
    return z.sub(mu, axis=0).div(sd, axis=0)


def sector_demean(signal: pd.DataFrame, sectors: pd.Series, min_group: int = 3) -> pd.DataFrame:
    """Subtract the sector mean from each day's cross-section.

    Sectors with fewer than ``min_group`` valid names are not demeaned
    (the mean would be the signal itself, wiping the information).
    """
    tick = sectors.reindex(signal.columns)
    out = signal.copy()
    for dt in signal.index:
        row = signal.loc[dt].dropna()
        if len(row) < 2:
            continue
        sec = tick.reindex(row.index).dropna()
        common = sec.index
        if len(common) < 2:
            continue
        groups = sec.loc[common]
        means = row.loc[common].groupby(groups).transform("mean")
        counts = row.loc[common].groupby(groups).transform("count")
        adj = row.loc[common] - means.where(counts >= min_group, 0.0)
        out.loc[dt, adj.index] = adj
    return out


def trailing_beta(panel: PricePanel, window: int = 126, benchmark: str = "SPY") -> pd.DataFrame:
    """Rolling beta of each name vs the benchmark, known at day t.

    ``beta = cov(r_i, r_m) / var(r_m)`` over the trailing ``window`` days
    (inclusive of t — same information set as the signal).  The rolling
    covariance is computed as ``E[r_i r_m] - E[r_i] E[r_m]``, fully
    vectorised (no per-name loops, no per-date Python).
    """
    rets = panel.returns
    mkt = rets[benchmark] if benchmark in rets.columns else rets.mean(axis=1)
    mp = max(window // 2, 2)
    mean_i = rets.rolling(window, min_periods=mp).mean()
    mean_m = mkt.rolling(window, min_periods=mp).mean()
    e_ij = rets.mul(mkt, axis=0).rolling(window, min_periods=mp).mean()
    cov = e_ij.sub(mean_i.mul(mean_m, axis=0), axis=0)
    var = mkt.rolling(window, min_periods=mp).var()
    beta = cov.div(var, axis=0)
    if benchmark in beta.columns:
        beta = beta.drop(columns=[benchmark])
    return beta


def beta_residual(signal: pd.DataFrame, beta: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Remove the linear beta exposure from each day's signal.

    Cross-sectional OLS ``signal ~ a + b * beta`` per date; residual keeps
    the stock-specific part.  Beta is the *trailing* beta known at t.
    """
    out = signal.copy()
    common_idx = signal.index.intersection(beta.index)
    beta = beta.reindex(index=common_idx, columns=signal.columns)
    sig = signal.loc[common_idx]
    for dt in common_idx:
        s = sig.loc[dt]
        b = beta.loc[dt]
        ok = s.notna() & b.notna() & np.isfinite(b)
        x = b[ok].to_numpy(float)
        y = s[ok].to_numpy(float)
        if len(x) < 10 or np.std(x) < 1e-12:
            continue
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        out.loc[dt, s[ok].index] = resid
    return out


def trailing_idio_vol(panel: PricePanel, window: int = 63, benchmark: str = "SPY") -> pd.DataFrame:
    """Rolling idiosyncratic (market-residual) volatility, known at t."""
    rets = panel.returns
    mkt = rets[benchmark] if benchmark in rets.columns else rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)
    return resid.rolling(window, min_periods=window // 2).std()


def vol_scale(signal: pd.DataFrame, idio_vol: pd.DataFrame, floor: float = 0.005) -> pd.DataFrame:
    """Divide the signal by trailing idio vol (alpha per unit of risk)."""
    common = signal.index.intersection(idio_vol.index)
    iv = idio_vol.reindex(index=common, columns=signal.columns).clip(lower=floor)
    return signal.loc[common].div(iv)


# --------------------------------------------------------------------- #
# Composable pipeline
# --------------------------------------------------------------------- #
@dataclass
class Neutralizer:
    """Configurable neutralisation pipeline for a date x ticker signal.

    Parameters
    ----------
    universe:
        Universe with sectors (required for ``sector=True``).
    panel:
        Price panel (required for ``beta``/``vol`` steps).
    sector:
        Subtract sector means.
    beta:
        Strip trailing market-beta exposure.
    vol:
        Divide by trailing idio volatility.
    winsorize:
        Winsorise + z-score before the other steps (recommended default).
    beta_window / vol_window:
        Trailing window sizes (days, inclusive of t).
    """

    universe: Universe | None = None
    panel: PricePanel | None = None
    sector: bool = True
    beta: bool = False
    vol: bool = False
    winsorize: bool = True
    clip: float = 3.0
    beta_window: int = 126
    vol_window: int = 63
    benchmark: str = "SPY"

    def __post_init__(self) -> None:
        if self.sector and (self.universe is None or self.universe.sectors is None):
            raise ValueError("sector neutralisation requires a Universe with sectors")
        if (self.beta or self.vol) and self.panel is None:
            raise ValueError("beta/vol neutralisation requires a PricePanel")

    # ------------------------------------------------------------------ #
    def transform(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured steps in order: winsor -> vol -> sector -> beta."""
        out = signal.copy()
        steps: list[str] = []
        if self.winsorize:
            out = winsorize_zscore(out, clip=self.clip)
            steps.append("winsor_z")
        if self.vol:
            iv = trailing_idio_vol(self.panel, window=self.vol_window, benchmark=self.benchmark)
            out = vol_scale(out, iv)
            steps.append("vol_scale")
            out = winsorize_zscore(out, clip=self.clip)  # re-standardise units
        if self.sector:
            out = sector_demean(out, self.universe.sectors)
            steps.append("sector_demean")
        if self.beta:
            b = trailing_beta(self.panel, window=self.beta_window, benchmark=self.benchmark)
            out = beta_residual(out, b, clip=self.clip)
            steps.append("beta_residual")
        self.steps_applied = steps
        return out

    def describe(self) -> dict:
        return {
            "sector": self.sector,
            "beta": self.beta,
            "vol": self.vol,
            "winsorize": self.winsorize,
            "beta_window": self.beta_window,
            "vol_window": self.vol_window,
            "benchmark": self.benchmark,
        }
