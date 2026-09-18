"""Shared numerical helpers for factor computation.

These helpers exist because pandas' implicit alignment semantics are a
common source of silent bugs in quant code: ``DataFrame / Series`` aligns
the Series along *columns* by default, while rolling regressions almost
always need row-wise (index) broadcast.  Every helper here makes the
broadcast direction explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(
    numerator: pd.DataFrame, denominator: pd.DataFrame | pd.Series, axis: int = 0
) -> pd.DataFrame:
    """Element-wise division where zero / non-finite denominators yield NaN.

    Parameters
    ----------
    axis:
        Broadcast direction when ``denominator`` is a Series:
        ``axis=0`` aligns it with the DataFrame index (row-wise),
        ``axis=1`` with the columns.  DataFrame / DataFrame ignores it.
    """
    if isinstance(denominator, pd.Series):
        den = denominator.where(denominator.ne(0) & np.isfinite(denominator))
        out = numerator.div(den, axis=axis)
    else:
        den = denominator.where(denominator.ne(0) & np.isfinite(denominator))
        out = numerator.div(den)
    return out.replace([np.inf, -np.inf], np.nan).astype(float)


def rolling_tstat(series: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling t-statistic of the OLS slope of each column against time.

    For y regressed on t = 1..W inside each window the slope t-stat equals
    ``slope / SE(slope)``.  All quantities are computed from rolling sums
    (O(N) total) instead of per-window regressions (O(N*W)):

        S_tt = sum t^2 - W * mean(t)^2
        S_ty = sum t*y - W * mean(t) mean(y)
        slope = S_ty / S_tt
        SSE   = S_yy - slope * S_ty
        SE    = sqrt( SSE / (W-2) / S_tt )
    """
    n = len(series)
    t = pd.Series(np.arange(1, n + 1, dtype=float), index=series.index)

    t_bar = t.rolling(window).mean()  # Series (dates)
    y_bar = series.rolling(window).mean()  # DF (dates x tickers)
    t_sq = (t**2).rolling(window).sum()  # Series
    y_sq = (series**2).rolling(window).sum()  # DF
    ty = series.mul(t, axis=0).rolling(window).sum()  # DF

    s_tt = t_sq - window * t_bar**2  # Series
    s_yy = y_sq - window * y_bar.pow(2)  # DF
    s_ty = ty - window * y_bar.mul(t_bar, axis=0)  # DF  (row broadcast!)

    slope = safe_div(s_ty, s_tt, axis=0)  # DF / Series(dates)
    s_ee = (s_yy - slope * s_ty).clip(lower=0)  # DF
    dof = max(window - 2, 1)
    se = np.sqrt(safe_div(s_ee, dof * s_tt, axis=0))
    return safe_div(slope, se)


def rolling_skew(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sample skewness (bias-corrected) of daily returns."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    m3 = ((returns - mean) ** 3).rolling(window).mean()
    skew = safe_div(m3, std.pow(3))
    # population third moment -> sample skewness adjustment
    adj = np.sqrt(window * (window - 1)) / max(window - 2, 1)
    return skew * adj


def cs_winsorize(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Cross-sectional winsorisation, applied row by row (per date)."""
    if not limits:
        return df
    return df.apply(lambda row: row.clip(row.quantile(limits[0]), row.quantile(limits[1])), axis=1)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date row (mean / std of the row)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    return safe_div(df.sub(mean, axis=0), std, axis=0)
