"""Performance and risk metrics for return series.

All statistics follow institutional conventions: annualisation factor 252,
Sharpe uses arithmetic excess return over the risk-free proxy (default 0),
Sortino penalises only downside deviation, and information ratio uses
active (benchmark-relative) returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = returns.notna().sum()
    if n == 0:
        return float("nan")
    years = n / periods_per_year
    tr = total_return(returns)
    if tr <= -1:
        return float("nan")
    return float((1.0 + tr) ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd = np.sqrt((downside**2).sum() / len(excess))
    return float(excess.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else float("nan")


def drawdown_series(cum: pd.Series) -> pd.Series:
    peak = cum.cummax()
    return cum / peak - 1.0


def max_drawdown(returns: pd.Series) -> tuple[float, float, float, float]:
    """Return (depth, peak_date, trough_date, recovery_days).

    recovery_days is measured peak->recovery in calendar days when the
    index is datetime-like, otherwise in index steps; ``inf`` when the
    drawdown never recovered.
    """
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    dd = drawdown_series(cum)
    if dd.empty or dd.min() >= 0:
        return 0.0, float("nan"), float("nan"), 0.0
    trough = dd.idxmin()
    peak = cum.loc[:trough].idxmax()
    after = cum.loc[trough:]
    recovered = after[after >= cum.loc[peak]]
    if len(recovered):
        delta = recovered.index[0] - peak
        recovery = float(delta.days) if hasattr(delta, "days") else float(delta)
    else:
        recovery = float("inf")
    return float(dd.min()), peak, trough, recovery


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    dd, *_ = max_drawdown(returns)
    ann = cagr(returns, periods_per_year)
    return float(ann / abs(dd)) if dd < 0 and ann == ann else float("nan")


def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR as a positive loss quantile (e.g. 0.02 = -2%)."""
    return float(-np.quantile(returns.dropna(), 1 - level))


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    tail = returns.dropna()[returns.dropna() <= -var_historical(returns, level)]
    return float(-tail.mean()) if len(tail) else float("nan")


def var_cornish_fisher(returns: pd.Series, level: float = 0.95) -> float:
    """Cornish-Fisher (moment-adjusted) VaR; captures fat tails.

    ``level`` is the confidence level (e.g. 0.95); the tail probability is
    ``1 - level`` and the normal quantile z is negative (left tail).
    """
    x = returns.dropna()
    z = {0.90: -1.2816, 0.95: -1.6449, 0.99: -2.3263}.get(round(level, 2), -1.6449)
    mu, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return 0.0
    s = float(x.skew())
    k = float(x.kurtosis())  # excess kurtosis
    cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(mu + cf * sd))


def var_historical_series(returns: pd.Series, window: int = 252, level: float = 0.95) -> pd.Series:
    return returns.rolling(window).apply(lambda x: -np.quantile(x, 1 - level), raw=True)


def tracking_error(active: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(active: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    te = active.std(ddof=1)
    return float(active.mean() / te * np.sqrt(periods_per_year)) if te > 0 else float("nan")


def alpha_beta(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> tuple[float, float]:
    df = pd.concat([returns, benchmark], axis=1, keys=["r", "b"]).dropna()
    if len(df) < 10 or df["b"].std() == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(df["r"], df["b"])[0, 1] / np.var(df["b"], ddof=1))
    alpha = float((df["r"].mean() - beta * df["b"].mean()) * periods_per_year)
    return alpha, beta


def up_down_capture(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    df = pd.concat([returns, benchmark], axis=1, keys=["r", "b"]).dropna()
    up, down = df["b"] > 0, df["b"] < 0
    up_cap = (df["r"][up].mean() / df["b"][up].mean()) if up.any() else float("nan")
    down_cap = (df["r"][down].mean() / df["b"][down].mean()) if down.any() else float("nan")
    return float(up_cap), float(down_cap)


def rolling_sharpe(
    returns: pd.Series, window: int = 126, periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    mean = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return (mean / sd.replace(0, np.nan)) * np.sqrt(periods_per_year)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year x month pivot table of monthly total returns."""
    m = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2020, int(c), 1).strftime("%b") for c in pivot.columns]
    return pivot


def performance_summary(
    returns: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0
) -> dict:
    """One-stop dictionary of headline performance statistics."""
    r = returns.dropna()
    out = {
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe(r, rf=rf),
        "sortino": sortino(r, rf=rf),
        "calmar": calmar(r),
        "max_drawdown": max_drawdown(r)[0],
        "var95_historical": var_historical(r, 0.95),
        "var95_cornish_fisher": var_cornish_fisher(r, 0.95),
        "cvar95": cvar_historical(r, 0.95),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "pct_positive_days": float((r > 0).mean()),
        "n_days": int(len(r)),
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        r_common = r.reindex(b.index)
        active = r_common - b
        a, beta = alpha_beta(r_common, b)
        up_cap, down_cap = up_down_capture(r_common, b)
        out.update(
            {
                "benchmark_total_return": total_return(b),
                "benchmark_cagr": cagr(b),
                "benchmark_sharpe": sharpe(b, rf=rf),
                "excess_cagr": cagr(r_common) - cagr(b),
                "tracking_error": tracking_error(active),
                "information_ratio": information_ratio(active),
                "alpha_ann": a,
                "beta": beta,
                "up_capture": up_cap,
                "down_capture": down_cap,
                "correlation_benchmark": float(r_common.corr(b)),
            }
        )
    return out
