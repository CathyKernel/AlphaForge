"""Risk analytics: tail risk, drawdown forensics, stress tests, rolling exposures.

The philosophy here is *diagnostic, not cosmetic*: every number is
something a risk committee would actually look at before allocating --
historical and moment-corrected VaR, expected shortfall, drawdown
episodes with recovery times, factor exposures of the strategy, and
replay of historical stress windows on the strategy's return stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.engine.metrics import (
    cvar_historical,
    max_drawdown,
    var_cornish_fisher,
    var_historical,
)


@dataclass
class DrawdownEpisode:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    depth: float
    length_days: int
    recovery_days: int | None

    def as_dict(self) -> dict:
        return {
            "peak": str(self.peak_date.date()),
            "trough": str(self.trough_date.date()),
            "recovery": (
                str(self.recovery_date.date())
                if self.recovery_date is not None
                else "not recovered"
            ),
            "depth": round(float(self.depth), 4),
            "length_days": self.length_days,
            "recovery_days": self.recovery_days,
        }


def drawdown_episodes(
    returns: pd.Series, top: int = 5, min_depth: float = 0.05
) -> list[DrawdownEpisode]:
    """Identify the deepest distinct drawdown episodes of a return series."""
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    in_dd = dd < -min_depth
    episodes: list[DrawdownEpisode] = []
    if in_dd.any():
        groups = (in_dd != in_dd.shift()).cumsum()
        for _, g in dd.groupby(groups):
            if g.min() >= -min_depth:
                continue
            trough = g.idxmin()
            p = cum.loc[:trough].idxmax()
            after = cum.loc[trough:]
            recovered = after[after >= cum.loc[p]]
            rec_date = recovered.index[0] if len(recovered) else None
            episodes.append(
                DrawdownEpisode(
                    peak_date=p,
                    trough_date=trough,
                    recovery_date=rec_date,
                    depth=float(g.min()),
                    length_days=int((trough - p).days),
                    recovery_days=(int((rec_date - p).days) if rec_date is not None else None),
                )
            )
    episodes.sort(key=lambda e: e.depth)
    return episodes[:top]


#: Historical stress windows replayed on strategy returns (US equity crises).
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "COVID crash (2020-02..03)": ("2020-02-19", "2020-03-23"),
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2015-16 mini-correction": ("2015-11-03", "2016-02-11"),
    "2024 Aug yen-carry unwind": ("2024-07-16", "2024-08-05"),
}


def stress_test(
    returns: pd.Series, windows: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Replay historical crisis windows on the strategy's own returns.

    This is NOT a scenario model -- it answers the humbler question
    "what did this return stream actually do during famous crises?",
    which is the first sanity check any allocator runs.
    """
    windows = windows or STRESS_WINDOWS
    rows = []
    for name, (s, e) in windows.items():
        seg = returns.loc[s:e]
        if len(seg) < 5:
            continue
        rows.append(
            {
                "episode": name,
                "n_days": int(len(seg)),
                "strategy_return": float((1 + seg).prod() - 1),
                "ann_vol_realised": float(seg.std(ddof=1) * np.sqrt(252)),
                "worst_day": float(seg.min()),
            }
        )
    return pd.DataFrame(rows).set_index("episode")


def rolling_risk(
    returns: pd.Series, benchmark: pd.Series | None = None, window: int = 126
) -> pd.DataFrame:
    """Rolling annualised vol, Sharpe, VaR95 and (optional) beta."""
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(252)
    mean = returns.rolling(window).mean() * 252
    sharpe = mean / vol.replace(0, np.nan)
    var95 = returns.rolling(window).apply(
        lambda x: -np.quantile(x, 0.05) if len(x) == window else np.nan, raw=True
    )
    out = pd.DataFrame({"ann_vol": vol, "ann_return": mean, "sharpe": sharpe, "var95_daily": var95})
    if benchmark is not None:
        b = benchmark.reindex(returns.index)
        cov = returns.rolling(window).cov(b)
        var_b = b.rolling(window).var()
        out["beta"] = cov / var_b.replace(0, np.nan)
    return out


def factor_exposure(
    returns: pd.Series, factor_returns: dict[str, pd.Series], window: int = 252
) -> pd.DataFrame:
    """Rolling beta of the strategy to long-short factor return series."""
    rows = []
    df = pd.DataFrame({"r": returns, **factor_returns}).dropna()
    if len(df) < window:
        return pd.DataFrame()
    for name in factor_returns:
        cov = df["r"].rolling(window).cov(df[name])
        var = df[name].rolling(window).var()
        beta = (cov / var.replace(0, np.nan)).rename(name)
        rows.append(beta)
    out = pd.concat(rows, axis=1)
    return out.dropna(how="all")


def risk_summary(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Headline tail-risk statistics for a return series."""
    out = {
        "var95_historical": var_historical(returns, 0.95),
        "var99_historical": var_historical(returns, 0.99),
        "var95_cornish_fisher": var_cornish_fisher(returns, 0.95),
        "cvar95": cvar_historical(returns, 0.95),
        "skew": float(returns.skew()),
        "excess_kurtosis": float(returns.kurtosis()),
        "worst_day": float(returns.min()),
        "worst_week": float((1 + returns).rolling(5).apply(np.prod, raw=True).min() - 1),
        "max_drawdown": max_drawdown(returns)[0],
    }
    if benchmark is not None:
        out["benchmark_max_drawdown"] = max_drawdown(benchmark)[0]
    return out
