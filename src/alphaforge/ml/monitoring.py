"""Model monitoring: has the alpha model's skill decayed, and when?

Production quant workflows re-train when the *out-of-sample* information
coefficient degrades.  This module implements four complementary
diagnostics:

* **Rolling IC** — trailing mean IC (default 63 trading days = one
  quarter) with confidence bands from the NW-style standard error.
* **CUSUM drift detection** — a cumulative-sum control chart
  (Page 1954) on standardised IC innovations; signals a *regime change*
  in skill, not just a bad week.
* **Regime report** — splits the OOS history into equal windows and
  reports IC / t-stat per window, making "the model worked in 2019 and
  died in 2023" visible at a glance.
* **Feature drift (PSI + KS)** — population stability index and a
  two-sample Kolmogorov-Smirnov statistic per feature, comparing the
  distribution a model was trained on against what it is scoring today.
  This is the *input* side of monitoring (IC is the *output* side):
  when the feature distribution moves, the model's calibration silently
  degrades long before IC reacts.

Everything is causal: every statistic at date t uses only IC/feature
observations up to t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def rolling_ic(ic: pd.Series, window: int = 63) -> pd.DataFrame:
    """Trailing mean IC and its standard-error band (causal)."""
    mean = ic.rolling(window, min_periods=window // 2).mean()
    std = ic.rolling(window, min_periods=window // 2).std()
    se = std / np.sqrt(window)
    return pd.DataFrame(
        {
            "ic_mean": mean,
            "ic_lower": mean - 1.96 * se,
            "ic_upper": mean + 1.96 * se,
            "ic_tstat": mean / se.replace(0, np.nan),
        }
    )


@dataclass(frozen=True)
class DriftAlert:
    """One CUSUM breach: skill drifted and the chart detected it."""

    date: pd.Timestamp
    direction: str  # 'skill_loss' | 'skill_gain'
    cusum: float


def cusum_drift(ic: pd.Series, threshold: float = 2.0, drift: float = 0.25) -> pd.DataFrame:
    """Two-sided CUSUM control chart on standardised IC innovations.

    Parameters
    ----------
    ic:
        Daily OOS IC series (mean can be non-zero — that's the point).
    threshold:
        Alarm level in units of the IC's own standard deviation.
    drift:
        Allowed drift (k) in SD units before the chart starts accumulating.

    Returns
    -------
    DataFrame indexed by date with the running ``cusum_pos`` / ``cusum_neg``
    statistics and a boolean ``alarm`` column.  Alerts accumulate while the
    statistic stays above threshold, so use ``alarm & ~alarm.shift(1)`` for
    breach *events*.
    """
    sd = float(ic.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return pd.DataFrame(index=ic.index)
    z = (ic - ic.mean()) / sd
    k = drift
    pos, neg = 0.0, 0.0
    rows = []
    for date, v in z.items():
        pos = max(0.0, pos + (v - k))
        neg = min(0.0, neg + (v + k))
        rows.append((date, pos, neg, pos > threshold or -neg > threshold))
    return pd.DataFrame(rows, columns=["date", "cusum_pos", "cusum_neg", "alarm"]).set_index("date")


def regime_report(ic: pd.Series, n_windows: int = 6) -> pd.DataFrame:
    """IC statistics per equal chronological window of the OOS history."""
    ic = ic.dropna()
    if len(ic) < n_windows * 10:
        raise ValueError("OOS history too short for a regime report")
    edges = np.array_split(np.arange(len(ic)), n_windows)
    rows = []
    for w in edges:
        seg = ic.iloc[w]
        se = seg.std(ddof=1) / np.sqrt(len(seg))
        rows.append(
            {
                "start": seg.index[0],
                "end": seg.index[-1],
                "n_days": int(len(seg)),
                "ic_mean": float(seg.mean()),
                "ic_tstat": float(seg.mean() / se) if se > 0 else np.nan,
                "ic_positive_rate": float((seg > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Feature drift: PSI + two-sample KS per feature
# --------------------------------------------------------------------- #
def psi(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    n_bins: int = 10,
    clip: float = 25.0,
) -> float:
    """Population stability index between two samples of one feature.

    PSI = sum over quantile bins of (a_i - e_i) * ln(a_i / e_i) with the
    bins cut on the *reference* (expected) distribution.  Industry
    thresholds: < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant
    drift.  ``clip`` caps per-bin contributions from empty bins.

    Both samples must be numeric; NaNs are dropped.
    """
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(e) < n_bins or len(a) < n_bins:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        # degenerate (near-constant) feature -> no meaningful drift
        return 0.0
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    contrib = (a_pct - e_pct) * np.log(a_pct / e_pct)
    return float(np.clip(contrib, -clip, clip).sum())


def ks_statistic(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), in [0, 1]."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:  # pragma: no cover - scipy is a hard dep
        return np.nan
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(e) < 2 or len(a) < 2:
        return np.nan
    return float(ks_2samp(e, a).statistic)


def drift_status(psi_value: float) -> str:
    """PSI band label (industry-standard thresholds)."""
    if not np.isfinite(psi_value):
        return "unknown"
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.25:
        return "moderate"
    return "significant"


def feature_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI / KS / mean-shift drift diagnostics.

    Parameters
    ----------
    reference:
        Feature matrix the model was trained on — (date, ticker) rows x
        feature columns (the ``X`` from :class:`FeaturePanelBuilder`).
    current:
        Feature matrix the model is scoring now, same columns.

    Returns
    -------
    DataFrame indexed by feature with ``psi``, ``status``, ``ks``,
    ``mean_ref``, ``mean_cur``, ``mean_shift_sd`` (shift in units of the
    reference SD).
    """
    cols = [c for c in reference.columns if c in current.columns]
    rows = []
    ref_sd = reference[cols].std(ddof=1)
    for c in cols:
        e, a = reference[c], current[c]
        p = psi(e, a, n_bins=n_bins)
        k = ks_statistic(e, a)
        sd = ref_sd.get(c, np.nan)
        shift = (
            (float(a.mean()) - float(e.mean())) / sd
            if sd and np.isfinite(sd) and sd > 0
            else np.nan
        )
        rows.append(
            {
                "feature": c,
                "psi": p,
                "status": drift_status(p),
                "ks": k,
                "mean_ref": float(e.mean()),
                "mean_cur": float(a.mean()),
                "mean_shift_sd": shift,
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("psi", ascending=False)


def rolling_feature_psi(
    X: pd.DataFrame, window: int = 252, n_bins: int = 10, step: int = 63
) -> pd.DataFrame:
    """Walk-forward PSI of every feature vs its trailing reference window.

    For each evaluation date t (every ``step`` days), compares the
    feature distribution over the last ``window`` days (reference =
    [t-2w, t-w)) with the following ``window`` (current = [t-w, t]).
    Returns a date-indexed frame with one column per feature; values are
    PSI.  Fully causal — usable to *backtest the retraining trigger*.
    """
    dates = X.index.get_level_values(0)
    unique = dates.unique().sort_values()
    if len(unique) < 2 * window:
        raise ValueError(f"need >= {2 * window} distinct dates for rolling PSI, got {len(unique)}")
    out = {}
    for i in range(window, len(unique) - window + 1, step):
        t = unique[i]
        ref_dates = unique[i - window : i]
        cur_dates = unique[i : i + window]
        ref = X[X.index.get_level_values(0).isin(ref_dates)]
        cur = X[X.index.get_level_values(0).isin(cur_dates)]
        row = {c: psi(ref[c], cur[c], n_bins=n_bins) for c in X.columns}
        out[t] = row
    return pd.DataFrame(out).T.sort_index()


@dataclass(frozen=True)
class MonitoringReport:
    """Bundle returned by :func:`monitoring_report`."""

    model: str
    n_days: int
    ic_mean: float
    ic_tstat: float
    last_63d_ic: float
    staleness_days: int
    alerts: list[DriftAlert] = field(default_factory=list)
    regimes: pd.DataFrame | None = None
    rolling: pd.DataFrame | None = None

    def as_dict(self) -> dict:
        d = {
            "model": self.model,
            "n_days": self.n_days,
            "ic_mean": round(self.ic_mean, 4),
            "ic_tstat": round(self.ic_tstat, 2),
            "last_63d_ic": round(self.last_63d_ic, 4),
            "staleness_days": self.staleness_days,
            "n_alerts": len(self.alerts),
            "alerts": [
                {"date": str(a.date.date()), "direction": a.direction, "cusum": round(a.cusum, 2)}
                for a in self.alerts
            ],
        }
        if self.regimes is not None:
            d["regimes"] = [
                {
                    "start": str(r["start"].date()),
                    "end": str(r["end"].date()),
                    "n_days": r["n_days"],
                    "ic_mean": round(r["ic_mean"], 4),
                    "ic_tstat": None if np.isnan(r["ic_tstat"]) else round(r["ic_tstat"], 2),
                    "ic_positive_rate": round(r["ic_positive_rate"], 3),
                }
                for _, r in self.regimes.iterrows()
            ]
        return d


def monitoring_report(
    ic: pd.Series, model: str = "model", as_of: pd.Timestamp | None = None
) -> MonitoringReport:
    """Full monitoring bundle for one model's OOS IC series.

    ``as_of`` truncates the series first (for backtesting the monitor
    itself); ``staleness_days`` counts trading days since the last
    observation relative to ``as_of`` (or the series' own last index).
    """
    ic = ic.dropna()
    if as_of is not None:
        ic = ic.loc[:as_of]
    if len(ic) < 30:
        raise ValueError(f"need >= 30 OOS days for monitoring, got {len(ic)}")

    se = ic.std(ddof=1) / np.sqrt(len(ic))
    chart = cusum_drift(ic)
    events = chart["alarm"] & ~chart["alarm"].shift(1, fill_value=False)
    alerts = []
    for date, row in chart[events].iterrows():
        direction = "skill_loss" if row["cusum_neg"] < -row["cusum_pos"] else "skill_gain"
        alerts.append(
            DriftAlert(
                date=pd.Timestamp(date),
                direction=direction,
                cusum=float(row["cusum_neg"] if direction == "skill_loss" else row["cusum_pos"]),
            )
        )

    reference = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(ic.index[-1])
    staleness = int((reference - pd.Timestamp(ic.index[-1])).days)

    return MonitoringReport(
        model=model,
        n_days=int(len(ic)),
        ic_mean=float(ic.mean()),
        ic_tstat=float(ic.mean() / se) if se > 0 else float("nan"),
        last_63d_ic=float(ic.iloc[-63:].mean()),
        staleness_days=staleness,
        alerts=alerts,
        regimes=regime_report(ic),
        rolling=rolling_ic(ic),
    )
