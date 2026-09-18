"""Equal-risk-contribution (risk parity) weights via cyclical coordinate descent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.engine.portfolio import PortfolioConstructor


def erc_weights(cov: np.ndarray, max_weight: float | None = None, iters: int = 100) -> np.ndarray:
    """Weights such that every asset contributes equally to portfolio variance.

    Cyclical coordinate descent on the convex reformulation (Spinu 2013,
    Griveau-Billion et al. 2013): minimise ``0.5 y' C y - (1/n) sum log y_i``.
    The coordinate-wise optimum has a closed form:
    ``y_i = (-b + sqrt(b^2 + 4 c_ii / n)) / (2 c_ii)`` with
    ``b = (C y)_i - c_ii y_i``.  Converges reliably, O(n^2) per sweep.
    """
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    cdiag = np.diag(cov).copy()
    cdiag[cdiag <= 0] = 1e-12
    y = 1.0 / np.sqrt(cdiag)
    for _ in range(iters):
        y_old = y.copy()
        for i in range(n):
            b_grad = float(cov[i] @ y - cov[i, i] * y[i])
            y[i] = (-b_grad + np.sqrt(b_grad * b_grad + 4.0 * cov[i, i] / n)) / (
                2.0 * max(cov[i, i], 1e-12)
            )
        if np.max(np.abs(y - y_old) / np.maximum(y, 1e-12)) < 1e-10:
            break
    w = y / y.sum()
    if max_weight is not None:
        # iteratively redistribute excess weight above the cap
        for _ in range(50):
            over = w > max_weight
            if not over.any():
                break
            excess = float((w[over] - max_weight).sum())
            w[over] = max_weight
            free = ~over
            if w[free].sum() > 0:
                w[free] += excess * w[free] / w[free].sum()
    return w


class EqualRiskContribution(PortfolioConstructor):
    """Risk-parity book over the whole signal universe (long-only).

    Ignores the alpha signal's *magnitudes* (uses only its support) — the
    point of comparison is: what does a pure risk-balanced portfolio do
    versus an alpha-seeking one?
    """

    side = "long"
    gross_exposure = 1.0

    def __init__(self, risk_model, max_weight: float | None = 0.05) -> None:
        self.risk_model = risk_model
        self.max_weight = max_weight
        self.weighting = "erc"

    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        sig = signal.dropna().astype(float)
        if len(sig) < 3 or date is None:
            return pd.Series(dtype=float)
        b_std = self.risk_model.standardized_exposures()
        known = set(next(iter(b_std.values())).columns)
        names = [n for n in sig.index if n in known]
        if len(names) < 3:
            return pd.Series(dtype=float)
        # diagonal-risk approximation of the structured covariance is not
        # good enough for risk parity: use the full structured matrix
        factor_names = list(b_std.keys())
        B = np.column_stack(
            [b_std[k].loc[date].reindex(names).to_numpy(dtype=float) for k in factor_names]
        )
        cov_f = (
            self.risk_model.factor_covariance(date=date).loc[factor_names, factor_names].to_numpy()
            * 252.0
        )
        spec = self.risk_model.specific_risk().loc[date].reindex(names).fillna(0.0).to_numpy(float)
        cov = B @ cov_f @ B.T + np.diag(spec**2) + np.eye(len(names)) * 1e-8
        w = erc_weights(cov, max_weight=self.max_weight)
        return pd.Series(w, index=names)
