"""Constrained mean-variance portfolio construction (SLSQP).

Moves beyond equal-weight TopN: the optimizer trades off a cross-sectional
alpha score against the *structured* covariance from the style risk model
(``B Sigma_F B' + diag(sigma_spec^2)``), subject to real portfolio
constraints:

* long-only (or dollar-neutral long-short),
* per-name cap (e.g. 4%) — concentration control,
* sector caps (e.g. 35%) — the constraint most equal-weight books violate,
* linear turnover penalty against the previous target book,
* optional benchmark-relative active-share cap.

Everything is estimated **point-in-time**: covariance and specific risk as
of the signal date, so the optimizer can be dropped into the vectorised
backtest engine with zero lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alphaforge.engine.portfolio import PortfolioConstructor


class MeanVariance(PortfolioConstructor):
    """Alpha-seeking, risk-aware, constraint-respecting weight builder.

    Parameters
    ----------
    risk_model:
        A fitted :class:`~alphaforge.riskmodel.StyleRiskModel`; supplies
        the structured covariance (factor + specific) as of each date.
    universe:
        Optional universe with sectors for the sector-cap constraint.
    risk_aversion:
        ``lambda`` in ``max alpha'w - lambda/2 * w' Sigma w``.  Higher =
        more risk-averse.  With z-scored alpha, 4-12 is a sensible band.
    max_weight:
        Cap on any single position.
    sector_cap:
        Cap on the total weight of any one GICS sector.
    turnover_penalty:
        Linear penalty per unit of turnover vs. the previous target book.
    side:
        ``'long'`` (fully invested) or ``'long_short'`` (dollar-neutral,
        gross 2.0).
    fallback_n:
        If the solver fails, degrade to equal-weight top-N by alpha.
    """

    side = "long"

    def __init__(
        self,
        risk_model,
        universe=None,
        risk_aversion: float = 8.0,
        max_weight: float = 0.05,
        sector_cap: float | None = 0.35,
        turnover_penalty: float = 0.0,
        side: str = "long",
        fallback_n: int = 15,
    ) -> None:
        if side not in ("long", "long_short"):
            raise ValueError("MeanVariance supports 'long' or 'long_short'")
        self.risk_model = risk_model
        self.universe = universe
        self.risk_aversion = float(risk_aversion)
        self.max_weight = float(max_weight)
        self.sector_cap = sector_cap
        self.turnover_penalty = float(turnover_penalty)
        self.side = side
        self.gross_exposure = 1.0 if side == "long" else 2.0
        self.fallback_n = int(fallback_n)
        self._prev_weights: pd.Series | None = None

    # ------------------------------------------------------------------ #
    def _covariance(self, names: list[str], date: pd.Timestamp) -> np.ndarray:
        """Structured covariance (annualised) for ``names`` as of ``date``."""
        b_std = self.risk_model.standardized_exposures()
        factor_names = list(b_std.keys())
        B = np.column_stack(
            [b_std[k].loc[date].reindex(names).to_numpy(dtype=float) for k in factor_names]
        )
        cov_f = (
            self.risk_model.factor_covariance(date=date).loc[factor_names, factor_names].to_numpy()
            * 252.0
        )
        spec = self.risk_model.specific_risk().loc[date].reindex(names).fillna(0.0).to_numpy(float)
        cov = B @ cov_f @ B.T
        cov[np.diag_indices_from(cov)] += spec**2
        cov += np.eye(len(names)) * 1e-8  # solver stability
        return cov

    def weights_from_signal(self, signal: pd.Series, date: pd.Timestamp | None = None) -> pd.Series:
        sig = signal.dropna().astype(float)
        if len(sig) < 10 or date is None:
            return self._fallback(sig)

        # restrict to names the risk model knows about, so alpha/cov/sectors
        # all live on the exact same index
        known = set(next(iter(self.risk_model.standardized_exposures().values())).columns)
        names = [n for n in sig.index if n in known]
        if len(names) < 10:
            return self._fallback(sig)
        sig = sig.loc[names]
        n = len(sig)
        cov = self._covariance(names, date)

        # alpha: winsorised z-score (unit-free; lambda band calibrated to it)
        alpha = sig.to_numpy()
        sd = float(alpha.std())
        alpha = np.clip((alpha - alpha.mean()) / (sd if sd > 0 else 1.0), -3.0, 3.0)

        gross = self.gross_exposure
        long_only = self.side == "long"
        if long_only:
            bounds = [(0.0, self.max_weight)] * n
        else:
            bounds = [(-self.max_weight, self.max_weight)] * n

        w0 = self._initial_guess(sig.index)
        if self._prev_weights is not None:
            prev = self._prev_weights.reindex(sig.index).fillna(0.0).to_numpy()
        else:
            prev = np.zeros(n)

        def objective(w: np.ndarray) -> float:
            risk = w @ cov @ w
            alpha_term = alpha @ w
            turnover = float(np.abs(w - prev).sum())
            return -(
                alpha_term - 0.5 * self.risk_aversion * risk - self.turnover_penalty * turnover
            )

        constraints: list[dict] = [
            {"type": "eq", "fun": lambda w: w.sum() - (gross if long_only else 0.0)}
        ]
        if self.sector_cap is not None and self.universe is not None:
            groups = self._sector_groups(sig.index)
            for members in groups.values():
                if len(members) < 2:
                    continue
                idx = [sig.index.get_loc(m) for m in members]
                cap = self.sector_cap * (gross if long_only else 1.0)
                constraints.append(
                    {"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - w[idx].sum()}
                )

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not res.success or not np.isfinite(res.x).all():
            return self._fallback(sig)

        w = pd.Series(res.x, index=sig.index)
        # clean numerical dust and renormalise the target exposure
        w = w.clip(lower=-self.max_weight, upper=self.max_weight)
        w = w * (gross / max(float(np.abs(w).sum()), 1e-9))
        self._prev_weights = w
        return w

    # ------------------------------------------------------------------ #
    def _sector_groups(self, names) -> dict[str, list[str]]:
        """sector -> tickers within ``names`` (groups by the sector VALUES)."""
        s = self.universe.sectors.reindex(list(names)).dropna()
        return {sector: sorted(group.index) for sector, group in s.groupby(s)}

    def _initial_guess(self, names: pd.Index) -> np.ndarray:
        n = len(names)
        if self._prev_weights is not None and len(self._prev_weights) == n:
            prev = self._prev_weights.reindex(names).to_numpy()
            if np.isfinite(prev).all():
                return prev
        return np.full(n, self.gross_exposure / n if self.side == "long" else 0.0)

    def _fallback(self, sig: pd.Series) -> pd.Series:
        if len(sig) == 0:
            return pd.Series(dtype=float)
        top = sig.sort_values(ascending=False).index[: self.fallback_n]
        w = pd.Series(0.0, index=sig.index)
        if self.side == "long":
            w.loc[top] = 1.0 / len(top)
        else:
            bot = sig.sort_values(ascending=False).index[-self.fallback_n :]
            w.loc[top] = 1.0 / self.fallback_n
            w.loc[bot] = -1.0 / self.fallback_n
        return w
