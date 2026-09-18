"""Ex-post factor attribution: which styles drove a strategy's returns?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttributionResult:
    """OLS attribution of a return stream to factor returns."""

    alpha_ann: float
    betas: dict[str, float]
    r_squared: float
    factor_pnl: dict[str, float]  # cumulative contribution (sum beta_k * f_k,t)
    residual_pnl: float
    n_obs: int


def attribute_returns(
    returns: pd.Series, factor_returns: pd.DataFrame, add_intercept: bool = True
) -> AttributionResult:
    """Regress ``returns`` on ``factor_returns`` (time-series OLS).

    The intercept is the *unexplained* mean return (annualised).  Factor
    P&L is the cumulative ``beta_k * f_k`` — the part of the cumulative
    strategy return explained by each style.  The residual P&L plus
    factor P&L plus alpha reconstitutes the total cumulative return.
    """
    df = pd.concat([returns.rename("y"), factor_returns], axis=1, join="inner").dropna()
    if len(df) < factor_returns.shape[1] + 5:
        raise ValueError("not enough overlapping observations for attribution")

    y = df["y"].to_numpy()
    X = factor_returns.loc[df.index].to_numpy()
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0

    names = list(factor_returns.columns)
    if add_intercept:
        intercept, coefs = float(beta[0]), beta[1:]
    else:
        intercept, coefs = 0.0, beta

    return AttributionResult(
        alpha_ann=intercept * 252.0,
        betas={n: float(c) for n, c in zip(names, coefs, strict=True)},
        r_squared=r2,
        factor_pnl={n: float((c * df[n]).sum()) for n, c in zip(names, coefs, strict=True)},
        residual_pnl=float(resid.sum()),
        n_obs=int(len(df)),
    )
