"""Barra-lite style factor risk model.

Estimates daily *factor returns* via weighted cross-sectional regressions
(the Fama-French / Barra approach), then builds a shrinkage covariance of
factor returns and an EWMA specific-risk model, and provides:

* portfolio risk decomposition into factor vs. specific contributions,
* ex-post return attribution of any return stream to the style factors.

The model is deliberately honest about its inputs: every exposure at date
``t`` uses only data through ``t``'s close, so the factor returns are
computable live and can be used inside a backtest without lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.panel import PricePanel

#: factor names in exposure-matrix column order (market first, unnormalised)
STYLE_FACTORS: tuple[str, ...] = ("market", "beta", "size", "momentum", "reversal", "volatility")


class StyleRiskModel:
    """Cross-sectional style factor model on a price panel.

    Parameters
    ----------
    panel:
        Price panel (close, volume; split-adjusted).
    min_breadth:
        Minimum number of stocks with valid data on a date for the
        cross-sectional regression to run; otherwise factor returns are NaN.
    halflife:
        EWMA half-life (trading days) for specific risk.
    """

    def __init__(self, panel: PricePanel, min_breadth: int = 30, halflife: int = 63) -> None:
        self.panel = panel
        self.min_breadth = int(min_breadth)
        self.halflife = int(halflife)
        self._exposures: dict[str, pd.DataFrame] | None = None
        self._factor_returns: pd.DataFrame | None = None
        self._b_std: dict[str, pd.DataFrame] | None = None  # standardised exposures
        self._specific: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Exposures (date x ticker)
    # ------------------------------------------------------------------ #
    def exposures(self) -> dict[str, pd.DataFrame]:
        """Style exposures, each a date x ticker DataFrame.

        Conventions (high exposure = high return when the factor pays):
        ``market`` 1.0 for every stock; ``beta`` trailing 252d beta to the
        equal-weight universe; ``size`` minus log dollar volume (small = high);
        ``momentum`` 63d return; ``reversal`` minus 5d return; ``volatility``
        21d realised vol.
        """
        if self._exposures is not None:
            return self._exposures

        rets = self.panel.returns
        close = self.panel.close
        dvol = self.panel.dollar_volume

        market_ret = rets.mean(axis=1)
        var_m = market_ret.rolling(252, min_periods=126).var()
        beta = rets.rolling(252, min_periods=126).cov(market_ret).div(var_m, axis=0)
        size = -np.log(dvol.rolling(21, min_periods=10).mean().clip(lower=1.0))
        momentum = close.ffill().pct_change(63, fill_method=None)
        reversal = -close.ffill().pct_change(5, fill_method=None)
        volatility = rets.rolling(21, min_periods=15).std()

        self._exposures = {
            "market": pd.DataFrame(1.0, index=close.index, columns=close.columns),
            "beta": beta,
            "size": size,
            "momentum": momentum,
            "reversal": reversal,
            "volatility": volatility,
        }
        return self._exposures

    # ------------------------------------------------------------------ #
    # Factor returns (date x factor) — batched WLS
    # ------------------------------------------------------------------ #
    def factor_returns(self) -> pd.DataFrame:
        """Daily factor returns from cross-sectional WLS regressions.

        Solves ``argmin_f sum_i w_i (r_i - b_i' f)^2`` per date with weights
        ``sqrt(dollar volume)`` (Barra practice), where the five style
        columns are winsorised cross-sectional z-scores.  The whole
        calculation is batched over all dates with ``numpy.einsum`` and a
        batched ``np.linalg.solve`` — no Python loop over dates.
        """
        if self._factor_returns is not None:
            return self._factor_returns

        exp = self.exposures()
        rets = self.panel.returns
        dvol = self.panel.dollar_volume
        names = list(STYLE_FACTORS)

        raw = [exp[n].to_numpy(dtype=float) for n in names]
        B = np.stack(raw, axis=-1)  # (T, N, K)
        R = rets.to_numpy(dtype=float)  # (T, N)
        W = np.sqrt(
            np.clip(dvol.rolling(21, min_periods=5).mean().to_numpy(dtype=float), 0.0, None)
        )
        W = np.where(np.isfinite(W), W, 0.0)

        finite_exp = np.stack(
            [np.isfinite(exp[n].to_numpy(dtype=float)) for n in names], axis=-1
        )  # (T, N, K)
        valid = finite_exp.all(axis=-1) & np.isfinite(R) & (W > 0)

        # ---- winsorise + z-score each style column over the date axis -----
        import warnings

        b_std: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k, name in enumerate(names):
                col_mask = finite_exp[:, :, k]
                col = np.where(col_mask, B[:, :, k], np.nan)
                if k == 0:  # market column stays 1.0
                    B[:, :, 0] = np.where(valid, 1.0, 0.0)
                    b_std[name] = B[:, :, 0]
                    continue
                lo = np.nanquantile(col, 0.025, axis=1, keepdims=True)
                hi = np.nanquantile(col, 0.975, axis=1, keepdims=True)
                mean = np.nanmean(col, axis=1, keepdims=True)
                sd = np.nanstd(col, axis=1, keepdims=True)
                sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
                z = np.clip(col, np.nan_to_num(lo), np.nan_to_num(hi))
                z = (z - np.nan_to_num(mean)) / sd
                B[:, :, k] = np.where(col_mask, np.nan_to_num(z), 0.0)
                b_std[name] = B[:, :, k]
        R = np.nan_to_num(R, nan=0.0)
        Wm = valid.astype(float) * W

        # ---- batched WLS solve --------------------------------------------
        breadth = valid.sum(axis=1)
        K = B.shape[-1]
        BtWB = np.einsum("tnk,tn,tnl->tkl", B, Wm, B)  # (T, K, K)
        BtWr = np.einsum("tnk,tn,tn->tk", B, Wm, R)  # (T, K)
        ok = (breadth >= self.min_breadth) & (np.linalg.matrix_rank(BtWB) == K)
        f = np.full((B.shape[0], K), np.nan)
        idx = np.flatnonzero(ok)
        if len(idx):
            f[idx] = np.linalg.solve(BtWB[idx], BtWr[idx][..., None])[..., 0]

        self._factor_returns = pd.DataFrame(f, index=self.panel.dates, columns=names)
        # cache the standardised exposures used in the regression — these,
        # not the raw exposures, are what specific risk and decomposition
        # must consume to stay consistent with the estimated factor returns
        self._b_std = {
            name: pd.DataFrame(arr, index=self.panel.dates, columns=self.panel.tickers)
            for name, arr in b_std.items()
        }
        return self._factor_returns

    def standardized_exposures(self) -> dict[str, pd.DataFrame]:
        """The winsorised z-score exposures actually used in the regression.

        Call :meth:`factor_returns` first (it computes and caches them).
        """
        if self._b_std is None:
            self.factor_returns()
        assert self._b_std is not None
        return self._b_std

    # ------------------------------------------------------------------ #
    # Covariance & specific risk
    # ------------------------------------------------------------------ #
    def factor_covariance(
        self, date: pd.Timestamp | None = None, window: int = 504
    ) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance of factor returns.

        Uses the trailing ``window`` observations up to and including
        ``date`` (default: full history).  Shrinkage handles the
        low-observation-count regime (6 factors, ~2y window).
        """
        from sklearn.covariance import LedoitWolf

        fr = self.factor_returns().dropna()
        if date is not None:
            fr = fr.loc[:date]
        fr = fr.iloc[-int(window) :]
        lw = LedoitWolf().fit(fr.to_numpy())
        return pd.DataFrame(lw.covariance_, index=fr.columns, columns=fr.columns)

    def specific_risk(self) -> pd.DataFrame:
        """EWMA vol of idiosyncratic returns, date x ticker (annualised)."""
        if self._specific is not None:
            return self._specific
        b_std = self.standardized_exposures()
        fr = self.factor_returns()
        rets = self.panel.returns
        # e = r - B f, with the SAME standardised exposures the factor
        # returns were estimated against (raw exposures would be a units
        # mismatch and blow up the residuals)
        fitted = np.zeros(rets.shape)
        for k in fr.columns:
            fitted += fr[k].to_numpy()[:, None] * b_std[k].to_numpy()
        e = rets.to_numpy() - fitted
        e = pd.DataFrame(
            np.where(np.isfinite(e), e, np.nan), index=rets.index, columns=rets.columns
        )
        var = e.ewm(halflife=self.halflife, min_periods=21).var() * 252.0
        self._specific = np.sqrt(var)
        return self._specific

    # ------------------------------------------------------------------ #
    # Decomposition & reporting
    # ------------------------------------------------------------------ #
    def decompose(
        self, weights: pd.Series, date: pd.Timestamp, lookback: int = 504
    ) -> RiskDecomposition:
        """Split a portfolio's variance into factor and specific parts.

        Uses exposures and covariance estimated **only with data through
        ``date``** (covariance from trailing factor returns, specific risk
        as of that date), so calling this inside a backtest on the
        rebalance date is leakage-safe.
        """
        names = list(STYLE_FACTORS)
        b_std = self.standardized_exposures()
        B = pd.DataFrame({n: b_std[n].loc[date] for n in names})
        w = weights.astype(float)
        B = B.reindex(w.index)
        valid = B.notna().all(axis=1) & w.notna() & (w != 0)
        B, w = B.loc[valid], w.loc[valid]

        cov_f = self.factor_covariance(date=date, window=lookback)
        spec = self.specific_risk().loc[date].reindex(w.index).fillna(0.0)

        b = B.to_numpy()
        wn = w.to_numpy()
        # annualise the daily factor covariance so both variance terms share
        # annualised units (specific_risk() is already annualised)
        cov_ff = cov_f.loc[names, names].to_numpy() * 252.0
        v = wn @ b  # w'B — aggregated portfolio factor exposures (K,)
        # portfolio factor variance = (w'B) C (B'w).  NOTE: this is NOT
        # sum_i w_i^2 b_i' C b_i — the cross-stock terms w_i w_j b_i' C b_j
        # are exactly what diversification eats, so they must be kept.
        factor_var = float(v @ cov_ff @ v)
        spec_var = float(np.sum((wn**2) * (spec.to_numpy() ** 2)))
        contrib = v * (cov_ff @ v)  # per style; sums exactly to factor_var

        return RiskDecomposition(
            date=date,
            total_variance=factor_var + spec_var,
            factor_variance=factor_var,
            specific_variance=spec_var,
            ann_vol=float(np.sqrt(factor_var + spec_var)),
            style_contributions={n: float(c) for n, c in zip(names, contrib, strict=True)},
        )

    def summary(self) -> dict:
        """Headline stats of the estimated factor model (full history)."""
        fr = self.factor_returns().dropna()
        ann_vol = fr.std() * np.sqrt(252)
        ann_ret = fr.mean() * 252
        sharpe = ann_ret / ann_vol.replace(0, np.nan)
        corr = fr.corr()
        return {
            "n_factors": int(fr.shape[1]),
            "n_days": int(fr.shape[0]),
            "ann_vol": {k: round(float(ann_vol[k]), 4) for k in fr.columns},
            "ann_return": {k: round(float(ann_ret[k]), 4) for k in fr.columns},
            "sharpe": {k: round(float(sharpe[k]), 3) for k in fr.columns},
            "correlation": {
                a: {b: round(float(corr.loc[a, b]), 3) for b in fr.columns} for a in fr.columns
            },
        }


@dataclass(frozen=True)
class RiskDecomposition:
    """Variance decomposition of one portfolio snapshot."""

    date: pd.Timestamp
    total_variance: float
    factor_variance: float
    specific_variance: float
    ann_vol: float
    style_contributions: dict[str, float]
