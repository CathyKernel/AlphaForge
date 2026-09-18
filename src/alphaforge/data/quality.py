"""Data-quality validation for price panels.

Free data is messy data.  Before any research is run, AlphaForge executes a
battery of integrity checks and produces a structured report:

* duplicate (date, ticker) rows
* non-positive or missing prices
* OHLC cross-field violations (high < low, close outside [low, high])
* stale prices (long runs of identical closes -- halted or illiquid names)
* extreme daily moves (|return| > 30%) -- flagged for inspection, not removed
* coverage gaps (missing trading days per ticker vs. the panel calendar)
* zero-volume days

Checks are *reporting* tools: they never silently mutate the panel.  The
strategy layer applies its own liquidity / price filters at backtest time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alphaforge.data.panel import PricePanel


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # 'error' | 'warning' | 'info'
    n_flagged: int = 0
    detail: pd.DataFrame | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "n_flagged": int(self.n_flagged),
            "message": self.message,
        }


@dataclass
class QualityReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.checks])

    def to_frame(self) -> pd.DataFrame:
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_pass = sum(c.passed for c in self.checks)
        return f"QualityReport({n_pass}/{len(self.checks)} checks passed)"


class DataQualityChecker:
    """Run all quality checks against a :class:`PricePanel`."""

    def __init__(
        self,
        stale_days: int = 10,
        extreme_ret: float = 0.30,
        max_missing_pct: float = 0.05,
        ohlc_tolerance: float = 0.0005,
    ) -> None:
        self.stale_days = stale_days
        self.extreme_ret = extreme_ret
        self.max_missing_pct = max_missing_pct
        self.ohlc_tolerance = ohlc_tolerance

    # ------------------------------------------------------------------ #
    def run(self, panel: PricePanel) -> QualityReport:
        checks = [
            self.check_duplicates(panel),
            self.check_non_positive(panel),
            self.check_ohlc_consistency(panel),
            self.check_stale_prices(panel),
            self.check_extreme_returns(panel),
            self.check_coverage_gaps(panel),
            self.check_zero_volume(panel),
        ]
        return QualityReport(checks=checks)

    # ------------------------------------------------------------------ #
    def check_duplicates(self, panel: PricePanel) -> CheckResult:
        long = panel.to_long()
        dup = long.duplicated(subset=["date", "ticker"], keep=False)
        n = int(dup.sum())
        return CheckResult(
            "duplicates",
            n == 0,
            "error",
            n,
            detail=long.loc[dup, ["date", "ticker"]].head(50),
            message=f"{n} duplicated (date, ticker) rows",
        )

    def check_non_positive(self, panel: PricePanel) -> CheckResult:
        bad = {}
        for f in ("open", "high", "low", "close"):
            df = panel._fields[f]
            mask = (df <= 0) & df.notna()
            n = int(mask.values.sum())
            if n:
                bad[f] = df.columns[mask.any()].tolist()[:20]
        n = sum(
            int((panel._fields[f] <= 0).values.sum())
            for f in ("open", "high", "low", "close")
            if f in panel._fields
        )
        return CheckResult("non_positive_prices", n == 0, "error", n, message=str(bad or "none"))

    def check_ohlc_consistency(self, panel: PricePanel) -> CheckResult:
        """close within [low, high] and high >= low, up to a small tolerance.

        The tolerance absorbs a known Yahoo Finance artifact: adjusted
        OHLC series apply slightly different rounding to each field, so
        close/high can disagree by <0.05%.  Gross violations still fail.
        """
        hi, lo = panel.high, panel.low
        valid = hi.notna() & lo.notna()
        tol = self.ohlc_tolerance
        bad_hl = (hi < lo * (1 - tol)) & valid
        bad_close = ((panel.close > hi * (1 + tol)) | (panel.close < lo * (1 - tol))) & valid
        n = int(bad_hl.values.sum() + bad_close.values.sum())
        return CheckResult(
            "ohlc_consistency",
            n == 0,
            "error",
            n,
            message=f"{n} rows violate high>=low or low<=close<=high (tolerance {tol:.2%})",
        )

    def check_stale_prices(self, panel: PricePanel) -> CheckResult:
        """Flag tickers with long runs of identical closes (halts / bad data)."""
        stale_tickers: list[str] = []
        close = panel.close
        unchanged = close.eq(close.shift())
        # run length per ticker
        for t in close.columns:
            run = unchanged[t] * (unchanged[t].groupby(unchanged[t].cumsum()).cumcount() + 1)
            if int(run.max() or 0) >= self.stale_days:
                stale_tickers.append(t)
        return CheckResult(
            "stale_prices",
            len(stale_tickers) == 0,
            "warning",
            len(stale_tickers),
            message=f"tickers with >= {self.stale_days} "
            f"consecutive unchanged closes: {stale_tickers[:10]}",
        )

    def check_extreme_returns(self, panel: PricePanel) -> CheckResult:
        rets = panel.returns
        mask = rets.abs() > self.extreme_ret
        n = int(mask.values.sum())
        detail = rets[mask].stack(future_stack=True).rename("ret").reset_index() if n else None
        return CheckResult(
            "extreme_returns",
            n == 0,
            "info",
            n,
            detail=detail,
            message=f"{n} daily returns beyond "
            f"+/-{self.extreme_ret:.0%} (earnings / corporate actions; "
            "informational only)",
        )

    def check_coverage_gaps(self, panel: PricePanel) -> CheckResult:
        """Per-ticker missing days vs. the union calendar of the panel."""
        close = panel.close
        notna = close.notna()
        missing_pct = 1.0 - notna.mean()
        bad = missing_pct[missing_pct > self.max_missing_pct]
        n = int(len(bad))
        detail = (
            pd.DataFrame({"missing_pct": bad, "n_days_missing": (~notna[bad.index]).sum()})
            if n
            else None
        )
        return CheckResult(
            "coverage_gaps",
            n == 0,
            "warning",
            n,
            detail=detail,
            message=f"{n} tickers have > {self.max_missing_pct:.0%} "
            f"missing days on the panel calendar",
        )

    def check_zero_volume(self, panel: PricePanel) -> CheckResult:
        vol, close = panel.volume, panel.close
        mask = (vol == 0) | vol.isna()
        traded = close.notna()
        n = int((mask & traded).values.sum())
        return CheckResult(
            "zero_volume_days", n == 0, "info", n, message=f"{n} days with a price but no volume"
        )
