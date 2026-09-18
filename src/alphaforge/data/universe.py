"""Universe management: S&P 500 membership with GICS sector classification.

Three data paths are supported (tried in order):

1. **Live refresh — GitHub `datasets` corpus** (default with network): the
   `datasets/s-and-p-500-companies` repository publishes the current
   constituents table (503 tickers = 500 companies + 3 dual-class listings:
   GOOGL/GOOG, FOXA/FOX, NWSA/NWS) as a versioned CSV.  Structured, stable
   and machine-readable.
2. **Live refresh — Wikipedia scrape** (fallback): the constituents table
   from the "List of S&P 500 companies" page.
3. **Bundled snapshot** (offline): a static CSV shipped with the repository,
   frozen at a known date.  Reproducible but subject to membership drift.

Survivorship caveat: the point-in-time membership history is not available
from free sources.  AlphaForge documents this openly -- see the README
section "Methodology & limitations".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_GITHUB_DATASETS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

#: Tickers of the S&P 100 subset (mega-cap liquid names) used for the
#: bundled demo dataset.  Ordered alphabetically.
SP100_TICKERS: tuple[str, ...] = tuple(
    sorted(
        {
            "AAPL",
            "ABBV",
            "ABT",
            "ACN",
            "ADBE",
            "AIG",
            "AMD",
            "AMGN",
            "AMT",
            "AMZN",
            "AVGO",
            "AXP",
            "BA",
            "BAC",
            "BNY",
            "BKNG",
            "BLK",
            "BMY",
            "BRK-B",
            "C",
            "CAT",
            "CHTR",
            "CL",
            "CMCSA",
            "COF",
            "COP",
            "COST",
            "CRM",
            "CSCO",
            "CVS",
            "CVX",
            "DE",
            "DHR",
            "DIS",
            "DUK",
            "EMR",
            "ETN",
            "F",
            "FDX",
            "GD",
            "GE",
            "GILD",
            "GM",
            "GOOG",
            "GOOGL",
            "GS",
            "HD",
            "HON",
            "IBM",
            "INTC",
            "INTU",
            "ISRG",
            "JNJ",
            "JPM",
            "KHC",
            "KO",
            "LIN",
            "LLY",
            "LMT",
            "LOW",
            "MA",
            "MCD",
            "MDLZ",
            "MDT",
            "MET",
            "META",
            "MMM",
            "MO",
            "MRK",
            "MS",
            "MSFT",
            "NEE",
            "NFLX",
            "NKE",
            "NVDA",
            "ORCL",
            "PEP",
            "PFE",
            "PG",
            "PLTR",
            "PM",
            "PYPL",
            "QCOM",
            "RTX",
            "SBUX",
            "SCHW",
            "SO",
            "SPG",
            "T",
            "TGT",
            "TMO",
            "TMUS",
            "TSLA",
            "TXN",
            "UNH",
            "UNP",
            "UPS",
            "USB",
            "V",
            "VZ",
            "WFC",
            "WMT",
            "XOM",
        }
    )
)


@dataclass(frozen=True)
class Universe:
    """An immutable stock universe with sector classification."""

    tickers: tuple[str, ...]
    sectors: pd.Series  # ticker -> GICS sector
    snapshot_date: str = "bundled snapshot"
    source: str = "bundled"
    additions: pd.Series | None = None  # ticker -> date added to the index
    sub_industries: pd.Series | None = None  # ticker -> GICS sub-industry

    def __len__(self) -> int:
        return len(self.tickers)

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._ticker_set

    @property
    def _ticker_set(self) -> frozenset[str]:
        return frozenset(self.tickers)

    def sector_counts(self) -> pd.Series:
        return self.sectors.value_counts()

    def by_sector(self) -> dict[str, list[str]]:
        df = self.sectors.reset_index()
        df.columns = ["ticker", "sector"]
        return {k: sorted(v) for k, v in df.groupby("sector")["ticker"]}

    def sector_dummies(self) -> pd.DataFrame:
        """Ticker x sector one-hot matrix (used for sector neutralisation)."""
        return pd.get_dummies(self.sectors).astype(float)

    def restrict(self, tickers: list[str]) -> Universe:
        keep = [t for t in tickers if t in self._ticker_set]
        return Universe(
            tuple(keep),
            self.sectors.reindex(keep),
            self.snapshot_date,
            self.source,
            self.additions.reindex(keep) if self.additions is not None else None,
            self.sub_industries.reindex(keep) if self.sub_industries is not None else None,
        )

    @classmethod
    def from_csv(cls, path: str | Path, source: str = "csv") -> Universe:
        df = pd.read_csv(path)
        df["ticker"] = df["ticker"].astype(str).str.strip()
        # Yahoo uses dashes for share classes (BRK-B), Wikipedia uses dots.
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df = df.dropna(subset=["ticker", "sector"]).drop_duplicates("ticker")
        snap = str(df.get("snapshot_date", ["bundled snapshot"]).iloc[0])
        additions = sub_ind = None
        if "date_added" in df.columns:
            additions = pd.to_datetime(df.set_index("ticker")["date_added"], errors="coerce")
        if "sub_industry" in df.columns:
            sub_ind = df.set_index("ticker")["sub_industry"]
        return cls(
            tuple(df["ticker"]),
            df.set_index("ticker")["sector"],
            snap,
            source,
            additions,
            sub_ind,
        )

    @classmethod
    def sp500(cls, refresh: bool = False, cache_dir: str | Path | None = None) -> Universe:
        """Load the S&P 500 universe.

        Parameters
        ----------
        refresh:
            When True, scrape the live table from Wikipedia and cache it.
            When False, use the freshest locally cached / bundled snapshot.
        cache_dir:
            Directory used to cache refreshed snapshots.  Defaults to
            ``<data_dir>/universe``.
        """
        import logging

        from alphaforge.config import get_config

        log = logging.getLogger(__name__)
        cache_dir = Path(cache_dir) if cache_dir else get_config().universe_dir
        cache_path = cache_dir / "sp500_live.csv"

        if refresh:
            try:
                live = _fetch_wikipedia_sp500()
                live["snapshot_date"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
                cache_dir.mkdir(parents=True, exist_ok=True)
                live.to_csv(cache_path, index=False)
                log.info("Refreshed S&P 500 snapshot: %d tickers", len(live))
            except Exception as exc:  # noqa: BLE001
                log.warning("Live refresh failed (%s); falling back to snapshot.", exc)

        if cache_path.exists():
            return cls.from_csv(cache_path, source="live-cache")
        bundled = _bundled_snapshot_path()
        if bundled.exists():
            return cls.from_csv(bundled, source="bundled")
        raise FileNotFoundError(
            "No S&P 500 snapshot found. Run `python -m alphaforge.cli universe --refresh` "
            "with network access to create one."
        )

    @classmethod
    def sp100(cls) -> Universe:
        """The S&P 100 mega-cap subset with sectors from the S&P 500 snapshot."""
        sp = cls.sp500()
        return sp.restrict(list(SP100_TICKERS))


def _bundled_snapshot_path() -> Path:
    from alphaforge.config import get_config

    return get_config().universe_dir / "sp500_snapshot.csv"


def _fetch_wikipedia_sp500() -> pd.DataFrame:
    """Fetch current S&P 500 constituents (network required).

    Primary source: the versioned CSV published by the ``datasets``
    organisation on GitHub (503 tickers, includes sub-industries and the
    date each security was added to the index).  Fallback: the Wikipedia
    constituents table.  Raises if both fail.
    """
    import io

    import requests

    headers = {"User-Agent": "AlphaForge/1.0 (quant research project)"}
    try:
        resp = requests.get(_GITHUB_DATASETS_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        keep = {
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Date added": "date_added",
        }
        cols = {c: keep[c] for c in df.columns if c in keep}
        out = df[list(cols)].rename(columns=cols)
        if len(out) < 480:  # sanity: full index has ~503 listings
            raise ValueError(f"implausible constituent count: {len(out)}")
        return out
    except Exception as github_exc:  # noqa: BLE001
        # fall through to the Wikipedia scrape
        resp = requests.get(_WIKI_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            raise ValueError("No tables found on the Wikipedia page.") from github_exc
        df = tables[0]
        keep = {"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
        cols = {c: keep[c] for c in df.columns if c in keep}
        df = df[list(cols)].rename(columns=cols)
        df["sub_industry"] = df["GICS Sub-Industry"] if "GICS Sub-Industry" in df.columns else ""
        return df
