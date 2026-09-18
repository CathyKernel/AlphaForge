"""Central configuration for AlphaForge.

Resolution order for the data directory:

1. ``ALPHAFORGE_DATA_DIR`` environment variable (if set).
2. The repository checkout that contains this package (``<repo>/data``),
   detected by walking up from the package file location.  This makes an
   editable install ``pip install -e .`` work out of the box and lets the
   bundled sample dataset be discovered automatically.
3. A per-user directory ``~/.alphaforge`` (created on demand) for regular
   ``pip install`` usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TRADING_DAYS_PER_YEAR = 252
DEFAULT_BPS_PER_SIDE = 10.0  # commission + slippage + spread, per side
DEFAULT_COST_BPS_TEARSHEET = 10.0


def _find_repo_data_dir() -> Path | None:
    """Walk up from this file looking for a repo root that has ``data/``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").exists():
            return parent / "data"
    return None


def resolve_data_dir() -> Path:
    """Return the directory holding datasets, with env-var override support."""
    env = os.environ.get("ALPHAFORGE_DATA_DIR")
    if env:
        path = Path(env).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    repo_dir = _find_repo_data_dir()
    if repo_dir is not None:
        return repo_dir
    user_dir = Path.home() / ".alphaforge"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration.

    Attributes
    ----------
    data_dir:
        Root directory for the local data repository (Parquet caches,
        universe snapshots, downloaded history).
    results_dir:
        Directory where backtest reports, tearsheets and model artefacts
        are written.
    trading_days_per_year:
        Annualisation factor (252 for US equities).
    bps_per_side:
        Default one-way transaction cost in basis points.  10 bps per side
        is a deliberately conservative assumption for large-cap US equities
        (commission + half-spread + market impact).
    """

    data_dir: Path = field(default_factory=resolve_data_dir)
    results_dir: Path = field(default_factory=lambda: resolve_data_dir().parent / "results")
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    bps_per_side: float = DEFAULT_BPS_PER_SIDE

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "universe").mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def universe_dir(self) -> Path:
        return self.data_dir / "universe"


_DEFAULT: Config | None = None


def get_config() -> Config:
    """Return the process-wide default :class:`Config` (lazily constructed)."""
    global _DEFAULT  # noqa: PLW0603
    if _DEFAULT is None:
        _DEFAULT = Config()
    return _DEFAULT
