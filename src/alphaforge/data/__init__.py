"""Data layer: downloading, caching, universe management and quality checks."""

from alphaforge.data.cache import DataRepository, Manifest
from alphaforge.data.downloader import DownloadReport, PriceDownloader
from alphaforge.data.panel import PricePanel
from alphaforge.data.pit import PITStats, PointInTimeUniverse
from alphaforge.data.quality import DataQualityChecker, QualityReport
from alphaforge.data.universe import SP100_TICKERS, Universe

__all__ = [
    "PricePanel",
    "PriceDownloader",
    "DownloadReport",
    "DataRepository",
    "Manifest",
    "DataQualityChecker",
    "QualityReport",
    "Universe",
    "SP100_TICKERS",
    "PointInTimeUniverse",
    "PITStats",
]
