"""Machine learning: feature panels, leakage-safe CV, LightGBM / LSTM models."""

from alphaforge.ml.features import FeatureConfig, FeaturePanelBuilder
from alphaforge.ml.lstm import LSTMAlphaModel
from alphaforge.ml.models import AlphaModel, EnsembleAlphaModel, LightGBMAlphaModel
from alphaforge.ml.pipeline import AlphaTrainingPipeline, MLPipelineResult
from alphaforge.ml.validation import Fold, PurgedWalkForwardCV

__all__ = [
    "FeatureConfig",
    "FeaturePanelBuilder",
    "PurgedWalkForwardCV",
    "Fold",
    "AlphaModel",
    "LightGBMAlphaModel",
    "LSTMAlphaModel",
    "EnsembleAlphaModel",
    "AlphaTrainingPipeline",
    "MLPipelineResult",
]


def lstm_available() -> bool:
    """True when PyTorch is importable (the optional ``[dl]`` extra)."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
