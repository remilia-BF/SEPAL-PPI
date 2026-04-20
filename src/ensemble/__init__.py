"""
Ensemble Learning Module

Provides ensemble learning functionality for protein-protein interaction prediction.
"""

from .ensemble_engine import EnsembleInferenceEngine
from .ensemble_predict_engine import EnsemblePredictEngine

__all__ = ['EnsembleInferenceEngine', 'EnsemblePredictEngine'] 