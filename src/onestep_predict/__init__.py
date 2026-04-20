"""
One-Step Prediction Module for SEPAL-PPI

This module provides a streamlined prediction pipeline that:
1. Extracts protein sequences from PDB files
2. Generates ESM embeddings (CIS + pooled) on-the-fly
3. Runs ensemble inference with meta-learner
4. Provides attention weight analysis and optional IG attribution

Usage:
    python sepal-ppi.predict.py --pdb-dir /path/to/pdbs --interaction-list /path/to/pairs.csv
"""

from .data_preparation import DataPreparator
from .embedding_generator import (
    EmbeddingGenerator, 
    EmbeddingResult,
    InputLayerProjector,
    AttentionPoolingUnit
)
from .ig_attribution import (
    IGAttributionAnalyzer,
    IGResult,
    IntegratedGradientsCalculator,
    PDBBFactorWriter
)
from .predict_engine import (
    OnestepPredictEngine,
    OnestepPredictConfig,
    PredictionResult
)
from .ig_singele import SingleProteinLMDBExporter
from .result_saver import (
    ResultSaver,
    format_evaluation_metrics_report
)

__all__ = [
    # Data Preparation
    'DataPreparator',
    
    # Embedding Generation
    'EmbeddingGenerator',
    'EmbeddingResult',
    'InputLayerProjector',
    'AttentionPoolingUnit',
    
    # IG Attribution
    'IGAttributionAnalyzer',
    'IGResult',
    'IntegratedGradientsCalculator',
    'PDBBFactorWriter',
    
    # Prediction Engine
    'OnestepPredictEngine',
    'OnestepPredictConfig',
    'PredictionResult',
    'SingleProteinLMDBExporter',
    
    # Result Saving
    'ResultSaver',
    'format_evaluation_metrics_report',
]
