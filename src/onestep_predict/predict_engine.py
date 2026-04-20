"""
One-Step Prediction Engine

Main engine integrating:
1. Data preparation (PDB parsing, validation)
2. Embedding generation (ESM + adapters)
3. Model inference (ensemble meta-learner)
4. Optional IG attribution analysis

Provides simplified one-step prediction from PDB files to results.
"""

import os
import sys
import json
import logging
import pickle
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.inference_engine import InferenceEngine

# Sklearn imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, 
        accuracy_score, f1_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Local imports
from .data_preparation import DataPreparator
from .embedding_generator import EmbeddingGenerator, EmbeddingResult
from .ig_singele import SingleProteinLMDBExporter


def _safe_torch_load(path: str, map_location: torch.device):
    """Safe checkpoint loading with proper warnings handling."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)
        return torch.load(path, map_location=map_location)


class MLPClassifier(nn.Module):
    """
    MLP classifier for protein-protein interaction prediction.
    Matches the structure from src/models/classifier.py
    
    NOTE: Uses GELU activation to match the training configuration.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False,
                 activation: str = 'gelu'):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 128, 16]  # Default architecture
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Select activation function
        if activation.lower() == 'gelu':
            act_fn = nn.GELU
        elif activation.lower() == 'relu':
            act_fn = nn.ReLU
        else:
            act_fn = nn.GELU  # Default to GELU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_dim == 1:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    @staticmethod
    def from_state_dict(state_dict: Dict[str, torch.Tensor], device: torch.device) -> 'MLPClassifier':
        """
        Create MLPClassifier from state dict, inferring architecture from weights.
        """
        # Extract classifier weights (keys like 'classifier.network.0.weight')
        classifier_state = {}
        for k, v in state_dict.items():
            if k.startswith('classifier.'):
                new_key = k[len('classifier.'):]
                classifier_state[new_key] = v
            elif k.startswith('network.'):
                classifier_state[k] = v
        
        if not classifier_state:
            # Try direct keys
            classifier_state = state_dict
        
        # Infer architecture from weight shapes
        # network.0.weight: [hidden_0, input_dim]
        # network.2.weight: [hidden_1, hidden_0]
        # ...
        linear_weights = [(k, v) for k, v in classifier_state.items() 
                         if 'weight' in k and 'bias' not in k]
        linear_weights.sort(key=lambda x: int(x[0].split('.')[1]) if x[0].split('.')[1].isdigit() else 0)
        
        if not linear_weights:
            raise ValueError("No linear weights found in state dict")
        
        # First layer: input_dim
        first_weight = linear_weights[0][1]
        input_dim = first_weight.shape[1]
        
        # Hidden dims: output of each layer except last
        hidden_dims = []
        for i, (k, v) in enumerate(linear_weights[:-1]):
            hidden_dims.append(v.shape[0])
        
        # Last layer: output_dim
        output_dim = linear_weights[-1][1].shape[0]
        
        # Create classifier
        classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Load weights
        classifier.load_state_dict(classifier_state, strict=False)
        classifier.to(device)
        classifier.eval()
        
        return classifier


@dataclass
class PredictionResult:
    """Container for ensemble prediction results."""
    protein1: str
    protein2: str
    ensemble_probability: float
    ensemble_prediction: int
    model_probabilities: Dict[str, float]
    model_weights: Dict[str, float]


@dataclass 
class OnestepPredictConfig:
    """Configuration for one-step prediction."""
    # Input paths
    pdb_dir: Optional[str]
    ensemble_config: str
    fasta_path: Optional[str] = None
    ensemble_config_explicit: bool = False
    legacy_predict_config: Optional[str] = None
    interaction_list: Optional[str] = None
    genome_mode: bool = False
    genome_max_length: int = 1022
    input_layer_ckpt: Optional[str] = None  # Override for input layer
    model_pretrain_ckpt: Optional[str] = None  # Override for model pretrain
    
    # ESM configuration
    esm_model: str = "esm2_15b"
    esm_precision: str = "bf16"
    force_hf_esm: bool = False
    
    # Multifeature
    multifeature_dir: Optional[str] = None
    
    # Output configuration
    output_dir: Optional[str] = None
    dataset_name: str = "onestep"
    
    # Processing options
    prebuild_embeddings: bool = False
    emb_batch: int = 16
    predict_batch: int = 128
    enable_ig: bool = False
    ig_mode: str = "single"
    single_lmdb_export_dir: Optional[str] = None
    ig_baseline_path: Optional[str] = None
    strict_mode: bool = False  # Enable strict validation mode
    concise_logging: bool = False
    output_log_file: Optional[str] = None
    
    # Optional true labels for evaluation
    has_true_labels: bool = False


class OnestepPredictEngine:
    """
    One-step prediction engine for protein-protein interaction.
    
    Pipeline:
    1. Parse PDB files and extract sequences
    2. Validate protein pairs
    3. Generate ESM embeddings (CIS + Pooled)
    4. Run ensemble inference with meta-learner
    5. Optional: Compute IG attribution analysis
    6. Save results to standardized output format
    """
    
    def __init__(
        self,
        config: OnestepPredictConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize one-step prediction engine.
        
        Args:
            config: Prediction configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set up output directory
        if config.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"sepal-ppi-outputdata/{config.dataset_name}_{timestamp}")
        else:
            self.output_dir = Path(config.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy loading)
        self.data_preparator: Optional[DataPreparator] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.ig_analyzer = None  # Lazy import
        self.single_lmdb_outputs: Dict[str, str] = {}
        
        # Meta-learner components
        self.meta_learner = None
        self.scaler = None
        self.model_names: List[str] = []
        self.training_mode: str = "logits"
        
        # Ensemble configuration
        self.ensemble_config: Dict[str, Any] = {}
        self.sequence_engines: Dict[str, InferenceEngine] = {}
        self.sequence_input_precisions: Dict[str, str] = {}
        
        # Data caches
        self.protein_sequences: Dict[str, str] = {}
        self.protein_pairs: List[Tuple[str, str]] = []
        self.true_labels: Optional[List[int]] = None
        self.protein_ids: List[str] = []
        self.total_pairs_evaluated: Optional[int] = None
        
        # Model checkpoints from ensemble config
        self.cis_model_config: Dict[str, Any] = {}
        self.pooled_model_config: Dict[str, Any] = {}
        self.pooled_classifier_ckpt: Optional[str] = None
        self.classifier_source_info: Dict[str, Dict[str, Any]] = {}
        self.embedding_component_sources: Dict[str, Dict[str, Any]] = {}
        self._ensemble_config_path: Optional[Path] = None
        self._ensemble_config_dir: Optional[Path] = None
        self._needs_multifeature: bool = False
        self._online_multifeature_mode: bool = False
        
        self.logger.info(f"One-step prediction engine initialized, output: {self.output_dir}")

    def _emit_concise_step(self, message: str) -> None:
        """Emit real-time step message in concise mode."""
        if self.config.concise_logging:
            print(message, flush=True)

    def _emit_concise_line(self, message: str) -> None:
        """Emit a concise status line in real time."""
        if self.config.concise_logging:
            print(message, flush=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete one-step prediction pipeline.
        
        Returns:
            Dict with prediction results, evaluation metrics (if available), and output paths
        """
        try:
            run_started_at = datetime.now().isoformat()

            if self.config.legacy_predict_config:
                self.logger.info("Legacy LMDB mode enabled: reusing legacy ensemble_predict data pipeline")
                legacy_results = self._run_legacy_lmdb_mode()
                legacy_results['summary_stats'] = {
                    'interaction_pairs': int(len(legacy_results.get('prediction_results', []))),
                    'pairs_inferred': int(len(legacy_results.get('prediction_results', []))),
                    'multifeature_dir': self.config.multifeature_dir,
                    'model_components_loaded': None,
                    'log_file': self.config.output_log_file,
                    'run_started_at': run_started_at,
                    'run_finished_at': datetime.now().isoformat(),
                }
                return legacy_results

            # Step 1: Prepare data
            self._emit_concise_step("Step 1: Preparing data...")
            self.logger.info("Step 1: Preparing data...")
            self._prepare_data()

            labels_positive = None
            labels_negative = None
            if self.true_labels is not None:
                labels_positive = int(sum(1 for v in self.true_labels if int(v) == 1))
                labels_negative = int(sum(1 for v in self.true_labels if int(v) == 0))

            self._emit_concise_line("Data loaded:")
            self._emit_concise_line(f"  - PDB files: {self._count_pdb_files()}")
            self._emit_concise_line(f"  - Unique proteins: {len(self.protein_sequences)}")
            if labels_positive is not None and labels_negative is not None:
                self._emit_concise_line(
                    f"  - Interaction pairs: {len(self.protein_pairs)} "
                    f"(labels: {labels_positive} positive, {labels_negative} negative)"
                )
            else:
                self._emit_concise_line(f"  - Interaction pairs: {len(self.protein_pairs)}")
            self._emit_concise_line("-" * 70)
            
            # Step 2: Load ensemble configuration and meta-learner
            self._emit_concise_step("Step 2: Loading ensemble components...")
            self.logger.info("Step 2: Loading ensemble components...")
            self._load_ensemble_components()
            model_components_loaded = int(len([k for k, v in self.classifiers.items() if v is not None]))
            self._emit_concise_line(
                f"Model loading: ensemble components loaded ({model_components_loaded} classifiers) ✓"
            )
            for model_name in self.model_names:
                source_info = self.classifier_source_info.get(model_name, {})
                if source_info.get('status') == 'loaded':
                    model_kind = 'cis' if source_info.get('is_cis') else 'residue'
                    self._emit_concise_line(
                        f"  - {model_name} ({model_kind}): {source_info.get('path')} [source={source_info.get('source')}]"
                    )
            if self.embedding_component_sources:
                emb_src = self.embedding_component_sources
                self._emit_concise_line(
                    f"  - residue input_layer: {emb_src.get('residue_input_layer', {}).get('path')}"
                )
                self._emit_concise_line(
                    f"  - residue complete_model: {emb_src.get('residue_complete_model', {}).get('path')}"
                )
                self._emit_concise_line(
                    f"  - cis input_layer: {emb_src.get('cis_input_layer', {}).get('path')}"
                )
            if self.config.multifeature_dir:
                self._emit_concise_line(
                    f"Multimodal features loaded from {self.config.multifeature_dir}"
                )
            elif self._online_multifeature_mode:
                self._emit_concise_line(
                    "Multimodal features: online generation from ESM residue embeddings"
                )
            else:
                self._emit_concise_line("Multimodal features: not enabled")
            self._emit_concise_line("-" * 70)
            
            # Step 3: Generate embeddings
            # Step 3/4 are emitted right before their corresponding tqdm in inference.
            self.logger.info("Step 3: Generating embeddings...")
            self._generate_embeddings()
            
            # Step 4: Run ensemble inference
            self.logger.info("Step 4: Running ensemble inference...")
            prediction_results = self._run_ensemble_inference()
            inferred_now = self.total_pairs_evaluated
            if inferred_now is None:
                inferred_now = len(self.protein_pairs) if self.protein_pairs else len(prediction_results)
            self._emit_concise_line(f"{int(inferred_now)} pairs inferred ✓")
            self._emit_concise_line("-" * 70)
            
            # Step 5: Compute evaluation metrics (if true labels available)
            evaluation_metrics = None
            if self.config.has_true_labels and self.true_labels is not None:
                self.logger.info("Step 5: Computing evaluation metrics...")
                evaluation_metrics = self._compute_evaluation_metrics(prediction_results)
                if evaluation_metrics and ('auroc' in evaluation_metrics) and ('aupr' in evaluation_metrics):
                    self._emit_concise_line(
                        f"Evaluation: AUROC = {evaluation_metrics['auroc']:.4f}, "
                        f"AUPR = {evaluation_metrics['aupr']:.4f}"
                    )
                    self._emit_concise_line("-" * 70)
            else:
                self._emit_concise_line("Evaluation: N/A (no labels)")
                self._emit_concise_line("-" * 70)
            
            # Step 6: IG attribution analysis (optional)
            ig_results = None
            if self.config.enable_ig:
                mode = (self.config.ig_mode or "single").strip().lower()
                if mode == "single":
                    self.logger.info("Step 6: Exporting single-protein LMDB representations...")
                    ig_results = self._export_single_mode_lmdb()
                else:
                    self.logger.info("Step 6: Running IG attribution analysis...")
                    ig_results = self._run_ig_analysis()
            
            # Step 7: Save results
            self._emit_concise_step("Step 5: Saving results...")
            self.logger.info("Step 7: Saving results...")
            output_paths = self._save_results(prediction_results, evaluation_metrics, ig_results)

            pairs_inferred = self.total_pairs_evaluated
            if pairs_inferred is None:
                pairs_inferred = len(self.protein_pairs) if self.protein_pairs else len(prediction_results)

            pdb_files_count = None
            if self.config.pdb_dir:
                try:
                    pdb_files_count = self._count_pdb_files()
                except Exception:
                    pdb_files_count = None

            summary_stats = {
                'pdb_files_count': pdb_files_count,
                'unique_proteins': int(len(self.protein_sequences)),
                'interaction_pairs': int(len(self.protein_pairs)),
                'labels_positive': labels_positive,
                'labels_negative': labels_negative,
                'embeddings_generated': int(len(self.protein_sequences)),
                'pairs_inferred': int(pairs_inferred),
                'multifeature_dir': self.config.multifeature_dir,
                'model_components_loaded': model_components_loaded,
                'model_classifier_sources': self.classifier_source_info,
                'single_lmdb_outputs': self.single_lmdb_outputs,
                'log_file': self.config.output_log_file,
                'run_started_at': run_started_at,
                'run_finished_at': datetime.now().isoformat(),
            }
            
            return {
                'prediction_results': prediction_results,
                'evaluation_metrics': evaluation_metrics,
                'ig_results': ig_results,
                'output_paths': output_paths,
                'output_dir': str(self.output_dir),
                'multifeature_used': self.config.multifeature_dir is not None,
                'summary_stats': summary_stats,
            }
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def _count_pdb_files(self) -> int:
        """Count PDB files in configured directory."""
        if not self.config.pdb_dir:
            return 0
        pdb_dir = Path(self.config.pdb_dir)
        return len(list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.PDB")))

    def _run_legacy_lmdb_mode(self) -> Dict[str, Any]:
        """Run inference by fully reusing legacy ensemble_predict data entry (FASTA + LMDB)."""
        from src.ensemble.ensemble_predict_engine import EnsemblePredictEngine
        from src.utils.helpers import set_random_seed

        with open(self.config.legacy_predict_config, 'r', encoding='utf-8') as f:
            legacy_cfg = yaml.safe_load(f)

        # Match sepal-ppi.py --mode ensemble_predict behavior for reproducibility.
        seed = legacy_cfg.get('seed')
        if seed is None:
            seed = legacy_cfg.get('training', {}).get('seed')
        if seed is None:
            seed = 42
        legacy_cfg['seed'] = seed
        set_random_seed(seed)
        self.logger.info(f"Legacy LMDB mode random seed set to: {seed}")

        # Allow one-step CLI to override feature root while keeping legacy data entry intact.
        if self.config.multifeature_dir:
            legacy_cfg['predict_prot_feature_folder'] = self.config.multifeature_dir

        # Keep one-step output dir behavior.
        legacy_engine = EnsemblePredictEngine(
            legacy_cfg,
            output_dir=str(self.output_dir),
            logger=self.logger,
        )
        legacy_results = legacy_engine.run_ensemble_predict()

        prediction_results = legacy_results['prediction_results']['ensemble_predictions']
        evaluation_metrics = legacy_results.get('evaluation_metrics')
        interpretability_results = legacy_results.get('interpretability_results')

        output_paths = {
            'predictions': str(self.output_dir / "ensemble_predictions.csv"),
        }
        attention_file = self.output_dir / "attention_weights.jsonl"
        if attention_file.exists():
            output_paths['attention'] = str(attention_file)

        summary_path = self.output_dir / "prediction_summary.json"
        summary = {
            'total_pairs': int(len(prediction_results)),
            'mode': 'legacy_lmdb_reuse',
            'legacy_predict_config': self.config.legacy_predict_config,
            'ensemble_config': self.config.ensemble_config,
            'timestamp': datetime.now().isoformat(),
        }
        if evaluation_metrics is not None:
            summary['evaluation_metrics'] = evaluation_metrics
        if interpretability_results is not None:
            summary['interpretability_results'] = interpretability_results

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        output_paths['summary'] = str(summary_path)

        return {
            'prediction_results': prediction_results,
            'evaluation_metrics': evaluation_metrics,
            'ig_results': None,
            'output_paths': output_paths,
            'output_dir': str(self.output_dir),
            'multifeature_used': self.config.multifeature_dir is not None,
        }
    
    def _prepare_data(self) -> None:
        """Prepare data from PDB files and interaction list."""
        # Initialize data preparator
        self.data_preparator = DataPreparator(
            pdb_dir=self.config.pdb_dir,
            fasta_path=self.config.fasta_path,
            interaction_list=self.config.interaction_list,
            output_dir=str(self.output_dir),
            genome_mode=self.config.genome_mode,
            max_sequence_length=self.config.genome_max_length,
            logger=self.logger
        )
        
        # Parse and validate data
        data = self.data_preparator.prepare()
        
        # Get validated data
        self.protein_sequences = data['protein_sequences']
        self.protein_pairs = data['protein_pairs']
        self.protein_ids = data.get('protein_ids', [])
        
        # Check for true labels
        if data.get('has_true_labels', False) or data.get('has_labels', False):
            self.true_labels = data.get('true_labels')
            self.config.has_true_labels = True
            self.logger.info(f"Detected {len(self.true_labels)} true labels")
        
        self.logger.info(
            f"Data prepared: {len(self.protein_sequences)} proteins, "
            f"{len(self.protein_pairs)} pairs"
        )
        
        # Check for multifeature - FASTA mode always enables online generation when
        # --multifeature-dir is not provided, so feature_concat models never fail
        # due to pre-check false negatives.
        self._needs_multifeature = self._check_ensemble_needs_multifeature()
        self._online_multifeature_mode = False

        if self.config.multifeature_dir is None:
            if self.config.fasta_path:
                self._online_multifeature_mode = True
                if self._needs_multifeature:
                    self.logger.info(
                        "Multifeature required: FASTA mode will generate RSA/Q3 online from each protein's ESM residue embedding"
                    )
                else:
                    self.logger.info(
                        "FASTA mode enabled without --multifeature-dir: online multifeature fallback is active if any model path needs feature_concat"
                    )
            elif self._needs_multifeature:
                raise RuntimeError(
                    "Model requires multifeature; in PDB mode please provide --multifeature-dir or precompute features first"
                )
            else:
                self.logger.info("Multifeature not required for this ensemble configuration")
    
    def _check_ensemble_needs_multifeature(self) -> bool:
        """
        Check if any ensemble model requires multifeature by:
        1. Checking if ensemble config has complete_model_path (pooled model needs multifeature)
        2. Checking if model config has preprocessing with multifeature
        """
        try:
            with open(self.config.ensemble_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            models_config = config.get('ensemble', {}).get('models', [])
            
            for model_info in models_config:
                # Check if model has complete_model_path (pooled/residue model)
                # These models typically require multifeature
                if model_info.get('complete_model_path'):
                    return True
                
                # Check if model config has preprocessing with multifeature
                model_config = model_info.get('model_config', {})
                preprocessing = model_config.get('preprocessing', {})
                
                # Check if preprocessing requires features
                if preprocessing.get('preprocessor') == 'feature_concat':
                    return True
                
                # Also check data_processing path
                data_processing = preprocessing.get('data_processing', {})
                if data_processing.get('all_feature_folder'):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not check multifeature requirement: {e}")
            # Default to True to be safe
            return True

    def _resolve_path_from_ensemble(self, raw_path: Optional[str]) -> Optional[str]:
        """Resolve a path, preferring location relative to the provided ensemble config."""
        if not raw_path:
            return None

        p = Path(raw_path)
        if p.is_absolute():
            return str(p)

        if self._ensemble_config_dir is not None:
            cand = (self._ensemble_config_dir / p)
            if cand.exists():
                return str(cand)

        return str(p)

    def _get_input_layer_input_dim(self, ckpt_path: str) -> Optional[int]:
        """Best-effort extraction of input dimension from an input-layer checkpoint."""
        try:
            ckpt = _safe_torch_load(ckpt_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}

            if isinstance(meta, dict) and ('input_dim' in meta):
                return int(meta['input_dim'])

            for key in ('proj1.weight', 'proj.weight'):
                w = state.get(key) if isinstance(state, dict) else None
                if w is not None and hasattr(w, 'shape') and len(w.shape) == 2:
                    return int(w.shape[1])
        except Exception:
            return None

        return None

    def _is_input_layer_compatible(self, ckpt_path: str, expected_input_dim: int) -> bool:
        """Check whether checkpoint input dim matches ESM residue embedding dim."""
        input_dim = self._get_input_layer_input_dim(ckpt_path)
        # If dimension cannot be inferred, keep backward compatibility and accept it.
        if input_dim is None:
            return True
        return input_dim == expected_input_dim
    
    def _load_ensemble_components(self) -> None:
        """Load ensemble configuration and meta-learner."""
        self._ensemble_config_path = Path(self.config.ensemble_config).resolve()
        self._ensemble_config_dir = self._ensemble_config_path.parent

        # Load ensemble config
        with open(self.config.ensemble_config, 'r', encoding='utf-8') as f:
            self.ensemble_config = yaml.safe_load(f)
        
        # Load meta-learner
        meta_learner_path = self._resolve_path_from_ensemble(
            self.ensemble_config['ensemble']['meta_learner_path']
        )
        self.logger.debug(f"Loading meta-learner: {meta_learner_path}")
        
        with open(meta_learner_path, 'rb') as f:
            meta_learner_data = pickle.load(f)
        
        self.meta_learner = meta_learner_data['meta_learner']
        self.scaler = meta_learner_data['scaler']
        self.model_names = meta_learner_data['model_names']
        self.training_mode = meta_learner_data.get('training_mode', 'logits')
        
        self.logger.debug(f"Meta-learner loaded, models: {self.model_names}")
        self.logger.debug(f"Training mode: {self.training_mode}")
        
        # Extract model configurations and load classifiers
        models_config = self.ensemble_config['ensemble']['models']
        
        # Initialize model containers
        self.classifiers: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.model_weights: Dict[str, float] = {}  # Store weights from config
        self.classifier_source_info = {}
        
        for model_info in models_config:
            model_name = model_info['model_name']
            data_config = model_info.get('data_config', {})
            model_config = model_info.get('model_config', {})
            embedding_dim = data_config.get('embedding_dim', 5120)
            interaction_type = model_config.get('interaction_type', 'hadamard')
            weight = model_info.get('weight', 0.5)  # Get weight from config
            has_complete_model = 'complete_model_path' in model_info
            is_pooled = has_complete_model or embedding_dim == 1280
            
            # Store model config and weight
            self.model_configs[model_name] = {
                'embedding_dim': embedding_dim,
                'interaction_type': interaction_type,
                'model_path': model_info.get('model_path'),
                'is_pooled': is_pooled,
            }
            self.model_weights[model_name] = weight
            
            # Distinguish models by embedding dimension:
            # 5120 = CIS (raw ESM embedding)
            # 1280 = pooled (after projection + pooling)
            if is_pooled:
                self.pooled_model_config = model_info
                self.pooled_classifier_ckpt = model_info.get('model_path')
                self.logger.debug(f"Pooled model: {model_name} (dim={embedding_dim}, interaction={interaction_type}, weight={weight})")
                # Use pool-residue classifier for pooled model
                classifier_path, classifier_source = self._resolve_classifier_path(model_name, model_info, is_cis=False)
            else:
                self.cis_model_config = model_info
                self.logger.debug(f"CIS model: {model_name} (dim={embedding_dim}, interaction={interaction_type}, weight={weight})")
                # Use cis classifier for CIS model
                classifier_path, classifier_source = self._resolve_classifier_path(model_name, model_info, is_cis=True)

            self.classifier_source_info[model_name] = {
                'path': classifier_path,
                'source': classifier_source,
                'is_cis': (not is_pooled),
                'status': 'pending',
            }
            
            # Load model component
            if classifier_path and Path(classifier_path).exists():
                # Two-stage non-genome path uses pooled single-protein embeddings + pairwise
                # interaction + classifier head. Always load classifier for scoring.
                self._load_classifier(model_name, classifier_path, embedding_dim, interaction_type)
                if self.classifiers.get(model_name) is not None:
                    self.classifier_source_info[model_name]['status'] = 'loaded'
                    self.logger.info(
                        f"Classifier loaded: model={model_name}, source={classifier_source}, path={classifier_path}"
                    )
                else:
                    self.classifier_source_info[model_name]['status'] = 'load_failed'
                if is_pooled:
                    # Keep compatibility field but do not route pooled model through sequence_engine
                    # in the default path.
                    self.sequence_engines[model_name] = None
            else:
                self.classifier_source_info[model_name]['status'] = 'missing'
                self.logger.warning(f"Classifier not found for {model_name}: {classifier_path}")
        
        # Determine checkpoint paths for embedding generation
        self._resolve_checkpoint_paths()
    
    def _resolve_classifier_path(self, model_name: str, model_info: Dict, is_cis: bool) -> Tuple[Optional[str], str]:
        """
        Resolve classifier path from the selected ensemble config.

        The active ensemble config (explicit or auto-resolved default) is the
        single source of truth for model artifacts.
        """
        config_path = self._resolve_path_from_ensemble(model_info.get('model_path'))

        if not config_path:
            raise FileNotFoundError(
                f"ensemble config is missing model_path for {model_name}"
            )
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"ensemble config model not found for {model_name}: {config_path}"
            )

        self.logger.debug(f"Using classifier from ensemble config: {config_path}")
        return str(config_path), 'ensemble_config'
    
    def _load_classifier(self, model_name: str, model_path: str, 
                        embedding_dim: int, interaction_type: str) -> None:
        """Load classifier from model checkpoint."""
        try:
            self.logger.debug(f"Loading classifier for {model_name} from {model_path}")
            
            ckpt = _safe_torch_load(model_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            
            # Create classifier from state dict
            classifier = MLPClassifier.from_state_dict(state, self.device)
            self.classifiers[model_name] = classifier
            
            self.logger.debug(f"Loaded classifier for {model_name}: input_dim={classifier.input_dim}")
            
        except Exception as e:
            self.logger.warning(f"Could not load classifier for {model_name}: {e}")
            self.classifiers[model_name] = None

    def _load_sequence_engine(self, model_name: str, model_info: Dict[str, Any], model_path: str) -> None:
        """Load full sequence model for pooled-path inference, matching legacy InferenceEngine."""
        try:
            temp_config = {
                'mode': 'inference',
                'model': model_info['model_config'],
                'data': model_info['data_config'],
                'inference': self.ensemble_config.get('inference', {}),
                'output': self.ensemble_config.get('output', {}),
            }

            # Keep one-step feature override behavior consistent with legacy mode.
            if self.config.multifeature_dir and isinstance(temp_config.get('model'), dict):
                model_cfg = temp_config['model'].setdefault('model_config', {})
                preprocessing_cfg = model_cfg.get('preprocessing') or {}
                if not isinstance(preprocessing_cfg, dict):
                    preprocessing_cfg = {}
                data_processing_cfg = preprocessing_cfg.get('data_processing') or {}
                if not isinstance(data_processing_cfg, dict):
                    data_processing_cfg = {}
                data_processing_cfg['all_feature_folder'] = self.config.multifeature_dir
                data_processing_cfg['log_missing_as_error'] = True
                preprocessing_cfg['data_processing'] = data_processing_cfg
                model_cfg['preprocessing'] = preprocessing_cfg
                temp_config['model']['model_config'] = model_cfg

            engine = InferenceEngine(temp_config, self.logger)
            engine.load_model(model_path, device=self.device)
            # Ensure model is in eval mode
            if engine.model:
                engine.model.eval()
            self.sequence_engines[model_name] = engine
            self.sequence_input_precisions[model_name] = model_info.get('data_config', {}).get('target_precision', 'fp32')

            # Also load the pooled classifier head so pair-stage hadamard+MLP can run
            # without falling back to cosine similarity in the GPU tensor path.
            embedding_dim = model_info.get('data_config', {}).get('embedding_dim', 1280)
            interaction_type = model_info.get('model_config', {}).get('interaction_type', 'hadamard')
            self._load_classifier(model_name, model_path, embedding_dim, interaction_type)

            self.logger.debug(f"Loaded full sequence engine for {model_name} from {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load full sequence engine for {model_name}: {e}")
            self.sequence_engines[model_name] = None
    
    def _resolve_checkpoint_paths(self) -> None:
        """
        Resolve checkpoint paths from config or overrides.

        Priority order:
        1. Command line overrides (config.input_layer_ckpt, config.model_pretrain_ckpt)
        2. Paths discoverable from the selected ensemble config hierarchy
        """
        esm_input_dim_map = {
            'esm2_15b': 5120,
            'esm2_3b': 2560,
            'esm2_650m': 1280,
            'esm2_150m': 640,
            'esm1b_650m': 1280,
        }
        expected_input_dim = esm_input_dim_map.get(self.config.esm_model, 5120)

        # ========== Input layer checkpoint (5120 -> 1280 projection) ==========
        if self.config.input_layer_ckpt:
            input_layer_ckpt = self.config.input_layer_ckpt
        else:
            input_layer_ckpt = None
            
            # Priority 2: Look in pooled model config directory (explicit mode prefers this)
            if not input_layer_ckpt and self.pooled_model_config:
                pooled_path = self._resolve_path_from_ensemble(self.pooled_model_config.get('model_path', ''))
                if pooled_path:
                    model_dir = Path(pooled_path).parent
                    config_root = None
                    if self._ensemble_config_dir is not None:
                        # ensemble config is usually at .../<esm_model>/cis_residue_ensemble/
                        # and input_layer for pooled embeddings is under sibling fusion dir.
                        config_root = self._ensemble_config_dir.parent
                    candidates = [
                        (config_root / "sepal-ppi-fusion_finetune_avg_to_1280" / "input_layer.pt") if config_root else None,
                        (config_root / "sepal-ppi-fusion_finetune_avg_to_1280" / "input_layer.pth") if config_root else None,
                        model_dir.parent / "sepal-ppi-fusion_finetune_avg_to_1280" / "input_layer.pt",
                        model_dir.parent / "sepal-ppi-fusion_finetune_avg_to_1280" / "input_layer.pth",
                        model_dir.parent / "input_layer" / "input_layer.pt",
                        model_dir.parent / "input_layer" / "input_layer.pth",
                        model_dir / "input_layer.pt",
                        model_dir / "input_layer.pth",
                        model_dir.parent / "input_layer.pt",
                        model_dir.parent / "input_layer.pth",
                    ]
                    
                    for candidate in candidates:
                        if candidate and candidate.exists() and self._is_input_layer_compatible(str(candidate), expected_input_dim):
                            input_layer_ckpt = str(candidate)
                            self.logger.debug(f"Found input_layer.pth at {candidate}")
                            break
        
        # ========== Model pretrain checkpoint (MLP adapter + pooling + preprocessing) ==========
        if self.config.model_pretrain_ckpt:
            model_pretrain_ckpt = self.config.model_pretrain_ckpt
        else:
            model_pretrain_ckpt = None
            
            # Priority 2: From pooled_model_config
            if not model_pretrain_ckpt and self.pooled_model_config:
                # First check complete_model_path (preferred for pooled embedding generation)
                model_pretrain_ckpt = self._resolve_path_from_ensemble(
                    self.pooled_model_config.get('complete_model_path')
                )
                if not model_pretrain_ckpt:
                    # Try to find complete_model.pth in model_path directory
                    model_path = self._resolve_path_from_ensemble(
                        self.pooled_model_config.get('model_path')
                    )
                    if model_path:
                        complete_path = Path(model_path).parent / "complete_model.pth"
                        if complete_path.exists():
                            model_pretrain_ckpt = str(complete_path)
                        else:
                            # Fallback to model_path (may not have all components)
                            model_pretrain_ckpt = model_path
                            self.logger.warning(
                                f"complete_model.pth not found, using {model_path}. "
                                f"Pooled embeddings may not include all preprocessing layers."
                            )

        
        self.input_layer_ckpt = input_layer_ckpt
        self.model_pretrain_ckpt = model_pretrain_ckpt
        
        # ========== CIS input layer checkpoint (5120 -> 2560 -> 5120 MLP) ==========
        cis_input_layer_ckpt = None

        # Fallback: infer CIS input layer from cis model path in ensemble config.
        if self.cis_model_config:
            cis_model_path = self._resolve_path_from_ensemble(self.cis_model_config.get('model_path'))
            if cis_model_path:
                cis_dir = Path(cis_model_path).parent
                cis_candidates = [
                    cis_dir / "input_layer.pt",
                    cis_dir / "input_layer.pth",
                    cis_dir.parent / "input_layer.pt",
                    cis_dir.parent / "input_layer.pth",
                ]
                for candidate in cis_candidates:
                    if candidate.exists() and self._is_input_layer_compatible(str(candidate), expected_input_dim):
                        cis_input_layer_ckpt = str(candidate)
                        self.logger.debug(f"Found CIS input_layer.pth at {candidate}")
                        break
        
        self.cis_input_layer_ckpt = cis_input_layer_ckpt
        self.embedding_component_sources = {
            'residue_input_layer': {
                'path': input_layer_ckpt,
                'source': 'ensemble_config_hierarchy',
            },
            'residue_complete_model': {
                'path': model_pretrain_ckpt,
                'source': 'ensemble_config_hierarchy',
            },
            'cis_input_layer': {
                'path': cis_input_layer_ckpt,
                'source': 'ensemble_config_hierarchy',
            },
        }
        
        if input_layer_ckpt:
            self.logger.info(f"Input layer checkpoint: {input_layer_ckpt}")
        else:
            self.logger.error(
                "Input layer checkpoint not found!\n"
                "Please specify with --input-layer-ckpt option.\n"
                "Example: --input-layer-ckpt results/Strings_plant50/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth"
            )
            raise FileNotFoundError("Input layer checkpoint required but not found")
        
        if model_pretrain_ckpt:
            self.logger.info(f"Model pretrain checkpoint: {model_pretrain_ckpt}")
        else:
            self.logger.error(
                "Model pretrain checkpoint not found!\n"
                "Please specify with --model-pretrain-ckpt option.\n"
                "Example: --model-pretrain-ckpt results/Strings_plant50/esm2_15b/sepal-ppi-projector-feature-contant/complete_model.pth"
            )
            raise FileNotFoundError("Model pretrain checkpoint required but not found")

        if cis_input_layer_ckpt is None:
            self.logger.error(
                "CIS input layer checkpoint not found!\n"
                "When using --ensemble-config explicitly, files must be discoverable from that config's hierarchy."
            )
            raise FileNotFoundError("CIS input layer checkpoint required but not found")
    
    def _generate_embeddings(self) -> None:
        """Generate ESM embeddings for all proteins."""
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            esm_model_name=self.config.esm_model,
            esm_precision=self.config.esm_precision,
            force_hf_esm=self.config.force_hf_esm,
            input_layer_ckpt=self.input_layer_ckpt,
            model_pretrain_ckpt=self.model_pretrain_ckpt,
            cis_input_layer_ckpt=self.cis_input_layer_ckpt,
            pooled_classifier_ckpt=self.pooled_classifier_ckpt,
            multifeature_dir=self.config.multifeature_dir,
            enable_online_multifeature=self._online_multifeature_mode,
            strict_mode=self.config.strict_mode,
            logger=self.logger
        )
        
        # Load models
        self.embedding_generator.load_models()
        
        # Generate embeddings
        if self.config.prebuild_embeddings:
            # Prebuild all embeddings
            self.logger.info(
                f"Pre-building all embeddings with emb_batch={max(1, int(self.config.emb_batch))}..."
            )
            self.embedding_generator.generate_embeddings_batch(
                self.protein_sequences,
                mode="both",
                show_progress=True,
                batch_size=self.config.emb_batch,
                return_residue_emb=self.config.enable_ig,
                collect_results=False,
            )
        else:
            # Online mode: generate embeddings per pair during inference
            self.logger.debug("Using online embedding generation mode")

    def _prepare_embeddings_for_proteins(
        self,
        protein_ids: List[str],
        show_progress: bool = True,
    ) -> None:
        """Prepare missing single-protein embeddings in emb_batch-sized chunks."""
        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator not initialized")

        missing_sequences: Dict[str, str] = {}
        for protein_id in protein_ids:
            cached_cis_t = self.embedding_generator.get_cached_cis_tensor(protein_id)
            cached_pooled_t = self.embedding_generator.get_cached_pooled_tensor(protein_id)
            if cached_cis_t is not None and cached_pooled_t is not None:
                continue

            if (
                self.embedding_generator.get_cached_cis_embedding(protein_id) is not None
                and self.embedding_generator.get_cached_pooled_embedding(protein_id) is not None
            ):
                continue

            seq = self.protein_sequences.get(protein_id)
            if seq is None:
                self.logger.warning(f"Missing sequence for protein {protein_id}, skipping")
                continue
            missing_sequences[protein_id] = seq

        if not missing_sequences:
            self.logger.info("All required protein embeddings are already cached")
            return

        effective_batch = max(1, int(self.config.emb_batch))
        self.logger.info(
            "Preparing single-protein embeddings in batches: "
            f"count={len(missing_sequences)}, emb_batch={effective_batch}, "
            f"mode={'genome' if self.config.genome_mode else 'standard'}"
        )
        self.embedding_generator.generate_embeddings_batch(
            missing_sequences,
            mode="both",
            show_progress=show_progress,
            batch_size=effective_batch,
            return_residue_emb=self.config.enable_ig,
            collect_results=False,
        )

    def _build_genome_embedding_matrices(
        self,
        protein_ids: List[str]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Build contiguous CIS/pooled embedding matrices for genome-mode inference.

        Returns:
            (valid_protein_ids, cis_matrix [N, D_cis], pooled_matrix [N, D_pool])
        """
        self._prepare_embeddings_for_proteins(protein_ids, show_progress=True)

        valid_ids: List[str] = []
        cis_list: List[np.ndarray] = []
        pooled_list: List[np.ndarray] = []

        for protein_id in protein_ids:
            try:
                cis_t = self.embedding_generator.get_cached_cis_tensor(protein_id)
                pooled_t = self.embedding_generator.get_cached_pooled_tensor(protein_id)
                cis_np = self.embedding_generator.get_cached_cis_embedding(protein_id)
                pooled_np = self.embedding_generator.get_cached_pooled_embedding(protein_id)

                if cis_np is None and cis_t is not None:
                    cis_np = cis_t.detach().float().cpu().numpy().astype(np.float32)
                if pooled_np is None and pooled_t is not None:
                    pooled_np = pooled_t.detach().float().cpu().numpy().astype(np.float32)

                if cis_np is None or pooled_np is None:
                    seq = self.protein_sequences.get(protein_id)
                    if seq is None:
                        self.logger.warning(f"Missing sequence for protein {protein_id}, skipping")
                        continue
                    cis_t, pooled_t = self.embedding_generator.generate_cis_and_pooled_tensors(
                        protein_id,
                        seq,
                        return_attention=True,
                        return_residue_emb=self.config.enable_ig,
                    )
                    cis_np = cis_t.detach().float().cpu().numpy().astype(np.float32)
                    pooled_np = pooled_t.detach().float().cpu().numpy().astype(np.float32)

                valid_ids.append(protein_id)
                cis_list.append(cis_np.astype(np.float32, copy=False))
                pooled_list.append(pooled_np.astype(np.float32, copy=False))
            except Exception as e:
                self.logger.error(f"Error preparing embeddings for {protein_id}: {e}")

        if not valid_ids:
            raise RuntimeError("No valid proteins available for genome-mode inference")

        cis_matrix = np.stack(cis_list, axis=0).astype(np.float32, copy=False)
        pooled_matrix = np.stack(pooled_list, axis=0).astype(np.float32, copy=False)

        return valid_ids, cis_matrix, pooled_matrix

    def _predict_sequence_model_logits_batch(
        self,
        model_name: str,
        residue_pairs: List[Tuple[np.ndarray, np.ndarray]],
        protein_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """Run full sequence model on residue-level embeddings, matching legacy inference math."""
        engine = self.sequence_engines.get(model_name)
        if engine is None or engine.model is None:
            raise RuntimeError(f"Full sequence engine unavailable for {model_name}")

        batch_size = len(residue_pairs)
        max_len1 = max(pair[0].shape[0] for pair in residue_pairs)
        max_len2 = max(pair[1].shape[0] for pair in residue_pairs)
        embedding_dim = residue_pairs[0][0].shape[1]

        # Always use float32: model weights are loaded from FP32 checkpoint and legacy LMDB
        # data is also stored as float32. The data_config target_precision only controls LMDB
        # storage precision, not model inference dtype.
        tensor_dtype = torch.float32

        protein1_seq = torch.zeros(batch_size, max_len1, embedding_dim, device=self.device, dtype=tensor_dtype)
        protein1_mask = torch.zeros(batch_size, max_len1, device=self.device, dtype=torch.bool)
        protein2_seq = torch.zeros(batch_size, max_len2, embedding_dim, device=self.device, dtype=tensor_dtype)
        protein2_mask = torch.zeros(batch_size, max_len2, device=self.device, dtype=torch.bool)

        protein1_ids: List[str] = []
        protein2_ids: List[str] = []

        for idx, ((residue1, residue2), (protein1, protein2)) in enumerate(zip(residue_pairs, protein_pairs)):
            len1 = residue1.shape[0]
            len2 = residue2.shape[0]
            protein1_seq[idx, :len1] = torch.as_tensor(residue1, dtype=tensor_dtype, device=self.device)
            protein1_mask[idx, :len1] = True
            protein2_seq[idx, :len2] = torch.as_tensor(residue2, dtype=tensor_dtype, device=self.device)
            protein2_mask[idx, :len2] = True
            protein1_ids.append(protein1)
            protein2_ids.append(protein2)

        with torch.no_grad():
            predictions = engine.model(
                protein1_seq,
                protein2_seq,
                protein1_mask,
                protein2_mask,
                protein_ids=(protein1_ids, protein2_ids)
            )
            predictions = predictions.reshape(-1)
            predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
            logits = torch.log(predictions / (1.0 - predictions))

        return logits.detach().float().cpu().numpy()

    def _compute_model_predictions_batch_torch(
        self,
        cis_batch: torch.Tensor,
        pooled_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        GPU-native batch prediction path for genome mode.

        Args:
            cis_batch: [batch, 2 * embedding_dim] tensor on self.device
            pooled_batch: [batch, 2 * embedding_dim] tensor on self.device

        Returns:
            Dict mapping model_name to logits tensor [batch] on self.device
        """
        model_logits: Dict[str, torch.Tensor] = {}

        for model_name in self.model_names:
            model_config = self.model_configs.get(model_name, {})
            embedding_dim = model_config.get('embedding_dim', 5120)
            interaction_type = model_config.get('interaction_type', 'hadamard')

            if embedding_dim == 5120:
                emb_dim = cis_batch.shape[1] // 2
                emb1 = cis_batch[:, :emb_dim]
                emb2 = cis_batch[:, emb_dim:]
            else:
                emb_dim = pooled_batch.shape[1] // 2
                emb1 = pooled_batch[:, :emb_dim]
                emb2 = pooled_batch[:, emb_dim:]

            if interaction_type == 'hadamard':
                interaction_emb = emb1 * emb2
            elif interaction_type == 'concatenation':
                interaction_emb = torch.cat([emb1, emb2], dim=1)
            elif interaction_type == 'difference':
                interaction_emb = torch.abs(emb1 - emb2)
            else:
                interaction_emb = emb1 * emb2

            classifier = self.classifiers.get(model_name)
            if classifier is not None:
                with torch.no_grad():
                    output = classifier(interaction_emb.float()).squeeze(-1)
                    prob = torch.clamp(output, 1e-7, 1.0 - 1e-7)
                    logit = torch.log(prob / (1.0 - prob))
            else:
                emb1_norm = torch.linalg.norm(emb1, dim=1)
                emb2_norm = torch.linalg.norm(emb2, dim=1)
                cos_sim = (emb1 * emb2).sum(dim=1) / (emb1_norm * emb2_norm + 1e-8)
                logit = cos_sim * 2.0
                self.logger.warning(f"Using fallback cosine similarity for {model_name}")

            model_logits[model_name] = logit

        return model_logits

    def _ensemble_predict_batch_arrays(
        self,
        model_logits_batch: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch ensemble prediction returning array outputs for fast downstream filtering.

        Returns:
            (model_order, ensemble_proba, ensemble_pred, model_prob_matrix, model_weights)
        """
        model_order = self.model_names
        sample_values = next(iter(model_logits_batch.values()))

        if isinstance(sample_values, torch.Tensor):
            logits_matrix = torch.stack([model_logits_batch[name] for name in model_order], dim=1)
            logits_matrix = logits_matrix.float().detach().cpu().numpy()
        else:
            logits_matrix = np.stack([model_logits_batch[name] for name in model_order], axis=1)

        features_scaled = self.scaler.transform(logits_matrix)
        ensemble_proba = self.meta_learner.predict_proba(features_scaled)[:, 1]
        ensemble_pred = self.meta_learner.predict(features_scaled)
        model_prob_matrix = 1.0 / (1.0 + np.exp(-logits_matrix))
        model_weights = np.asarray(self.meta_learner.coef_[0], dtype=np.float32)

        return model_order, ensemble_proba, ensemble_pred, model_prob_matrix, model_weights
    
    def _run_ensemble_inference(self) -> pd.DataFrame:
        """Run ensemble inference on protein pairs."""
        results = []
        from tqdm import tqdm

        if self.config.genome_mode:
            protein_ids = self.protein_ids or sorted(self.protein_sequences.keys())
            self._emit_concise_step("Step 3: Generating embeddings...")
            valid_ids, cis_all, pooled_all = self._build_genome_embedding_matrices(protein_ids)

            n = len(valid_ids)
            total_pairs = n * (n + 1) // 2
            self.total_pairs_evaluated = total_pairs
            self._emit_concise_step("Step 4: Running ensemble inference...")
            progress = tqdm(total=total_pairs, desc="Running ensemble inference (genome mode)")
            valid_ids_arr = np.asarray(valid_ids, dtype=object)

            pair_batch = max(1, int(self.config.predict_batch))
            protein_tile = max(128, min(4096, int(np.sqrt(pair_batch * 8))))
            gpu_available = self.device.type == 'cuda'

            if gpu_available:
                tensor_dtype = torch.bfloat16 if self.config.esm_precision == 'bf16' else torch.float32
                cis_all_t = torch.as_tensor(cis_all, dtype=tensor_dtype, device=self.device)
                pooled_all_t = torch.as_tensor(pooled_all, dtype=tensor_dtype, device=self.device)
            else:
                cis_all_t = None
                pooled_all_t = None

            self.logger.info(
                f"Genome fast path enabled: proteins={n}, pair_batch={pair_batch}, tile={protein_tile}, gpu={gpu_available}"
            )

            for i_start in range(0, n, protein_tile):
                i_end = min(i_start + protein_tile, n)

                for j_start in range(i_start, n, protein_tile):
                    j_end = min(j_start + protein_tile, n)

                    if i_start == j_start:
                        local_i, local_j = np.triu_indices(i_end - i_start, k=0)
                        idx_i_all = local_i + i_start
                        idx_j_all = local_j + j_start
                    else:
                        i_idx = np.arange(i_start, i_end, dtype=np.int64)
                        j_idx = np.arange(j_start, j_end, dtype=np.int64)
                        idx_i_all = np.repeat(i_idx, len(j_idx))
                        idx_j_all = np.tile(j_idx, len(i_idx))

                    pair_count = idx_i_all.shape[0]

                    for pair_start in range(0, pair_count, pair_batch):
                        pair_end = min(pair_start + pair_batch, pair_count)
                        idx_i = idx_i_all[pair_start:pair_end]
                        idx_j = idx_j_all[pair_start:pair_end]

                        if gpu_available:
                            idx_i_t = torch.as_tensor(idx_i, dtype=torch.long, device=self.device)
                            idx_j_t = torch.as_tensor(idx_j, dtype=torch.long, device=self.device)

                            cis_batch_t = torch.cat([
                                cis_all_t.index_select(0, idx_i_t),
                                cis_all_t.index_select(0, idx_j_t)
                            ], dim=1)
                            pooled_batch_t = torch.cat([
                                pooled_all_t.index_select(0, idx_i_t),
                                pooled_all_t.index_select(0, idx_j_t)
                            ], dim=1)

                            model_logits_batch = self._compute_model_predictions_batch_torch(
                                cis_batch_t,
                                pooled_batch_t
                            )
                        else:
                            cis_batch = np.concatenate([cis_all[idx_i], cis_all[idx_j]], axis=1)
                            pooled_batch = np.concatenate([pooled_all[idx_i], pooled_all[idx_j]], axis=1)
                            model_logits_batch = self._compute_model_predictions_batch(
                                cis_batch,
                                pooled_batch
                            )

                        (model_order,
                         ensemble_proba,
                         ensemble_pred,
                         model_prob_matrix,
                         model_weights) = self._ensemble_predict_batch_arrays(model_logits_batch)

                        positive_idx = np.where(ensemble_pred == 1)[0]
                        for row_idx in positive_idx:
                            protein1 = valid_ids_arr[idx_i[row_idx]]
                            protein2 = valid_ids_arr[idx_j[row_idx]]

                            rounded_probs = {
                                f"{name}_probability": round(float(model_prob_matrix[row_idx, col_idx]), 4)
                                for col_idx, name in enumerate(model_order)
                            }

                            results.append({
                                'protein1': protein1,
                                'protein2': protein2,
                                'ensemble_probability': round(float(ensemble_proba[row_idx]), 4),
                                'ensemble_prediction': int(ensemble_pred[row_idx]),
                                **rounded_probs,
                                **{
                                    f'{name}_weight': float(model_weights[col_idx])
                                    for col_idx, name in enumerate(model_order)
                                }
                            })

                        progress.update(pair_end - pair_start)

            progress.close()
            return pd.DataFrame(results)

        # Non-genome mode: two-stage inference (single-protein cache -> pair scoring)
        pairs = list(self.protein_pairs)
        self.total_pairs_evaluated = len(pairs)
        protein_tensor_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        unique_proteins = sorted({protein for pair in pairs for protein in pair})
        self._emit_concise_step("Step 3: Generating embeddings...")
        self._prepare_embeddings_for_proteins(unique_proteins, show_progress=True)
        for protein_id in unique_proteins:
            cis_t = self.embedding_generator.get_cached_cis_tensor(protein_id)
            pooled_t = self.embedding_generator.get_cached_pooled_tensor(protein_id)

            if cis_t is not None and pooled_t is not None:
                protein_tensor_cache[protein_id] = (cis_t.to(self.device), pooled_t.to(self.device))
                continue

            cis_np = self.embedding_generator.get_cached_cis_embedding(protein_id)
            pooled_np = self.embedding_generator.get_cached_pooled_embedding(protein_id)
            if cis_np is None or pooled_np is None:
                self.logger.warning(f"Missing cached embedding for protein {protein_id}")
                continue

            cis_t = torch.as_tensor(cis_np, dtype=torch.float32, device=self.device)
            pooled_t = torch.as_tensor(pooled_np, dtype=torch.float32, device=self.device)
            protein_tensor_cache[protein_id] = (cis_t, pooled_t)

        self._emit_concise_step("Step 4: Running ensemble inference...")
        progress = tqdm(total=len(pairs), desc="Running ensemble inference")
        for start in range(0, len(pairs), self.config.predict_batch):
            batch = pairs[start:start + self.config.predict_batch]
            batch_records = []
            cis_pair_tensors: List[torch.Tensor] = []
            pooled_pair_tensors: List[torch.Tensor] = []
            valid_pairs: List[Tuple[str, str]] = []

            for protein1, protein2 in batch:
                emb1 = protein_tensor_cache.get(protein1)
                emb2 = protein_tensor_cache.get(protein2)

                if emb1 is None or emb2 is None:
                    self.logger.warning(f"Missing cached embedding for pair ({protein1}, {protein2})")
                    progress.update(1)
                    continue

                cis1, pooled1 = emb1
                cis2, pooled2 = emb2
                cis_pair_tensors.append(torch.cat([cis1, cis2], dim=0))
                pooled_pair_tensors.append(torch.cat([pooled1, pooled2], dim=0))
                valid_pairs.append((protein1, protein2))

            if not valid_pairs:
                continue

            cis_batch_t = torch.stack(cis_pair_tensors, dim=0)
            pooled_batch_t = torch.stack(pooled_pair_tensors, dim=0)

            model_logits_batch = self._compute_model_predictions_batch_torch(
                cis_batch_t,
                pooled_batch_t
            )

            predictions = self._ensemble_predict_batch(model_logits_batch)

            for (protein1, protein2), prediction in zip(valid_pairs, predictions):
                batch_records.append({
                    'protein1': protein1,
                    'protein2': protein2,
                    'ensemble_probability': prediction['ensemble_probability'],
                    'ensemble_prediction': prediction['ensemble_prediction'],
                    **{f'{name}_probability': prob
                       for name, prob in prediction['model_probabilities'].items()},
                    **{f'{name}_weight': weight
                       for name, weight in prediction['model_weights'].items()}
                })

            results.extend(batch_records)
            progress.update(len(valid_pairs))

        progress.close()
        
        return pd.DataFrame(results)
    
    def _compute_model_predictions(
        self, 
        cis_pair_emb: np.ndarray, 
        pooled_pair_emb: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute individual model predictions.
        
        Note: This is a simplified version. For full accuracy, load the actual
        classifier layers from the checkpoints.
        
        Args:
            cis_pair_emb: CIS pair embedding [2 * embedding_dim]
            pooled_pair_emb: Pooled pair embedding [2 * embedding_dim]
            
        Returns:
            Dict mapping model_name to logit
        """
        model_logits = {}
        
        # For each model in ensemble
        for model_name in self.model_names:
            model_config = self.model_configs.get(model_name, {})
            embedding_dim = model_config.get('embedding_dim', 5120)
            interaction_type = model_config.get('interaction_type', 'hadamard')
            
            # Choose embedding based on dimension
            if embedding_dim == 5120:
                # Use CIS embedding (raw ESM)
                emb_dim = len(cis_pair_emb) // 2
                emb1, emb2 = cis_pair_emb[:emb_dim], cis_pair_emb[emb_dim:]
            else:
                # Use pooled embedding (projected)
                emb_dim = len(pooled_pair_emb) // 2
                emb1, emb2 = pooled_pair_emb[:emb_dim], pooled_pair_emb[emb_dim:]
            
            # Apply interaction
            if interaction_type == 'hadamard':
                interaction_emb = emb1 * emb2  # Element-wise multiplication
            elif interaction_type == 'concatenation':
                interaction_emb = np.concatenate([emb1, emb2])
            elif interaction_type == 'difference':
                interaction_emb = np.abs(emb1 - emb2)
            else:
                # Default to hadamard
                interaction_emb = emb1 * emb2
            
            # Get classifier prediction
            classifier = self.classifiers.get(model_name)
            if classifier is not None:
                # Use actual classifier
                with torch.no_grad():
                    input_tensor = torch.tensor(interaction_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
                    output = classifier(input_tensor)
                    prob = output.squeeze().cpu().item()
                    # Convert probability to logit for meta-learner
                    # prob = sigmoid(logit) => logit = log(prob / (1 - prob))
                    prob = max(min(prob, 1.0 - 1e-7), 1e-7)  # Clamp to avoid inf
                    logit = np.log(prob / (1 - prob))
            else:
                # Fallback: cosine similarity (less accurate)
                cos_sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                logit = cos_sim * 2.0
                self.logger.warning(f"Using fallback cosine similarity for {model_name}")
            
            model_logits[model_name] = logit
        
        return model_logits

    def _compute_model_predictions_batch(
        self,
        cis_batch: np.ndarray,
        pooled_batch: np.ndarray,
        residue_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        protein_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute individual model predictions for a batch.

        Args:
            cis_batch: [batch, 2 * embedding_dim]
            pooled_batch: [batch, 2 * embedding_dim]

        Returns:
            Dict mapping model_name to logits array [batch]
        """
        model_logits = {}

        for model_name in self.model_names:
            model_config = self.model_configs.get(model_name, {})
            embedding_dim = model_config.get('embedding_dim', 5120)
            interaction_type = model_config.get('interaction_type', 'hadamard')

            if embedding_dim == 5120:
                emb_dim = cis_batch.shape[1] // 2
                emb1 = cis_batch[:, :emb_dim]
                emb2 = cis_batch[:, emb_dim:]
            else:
                if residue_pairs is not None and protein_pairs is not None and self.sequence_engines.get(model_name) is not None:
                    model_logits[model_name] = self._predict_sequence_model_logits_batch(
                        model_name,
                        residue_pairs,
                        protein_pairs
                    )
                    continue

                emb_dim = pooled_batch.shape[1] // 2
                emb1 = pooled_batch[:, :emb_dim]
                emb2 = pooled_batch[:, emb_dim:]

            if interaction_type == 'hadamard':
                interaction_emb = emb1 * emb2
            elif interaction_type == 'concatenation':
                interaction_emb = np.concatenate([emb1, emb2], axis=1)
            elif interaction_type == 'difference':
                interaction_emb = np.abs(emb1 - emb2)
            else:
                interaction_emb = emb1 * emb2

            classifier = self.classifiers.get(model_name)
            if classifier is not None:
                with torch.no_grad():
                    input_tensor = torch.tensor(
                        interaction_emb, dtype=torch.float32, device=self.device
                    )
                    output = classifier(input_tensor).squeeze()
                    prob = output.detach().float().cpu().numpy()
                    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
                    logit = np.log(prob / (1.0 - prob))
            else:
                emb1_norm = np.linalg.norm(emb1, axis=1)
                emb2_norm = np.linalg.norm(emb2, axis=1)
                cos_sim = (emb1 * emb2).sum(axis=1) / (emb1_norm * emb2_norm + 1e-8)
                logit = cos_sim * 2.0
                self.logger.warning(f"Using fallback cosine similarity for {model_name}")

            model_logits[model_name] = logit

        return model_logits
    
    def _ensemble_predict_single(
        self, 
        model_logits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform ensemble prediction for a single pair.
        
        Args:
            model_logits: Dict mapping model_name to logit
            
        Returns:
            Dict with ensemble_probability, ensemble_prediction, model_probabilities, model_weights
        """
        # Prepare features for meta-learner
        if self.training_mode == 'logits':
            features = np.array([model_logits[name] for name in self.model_names])
        else:
            # For features mode, would need actual feature extraction
            features = np.array([model_logits[name] for name in self.model_names])
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Meta-learner prediction
        ensemble_proba = self.meta_learner.predict_proba(features_scaled)[0, 1]
        ensemble_pred = self.meta_learner.predict(features_scaled)[0]
        
        # Get model weights
        weights = self.meta_learner.coef_[0]
        
        # Compute model probabilities
        model_probabilities = {
            name: 1.0 / (1.0 + np.exp(-logit))  # sigmoid
            for name, logit in model_logits.items()
        }
        
        model_weights = {
            name: float(weights[i]) 
            for i, name in enumerate(self.model_names)
        }
        
        return {
            'ensemble_probability': float(ensemble_proba),
            'ensemble_prediction': int(ensemble_pred),
            'model_probabilities': model_probabilities,
            'model_weights': model_weights
        }

    def _ensemble_predict_batch(
        self,
        model_logits_batch: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        model_order, ensemble_proba, ensemble_pred, model_prob_matrix, model_weights = \
            self._ensemble_predict_batch_arrays(model_logits_batch)
        predictions = []

        for row_idx in range(len(ensemble_proba)):
            model_probabilities = {
                name: float(model_prob_matrix[row_idx, i])
                for i, name in enumerate(model_order)
            }
            weights_dict = {name: float(model_weights[i]) for i, name in enumerate(model_order)}
            predictions.append({
                'ensemble_probability': float(ensemble_proba[row_idx]),
                'ensemble_prediction': int(ensemble_pred[row_idx]),
                'model_probabilities': model_probabilities,
                'model_weights': weights_dict
            })

        return predictions
    
    def _compute_evaluation_metrics(
        self, 
        predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute evaluation metrics if true labels are available."""
        if self.true_labels is None:
            return {}
        
        predicted_proba = predictions['ensemble_probability'].values
        predicted_labels = predictions['ensemble_prediction'].values
        
        # Align labels with predictions
        true_labels = np.array(self.true_labels[:len(predicted_labels)])
        
        try:
            metrics = {
                'auroc': float(roc_auc_score(true_labels, predicted_proba)),
                'aupr': float(average_precision_score(true_labels, predicted_proba)),
                'accuracy': float(accuracy_score(true_labels, predicted_labels)),
                'f1_binary': float(f1_score(true_labels, predicted_labels, average='binary')),
                'f1_macro': float(f1_score(true_labels, predicted_labels, average='macro')),
                'confusion_matrix': confusion_matrix(true_labels, predicted_labels).tolist(),
                'total_samples': len(true_labels),
                'positive_samples': int(sum(true_labels)),
                'negative_samples': int(len(true_labels) - sum(true_labels))
            }
            
            self.logger.info(f"Evaluation metrics: AUROC={metrics['auroc']:.4f}, AUPR={metrics['aupr']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute evaluation metrics: {e}")
            return {}
    
    def _run_ig_analysis(self) -> Dict[str, Any]:
        """Run IG attribution analysis."""
        try:
            from .ig_attribution import (
                IGAttributionAnalyzer,
                PairIGAttributionAnalyzer,
                _load_ig_baselines,
            )

            baselines = _load_ig_baselines(self.config.ig_baseline_path, self.logger)
            
            # Initialize IG analyzer
            self.ig_analyzer = IGAttributionAnalyzer(
                embedding_generator=self.embedding_generator,
                residue_baseline_vector=baselines.get('residue'),
                n_steps=50,
                logger=self.logger
            )
            self.ig_analyzer.initialize()
            
            mode = (self.config.ig_mode or "single").strip().lower()
            if mode not in {"single", "pair", "both"}:
                self.logger.warning(f"Unknown ig_mode={mode}, fallback to single")
                mode = "single"

            output: Dict[str, Any] = {
                'ig_mode': mode,
                'ig_baseline_path': self.config.ig_baseline_path,
                'ig_results': {},
            }

            if mode in {"single", "both"}:
                ig_results = self.ig_analyzer.analyze_batch(
                    self.protein_sequences,
                    show_progress=True
                )

                ig_json_path = self.output_dir / "ig_attributions.jsonl"
                self.ig_analyzer.save_ig_json(ig_results, str(ig_json_path))

                if self.config.pdb_dir:
                    ig_pdb_dir = self.output_dir / "ig_pdbs"
                    output_paths = self.ig_analyzer.save_to_pdb(
                        ig_results,
                        self.config.pdb_dir,
                        str(ig_pdb_dir)
                    )
                    self.logger.info(f"Saved IG scores to {len(output_paths)} PDB files")

                output['ig_results'] = ig_results
                output['ig_json_path'] = str(ig_json_path)

            if mode in {"pair", "both"}:
                if self.config.genome_mode:
                    self.logger.warning("Pair-level IG is skipped in genome_mode due to scale")
                elif not self.protein_pairs:
                    self.logger.warning("No protein pairs available for pair-level IG")
                else:
                    pair_analyzer = PairIGAttributionAnalyzer(
                        embedding_generator=self.embedding_generator,
                        model_names=self.model_names,
                        model_configs=self.model_configs,
                        model_weights=self.model_weights,
                        classifiers=self.classifiers,
                        device=self.device,
                        cis_baseline_vector=baselines.get('cis'),
                        pooled_baseline_vector=baselines.get('pooled'),
                        n_steps=50,
                        logger=self.logger,
                    )

                    pair_ig_results = pair_analyzer.analyze_pairs(
                        protein_pairs=self.protein_pairs,
                        protein_sequences=self.protein_sequences,
                        show_progress=True,
                    )
                    pair_ig_path = self.output_dir / "pair_ig_attributions.jsonl"
                    pair_analyzer.save_pair_ig_json(pair_ig_results, str(pair_ig_path))

                    output['pair_ig_results'] = pair_ig_results
                    output['pair_ig_json_path'] = str(pair_ig_path)

            return output
            
        except Exception as e:
            self.logger.error(f"IG analysis failed: {e}")
            return {}

    def _export_single_mode_lmdb(self) -> Dict[str, Any]:
        """Export 4 single-protein representations to LMDB for single ig_mode."""
        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator is not initialized")

        export_root = Path(self.config.single_lmdb_export_dir) if self.config.single_lmdb_export_dir else self.output_dir
        exporter = SingleProteinLMDBExporter(
            embedding_generator=self.embedding_generator,
            logger=self.logger,
        )
        output_paths = exporter.export(self.protein_sequences, export_root)
        self.single_lmdb_outputs = dict(output_paths)

        return {
            'ig_mode': 'single',
            'single_mode_action': 'export_single_lmdb',
            'single_lmdb_outputs': output_paths,
            'ig_results': {},
        }
    
    def _save_results(
        self,
        predictions: pd.DataFrame,
        evaluation_metrics: Optional[Dict[str, Any]],
        ig_results: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Save all results to output directory."""
        output_paths = {}
        float_format = '%.4f' if self.config.genome_mode else None
        
        # 1. Save ensemble predictions CSV
        predictions_path = self.output_dir / "ensemble_predictions.csv"
        predictions.to_csv(predictions_path, index=False, float_format=float_format)
        output_paths['predictions'] = str(predictions_path)
        self.logger.info(f"Saved predictions to {predictions_path}")
        
        # 2. Save attention weights JSONL
        attention_path = self.output_dir / "attention_weights.jsonl"
        with open(attention_path, 'w', encoding='utf-8') as f:
            for protein_id in self.protein_sequences.keys():
                attention_data = self.embedding_generator.get_cached_attention(protein_id)
                if attention_data:
                    f.write(json.dumps(attention_data, ensure_ascii=False) + '\n')
        output_paths['attention'] = str(attention_path)
        self.logger.info(f"Saved attention weights to {attention_path}")
        
        # 3. Save prediction summary JSON
        summary = {
            'total_pairs': len(predictions),
            'model_info': {
                'esm_model': self.config.esm_model,
                'ensemble_config': self.config.ensemble_config,
                'models': self.model_names
            },
            'timestamp': datetime.now().isoformat()
        }

        if self.total_pairs_evaluated is not None:
            summary['total_pairs_evaluated'] = self.total_pairs_evaluated
            if self.config.genome_mode:
                summary['stored_positive_pairs'] = len(predictions)
        
        if evaluation_metrics:
            summary['evaluation_metrics'] = evaluation_metrics
        
        if ig_results:
            if ig_results.get('single_mode_action') == 'export_single_lmdb':
                summary['ig_analysis'] = {
                    'enabled': True,
                    'mode': 'single',
                    'action': 'export_single_lmdb',
                    'single_lmdb_outputs': ig_results.get('single_lmdb_outputs', {}),
                }
                output_paths.update({
                    f"single_lmdb_{k}": v for k, v in ig_results.get('single_lmdb_outputs', {}).items()
                })
            else:
                summary['ig_analysis'] = {
                    'enabled': True,
                    'mode': ig_results.get('ig_mode', self.config.ig_mode),
                    'proteins_analyzed': len(ig_results.get('ig_results', {}))
                }
        
        summary_path = self.output_dir / "prediction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        output_paths['summary'] = str(summary_path)
        self.logger.info(f"Saved summary to {summary_path}")

        return output_paths
    
    # ==================== Utility Methods ====================
    
    def get_multifeature_command(self) -> str:
        """Get command for generating multifeature files."""
        if self.data_preparator is None:
            return "Data not yet prepared"
        
        return self.data_preparator.get_multifeature_command(
            output_dir=str(self.output_dir / "multifeature")
        )
    
    def prompt_multifeature_generation(self) -> bool:
        """Interactive prompt for multifeature generation."""
        if self.data_preparator is None:
            self.logger.error("Data not yet prepared")
            return False
        
        return self.data_preparator.prompt_multifeature_generation(
            output_dir=str(self.output_dir / "multifeature")
        )
