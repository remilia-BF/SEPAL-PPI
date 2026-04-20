"""
Ensemble prediction engine

Provides prediction utilities using a pre-trained meta-learner for
protein-protein interaction, including residue-level interpretability
and CIS-mode FAISS retrieval.
"""

import logging
import yaml
import pickle
import json
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    faiss = None
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn

from src.utils.helpers import get_device
from src.utils.logger import create_output_directory
from src.inference import InferenceEngine


class EnsemblePredictEngine:
    """Ensemble prediction engine, supports prediction for new protein pairs and interpretability analysis"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[str] = None, logger=None):
        """
        Initialize the ensemble prediction engine.

        Args:
            config: prediction configuration dictionary
            output_dir: output directory; if None a directory will be created
            logger: optional logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # set up output directory
        if output_dir is None:
            if 'out_dir' in config:
                self.output_dir = Path(config['out_dir'])
            else:
                # use create_output_directory to create a predictable output dir
                self.output_dir = Path(create_output_directory(prefix="predict"))
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # device
        self.device = get_device()

        # initialize components
        self.meta_learner = None
        self.scaler = None
        self.model_names = []
        self.inference_engines = []
        self.ensemble_config = None
        
        # cache directory
        self.cache_dir = Path("cache/faiss")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Ensemble prediction engine initialized, output directory: {self.output_dir}")
    
    def run_ensemble_predict(self) -> Dict[str, Any]:
        """
        Run ensemble prediction

        Returns:
            prediction results dictionary
        """
        try:
            # 1. load meta-learner and models
            self._load_ensemble_components()
            
            # 2. prepare prediction data
            prediction_data = self._prepare_prediction_data()
            
            # 3. run ensemble inference
            prediction_results = self._run_ensemble_inference(prediction_data)
            
            # 4. compute evaluation metrics (if true labels available)
            evaluation_metrics = None
            if prediction_data.get('has_true_labels', False):
                self.logger.info("Detected true labels, computing evaluation metrics...")
                try:
                    true_labels = prediction_data['true_labels']
                    ensemble_predictions = prediction_results['ensemble_predictions']
                    predicted_proba = ensemble_predictions['ensemble_probability'].values
                    predicted_labels = ensemble_predictions['ensemble_prediction'].values
                    
                    evaluation_metrics = self._calculate_evaluation_metrics(
                        true_labels, predicted_proba, predicted_labels
                    )
                    self.logger.info("Evaluation metrics computed")
                except Exception as e:
                    self.logger.error(f"Error while computing evaluation metrics: {e}")
                    evaluation_metrics = None
            
            # 5. process interpretability analysis
            interpretability_results = self._process_interpretability(prediction_results)
            
            # 6. save results
            self._save_results(prediction_results, interpretability_results, evaluation_metrics)
            
            return {
                'prediction_results': prediction_results,
                'evaluation_metrics': evaluation_metrics,
                'interpretability_results': interpretability_results,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error during ensemble prediction: {str(e)}")
            raise
    
    def _load_ensemble_components(self):
        """Load ensemble inference configuration and meta-learner."""
        self.logger.debug("Loading ensemble inference configuration...")
        
        ensemble_config_path = self.config['ensemble_config']
        
        # load ensemble inference config
        with open(ensemble_config_path, 'r', encoding='utf-8') as f:
            self.ensemble_config = yaml.safe_load(f)
        
        # load meta-learner
        meta_learner_path = self.ensemble_config['ensemble']['meta_learner_path']
        self.logger.info(f"Loading meta-learner: {meta_learner_path}")
        
        with open(meta_learner_path, 'rb') as f:
            meta_learner_data = pickle.load(f)
        
        self.meta_learner = meta_learner_data['meta_learner']
        self.scaler = meta_learner_data['scaler']
        self.model_names = meta_learner_data['model_names']
        
        # obtain training mode information
        self.training_mode = meta_learner_data.get('training_mode', 'logits')
        
        self.logger.debug(f"Meta-learner loaded, available models: {self.model_names}")
        self.logger.debug(f"Training mode: {self.training_mode}")
        
        # load per-model inference engines
        self._load_individual_models()
    
    def _load_individual_models(self):
        """Load per-model inference engines."""
        self.logger.debug("Loading individual models...")
        
        models_config = self.ensemble_config['ensemble']['models']
        
        # optional: override multimodal feature root during prediction
        predict_feature_root = self.config.get('predict_prot_feature_folder', None)

        for model_info in models_config:
            model_name = model_info['model_name']
            model_path = model_info['model_path']
            
            self.logger.info(f"Loading model {model_name}: {model_path}")
            
            # create temporary inference config
            temp_config = {
                'mode': 'inference',
                'model': model_info['model_config'],
                'data': model_info['data_config'],
                'inference': self.ensemble_config.get('inference', {}),
                'output': self.ensemble_config.get('output', {})
            }

            # override all_feature_folder in preprocessing (only if provided)
            try:
                if predict_feature_root and isinstance(temp_config.get('model'), dict):
                    mc = temp_config['model']

                    # Prefer modular path: model_config.preprocessing.data_processing
                    if not isinstance(mc.get('model_config'), dict):
                        mc['model_config'] = {}
                    model_cfg = mc['model_config']

                    preproc = model_cfg.get('preprocessing') or {}
                    if not isinstance(preproc, dict):
                        preproc = {}
                    dp = preproc.get('data_processing') or {}
                    if not isinstance(dp, dict):
                        dp = {}
                    dp['all_feature_folder'] = predict_feature_root
                    # treat missing features as errors during prediction
                    dp['log_missing_as_error'] = True
                    preproc['data_processing'] = dp
                    model_cfg['preprocessing'] = preproc
                    mc['model_config'] = model_cfg

                    # Also support legacy path: preprocessing at model root (if present)
                    if isinstance(mc.get('preprocessing'), dict):
                        dp_root = mc['preprocessing'].get('data_processing') or {}
                        if not isinstance(dp_root, dict):
                            dp_root = {}
                        dp_root['all_feature_folder'] = predict_feature_root
                        dp_root['log_missing_as_error'] = True
                        mc['preprocessing']['data_processing'] = dp_root

                    self.logger.info(
                        f"Model {model_name} preprocessing feature root overridden to: {predict_feature_root}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to override feature root for model {model_name}: {e}")
            
            # create inference engine
            engine = InferenceEngine(temp_config, self.logger)
            engine.load_model(model_path, device=self.device)
            
            self.inference_engines.append(engine)
            self.logger.debug(f"Model {model_name} loaded")
        
        self.logger.debug(f"All {len(self.inference_engines)} models loaded")
    
    def _prepare_prediction_data(self) -> Dict[str, Any]:
        """Prepare prediction data."""
        self.logger.debug("Preparing prediction data...")
        
        # read prediction ID file
        test_id_file = self.config['test_id_file']
        test_protein_fasta = self.config['test_protein_fasta']
        
        # read protein pair IDs and optional true labels
        protein_pairs = []
        true_labels = []
        has_labels = False
        
        with open(test_id_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        protein1 = parts[0].strip()
                        protein2 = parts[1].strip()
                        protein_pairs.append((protein1, protein2))
                        
                        # check for optional third column (true labels)
                        if len(parts) >= 3:
                            try:
                                label = int(parts[2].strip())
                                if label in [0, 1]:
                                    true_labels.append(label)
                                    has_labels = True
                                else:
                                    # third column is not 0 or 1, ignore labels
                                    if has_labels:
                                        self.logger.warning(f"Third column contains non-0/1 value: {parts[2]}, ignoring all labels")
                                        has_labels = False
                                        true_labels = []
                                    break
                            except ValueError:
                                # third column is not an integer, ignore labels
                                if has_labels:
                                    self.logger.warning(f"Third column is not an integer: {parts[2]}, ignoring all labels")
                                    has_labels = False
                                    true_labels = []
                                break
                        elif has_labels:
                            # previously had labels but this line does not, maintain consistency
                            self.logger.warning("Inconsistent data format, some rows missing labels, ignoring all labels")
                            has_labels = False
                            true_labels = []
                            break
        
        if has_labels and len(true_labels) != len(protein_pairs):
            self.logger.warning("Label count does not match protein pair count, ignoring labels")
            has_labels = False
            true_labels = []
        
        if has_labels:
            self.logger.info(f"Read {len(protein_pairs)} protein pairs with true labels")
            self.logger.info(f"Label distribution: positive={sum(true_labels)}, negative={len(true_labels)-sum(true_labels)}")
        else:
            self.logger.debug(f"Read {len(protein_pairs)} protein pairs (no labels)")
        
        # read FASTA sequences
        fasta_sequences = self._read_fasta_sequences(test_protein_fasta)
        self.logger.debug(f"Read {len(fasta_sequences)} protein sequences")
        
        return {
            'protein_pairs': protein_pairs,
            'fasta_sequences': fasta_sequences,
            'test_protein_lmdb_dir': self.config['test_protein_lmdb_dir'],
            'has_true_labels': has_labels,
            'true_labels': true_labels if has_labels else None
        }
    
    def _read_fasta_sequences(self, fasta_file: str) -> Dict[str, str]:
        """Read FASTA sequence file."""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]  # use part before first space as ID
                    current_seq = []
                elif line:
                    current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def _run_ensemble_inference(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble inference."""
        self.logger.debug("Starting ensemble inference...")
        
        protein_pairs = prediction_data['protein_pairs']
        fasta_sequences = prediction_data['fasta_sequences']
        test_protein_lmdb_dir = prediction_data['test_protein_lmdb_dir']
        
        # collect predictions from all models
        all_model_predictions = {}
        
        for i, (engine, model_name) in enumerate(zip(self.inference_engines, self.model_names)):
            self.logger.info(f"Predicting with model {model_name}...")
            
            # get corresponding LMDB directory
            model_key = f'model{i+1}'
            if model_key in test_protein_lmdb_dir:
                lmdb_path = test_protein_lmdb_dir[model_key]
            else:
                self.logger.warning(f"LMDB path for model {model_name} not found, skipping")
                continue
            
            # 创建临时数据集进行预测
            model_predictions = self._predict_with_single_model(
                engine, model_name, protein_pairs, fasta_sequences, lmdb_path, i
            )
            
            all_model_predictions[model_name] = model_predictions
        
        # 使用元学习器进行集成预测
        ensemble_predictions = self._ensemble_predict(all_model_predictions, protein_pairs)
        
        return {
            'individual_predictions': all_model_predictions,
            'ensemble_predictions': ensemble_predictions,
            'protein_pairs': protein_pairs
        }
    
    def _predict_with_single_model(self, engine: InferenceEngine, model_name: str, 
                                 protein_pairs: List[Tuple[str, str]], 
                                 fasta_sequences: Dict[str, str], 
                                 lmdb_path: str, model_index: int) -> Dict[str, Any]:
        """Predict with a single model."""
        # check if this is a CIS model
        # Prioritize checking the model's own configuration from the ensemble as it is the source of truth
        model_config_ensemble = self.ensemble_config['ensemble']['models'][model_index]
        is_cis_model = model_config_ensemble.get('data_config', {}).get('cis_type', False)
        
        # Fallback to user config if needed (though ensemble config should determine architecture)
        if is_cis_model is None:
            model_key = f'model{model_index+1}'
            model_config = self.config.get(model_key, {})
            is_cis_model = model_config.get('cis_type', False) or model_config.get('cis_only', False)
        
        if is_cis_model:
            return self._predict_with_cis_model(engine, model_name, protein_pairs, lmdb_path, model_index)
        else:
            return self._predict_with_sequence_model(engine, model_name, protein_pairs, 
                                                   fasta_sequences, lmdb_path, model_index)
    
    def _predict_with_sequence_model(self, engine: InferenceEngine, model_name: str,
                                   protein_pairs: List[Tuple[str, str]], 
                                   fasta_sequences: Dict[str, str], 
                                   lmdb_path: str, model_index: int) -> Dict[str, Any]:
        """Predict with sequence-level model."""
        from src.data_processing.unified_cache_loader import create_unified_data_loaders
        
        # create temporary test file
        temp_test_file = self.output_dir / f"temp_{model_name}_test.txt"
        with open(temp_test_file, 'w') as f:
            for p1, p2 in protein_pairs:
                f.write(f"{p1}\t{p2}\t0\n")  # label set to 0, not used in prediction
        
        # create FASTA file
        temp_fasta_file = self._create_temp_fasta_file(fasta_sequences, model_name)
        
        # create data loader config
        # get embedding_dim from current model config, not from global config
        model_config = self.ensemble_config['ensemble']['models'][model_index]
        embedding_dim = model_config['data_config']['embedding_dim']
        target_precision = model_config['data_config'].get('target_precision', 'fp32')
        data_config = {
            'embedding_file': lmdb_path,
            'fasta_file': str(temp_fasta_file),
            'test_files': {'predict': str(temp_test_file)},
            'batch_size': 128,
            'cache_size': 8000,
            'embedding_dim': embedding_dim,  # get embedding_dim from current model config
            'target_precision': target_precision,
            'seed': self.config.get('seed', 42)
        }
        
        # create data loaders
        batch_data = create_unified_data_loaders(data_config)
        data_loaders = batch_data['data_loaders']
        
        # extract dataset
        test_dataset = data_loaders['predict'].dataset
        
        # check if residue-level attention results are needed
        model_key = f'model{model_index+1}'
        should_save_attention = self.config.get(model_key, {}).get('Residue_Interpretability', False)
        
        # temporarily modify engine config to support attention weight collection
        if should_save_attention:
            # set output directory and FASTA file path for attention weight collection
            original_fasta = engine.config.get('data', {}).get('fasta_file')
            engine.config['data'] = engine.config.get('data', {})
            engine.config['data']['fasta_file'] = str(temp_fasta_file)
            engine.config['data']['cis_type'] = False  # ensure non-CIS mode to enable attention weight collection
        
        # check if features need to be extracted
        config_mode = self.config.get('ensemble_mode', 'logits')
        training_mode = getattr(self, 'training_mode', 'logits')
        
        # check mode consistency
        if config_mode and training_mode and config_mode != training_mode:
            self.logger.warning(f"Mode mismatch: config={config_mode}, training={training_mode}")
            self.logger.warning(f"Auto-adjusting to training mode {training_mode} to ensure meta-learner input dimension consistency")
            inference_mode = training_mode
        else:
            # prioritize config mode, else use training mode
            inference_mode = config_mode if config_mode else training_mode
        
        # run prediction
        results = engine._evaluate_single_dataset(
            test_dataset, 
            f'{model_name}_predict', 
            show_progress=True,
            return_logits=True,
            return_features=(inference_mode == 'features'),
            save_results=should_save_attention,  # only save results when attention weights needed
            output_dir=str(self.output_dir) if should_save_attention else None
        )
        
        # restore original config
        if should_save_attention and original_fasta:
            engine.config['data']['fasta_file'] = original_fasta
        
        # cleanup temporary files
        temp_test_file.unlink()
        temp_fasta_file.unlink()

        # if return_logits=True, predictions are actually logits
        return_logits = True
        if return_logits:
            logits = results['predictions']
            predictions = 1 / (1 + np.exp(-np.array(logits)))  # sigmoid conversion to probability
        else:
            predictions = results['predictions']
            logits = np.log(np.array(predictions) / (1 - np.array(predictions) + 1e-7))  # probability to logits
        
        result = {
            'predictions': predictions,
            'logits': logits,
            'model_type': 'sequence',
            'feature_gating_summary': results.get('feature_gating_summary', {})
        }
        
        # if features were extracted, add them to result
        if inference_mode == 'features' and 'features' in results:
            result['features'] = np.array(results['features'])
            self.logger.debug(f"Model {model_name} extracted features, shape: {result['features'].shape}")
        
        # if attention weight file was generated, add it to result
        if should_save_attention and results.get('attention_weights_file'):
            attention_file = self.output_dir / results['attention_weights_file']
            if attention_file.exists():
                result['attention_weights_file'] = str(attention_file)
                self.logger.info(f"Residue-level attention weights for model {model_name} saved to: {attention_file}")
        
        return result
    
    def _predict_with_cis_model(self, engine: InferenceEngine, model_name: str,
                              protein_pairs: List[Tuple[str, str]], 
                              lmdb_path: str, model_index: int) -> Dict[str, Any]:
        """Predict with CIS-level model."""
        from src.data_processing.cis_data_loader import create_cis_data_loaders
        
        # create temporary test file
        temp_test_file = self.output_dir / f"temp_{model_name}_test.txt"
        with open(temp_test_file, 'w') as f:
            for p1, p2 in protein_pairs:
                f.write(f"{p1}\t{p2}\t0\n")  # label set to 0, not used in prediction
        
        # create CIS data loader config
        # get embedding_dim from current model config, not from global config
        model_config = self.ensemble_config['ensemble']['models'][model_index]
        embedding_dim = model_config['data_config']['embedding_dim']
        target_precision = model_config['data_config'].get('target_precision', 'fp32')
        cis_config = {
            'embedding_file': lmdb_path,
            'test_files': {'predict': str(temp_test_file)},
            'batch_size': 128,
            'cache_size': 8000,
            'embedding_dim': embedding_dim,  # get embedding_dim from current model config
            'target_precision': target_precision,
            'seed': self.config.get('seed', 42)
        }
        
        # create data loaders
        cis_batch_data = create_cis_data_loaders(cis_config)
        data_loaders = cis_batch_data['data_loaders']
        
        # extract dataset
        test_dataset = data_loaders['predict'].dataset
        
        # check if features need to be extracted
        config_mode = self.config.get('ensemble_mode', 'logits')
        training_mode = getattr(self, 'training_mode', 'logits')
        
        # check mode consistency
        if config_mode and training_mode and config_mode != training_mode:
            self.logger.warning(f"Mode mismatch: config={config_mode}, training={training_mode}")
            self.logger.warning(f"Auto-adjusting to training mode {training_mode} to ensure meta-learner input dimension consistency")
            inference_mode = training_mode
        else:
            # prioritize config mode, else use training mode
            inference_mode = config_mode if config_mode else training_mode
        
        # run prediction
        results = engine._evaluate_single_dataset(
            test_dataset, 
            'predict', 
            show_progress=True,
            return_logits=True,
            return_features=(inference_mode == 'features'),
            save_results=False
        )
        
        # cleanup temporary files
        temp_test_file.unlink()
        
        # if return_logits=True, predictions are actually logits
        return_logits = True
        if return_logits:
            logits = results['predictions']
            predictions = 1 / (1 + np.exp(-np.array(logits)))  # sigmoid conversion to probability
        else:
            predictions = results['predictions']
            logits = np.log(np.array(predictions) / (1 - np.array(predictions) + 1e-7))  # probability to logits
        
        result = {
            'predictions': predictions,
            'logits': logits,
            'model_type': 'cis',
            'embedding_cache': cis_batch_data.get('embedding_cache'),
            'feature_gating_summary': results.get('feature_gating_summary', {})
        }
        
        # if features were extracted, add them to result
        if inference_mode == 'features' and 'features' in results:
            result['features'] = np.array(results['features'])
            self.logger.debug(f"Model {model_name} extracted features, shape: {result['features'].shape}")
        
        return result
    
    def _ensemble_predict(self, all_model_predictions: Dict[str, Dict], 
                         protein_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Perform ensemble prediction with meta-learner."""
        self.logger.debug("Performing ensemble prediction...")
        
        # determine mode to use
        config_mode = self.config.get('ensemble_mode', 'logits')
        training_mode = getattr(self, 'training_mode', 'logits')
        
        # check mode consistency
        if config_mode and training_mode and config_mode != training_mode:
            self.logger.warning(f"Mode mismatch: config={config_mode}, training={training_mode}")
            self.logger.warning(f"Auto-adjusting to training mode {training_mode} to ensure meta-learner input dimension consistency")
            inference_mode = training_mode
        else:
            # prioritize config mode, else use training mode
            inference_mode = config_mode if config_mode else training_mode
        
        self.logger.info(f"Using ensemble inference mode: {inference_mode} (config: {config_mode}, training: {training_mode})")
        
        num_samples = len(protein_pairs)
        num_models = len(self.model_names)
        
        if inference_mode == 'features':
            # 使用features模式
            self.logger.debug("使用features模式进行集成预测...")
            
            # 准备特征矩阵 (features)
            # 每个模型贡献16维features，总共2个模型，所以是32维
            X_test = np.zeros((num_samples, 32))
            
            for i, model_name in enumerate(self.model_names):
                if model_name in all_model_predictions:
                    if 'features' in all_model_predictions[model_name]:
                        features = all_model_predictions[model_name]['features']
                        # handle features dimensions
                        if features.shape[1] == 16:
                            # if features are 16-dim, use directly
                            self.logger.info(f"Model {model_name} features dimension is 16")
                            start_idx = i * 16  # each model uses 16 dims
                            end_idx = start_idx + 16
                            self.logger.debug(f"Model {model_name} (i={i}): start_idx={start_idx}, end_idx={end_idx}, X_test.shape={X_test.shape}")
                            X_test[:, start_idx:end_idx] = features
                        elif features.shape[1] == 32:
                            # if features are already 32-dim, take first 16 dims
                            self.logger.info(f"Model {model_name} features dimension is 32, taking first 16 dims")
                            start_idx = i * 16
                            end_idx = start_idx + 16
                            X_test[:, start_idx:end_idx] = features[:, :16]
                        else:
                            self.logger.warning(f"Model {model_name} features dimension incorrect: {features.shape}, using logits as fallback")
                            logits = all_model_predictions[model_name]['logits']
                            # expand logits to 16-dim features
                            start_idx = i * 16
                            end_idx = start_idx + 2
                            X_test[:, start_idx:end_idx] = logits.reshape(-1, 2)
                            X_test[:, end_idx:start_idx+16] = 0  # pad remaining dims with 0
                    else:
                        self.logger.warning(f"Model {model_name} has no features data, using logits as fallback")
                        logits = all_model_predictions[model_name]['logits']
                        # expand logits to 16-dim features
                        start_idx = i * 16
                        end_idx = start_idx + 2
                        X_test[:, start_idx:end_idx] = logits.reshape(-1, 2)
                        X_test[:, end_idx:start_idx+16] = 0  # pad remaining dims with 0
        else:
            # use logits mode
            self.logger.debug("Using logits mode for ensemble prediction...")
            
            # prepare feature matrix (logits)
            X_test = np.zeros((num_samples, num_models))
            
            for i, model_name in enumerate(self.model_names):
                if model_name in all_model_predictions:
                    logits = all_model_predictions[model_name]['logits']
                    X_test[:, i] = logits
        
        # check feature dimension match
        current_features = X_test.shape[1]
        expected_features = self.scaler.n_features_in_
        
        if current_features != expected_features:
            self.logger.warning(f"Feature dimension mismatch: current={current_features}, expected={expected_features}")
            
            if inference_mode == 'features' and training_mode == 'features' and current_features == 2:
                self.logger.warning("Training used features mode, but inference used logits mode, attempting to adjust feature dimensions")
                
                # expand 2-dim logits to 32-dim features
                X_test_expanded = np.zeros((num_samples, 32))
                X_test_expanded[:, :2] = X_test  # first 2 dims use logits
                X_test_expanded[:, 2:] = 0  # pad remaining dims with 0
                
                X_test = X_test_expanded
                self.logger.info(f"Feature dimensions adjusted: {X_test.shape[1]}")
            elif inference_mode == 'logits' and training_mode == 'features':
                self.logger.warning("Training used features mode, but inference used logits mode, attempting to adjust feature dimensions")
                
                # expand 2-dim logits to 32-dim features
                X_test_expanded = np.zeros((num_samples, 32))
                X_test_expanded[:, :2] = X_test  # first 2 dims use logits
                X_test_expanded[:, 2:] = 0  # pad remaining dims with 0
                
                X_test = X_test_expanded
                self.logger.info(f"Feature dimensions adjusted: {X_test.shape[1]}")
            else:
                self.logger.error("Cannot match feature dimensions, please check model configuration")
                raise ValueError(f"Feature dimension mismatch: current={current_features}, expected={expected_features}")
        
        # scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # meta-learner prediction
        ensemble_proba = self.meta_learner.predict_proba(X_test_scaled)[:, 1]
        ensemble_pred = self.meta_learner.predict(X_test_scaled)
        
        # get model weights
        weights = self.meta_learner.coef_[0]
        
        # build result DataFrame
        result_data = {
            'protein1': [pair[0] for pair in protein_pairs],
            'protein2': [pair[1] for pair in protein_pairs],
            'ensemble_probability': ensemble_proba,
            'ensemble_prediction': ensemble_pred,
        }
        
        # add per-model predictions and weights
        for i, model_name in enumerate(self.model_names):
            if model_name in all_model_predictions:
                if inference_mode == 'features':
                    # for features mode, use original logits to compute probability
                    model_logits = all_model_predictions[model_name]['logits']
                    model_proba = 1 / (1 + np.exp(-model_logits))  # sigmoid
                else:
                    # for logits mode, directly use logits to compute probability
                    model_logits = X_test[:, i] if X_test.shape[1] == num_models else X_test[:, i*32:(i+1)*32].mean(axis=1)
                    model_proba = 1 / (1 + np.exp(-model_logits))  # sigmoid
                
                result_data[f'{model_name}_probability'] = model_proba
                result_data[f'{model_name}_weight'] = [weights[i]] * num_samples
        
        return pd.DataFrame(result_data)
    
    def _calculate_evaluation_metrics(self, true_labels: List[int], 
                                    predicted_proba: np.ndarray, 
                                    predicted_labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.
        
        Args:
            true_labels: true labels
            predicted_proba: predicted probabilities
            predicted_labels: predicted labels (0 or 1)
            
        Returns:
            dict containing various evaluation metrics
        """
        try:
            # basic metrics
            auroc = roc_auc_score(true_labels, predicted_proba)
            aupr = average_precision_score(true_labels, predicted_proba)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            # F1 scores
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')
            f1_micro = f1_score(true_labels, predicted_labels, average='micro')
            f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
            f1_binary = f1_score(true_labels, predicted_labels, average='binary')
            
            # confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            tn, fp, fn, tp = cm.ravel()
            
            # other metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # classification report
            class_report = classification_report(true_labels, predicted_labels, 
                                               target_names=['Negative', 'Positive'],
                                               output_dict=True)
            
            metrics = {
                'auroc': float(auroc),
                'aupr': float(aupr),
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_micro': float(f1_micro),
                'f1_weighted': float(f1_weighted),
                'f1_binary': float(f1_binary),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                },
                'classification_report': class_report,
                'label_distribution': {
                    'true_positive': int(sum(true_labels)),
                    'true_negative': int(len(true_labels) - sum(true_labels)),
                    'predicted_positive': int(sum(predicted_labels)),
                    'predicted_negative': int(len(predicted_labels) - sum(predicted_labels))
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {e}")
            return {}
    
    def _process_interpretability(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interpretability analysis."""
        self.logger.debug("Processing interpretability analysis...")
        
        interpretability_results = {}
        
        # handle residue-level interpretability
        if self._should_generate_residue_interpretability():
            interpretability_results['residue'] = self._generate_residue_interpretability(prediction_results)
        
        # handle CIS interpretability
        if self._should_generate_cis_interpretability():
            interpretability_results['cis'] = self._generate_cis_interpretability(prediction_results)
        
        return interpretability_results
    
    def _should_generate_residue_interpretability(self) -> bool:
        """检查是否需要生成残基级别可解释性"""
        return any(
            self.config.get(f'model{i+1}', {}).get('Residue_Interpretability', False)
            for i in range(len(self.model_names))
        )
    
    def _should_generate_cis_interpretability(self) -> bool:
        """检查是否需要生成CIS可解释性"""
        return any(
            self.config.get(f'model{i+1}', {}).get('cis_Interpretability', False)
            for i in range(len(self.model_names))
        )
    
    def _generate_cis_interpretability(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成CIS可解释性结果（FAISS检索）"""
        self.logger.debug("生成CIS可解释性结果...")
        
        from .cis_interpretability import CisInterpretabilityEngine
        
        cis_results = {}
        
        # 获取所有待预测的蛋白质对和预测结果
        ensemble_predictions = prediction_results['ensemble_predictions']
        test_protein_pairs = []
        all_predictions = []
        
        for _, row in ensemble_predictions.iterrows():
            test_protein_pairs.append((row['protein1'], row['protein2']))
            all_predictions.append(row['ensemble_prediction'])
        
        if not test_protein_pairs:
            self.logger.warning("没有预测结果，跳过CIS可解释性分析")
            return {'message': '没有预测结果'}
        
        # 初始化CIS可解释性引擎（全局共享）
        if not hasattr(self, '_cis_engine'):
            self._cis_engine = CisInterpretabilityEngine()
        
        for i, model_name in enumerate(self.model_names):
            model_key = f'model{i+1}'
            model_config = self.config.get(model_key, {})
            
            if not model_config.get('cis_Interpretability', False):
                continue
            
            self.logger.info(f"为模型 {model_name} 生成CIS可解释性...")
            
            # 注入该模型对应的 test_protein_lmdb 路径（ensemble 配置中可能包含 test_protein_lmdb_dir 映射）
            try:
                test_lmdb_mapping = self.config.get('test_protein_lmdb_dir', {})
                test_lmdb_for_model = test_lmdb_mapping.get(model_key)
                if test_lmdb_for_model:
                    # 复制一份 model_config 避免修改原始配置
                    model_config = dict(model_config)
                    model_config['test_protein_lmdb_path'] = test_lmdb_for_model
                else:
                    self.logger.debug(f"未找到 {model_key} 对应的 test_protein_lmdb_dir 映射，使用默认配置")
            except Exception as e:
                self.logger.warning(f"注入 test_protein_lmdb_path 失败: {e}")

            # 获取该模型的预测结果
            individual_predictions = prediction_results.get('individual_predictions', {})
            model_predictions = []
            
            if model_name in individual_predictions:
                model_pred_data = individual_predictions[model_name].get('predictions', [])
                # 处理不同的预测数据格式
                if isinstance(model_pred_data, list) and len(model_pred_data) > 0:
                    if isinstance(model_pred_data[0], dict):
                        # 如果是字典列表
                        model_predictions = [pred.get('prediction', 0) for pred in model_pred_data]
                    else:
                        # 如果是数值列表
                        model_predictions = [float(pred) for pred in model_pred_data]
                elif hasattr(model_pred_data, '__iter__'):
                    # 如果是numpy数组或其他可迭代对象
                    model_predictions = [float(pred) for pred in model_pred_data]
            
            # 如果没有单独的模型预测，使用集成预测
            if not model_predictions:
                model_predictions = all_predictions
            
            # 生成CIS可解释性结果
            cis_result = self._cis_engine.generate_cis_interpretability(
                model_name, model_config, test_protein_pairs, np.array(model_predictions), self.config, self.output_dir
            )
            
            cis_results[model_name] = cis_result
        
        return cis_results
    
    def _collect_attention_weights_for_model(self, model_index: int, prediction_results: Dict[str, Any]) -> List[Dict]:
        """为指定模型收集注意力权重"""
        self.logger.debug(f"收集模型 {model_index} 的注意力权重...")
        
        # 检查模型预测结果中是否包含注意力权重文件
        individual_predictions = prediction_results.get('individual_predictions', {})
        
        for model_name, model_results in individual_predictions.items():
            if model_results.get('attention_weights_file'):
                attention_file = Path(model_results['attention_weights_file'])
                
                if attention_file.exists():
                    self.logger.debug(f"找到模型 {model_name} 的注意力权重文件: {attention_file}")
                    
                    # 读取注意力权重文件
                    attention_data = []
                    try:
                        with open(attention_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    attention_data.append(json.loads(line.strip()))
                        
                        return attention_data
                        
                    except Exception as e:
                        self.logger.error(f"读取注意力权重文件时出错: {e}")
        
        # 如果没有找到现有的注意力权重文件，返回空列表
        self.logger.warning(f"未找到模型 {model_index} 的注意力权重数据")
        return []
    
    def _generate_residue_interpretability(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成残基级别可解释性结果"""
        self.logger.debug("生成残基级别可解释性结果...")
        
        residue_results = {}
        
        # 检查个别模型的预测结果中是否有注意力权重文件
        individual_predictions = prediction_results.get('individual_predictions', {})
        
        for i, model_name in enumerate(self.model_names):
            model_key = f'model{i+1}'
            if not self.config.get(model_key, {}).get('Residue_Interpretability', False):
                continue
            
            self.logger.debug(f"为模型 {model_name} 生成残基级别可解释性...")
            
            # 查找对应的注意力权重文件
            model_results = individual_predictions.get(model_name, {})
            attention_file = model_results.get('attention_weights_file')
            
            if attention_file and Path(attention_file).exists():
                # 直接使用现有的注意力权重文件作为残基可解释性结果
                residue_results[model_name] = str(attention_file)
                self.logger.info(f"残基级别可解释性结果: {attention_file}")
                
                # 统计注意力权重记录数
                record_count = 0
                try:
                    with open(attention_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                record_count += 1
                    
                    self.logger.debug(f"模型 {model_name} 包含 {record_count} 条残基级注意力权重记录")
                except Exception as e:
                    self.logger.error(f"读取注意力权重文件时出错: {e}")
            else:
                self.logger.warning(f"未找到模型 {model_name} 的注意力权重文件")
        
        return residue_results
    
    def _read_protein_ids(self, protein_id_file: str) -> List[str]:
        """读取蛋白ID文件"""
        protein_ids = []
        try:
            with open(protein_id_file, 'r') as f:
                for line in f:
                    protein_id = line.strip()
                    if protein_id:
                        protein_ids.append(protein_id)
        except Exception as e:
            self.logger.error(f"读取蛋白ID文件时出错: {e}")
        
        return protein_ids
    
    def _build_or_load_faiss_index(self, lmdb_dir: str, protein_ids: List[str], 
                                 model_name: str) -> Tuple[Any, Dict[str, int]]:
        """构建或加载FAISS索引 - 已移动到CisInterpretabilityEngine"""
        # 这个方法已经移动到CisInterpretabilityEngine中
        # 保留这里是为了向后兼容，实际上不应该被调用
        self.logger.warning("_build_or_load_faiss_index 已弃用，请使用 CisInterpretabilityEngine")
        return None, {}
    
    def _compute_hadamard_product(self, protein1: str, protein2: str, lmdb_dir: str) -> Optional[np.ndarray]:
        """计算两个蛋白的哈达玛积 - 已移动到CisInterpretabilityEngine"""
        # 这个方法已经移动到CisInterpretabilityEngine中
        # 保留这里是为了向后兼容，实际上不应该被调用
        self.logger.warning("_compute_hadamard_product 已弃用，请使用 CisInterpretabilityEngine")
        return None
    
    def _faiss_search(self, faiss_index: Any, query_vector: np.ndarray, 
                     protein_id_to_index: Dict[str, int], top_k: int) -> List[Dict[str, Any]]:
        """使用FAISS进行相似性搜索 - 已移动到CisInterpretabilityEngine"""
        # 这个方法已经移动到CisInterpretabilityEngine中
        # 保留这里是为了向后兼容，实际上不应该被调用
        self.logger.warning("_faiss_search 已弃用，请使用 CisInterpretabilityEngine")
        return []
    
    def _save_results(self, prediction_results: Dict[str, Any], 
                     interpretability_results: Dict[str, Any],
                     evaluation_metrics: Optional[Dict[str, Any]] = None):
        """保存预测结果"""
        self.logger.debug("保存预测结果...")
        
        # 保存主要预测结果CSV
        ensemble_predictions = prediction_results['ensemble_predictions']
        output_csv = self.output_dir / "ensemble_predictions.csv"
        ensemble_predictions.to_csv(output_csv, index=False)
        self.logger.info(f"集成预测结果保存到: {output_csv}")
        
        # 保存详细结果JSON
        detailed_results = {
            'prediction_summary': {
                'total_pairs': len(prediction_results['protein_pairs']),
                'positive_predictions': int(ensemble_predictions['ensemble_prediction'].sum()),
                'average_probability': float(ensemble_predictions['ensemble_probability'].mean())
            },
            'model_info': {
                'model_names': self.model_names,
                'meta_learner_weights': self.meta_learner.coef_[0].tolist() if hasattr(self.meta_learner, 'coef_') else None
            },
            'interpretability_summary': interpretability_results
        }
        
        # 添加评估指标（如果有真实标签）
        if evaluation_metrics:
            detailed_results['evaluation_metrics'] = evaluation_metrics
            self.logger.info("评估指标:")
            self.logger.info(f"  AUROC: {evaluation_metrics.get('auroc', 'N/A'):.4f}")
            self.logger.info(f"  AUPR: {evaluation_metrics.get('aupr', 'N/A'):.4f}")
            self.logger.info(f"  Accuracy: {evaluation_metrics.get('accuracy', 'N/A'):.4f}")
            self.logger.info(f"  F1 (binary): {evaluation_metrics.get('f1_binary', 'N/A'):.4f}")
            self.logger.info(f"  F1 (macro): {evaluation_metrics.get('f1_macro', 'N/A'):.4f}")
            self.logger.info(f"  F1 (micro): {evaluation_metrics.get('f1_micro', 'N/A'):.4f}")
            
            # 单独保存评估指标CSV
            metrics_df = pd.DataFrame([evaluation_metrics])
            metrics_csv = self.output_dir / "evaluation_metrics.csv"
            metrics_df.to_csv(metrics_csv, index=False)
            self.logger.info(f"评估指标保存到: {metrics_csv}")
            
            # 保存混淆矩阵
            if 'confusion_matrix' in evaluation_metrics:
                cm = evaluation_metrics['confusion_matrix']
                cm_df = pd.DataFrame([
                    ['True Negative', cm['tn']],
                    ['False Positive', cm['fp']],
                    ['False Negative', cm['fn']],
                    ['True Positive', cm['tp']]
                ], columns=['Type', 'Count'])
                cm_csv = self.output_dir / "confusion_matrix.csv"
                cm_df.to_csv(cm_csv, index=False)
                self.logger.info(f"混淆矩阵保存到: {cm_csv}")
        
        output_json = self.output_dir / "prediction_summary.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        # 导出按模型的 feature gating 统计
        try:
            gating_rows = []
            individual_predictions = prediction_results.get('individual_predictions', {}) or {}
            for model_name, model_result in individual_predictions.items():
                if not isinstance(model_result, dict):
                    continue
                summary = model_result.get('feature_gating_summary') or {}
                if not isinstance(summary, dict):
                    continue
                dataset_name = summary.get('dataset_name') or 'predict'
                feature_stats = summary.get('features') or {}
                if not isinstance(feature_stats, dict):
                    continue

                for feature_name, stat in feature_stats.items():
                    if not isinstance(stat, dict):
                        continue
                    gating_rows.append({
                        'dataset': dataset_name,
                        'model_name': model_name,
                        'feature_name': feature_name,
                        'mean': float(stat.get('mean', 0.0)),
                        'var': float(stat.get('var', 0.0)),
                        'n': int(stat.get('count', 0)),
                    })

            if gating_rows:
                gating_csv = self.output_dir / "feature_gating.csv"
                pd.DataFrame(gating_rows).to_csv(gating_csv, index=False)
                self.logger.info(f"Feature gating summary saved to: {gating_csv}")
        except Exception as e:
            self.logger.warning(f"Failed to save feature_gating.csv: {e}")
        
        self.logger.debug(f"预测摘要保存到: {output_json}")
        
        # 记录结果统计
        total_pairs = len(prediction_results['protein_pairs'])
        positive_count = int(ensemble_predictions['ensemble_prediction'].sum())
        avg_prob = float(ensemble_predictions['ensemble_probability'].mean())
        
        self.logger.debug(f"预测完成统计:")
        self.logger.debug(f"  总蛋白对数: {total_pairs}")
        self.logger.debug(f"  阳性预测数: {positive_count}")
        self.logger.debug(f"  平均预测概率: {avg_prob:.4f}") 

    def _create_temp_fasta_file(self, fasta_sequences: Dict[str, str], model_name: str) -> Path:
        """创建临时FASTA文件"""
        temp_fasta_file = self.output_dir / f"temp_{model_name}_sequences.fasta"
        
        with open(temp_fasta_file, 'w') as f:
            for protein_id, sequence in fasta_sequences.items():
                f.write(f">{protein_id}\n{sequence}\n")
        
        return temp_fasta_file


def main():
    """主入口函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python -m src.ensemble.ensemble_predict_engine <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        # 加载配置
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建预测引擎并运行
        engine = EnsemblePredictEngine(config)
        results = engine.run_ensemble_predict()
        
        print("预测完成！")
        print(f"结果保存到: {engine.output_dir}")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 