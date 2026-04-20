"""
Ensemble inference engine

Provides multi-model ensemble learning and inference utilities.
"""

import logging
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from src.utils.helpers import get_device
from src.inference import InferenceEngine


class WeightedEnsemble:
    """Simple weighted ensemble model."""
    def __init__(self, weights: List[float]):
        self.weights = np.array(weights)
        self.classes_ = np.array([0, 1])
        # Mock sklearn attributes for compatibility
        self.coef_ = np.array([weights])
        self.intercept_ = np.array([0.0])

    def fit(self, X, y):
        # No training needed
        pass

    def predict_proba(self, X):
        # X contains logits from sub-models
        # Weighted sum of logits
        final_logits = np.dot(X, self.weights)
        # Convert to probabilities using sigmoid
        final_prob = 1 / (1 + np.exp(-final_logits))
        return np.vstack([1 - final_prob, final_prob]).T
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def _extract_pooling_type_from_config(model_config: Dict[str, Any]) -> str:
    """
    Extract pooling type from a model configuration.

    Supports the new modular YAML architecture or the legacy configuration

    Args:
        model_config: model configuration dictionary

    Returns:
        str: pooling type ('avg', 'max', 'attention')
    """
    # Prefer reading from the combined modular configuration
    if 'model_config' in model_config and isinstance(model_config['model_config'], dict):
        pooling_method = model_config['model_config'].get('pooling', {}).get('method', 'average_pooling')
        method_mapping = {
            'average_pooling': 'avg',
            'max_pooling': 'max',
            'attention_pooling': 'attention'
        }
        return method_mapping.get(pooling_method, 'avg')

    # Next check for a YAML architecture file (legacy path)
    if 'model_architecture_file' in model_config and model_config['model_architecture_file']:
        # Read pooling configuration from the YAML architecture file
        try:
            import yaml
            from pathlib import Path
            
            architecture_file = model_config['model_architecture_file']
            config_path = Path(architecture_file)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                pooling_method = yaml_config.get('model_config', {}).get('pooling', {}).get('method', 'average_pooling')
                
                # Map YAML method to the pooling_type expected by data loaders
                method_mapping = {
                    'average_pooling': 'avg',
                    'max_pooling': 'max', 
                    'attention_pooling': 'attention'
                }
                
                return method_mapping.get(pooling_method, 'avg')
            else:
                logging.getLogger(__name__).warning(
                    f"Warning: YAML architecture file not found: {architecture_file}, using default pooling type"
                )
                return 'avg'
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Warning: Failed to read YAML architecture file: {e}, using default pooling type"
            )
            return 'avg'
    else:
        # fallback to legacy config format
        return model_config.get('pooling_type', 'avg')


class EnsembleInferenceEngine:
    """Ensemble inference engine with meta-learner support."""
    
    def __init__(self, config_paths: List[str], output_dir: str, logger=None):
        self.config_paths = config_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # store per-model inference engines
        self.inference_engines = []
        self.configs = []
        self.model_names = []
        
        # meta-learner and scaler
        self.meta_learner = None
        self.scaler = StandardScaler()
        
        # compute device
        self.device = get_device()
        
        self._load_configs_and_engines()
    
    def _load_configs_and_engines(self):
        """Load model configuration files and create inference engines."""
        self.logger.info(f"Loading {len(self.config_paths)} model configurations...")
        
        for i, config_path in enumerate(self.config_paths):
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # load config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # ensure config is set to inference mode
            config['mode'] = 'inference'
            
            # generate a model name
            model_name = f"model_{i+1}_{config_path.parent.name}"
            self.model_names.append(model_name)
            
            # create inference engine
            inference_engine = InferenceEngine(config, self.logger)
            
            # check that the model file exists
            model_path = config['model']['model_path']
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # load model
            inference_engine.load_model(model_path, device=self.device)
            
            self.inference_engines.append(inference_engine)
            self.configs.append(config)
            
            self.logger.info(f"Loaded model {model_name}: {model_path}")
        
        self.logger.info(f"All {len(self.inference_engines)} models loaded")
    
    def _create_datasets(self, config: Dict) -> Dict:
        """Create datasets for a given model configuration."""
        data_config = config['data']
        
        # handle different types of data configurations
        if data_config['use_sequence_data']:
            # check if this is CIS-type data
            is_cis_data = data_config.get('cis_type', False)
            
            # build a mapping of all required files
            all_files = {}
            if 'train_file' in data_config:
                all_files['train'] = data_config['train_file']
            
            # handle test files
            test_files = data_config.get('test_files', {})
            if test_files:
                all_files.update(test_files)
            else:
                # if no test_files provided, build mapping from individual fields
                if 'validation_file' in data_config:
                    all_files['val'] = data_config['validation_file']  # use 'val' as key
                if 'test_file' in data_config:
                    all_files['test'] = data_config['test_file']  # use 'test' as key
            
            if is_cis_data:
                # use CIS-specific data loader
                from src.data_processing.cis_data_loader import create_cis_data_loaders
                
                cis_config = {
                    'embedding_file': data_config['embedding_file'],
                    'train_file': data_config.get('train_file'),
                    'test_files': {k: v for k, v in all_files.items() if k != 'train'},
                    'batch_size': 32,  # use smaller batch size for inference
                    'cache_size': data_config.get('cache_size', 8000),
                    # parse LMDB at embedding source dimension
                    'embedding_dim': data_config.get('embedding_dim', config['model'].get('embedding_dim', 1280)),
                    'target_precision': data_config.get('target_precision', 'fp32')
                }
                
                # if there is a training file, include it in the test_files mapping
                if 'train' in all_files:
                    cis_config['test_files']['train'] = all_files['train']
                
                cis_batch_data = create_cis_data_loaders(cis_config)
                data_loaders = cis_batch_data['data_loaders']
                
                # extract Dataset objects from DataLoaders
                datasets = {}
                for key, loader in data_loaders.items():
                    if hasattr(loader, 'dataset'):
                        datasets[key] = loader.dataset
                
                return datasets
            else:
                # use smart-batch data loader
                from src.data_processing.smart_batch_loader import create_smart_batch_data_loaders
                
                compat_config = {
                    'embedding_file': data_config['embedding_file'],
                    'train_file': data_config.get('train_file'),
                    'test_files': {k: v for k, v in all_files.items() if k != 'train'},
                    'fasta_file': data_config['fasta_file'],
                    'batch_size': 32,
                    'cache_size': data_config.get('cache_size', 8000),
                    # parse LMDB at embedding source dimension
                    'embedding_dim': data_config.get('embedding_dim', config['model'].get('embedding_dim', 1280)),
                    'max_length': data_config.get('max_length', 1024),
                    'pooling_type': _extract_pooling_type_from_config(config['model']),
                    'target_precision': data_config.get('target_precision'),
                    'cis_type': data_config.get('cis_type', False)
                }
                
                # 如果有训练文件，添加到test_files中
                if 'train' in all_files:
                    compat_config['test_files']['train'] = all_files['train']
                
                smart_batch_data = create_smart_batch_data_loaders(compat_config)
                data_loaders = smart_batch_data['data_loaders']
                
                # extract Dataset objects from DataLoaders
                datasets = {}
                for key, loader in data_loaders.items():
                    if hasattr(loader, 'dataset'):
                        datasets[key] = loader.dataset
                
                return datasets
        else:
            raise NotImplementedError("Ensemble inference currently supports sequence data only")
    
    def collect_logits(self, mode: str = 'logits') -> Dict[str, Dict]:
        """
        Collect logits or features from each model.

        Args:
            mode: 'logits' or 'features' — collect logits or penultimate-layer features

        Returns:
            Dict: mapping of datasets to per-model logits/features
        """
        if mode not in ['logits', 'features']:
            raise ValueError("mode must be 'logits' or 'features'")

        self.logger.info(f"Starting to collect {mode} from each model...")
        all_data = {}
        
        for i, (engine, config, model_name) in enumerate(zip(self.inference_engines, self.configs, self.model_names)):
            self.logger.info(f"Creating dedicated datasets for model {model_name}...")
            
            # create model-specific datasets for each model
            model_datasets = self._create_datasets(config)
            
            self.logger.info(f"Model {model_name} datasets: {list(model_datasets.keys())}")
            
            # collect data for each dataset
            for dataset_name, dataset in model_datasets.items():
                if dataset_name not in all_data:
                    all_data[dataset_name] = {}
                
                self.logger.info(f"Processing dataset {dataset_name} with model {model_name}")
                
                # evaluate dataset using the inference engine
                if mode == 'logits':
                    results = engine.evaluate_datasets(
                        {dataset_name: dataset},
                        show_progress=True,
                        save_results=False,
                        return_logits=True
                    )
                else:  # features
                    results = engine.evaluate_datasets(
                        {dataset_name: dataset},
                        show_progress=True,
                        save_results=False,
                        return_logits=False,
                        return_features=True
                    )
                
                if dataset_name in results:
                    result = results[dataset_name]
                    if mode == 'logits':
                        all_data[dataset_name][model_name] = {
                            'logits': result.get('predictions', []),
                            'labels': result.get('true_labels', []),
                            'protein_ids': result.get('protein_ids', [])
                        }
                    else:  # features
                        all_data[dataset_name][model_name] = {
                            'features': result.get('features', []),
                            'labels': result.get('true_labels', []),
                            'protein_ids': result.get('protein_ids', [])
                        }
                    
                    data_count = len(all_data[dataset_name][model_name]['labels'])
                    self.logger.info(f"  Collected {data_count} samples of {mode}")
        
        # validate data consistency
        self._validate_data_consistency(all_data, mode)
        
        return all_data
    
    def _validate_data_consistency(self, all_data: Dict[str, Dict], mode: str = 'logits') -> None:
        """
        Validate that data from different models are consistent.

        Args:
            all_data: dictionary containing datasets and per-model data
            mode: 'logits' or 'features'
        """
        self.logger.info(f"Validating {mode} consistency...")
        
        for dataset_name, model_data in all_data.items():
            if not model_data:
                continue
            
            # check that all models have the same sample counts
            if mode == 'logits':
                sample_counts = {model_name: len(data['logits']) for model_name, data in model_data.items()}
            else:  # features
                sample_counts = {model_name: len(data['features']) for model_name, data in model_data.items()}
            
            unique_counts = set(sample_counts.values())
            
            if len(unique_counts) > 1:
                self.logger.warning(f"Dataset {dataset_name} has inconsistent sample counts across models: {sample_counts}")
                # truncate to the smallest length
                min_count = min(sample_counts.values())
                self.logger.warning(f"Truncating to minimum length: {min_count}")
                
                for model_name, data in model_data.items():
                    if mode == 'logits':
                        data['logits'] = data['logits'][:min_count]
                    else:  # features
                        data['features'] = data['features'][:min_count]
                    data['labels'] = data['labels'][:min_count]
                    if data['protein_ids']:
                        data['protein_ids'] = data['protein_ids'][:min_count]
            else:
                self.logger.info(f"Dataset {dataset_name}: all models have consistent sample counts ({list(unique_counts)[0]} samples)")
    
    def train_meta_learner(self, train_data: Dict[str, Dict], mode: str = 'logits', model_type: str = 'linear') -> None:
        """
        Train the meta-learner.

        Args:
            train_data: dictionary with training data
            mode: 'logits' or 'features'
            model_type: 'linear' or 'weighted'
        """
        if mode not in ['logits', 'features']:
            raise ValueError("mode must be 'logits' or 'features'")
        
        self.logger.info(f"Starting meta-learner training (mode={mode}, type={model_type})...")
        
        # collect training data - allow using validation as training (for CV)
        data = None
        data_c1 = None
        data_c3 = None
        
        # Try to find c1, c2, c3
        if 'train' in train_data:
            data_c1 = train_data['train']
        elif 'c1' in train_data:
            data_c1 = train_data['c1']
            
        if 'c3' in train_data:
            data_c3 = train_data['c3']
        elif 'test' in train_data:
            data_c3 = train_data['test']

        # Determine optimization target (data)
        # Priority: c2 > val > train
        if 'c2' in train_data:
            # Explicitly use c2 (validation set) if available
            data = train_data['c2']
            self.logger.info("Using dataset 'c2' (validation set) as training data for weight optimization")
        elif 'val' in train_data:
            # In cross-validation, the validation set is used as training data
            data = train_data['val']
            self.logger.info("Using validation set as training data (cross-validation mode)")
        elif 'train' in train_data:
            data = train_data['train']
            self.logger.warning("Using 'train' dataset for weight optimization (c2/val not found). This may lead to overfitting!")
        else:
            # Try to find any available dataset
            available_keys = list(train_data.keys())
            if available_keys:
                data = train_data[available_keys[0]]
                self.logger.info(f"Using dataset '{available_keys[0]}' as training data")
            else:
                raise ValueError("No training or validation data found")
        
        # Helper to extract features/labels for a dataset
        def extract_features_labels(dataset_dict):
            if dataset_dict is None:
                return None, None
            X = []
            y = None
            for model_name in self.model_names:
                if model_name in dataset_dict:
                    if mode == 'logits':
                        features = np.array(dataset_dict[model_name]['logits'])
                    else:
                        features = np.array(dataset_dict[model_name]['features'])
                    X.append(features)
                    if y is None:
                        y = np.array(dataset_dict[model_name]['labels'])
            if not X:
                return None, None
            return np.column_stack(X), y

        # organize features and labels for optimization target (c2)
        X_train, y_train = extract_features_labels(data)
        
        # Extract for c1 and c3 if available
        X_c1, y_c1 = extract_features_labels(data_c1)
        X_c3, y_c3 = extract_features_labels(data_c3)
        
        if X_train is None:
            raise ValueError("No valid training data collected")
        
        self.logger.info(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
        
        # show per-model feature ranges
        feature_ranges = []
        for i, model_name in enumerate(self.model_names):
            if model_name in data:
                features = X_train[:, i]
                feature_ranges.append(f"{features.min():.3f}-{features.max():.3f}")
        self.logger.info(f"Per-model {mode} ranges: {feature_ranges}")
        
        # scale features
        if model_type == 'weighted':
            # For weighted ensemble, we don't want to scale logits/probs
            self.scaler = StandardScaler(with_mean=False, with_std=False)
        else:
            self.scaler = StandardScaler()
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == 'weighted':
            if len(self.model_names) != 2:
                self.logger.warning(f"Weighted ensemble currently optimized for 2 models, but got {len(self.model_names)}. Using equal weights.")
                best_weights = [1.0 / len(self.model_names)] * len(self.model_names)
            else:
                self.logger.info("Searching for best weights for 2 models (0.0 to 1.0)...")
                best_ap = -1
                best_w = 0.5
                
                # Convert logits to probs for optimization if mode is logits
                if mode == 'logits':
                    # Use logits directly for weighted sum
                    logits_train = X_train
                    logits_c1 = X_c1 if X_c1 is not None else None
                    logits_c3 = X_c3 if X_c3 is not None else None
                else:
                    self.logger.warning("Weighted ensemble with mode='features' might not work as expected. Assuming features are logits.")
                    logits_train = X_train
                    logits_c1 = X_c1
                    logits_c3 = X_c3

                # Stage 1: Coarse search (0.0 to 1.0, step 0.01)
                self.logger.info("Stage 1: Coarse search (step 0.01)")
                for w in np.linspace(0, 1, 101):
                    # w for model 0, (1-w) for model 1
                    # Weighted sum of logits
                    l = w * logits_train[:, 0] + (1-w) * logits_train[:, 1]
                    # Convert to probability
                    p = 1 / (1 + np.exp(-l))
                    try:
                        ap = average_precision_score(y_train, p)
                    except Exception:
                        ap = 0.0
                        
                    if ap > best_ap:
                        best_ap = ap
                        best_w = w
                
                self.logger.info(f"Stage 1 best weight: {best_w:.2f}, C2 AP: {best_ap:.4f}")

                # Stage 2: Fine search (best_w +/- 0.01, step 0.0001)
                self.logger.info("Stage 2: Fine search (step 0.0001)")
                
                # Define search range around best_w
                w_min = max(0.0, best_w - 0.01)
                w_max = min(1.0, best_w + 0.01)
                
                # 201 steps to cover +/- 0.01 with 0.0001 precision
                for w in np.linspace(w_min, w_max, 201):
                    l = w * logits_train[:, 0] + (1-w) * logits_train[:, 1]
                    p = 1 / (1 + np.exp(-l))
                    try:
                        ap = average_precision_score(y_train, p)
                    except Exception:
                        ap = 0.0
                        
                    if ap > best_ap:
                        best_ap = ap
                        best_w = w
                
                # Calculate AP for C1 and C3 with best weight
                ap_c1_str = "N/A"
                if logits_c1 is not None:
                    l_c1 = best_w * logits_c1[:, 0] + (1-best_w) * logits_c1[:, 1]
                    p_c1 = 1 / (1 + np.exp(-l_c1))
                    try:
                        ap_c1 = average_precision_score(y_c1, p_c1)
                        ap_c1_str = f"{ap_c1:.4f}"
                    except:
                        pass

                ap_c3_str = "N/A"
                if logits_c3 is not None:
                    l_c3 = best_w * logits_c3[:, 0] + (1-best_w) * logits_c3[:, 1]
                    p_c3 = 1 / (1 + np.exp(-l_c3))
                    try:
                        ap_c3 = average_precision_score(y_c3, p_c3)
                        ap_c3_str = f"{ap_c3:.4f}"
                    except:
                        pass

                self.logger.info(f"Best weight for {self.model_names[0]}: {best_w:.4f}")
                self.logger.info(f"Performance: C1 AP: {ap_c1_str}, C2 AP: {best_ap:.4f}, C3 AP: {ap_c3_str}")
                best_weights = [best_w, 1.0 - best_w]
            
            self.meta_learner = WeightedEnsemble(best_weights)
        else:
            # train logistic regression
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            self.meta_learner.fit(X_train_scaled, y_train)
        
        self.logger.info("Meta-learner training complete!")
        
        # record per-model weights
        weights = {name: float(self.meta_learner.coef_[0][i]) 
                  for i, name in enumerate(self.model_names)}
        
        self.logger.info(f"Per-model weights: {weights}")
        
        # save the meta-learner and weights
        self._save_meta_learner(weights, mode)
    
    def _save_meta_learner(self, weights: Dict[str, float], training_mode: str = 'logits') -> None:
        """
        Save the meta-learner and weight information.

        Args:
            weights: dictionary of per-model weights
            training_mode: training mode ('logits' or 'features')
        """
        import pickle
        import json
        
        # 保存元学习器
        meta_learner_path = self.output_dir / "meta_learner.pkl"
        meta_learner_data = {
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'model_names': self.model_names,
            'training_mode': training_mode
        }
        with open(meta_learner_path, 'wb') as f:
            pickle.dump(meta_learner_data, f)
        
        # 保存标准化器
        scaler_path = self.output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存权重信息
        weights_info = {
            'weights': weights,
            'model_names': self.model_names,
            'intercept': float(self.meta_learner.intercept_[0]),
            'coefficients': [float(w) for w in self.meta_learner.coef_[0]]
        }
        
        weights_path = self.output_dir / "meta_learner_weights.json"
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Meta-learner saved to: {meta_learner_path}")
        self.logger.info(f"Scaler saved to: {scaler_path}")
        self.logger.info(f"Weights info saved to: {weights_path}")
    
    def generate_inference_config(self, training_mode: str = 'logits') -> Dict[str, Any]:
        """
        Generate an ensemble inference configuration dictionary.

        Args:
            training_mode: training mode ('logits' or 'features')

        Returns:
            Dict: ensemble inference configuration
        """
        # use the first config as base (data configuration)
        base_config = self.configs[0].copy()
        
        # create ensemble inference configuration
        ensemble_config = {
            'mode': 'ensemble_inference',
            'ensemble': {
                'meta_learner_path': str(self.output_dir / "meta_learner.pkl"),
                'scaler_path': str(self.output_dir / "scaler.pkl"),
                'models': []
            },
            'data': base_config['data'].copy(),
            'inference': base_config.get('inference', {}).copy(),
            'output': {
                'generate_confusion_matrix': True,
                'save_predictions': True,
                'output_dir': str(self.output_dir)
            }
        }
        
        # add per-model configuration info
        for i, (model_name, config) in enumerate(zip(self.model_names, self.configs)):
            model_info = {
                'model_id': i + 1,
                'model_name': model_name,
                'model_path': config['model']['model_path'],
                'config_path': self.config_paths[i],
                'weight': float(self.meta_learner.coef_[0][i]) if self.meta_learner else 1.0,
                'model_config': config['model'].copy(),
                'data_config': config['data'].copy()
            }
            ensemble_config['ensemble']['models'].append(model_info)
        
        # add meta-learner information
        if self.meta_learner:
            ensemble_config['ensemble']['meta_learner_info'] = {
                'intercept': float(self.meta_learner.intercept_[0]),
                'weights': [float(w) for w in self.meta_learner.coef_[0]],
                'model_names': self.model_names,
                'training_mode': training_mode,
                'training_score': {
                    'train_auc': float(getattr(self, '_train_auc', 0.0)),
                    'train_ap': float(getattr(self, '_train_ap', 0.0))
                }
            }
        
        return ensemble_config
    
    def ensemble_predict(self, test_data: Dict[str, Dict], mode: str = 'logits') -> Dict[str, pd.DataFrame]:
        """
        Use the trained meta-learner to perform ensemble prediction.

        Args:
            test_data: dictionary of test data per dataset
            mode: 'logits' or 'features'

        Returns:
            Dict: mapping of dataset names to prediction DataFrames
        """
        if mode not in ['logits', 'features']:
            raise ValueError("mode must be 'logits' or 'features'")

        if self.meta_learner is None:
            raise RuntimeError("Meta-learner has not been trained. Call train_meta_learner first.")

        self.logger.info(f"Starting ensemble prediction using {mode}...")
        
        results = {}
        
        for dataset_name, model_data in test_data.items():
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            # assemble features
            X_test = []
            y_test = None
            protein_ids = None
            
            for model_name in self.model_names:
                if model_name in model_data:
                    if mode == 'logits':
                        features = np.array(model_data[model_name]['logits'])
                    else:  # features
                        features = np.array(model_data[model_name]['features'])
                    X_test.append(features)
                    
                    if y_test is None:
                        y_test = np.array(model_data[model_name]['labels'])
                    
                    if protein_ids is None and 'protein_ids' in model_data[model_name]:
                        protein_ids = model_data[model_name]['protein_ids']
            
            if len(X_test) == 0:
                self.logger.warning(f"No valid model data found for dataset {dataset_name}")
                continue
            
            # combine features
            X_test = np.column_stack(X_test)
            
            # standardize features
            X_test_scaled = self.scaler.transform(X_test)
            
            # predict with meta-learner
            ensemble_proba = self.meta_learner.predict_proba(X_test_scaled)[:, 1]
            ensemble_pred = self.meta_learner.predict(X_test_scaled)
            
            # create result DataFrame
            result_df = pd.DataFrame({
                'ensemble_prediction': ensemble_proba,
                'ensemble_label': ensemble_pred,
                'true_label': y_test
            })

            # Add protein ID columns if available
            if protein_ids is not None and len(protein_ids) == len(y_test):
                p1_list = []
                p2_list = []
                for pid in protein_ids:
                    if isinstance(pid, (list, tuple)) and len(pid) >= 2:
                        p1_list.append(pid[0])
                        p2_list.append(pid[1])
                    elif isinstance(pid, str) and "_" in pid:
                        parts = pid.split("_", 1)
                        p1_list.append(parts[0])
                        p2_list.append(parts[1])
                    else:
                        p1_list.append(str(pid))
                        p2_list.append("")
                
                result_df['protein1'] = p1_list
                result_df['protein2'] = p2_list
            
            # add per-model predictions
            for i, model_name in enumerate(self.model_names):
                if model_name in model_data:
                    if mode == 'logits':
                        # convert logits to probabilities
                        model_logits = np.array(model_data[model_name]['logits'])
                        model_proba = 1 / (1 + np.exp(-model_logits))
                    else:  # features
                        # in features mode we only have features, not direct predictions
                        # a simple linear combination could be used as a baseline
                        model_proba = X_test[:, i]  # use feature value as prediction
                    
                    result_df[f'{model_name}_prediction'] = model_proba
            
            results[dataset_name] = result_df
            
            # compute performance metrics
            auc = roc_auc_score(y_test, ensemble_proba)
            ap = average_precision_score(y_test, ensemble_proba)
            acc = accuracy_score(y_test, ensemble_pred)
            f1 = f1_score(y_test, ensemble_pred)
            
            self.logger.info(f"{dataset_name} ensemble performance: AUC={auc:.4f}, AP={ap:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        
        return results
    
    def save_results(self, ensemble_results: Dict[str, pd.DataFrame]) -> None:
        """Save ensemble results to disk."""
        self.logger.info("Saving results...")
        
        # save prediction CSVs
        for dataset_name, df in ensemble_results.items():
            output_file = self.output_dir / f"{dataset_name}_ensemble_predictions.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {dataset_name} predictions to: {output_file}")
        
        # create ensemble inference config file
        self._create_ensemble_inference_config(self.output_dir / "meta_learner.pkl")
        
        # save meta-learner weights
        weights_dict = {}
        for i, model_name in enumerate(self.model_names):
            weights_dict[model_name] = float(self.meta_learner.coef_[0][i])
        
        weights_file = self.output_dir / "meta_learner_weights.json"
        with open(weights_file, 'w', encoding='utf-8') as f:
            json.dump(weights_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Meta-learner weights saved to: {weights_file}")
        
        # 保存元学习器模型
        meta_learner_file = self.output_dir / "meta_learner.pkl"
        with open(meta_learner_file, 'wb') as f:
            pickle.dump({
                'meta_learner': self.meta_learner,
                'scaler': self.scaler,
                'model_names': self.model_names
            }, f)
        
        self.logger.info(f"Meta-learner model saved to: {meta_learner_file}")
        
        # save performance summary
        summary = {}
        for dataset_name, df in ensemble_results.items():
            y_true = df['true_label'].values
            y_proba = df['ensemble_probability'].values
            y_pred = df['ensemble_prediction'].values
            
            summary[dataset_name] = {
                'sample_count': len(df),
                'auc': float(roc_auc_score(y_true, y_proba)),
                'aupr': float(average_precision_score(y_true, y_proba)),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred))
            }
        
        summary_file = self.output_dir / "ensemble_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Performance summary saved to: {summary_file}")
    
    def _create_ensemble_inference_config(self, meta_learner_file: Path) -> None:
        """Create and save an ensemble inference configuration file."""
        self.logger.info("Creating ensemble inference configuration file...")
        
        # 使用第一个配置作为基准（数据配置）
        base_config = self.configs[0].copy()
        
        # 创建集成推理配置
        ensemble_config = {
            'mode': 'ensemble_inference',
            'ensemble': {
                'meta_learner_path': str(meta_learner_file),
                'models': []
            },
            'data': base_config['data'].copy(),
            'inference': base_config.get('inference', {}).copy(),
            'output': {
                'generate_confusion_matrix': True,
                'save_predictions': True,
                'output_dir': str(self.output_dir)
            }
        }
        
        # 添加各个模型的配置信息
        for i, (model_name, config) in enumerate(zip(self.model_names, self.configs)):
            model_info = {
                'model_id': i + 1,
                'model_name': model_name,
                'model_path': config['model']['model_path'],
                'config_path': self.config_paths[i],
                'weight': float(self.meta_learner.coef_[0][i]),
                'model_config': config['model'].copy(),
                'data_config': config['data'].copy()
            }
            ensemble_config['ensemble']['models'].append(model_info)
        
        # 添加元学习器信息
        ensemble_config['ensemble']['meta_learner_info'] = {
            'intercept': float(self.meta_learner.intercept_[0]),
            'weights': [float(w) for w in self.meta_learner.coef_[0]],
            'model_names': self.model_names,
            'training_mode': self.mode,  # 添加训练时使用的模式
            'training_score': {
                'train_auc': float(getattr(self, '_train_auc', 0.0)),
                'train_ap': float(getattr(self, '_train_ap', 0.0))
            }
        }
        
        # save ensemble inference config
        ensemble_config_file = self.output_dir / "ensemble_inference_config.yaml"
        import yaml
        with open(ensemble_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(ensemble_config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        self.logger.info(f"Ensemble inference config saved to: {ensemble_config_file}")
        self.logger.info("You can run ensemble inference with this config:")
        self.logger.info(f"  python sepal-single.py --mode ensemble_inference --config {ensemble_config_file}")
    
    def run_ensemble_inference(self) -> Dict[str, pd.DataFrame]:
        """Run the full ensemble inference workflow."""
        # 1. collect logits from each model
        all_logits = self.collect_logits()
        
        # 2. train the meta-learner
        self.train_meta_learner(all_logits)
        
        # 3. perform ensemble prediction
        ensemble_results = self.ensemble_predict(all_logits)
        
        # 4. save results
        self.save_results(ensemble_results)
        
        return ensemble_results
    
    def ensemble_predict_validation_only(self, val_file_path: str) -> Optional[pd.DataFrame]:
        """Perform ensemble prediction only for a validation file (for cross-validation)."""
        self.logger.info(f"Running validation-only ensemble prediction for: {val_file_path}")
        
        # create temporary configs that only include the validation set
        temp_configs = []
        for i, config in enumerate(self.configs):
            temp_config = config.copy()
            # 修改数据配置，只包含验证集
            temp_config['data']['test_files'] = {'val': val_file_path}
            # 移除训练文件，避免冲突
            if 'train_file' in temp_config['data']:
                del temp_config['data']['train_file']
            
            temp_configs.append(temp_config)
        
        # 收集验证集的logits
        val_logits = {}
        for i, (engine, config, model_name) in enumerate(zip(self.inference_engines, temp_configs, self.model_names)):
            # create validation dataset per model
            model_datasets = self._create_datasets(config)
            
            if 'val' in model_datasets:
                self.logger.info(f"Processing validation set with model {model_name}...")
                
                # 使用推理引擎评估验证集，返回logits
                results = engine.evaluate_datasets(
                    {'val': model_datasets['val']},
                    show_progress=True,
                    save_results=False,
                    return_logits=True
                )
                
                if 'val' in results:
                    result = results['val']
                    if 'val' not in val_logits:
                        val_logits['val'] = {}
                    
                    val_logits['val'][model_name] = {
                        'logits': result.get('predictions', []),
                        'labels': result.get('true_labels', []),
                        'protein_ids': result.get('protein_ids', [])
                    }
        
        # use trained meta-learner for prediction
        if 'val' in val_logits and val_logits['val']:
            ensemble_results = self.ensemble_predict(val_logits)
            if 'val' in ensemble_results:
                return ensemble_results['val']
            else:
                self.logger.warning("Ensemble prediction results contain no validation data")
                return None
        else:
            self.logger.warning("Unable to obtain validation logits")
            return None 