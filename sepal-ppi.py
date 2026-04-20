#!/usr/bin/env python3
"""
SEPAL-PPI: Protein-Protein Interaction Prediction using ESM Embeddings

Main script for training and evaluating protein-protein interaction prediction models.
This script supports both training and inference modes with YAML configuration.

Usage:
    python sepal.py [options]
    python sepal.py --mode inference --config inference_config.yaml

    SEPAL-PPI
    Version: 0.9.3
"""

import sys
import argparse
import time
import logging
import lmdb
import json
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.adam import Adam
import json
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm

# 抑制sklearn版本警告和ROC AUC警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

# 精确抑制 PyTorch 关于只读 NumPy 数组转换为 Tensor 的 UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*",
)

# Import our modules
from src.utils.helpers import set_random_seed, get_device, format_metrics
from src.utils.logger import setup_logger, create_output_directory, format_time, format_memory
from src.utils.config_manager import ConfigManager
from src.utils.hydra_config_manager import HydraConfigManager, create_hydra_config_manager
from src.data_processing import create_default_loaders, get_default_config
from src.data_processing.sequence_dataset import create_sequence_dataset_loaders, get_sequence_default_config
from src.models import create_avg_pool_mlp, train_model, get_default_training_config, create_yaml_model
from src.evaluation import evaluate_multiple_datasets, print_multiple_results, create_evaluation_summary
from src.inference import InferenceEngine
from src.training.training_runner import run_training_mode
from src.ensemble.ensemble_runner import (
    run_ensemble_inference_mode,
    run_ensemble_predict_mode,
    run_ensemble_training_mode,
)


def _extract_pooling_type_from_config(model_config: Dict[str, Any]) -> str:
    """
    从模型配置中提取池化类型
    
    支持新的YAML架构配置和传统配置格式
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        str: 池化类型 ('avg', 'max', 'attention')
    """
    # 优先从组合后的模块化配置中读取
    if 'model_config' in model_config and isinstance(model_config['model_config'], dict):
        pooling_method = model_config['model_config'].get('pooling', {}).get('method', 'average_pooling')
        method_mapping = {
            'average_pooling': 'avg',
            'max_pooling': 'max',
            'attention_pooling': 'attention'
        }
        return method_mapping.get(pooling_method, 'avg')

    # 其次检查是否使用YAML架构配置（旧路径）
    if 'model_architecture_file' in model_config and model_config['model_architecture_file']:
        # 从YAML架构文件中读取池化配置
        try:
            import yaml
            from pathlib import Path
            
            architecture_file = model_config['model_architecture_file']
            config_path = Path(architecture_file)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                pooling_method = yaml_config.get('model_config', {}).get('pooling', {}).get('method', 'average_pooling')
                
                # 将YAML中的method映射到data loader期望的pooling_type
                method_mapping = {
                    'average_pooling': 'avg',
                    'max_pooling': 'max', 
                    'attention_pooling': 'attention'
                }
                
                return method_mapping.get(pooling_method, 'avg')
            else:
                print(f"Warning: YAML architecture file not found: {architecture_file}, using default pooling type")
                return 'avg'
                
        except Exception as e:
            print(f"Warning: Failed to read YAML architecture file: {e}, using default pooling type")
            return 'avg'
    else:
        # 使用传统配置格式
        return model_config.get('pooling_type', 'avg')


def _extract_interaction_type_from_config(model_config: Dict[str, Any]) -> str:
    """
    从模型配置中提取交互类型
    
    支持新的YAML架构配置和传统配置格式
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        str: 交互类型 ('hadamard', 'concatenation', 'fast_compact_bilinear', etc.)
    """
    # 检查是否使用YAML架构配置
    if 'model_architecture_file' in model_config and model_config['model_architecture_file']:
        # 从YAML架构文件中读取交互配置
        try:
            import yaml
            from pathlib import Path
            
            architecture_file = model_config['model_architecture_file']
            config_path = Path(architecture_file)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                interaction_operation = yaml_config.get('model_config', {}).get('post_pooling', {}).get('operation', 'hadamard_product')
                
                # 将YAML中的operation映射到传统配置期望的interaction_type
                operation_mapping = {
                    'hadamard_product': 'hadamard',
                    'concatenation': 'concatenation',
                    'difference': 'difference',
                    'cosine': 'cosine',
                    'outer_product': 'outer_product',
                    'fast_compact_bilinear': 'fast_compact_bilinear'
                }
                
                return operation_mapping.get(interaction_operation, 'hadamard')
            else:
                print(f"Warning: YAML architecture file not found: {architecture_file}, using default interaction type")
                return 'hadamard'
                
        except Exception as e:
            print(f"Warning: Failed to read YAML architecture file: {e}, using default interaction type")
            return 'hadamard'
    else:
        # 使用传统配置格式
        return model_config.get('interaction_type', 'hadamard')


def _check_bucketed_lmdb(lmdb_path: str, logger) -> bool:
    """
    Check if LMDB is properly bucketed by looking for metadata keys
    
    Args:
        lmdb_path (str): Path to LMDB file or directory
        logger: Logger instance
        
    Returns:
        bool: True if bucketed, False otherwise
    """
    try:
        lmdb_file = Path(lmdb_path)
        
        # Check if it's a directory with data.mdb (standard LMDB database)
        if lmdb_file.is_dir() and (lmdb_file / "data.mdb").exists():
            # This is a single LMDB database directory (new format)
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        elif lmdb_file.is_dir():
            # Directory with multiple buckets (old format) - look for bucket_0.lmdb
            bucket_0_path = lmdb_file / "bucket_0.lmdb"
            if bucket_0_path.exists():
                lmdb_path = str(bucket_0_path)
                env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            else:
                # Look for any bucket file
                bucket_files = list(lmdb_file.glob("bucket_*.lmdb"))
                if bucket_files:
                    lmdb_path = str(bucket_files[0])
                    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
                else:
                    logger.warning(f"No bucketed LMDB files found in directory {lmdb_file}")
                    return False
        elif lmdb_file.is_file() and lmdb_file.suffix == '.lmdb':
            # Single LMDB file (less common)
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        else:
            logger.warning(f"Invalid LMDB path: {lmdb_path}")
            return False
        
        try:
            with env.begin() as txn:
                # Check for required metadata keys
                required_keys = [b'_bucket_info', b'_precision_info', b'_creation_info']
                found_keys = []
                
                for key in required_keys:
                    if txn.get(key):
                        found_keys.append(key.decode())
                
                if len(found_keys) == len(required_keys):
                    logger.debug("Detected complete bucketed LMDB metadata")
                    
                    # Log bucket info
                    bucket_info = txn.get(b'_bucket_info')
                    if bucket_info:
                        import json
                        bucket_data = json.loads(bucket_info.decode())
                        logger.debug(f"Bucketing strategy: {bucket_data.get('strategy', 'unknown')}")
                        logger.debug(f"Number of buckets: {bucket_data.get('num_buckets', 'unknown')}")
                        logger.debug(f"Created at: {bucket_data.get('created_at', 'unknown')}")
                    
                    return True
                else:
                    logger.warning(f"Incomplete LMDB metadata, found: {found_keys}")
                    return False
        
        finally:
            env.close()
    
    except Exception as e:
        logger.warning(f"Failed to check LMDB bucketing status: {e}")
        return False




## run_ensemble_inference_mode 已迁移到 src/ensemble/ensemble_runner.py
## run_ensemble_predict_mode 已迁移到 src/ensemble/ensemble_runner.py


def run_inference_mode(config: Dict[str, Any], output_dir: str, logger) -> Dict[str, Any]:
    """
    Run inference mode
    
    Args:
        config: Complete inference configuration
        output_dir: Output directory  
        logger: Logger instance
    
    Returns:
        Inference results
    """
    logger.debug("=== Inference mode ===")
    
    # Get device
    device = get_device()
    logger.info(f"Compute device: {device}")
    
    # Initialize inference engine
    inference_engine = InferenceEngine(config, logger)
    
    # Load model
    model_path = config['model']['model_path']
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    inference_engine.load_model(model_path, device=device)
    
    # Prepare data
    data_config = config['data']
    if data_config['use_sequence_data']:
        # Check if embedding file is bucketed (skip for cis data)
        embedding_file = data_config['embedding_file']
        is_cis_data = data_config.get('cis_type', False)
        
        if not is_cis_data and not _check_bucketed_lmdb(embedding_file, logger):
            logger.error("Detected a non-bucketed LMDB file!")
            logger.error("Please bucket the LMDB using the preprocessing script first:")
            logger.error(f"python emb_tools/preprocess_bucketed_lmdb.py --source-lmdb {embedding_file} --output-dir bucketed_embeddings/")
            raise ValueError("LMDB must be bucketed beforehand")
        elif is_cis_data:
            logger.info("Detected CIS-level data configuration, skipping bucketing check")
        
            # Process test file configuration
        test_files = data_config.get('test_files', {})
        if not test_files:
            # If no test_files, build from individual fields
            test_files = {}
            if 'validation_file' in data_config and data_config['validation_file']:
                test_files['c2'] = data_config['validation_file']
            if 'test_file' in data_config and data_config['test_file']:
                test_files['c3'] = data_config['test_file']
            
            if not test_files:
                raise ValueError("No test files found in inference configuration")
            
            logger.info(f"Built test files mapping from individual fields: {test_files}")
        
        if is_cis_data:
            # 使用CIS专用数据加载器
            from src.data_processing.cis_data_loader import create_cis_data_loaders
            
            # CIS数据配置（简化）
            cis_config = {
                'embedding_file': embedding_file,
                'test_files': test_files,
                'batch_size': config['inference']['batch_size'],
                'cache_size': data_config.get('cache_size', 8000),
                # 以数据源维度解析LMDB
                'embedding_dim': data_config.get('embedding_dim', config['model'].get('embedding_dim', 1280)),
                'target_precision': data_config.get('target_precision', 'fp32'),
                'seed': config.get('training', {}).get('seed', 42)  # 传递种子
            }
            
            # Create CIS dataloaders
            try:
                cis_batch_data = create_cis_data_loaders(cis_config)
                data_loaders = cis_batch_data['data_loaders']
                embedding_cache = cis_batch_data['embedding_cache']
            except Exception as e:
                logger.error(f"Failed to create CIS data loader during inference: {e}")
                logger.error("Please check:")
                logger.error(f"  1. Whether the LMDB file exists: {embedding_file}")
                logger.error(f"  2. Whether the test files exist: {test_files}")
                logger.error(f"  3. Whether embedding_dim is correctly configured: {config['model'].get('embedding_dim')}")
                raise RuntimeError(f"Failed to initialize CIS data loader during inference: {e}") from e
            
            logger.info(f"Inference CIS cache stats: {embedding_cache.get_cache_stats()}")
            logger.info(f"Inference pooling type: {cis_batch_data['pooling_type']}")
            
            # For CIS loader, need to extract Dataset from DataLoader  
            test_datasets = {}
            for key, loader in data_loaders.items():
                if hasattr(loader, 'dataset'):
                    test_datasets[key] = loader.dataset
                else:
                    logger.warning(f"{key} dataloader has no 'dataset' attribute, skipping")
            
            logger.debug(f"Test datasets extracted from CIS DataLoader: {list(test_datasets.keys())}")
        else:
            # 使用智能批次数据加载器
            from src.data_processing.smart_batch_loader import create_smart_batch_data_loaders
            
            # Load test datasets
            # Create unified config for all test files
            compat_config = {
                'embedding_file': embedding_file,
                'test_files': test_files,
                'fasta_file': data_config['fasta_file'],
                'batch_size': config['inference']['batch_size'],
                'cache_size': data_config.get('cache_size', 8000),
                # 以数据源维度解析LMDB
                'embedding_dim': data_config.get('embedding_dim', config['model'].get('embedding_dim', 1280)),
                'max_length': data_config.get('max_length', 1024),
                'pooling_type': _extract_pooling_type_from_config(config['model']),  # 提取池化类型
                'target_precision': data_config.get('target_precision'),
                'cis_type': data_config.get('cis_type', False),  # 添加cis_type配置
                'seed': config.get('training', {}).get('seed', 42)  # 传递种子
            }
            
            # Create smart-batch dataloaders
            smart_batch_data = create_smart_batch_data_loaders(compat_config)
            data_loaders = smart_batch_data['data_loaders']
            embedding_cache = smart_batch_data['embedding_cache']
            
            logger.info(f"Inference cache stats: {embedding_cache.get_cache_stats()}")
            logger.info(f"Inference pooling type: {smart_batch_data['pooling_type']}")
            
            # For smart-batch loader, extract Dataset from DataLoader
            test_datasets = {}
            for key, loader in data_loaders.items():
                if hasattr(loader, 'dataset'):
                    test_datasets[key] = loader.dataset
                else:
                    logger.warning(f"{key} dataloader has no 'dataset' attribute, skipping")
            
            logger.debug(f"Test datasets extracted from DataLoader: {list(test_datasets.keys())}")
        
        # Run inference
        results = inference_engine.evaluate_datasets(
            test_datasets, 
            show_progress=True, 
            save_results=True,
            output_dir=output_dir
        )
        
        # Log results
        logger.info("=== Inference results ===")
        for dataset_name, metrics in results.items():
            logger.info(f"{dataset_name.upper()}: ROC-AUC={metrics.get('roc_auc', 0.0):.4f}, "
                      f"AUPR={metrics.get('pr_auc', 0.0):.4f}")
        logger.info("=================")
        
        return {
            'inference_results': results,
            'config': config,
            'output_dir': output_dir
        }
    
    else:
        raise NotImplementedError("Inference mode currently only supports sequence data")


def run_training_with_callback(config: Dict[str, Any], epoch_callback=None) -> Dict[str, Any]:
    """
    简化的训练接口，专为Optuna优化设计
    
    Args:
        config: 完整的训练配置
        epoch_callback: Epoch级别的回调函数
    
    Returns:
        训练结果，包含最佳验证指标
    """
    # Create output directory
    output_dir = create_output_directory()
    
    # Setup logger with minimal output for Optuna
    logger = setup_logger(output_dir, log_level=logging.WARNING)
    
    try:
        # Add callback to config
        if epoch_callback:
            if 'optuna' not in config:
                config['optuna'] = {}
            config['optuna']['epoch_callback'] = epoch_callback
        
        # Run training
        results = run_training_mode(config, output_dir, logger)
        
        return {
            'best_val_metric': results['training_results']['best_val_metric'],
            'best_epoch': results['training_results']['best_epoch'],
            'final_evaluation': results['final_evaluation'],
            'output_dir': output_dir,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Main function with enhanced command line interface and Hydra support"""
    parser = argparse.ArgumentParser(description='SEPAL-PPI: Protein-Protein Interaction Prediction')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='training', 
                       choices=['training', 'inference', 'ensemble_inference', 'ensemble_predict', 'ensemble_train', 'fusion_finetune'],
                       help='Run mode: training, inference, ensemble_inference, ensemble_predict, ensemble_train, fusion_finetune')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    # Ensemble training parameters
    parser.add_argument('--ensemble-config', type=str, default=None,
                       help='Path to ensemble config file (YAML)')
    parser.add_argument('--configs', nargs='+', required=False,
                       help='Paths to multiple best_config.yaml files (required if --ensemble-config is not used)')
    parser.add_argument('--k-fold', type=int, default=5,
                       help='Number of folds for K-fold cross-validation (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--ensemble-mode', type=str, default='logits', choices=['logits', 'features'],
                       help='Meta-learner input mode: logits (default) or features (penultimate layer features)')
    parser.add_argument('--train-model', type=str, default='linear', choices=['linear', 'XGBoost', 'MLP'],
                       help='Meta-learner model type: linear (linear regression), XGBoost (gradient boosting), or MLP (multilayer perceptron)')
    
    # Hydra support
    parser.add_argument('--hydra', action='store_true', 
                       help='Use Hydra configuration management')
    parser.add_argument('--hydra-config-dir', type=str, default='config',
                       help='Hydra config directory')
    
    # Training parameters (used when config is not provided)
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--pooling-type', type=str, default='avg', 
                       choices=['avg', 'max', 'attention'], help='Pooling method')
    parser.add_argument('--interaction-type', type=str, default='hadamard', 
                       choices=['hadamard', 'concatenation', 'cosine', 'fast_compact_bilinear'], help='Interaction method')
    
    # General parameters
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    # New: override multimodal feature root folder during prediction
    parser.add_argument('--predict-prot-feature-folder', type=str, default=None,
                       help='Override all_feature_folder in preprocessing config during prediction; specify a new multimodal feature root folder')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--use-pooled-data', action='store_true', 
                       help='Use pre-pooled data instead of sequence data')
    # Fusion finetune specific parameters
    parser.add_argument('--base-checkpoint', type=str, default=None,
                       help='Fusion finetune: path to best/last weights of the no-feature model (.pth)')
    parser.add_argument('--freeze-except', nargs='*', default=None,
                       help='Fusion finetune: train only these modules; freeze others (e.g., preprocessing, classifier)')
    
    args = parser.parse_args()
    
    # Determine output directory based on mode
    if args.output_dir is None:
        if args.mode == 'ensemble_predict':
            output_dir = create_output_directory(prefix="predict")
        elif args.mode == 'ensemble_inference':
            output_dir = create_output_directory(prefix="ensemble_inference")
        elif args.mode == 'ensemble_train':
            output_dir = create_output_directory(prefix="ensemble_training")
        elif args.mode == 'inference':
            output_dir = create_output_directory(prefix="inference")
        elif args.mode == 'fusion_finetune':
            output_dir = create_output_directory(prefix="fusion_finetune")
        else:
            output_dir = create_output_directory(prefix="training")
    else:
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(output_dir, log_level=log_level)
    
    try:
        if args.mode == 'training':
            # Training mode
            if args.hydra:
                # Use Hydra configuration management
                logger.info("Using Hydra configuration management")
                hydra_config_manager = create_hydra_config_manager(args.hydra_config_dir)
                
                if args.config:
                    # Load Hydra config
                    config = hydra_config_manager.load_config(args.config)
                    
                    # Validate config
                    if not hydra_config_manager.validate_config(config):
                        raise ValueError("Configuration validation failed")
                    
                    # Resolve config
                    resolved_config = hydra_config_manager.resolve_config(config)
                    
                    logger.info(f"Loaded Hydra config: {args.config}")
                    
                    # 应用命令行覆盖
                    overrides = []
                    if args.epochs is not None:
                        resolved_config['training']['epochs'] = args.epochs
                        overrides.append(f"training.epochs={args.epochs}")
                    if args.batch_size is not None:
                        resolved_config['training']['batch_size'] = args.batch_size
                        overrides.append(f"training.batch_size={args.batch_size}")
                    if args.learning_rate is not None:
                        resolved_config['training']['learning_rate'] = args.learning_rate
                        overrides.append(f"training.learning_rate={args.learning_rate}")
                    
                    if args.seed is not None:
                        resolved_config['training']['seed'] = args.seed
                        overrides.append(f"training.seed={args.seed}")
                    elif 'training' in resolved_config and 'seed' not in resolved_config['training']:
                        # Default fallback if not in config and not in args
                        resolved_config['training']['seed'] = 42
                        logger.info("Seed not specified in config or args, using default: 42")
                    elif 'training' not in resolved_config:
                        # Should unlikely happen given validation, but for safety
                         resolved_config['training'] = {'seed': 42}
                         logger.info("Training config not found, using default seed: 42")

                    if args.pooling_type != 'avg':
                        resolved_config['model']['pooling_type'] = args.pooling_type
                        overrides.append(f"model.pooling_type={args.pooling_type}")
                    if args.interaction_type != 'hadamard':
                        resolved_config['model']['interaction_type'] = args.interaction_type
                        overrides.append(f"model.interaction_type={args.interaction_type}")
                    
                    resolved_config['logging']['level'] = 'DEBUG' if args.debug else 'INFO'
                    resolved_config['logging']['verbose'] = not args.quiet
                    resolved_config['data']['use_sequence_data'] = not args.use_pooled_data
                    resolved_config['output']['output_dir'] = output_dir
                    
                    # Reset log level to ensure model creation uses the correct level
                    new_log_level = logging.DEBUG if args.debug else logging.INFO
                    logger.logger.setLevel(new_log_level)
                    for handler in logger.logger.handlers:
                        if isinstance(handler, logging.StreamHandler):
                            handler.setLevel(new_log_level)
                    
                    # Also set root logger so child loggers inherit the correct level
                    root_logger = logging.getLogger()
                    root_logger.setLevel(new_log_level)
                    
                    if overrides:
                        logger.info(f"Command-line overrides: {', '.join(overrides)}")
                        
                    logger.info(f"Log level updated to: {'DEBUG' if args.debug else 'INFO'}")
                    
                else:
                    # Use default Hydra configuration
                    config = hydra_config_manager.load_config("hydra_config")
                    resolved_config = hydra_config_manager.resolve_config(config)
                    logger.debug("Using default Hydra configuration")
                
                # 保存解析后的配置
                config_save_path = Path(output_dir) / "resolved_config.yaml"
                hydra_config_manager.save_config(config, str(config_save_path))
                
            else:
                # Use traditional configuration management
                config_manager = ConfigManager()
                
                if args.config:
                    # Load from YAML config
                    config = config_manager.load_yaml_config(args.config)
                    logger.info(f"Loaded training config file: {args.config}")
                    
                    # Override config with command line arguments (only if specified)
                    overrides = []
                    if args.epochs is not None:
                        config['training']['epochs'] = args.epochs
                        overrides.append(f"epochs={args.epochs}")
                    if args.batch_size is not None:
                        config['training']['batch_size'] = args.batch_size
                        overrides.append(f"batch_size={args.batch_size}")
                    if args.learning_rate is not None:
                        config['training']['learning_rate'] = args.learning_rate
                        overrides.append(f"learning_rate={args.learning_rate}")
                    if args.seed is not None:
                        if 'training' not in config:
                            config['training'] = {}
                        config['training']['seed'] = args.seed
                        overrides.append(f"seed={args.seed}")
                    if args.pooling_type != 'avg':  # override only when non-default
                        config['model']['pooling_type'] = args.pooling_type
                        overrides.append(f"pooling_type={args.pooling_type}")
                    if args.interaction_type != 'hadamard':  # override only when non-default
                        config['model']['interaction_type'] = args.interaction_type
                        overrides.append(f"interaction_type={args.interaction_type}")
                    config['logging']['level'] = 'DEBUG' if args.debug else 'INFO'
                    config['logging']['verbose'] = not args.quiet
                    config['data']['use_sequence_data'] = not args.use_pooled_data
                    config['output']['output_dir'] = output_dir
                    
                    # Reset log level to ensure model creation uses the correct level
                    new_log_level = logging.DEBUG if args.debug else logging.INFO
                    logger.logger.setLevel(new_log_level)
                    for handler in logger.logger.handlers:
                        if isinstance(handler, logging.StreamHandler):
                            handler.setLevel(new_log_level)
                    
                    # Also set root logger so child loggers inherit the correct level
                    root_logger = logging.getLogger()
                    root_logger.setLevel(new_log_level)
                    
                    if overrides:
                        logger.info(f"Command-line overrides: {', '.join(overrides)}")
                    else:
                        logger.info("Using all parameters from config file, no command-line overrides")
                        
                    logger.info(f"Log level updated to: {'DEBUG' if args.debug else 'INFO'}")
                else:
                    # Generate config from command line arguments
                    config = config_manager.generate_training_config(args, output_dir)
                    if args.seed is not None:
                        config['training']['seed'] = args.seed
                    logger.info("Generated training configuration")
                
                resolved_config = config
            
            # Run training
            results = run_training_mode(resolved_config, output_dir, logger)
            
            logger.info("SEPAL-PPI training finished!")
            
        elif args.mode == 'ensemble_train':
            # Ensemble training mode
            logger.info("=== Ensemble training mode ===")
            
            # Get device
            device = get_device()
            logger.info(f"Compute device: {device}")
            
            # 处理配置文件参数
            if args.ensemble_config:
                # 从集成配置文件读取参数
                import yaml
                with open(args.ensemble_config, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 从配置文件更新参数
                if 'configs' in config_data:
                    args.configs = config_data['configs']
                if 'output_dir' in config_data:
                    args.output_dir = config_data['output_dir']
                if 'k_fold' in config_data:
                    args.k_fold = config_data['k_fold']
                if 'seed' in config_data and args.seed is None:
                    args.seed = config_data['seed']
                if 'mode' in config_data:
                    args.ensemble_mode = config_data['mode']
                if 'train_model' in config_data:
                    args.train_model = config_data['train_model']
                if 'debug' in config_data:
                    args.debug = config_data['debug']
                
                # 保存超参数配置
                model_params = {}
                if args.train_model == 'XGBoost' and 'xgboost_params' in config_data:
                    model_params = config_data['xgboost_params']
                elif args.train_model == 'linear' and 'linear_params' in config_data:
                    model_params = config_data['linear_params']
                elif args.train_model == 'MLP' and 'mlp_params' in config_data:
                    model_params = config_data['mlp_params']
            else:
                model_params = {}
            
            # Finalize seed
            if args.seed is None:
                args.seed = 42

            # 设置随机种子
            set_random_seed(args.seed)
            logger.debug(f"Random seed set to: {args.seed}")
            
            # Validate config files
            if not args.configs:
                raise ValueError("Ensemble training mode requires --configs or --ensemble-config")
            
            # Expand wildcard paths
            config_paths = []
            for pattern in args.configs:
                paths = list(Path().glob(pattern))
                if paths:
                    config_paths.extend([str(p) for p in paths])
                else:
                    # If not a wildcard, add directly
                    config_paths.append(pattern)
            
            # Deduplicate preserving order
            config_paths = list(dict.fromkeys(config_paths))
            
            # Validate config files
            valid_configs = []
            for config_path in config_paths:
                config_file = Path(config_path)
                if config_file.exists() and config_file.suffix in ['.yaml', '.yml']:
                    valid_configs.append(str(config_file))
                else:
                    logger.warning(f"Skipping invalid config file: {config_path}")
            
            if len(valid_configs) < 2:
                raise ValueError(f"At least 2 valid config files required, found {len(valid_configs)}")
            
            logger.info(f"Found {len(valid_configs)} valid config files:")
            for config_path in valid_configs:
                logger.info(f"  - {config_path}")
            
            # 运行集成训练
            results = run_ensemble_training_mode(
                valid_configs, output_dir, logger, device,
                seed=args.seed,
                mode=args.ensemble_mode,
                train_model=args.train_model,
                model_params=model_params
            )
            
            logger.info("SEPAL-PPI ensemble training finished!")
            
        elif args.mode == 'ensemble_inference':
            # Ensemble inference mode
            if not args.config:
                raise ValueError("Ensemble inference mode requires --config")
            
            if args.hydra:
                # Use Hydra config management
                hydra_config_manager = create_hydra_config_manager(args.hydra_config_dir)
                config = hydra_config_manager.load_config(args.config)
                resolved_config = hydra_config_manager.resolve_config(config)
                logger.info(f"Loaded Hydra ensemble inference config: {args.config}")
            else:
                # Use traditional config management
                config_manager = ConfigManager()
                config = config_manager.load_yaml_config(args.config)
                resolved_config = config
                logger.info(f"Loaded ensemble inference config file: {args.config}")
            
            # Run ensemble inference
            results = run_ensemble_inference_mode(resolved_config, output_dir, logger)
            
            logger.info("SEPAL-PPI ensemble inference finished!")
            
        elif args.mode == 'ensemble_predict':
            # Ensemble predict mode
            if not args.config:
                raise ValueError("Ensemble predict mode requires --config")
            
            if args.hydra:
                # Use Hydra config management
                hydra_config_manager = create_hydra_config_manager(args.hydra_config_dir)
                config = hydra_config_manager.load_config(args.config)
                resolved_config = hydra_config_manager.resolve_config(config)
                logger.info(f"Loaded Hydra ensemble prediction config: {args.config}")
            else:
                # Use traditional config management
                config_manager = ConfigManager()
                config = config_manager.load_yaml_config(args.config)
                resolved_config = config
                logger.info(f"Loaded ensemble prediction config file: {args.config}")

            # Finalize and apply seed for deterministic ensemble prediction
            seed = args.seed
            if seed is None:
                seed = resolved_config.get('seed')
            if seed is None:
                seed = resolved_config.get('training', {}).get('seed')
            if seed is None:
                seed = 42

            resolved_config['seed'] = seed
            set_random_seed(seed)
            logger.info(f"Random seed set to: {seed}")
            
            # 添加ensemble_mode参数到配置中
            if args.ensemble_mode:
                resolved_config['ensemble_mode'] = args.ensemble_mode
                logger.info(f"Using ensemble mode: {args.ensemble_mode}")
            # 覆盖多模态特征根目录
            if args.predict_prot_feature_folder:
                resolved_config['predict_prot_feature_folder'] = args.predict_prot_feature_folder
                logger.info(f"Prediction feature root folder overridden to: {args.predict_prot_feature_folder}")
            
            # Run ensemble predict
            results = run_ensemble_predict_mode(resolved_config, output_dir, logger)
            
            # Generate HTML visualization
            try:
                from src.visualization.html_generator import HTMLGenerator
                # Fix: pass the correct argument types
                html_generator = HTMLGenerator(str(output_dir), config_path=args.config, logger=logger)
                html_file = html_generator.generate_prediction_results_html(results)
                if html_file:
                    logger.debug(f"Prediction results HTML generated: {html_file}")
                else:
                    logger.warning("Failed to generate HTML")
            except Exception as e:
                logger.error(f"Error generating HTML: {e}")
            
            logger.info("SEPAL-PPI ensemble prediction finished!")
            
        elif args.mode == 'inference':
            # Inference mode
            if not args.config:
                raise ValueError("Inference mode requires --config")
            
            if args.hydra:
                # Use Hydra config management
                hydra_config_manager = create_hydra_config_manager(args.hydra_config_dir)
                config = hydra_config_manager.load_config(args.config)
                resolved_config = hydra_config_manager.resolve_config(config)
                logger.info(f"Loaded Hydra inference config: {args.config}")
            else:
                # Use traditional config management
                config_manager = ConfigManager()
                config = config_manager.load_yaml_config(args.config)
                resolved_config = config
                logger.info(f"Loaded inference config file: {args.config}")
            
            # Run inference
            results = run_inference_mode(resolved_config, output_dir, logger)
            
            logger.info("SEPAL-PPI inference finished!")

        elif args.mode == 'fusion_finetune':
            # Fusion finetune mode
            if not args.config:
                raise ValueError("Fusion finetune mode requires --config (should include feature_cross_transformer)")

            # Read config (supports Hydra and traditional)
            if args.hydra:
                hydra_config_manager = create_hydra_config_manager(args.hydra_config_dir)
                config = hydra_config_manager.load_config(args.config)
                resolved_config = hydra_config_manager.resolve_config(config)
                logger.info(f"Loaded Hydra fusion finetune config: {args.config}")
            else:
                config_manager = ConfigManager()
                config = config_manager.load_yaml_config(args.config)
                resolved_config = config
                logger.info(f"Loaded fusion finetune config file: {args.config}")

            # Write finetune section
            if 'finetune' not in resolved_config:
                resolved_config['finetune'] = {}
            if args.base_checkpoint:
                resolved_config['finetune']['base_checkpoint'] = args.base_checkpoint
            if args.freeze_except is not None:
                resolved_config['finetune']['freeze_except'] = args.freeze_except

            # Basic overrides
            overrides = []
            if args.epochs is not None:
                resolved_config['training']['epochs'] = args.epochs
                overrides.append(f"training.epochs={args.epochs}")
            if args.batch_size is not None:
                resolved_config['training']['batch_size'] = args.batch_size
                overrides.append(f"training.batch_size={args.batch_size}")
            if args.learning_rate is not None:
                resolved_config['training']['learning_rate'] = args.learning_rate
                overrides.append(f"training.learning_rate={args.learning_rate}")
            resolved_config['logging']['level'] = 'DEBUG' if args.debug else 'INFO'
            resolved_config['logging']['verbose'] = not args.quiet
            resolved_config['data']['use_sequence_data'] = not args.use_pooled_data
            resolved_config['output']['output_dir'] = output_dir

            if overrides:
                logger.info(f"Command-line overrides: {', '.join(overrides)}")

            # Call training runner
            from src.training.training_runner import run_fusion_finetune_mode
            results = run_fusion_finetune_mode(resolved_config, output_dir, logger)
            logger.info("SEPAL-PPI fusion finetune finished!")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


## run_ensemble_training_mode 已迁移到 src/ensemble/ensemble_runner.py


if __name__ == "__main__":
    main()