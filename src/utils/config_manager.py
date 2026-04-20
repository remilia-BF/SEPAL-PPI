"""
Configuration manager for SEPAL-PPI
Handles YAML configuration loading and generation
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ConfigManager:
    """Manages training and inference configurations"""
    
    def __init__(self):
        self.training_config = None
        self.inference_config = None
    
    @staticmethod
    def load_yaml_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def save_yaml_config(config: Dict[str, Any], output_path: str):
        """Save configuration to YAML file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    @staticmethod
    def create_default_training_config() -> Dict[str, Any]:
        """Create default training configuration"""
        return {
            'mode': 'training',
            'model': {
                'pooling_type': 'avg',
                'interaction_type': 'hadamard',
                'classifier_type': 'standard',
                'embedding_dim': 1280
            },
            'training': {
                'epochs': 40,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'loss_type': 'bce',
                'seed': 42
            },
            'data': {
                'use_sequence_data': True,
                'embedding_file': 'emb/esm1b_S1_all.NOcls_NOeos.lmdb',
                'train_file': 'dataset/S1/c1Train.txt',
                'validation_file': 'dataset/S1/c2Validation.txt',
                'test_file': 'dataset/S1/c3Test.txt',
                'fasta_file': 'dataset/S1/protein.fasta',
                'cache_size': 10000
            },
            'logging': {
                'level': 'INFO',
                'verbose': True
            },
            'output': {
                'save_best_model': True,
                'save_last_model': True,
                'generate_confusion_matrix': True,
                'output_dir': None  # Will be auto-generated
            }
        }
    
    @staticmethod
    def create_default_inference_config(model_path: Optional[str] = None, best_epoch: int = 1) -> Dict[str, Any]:
        """Create default inference configuration"""
        return {
            'mode': 'inference',
            'model': {
                'model_path': model_path or 'results/latest/best_model.pth',
                'pooling_type': 'avg',
                'interaction_type': 'hadamard',
                'classifier_type': 'standard',
                'embedding_dim': 1280
            },
            'inference': {
                'batch_size': 32,
                'threshold': 0.5,
                'best_epoch': best_epoch
            },
            'data': {
                'use_sequence_data': True,
                'embedding_file': 'emb/esm1b_S1_all.NOcls_NOeos.lmdb',
                'test_files': {
                    'c2': 'dataset/S1/c2Validation.txt',
                    'c3': 'dataset/S1/c3Test.txt'
                },
                'fasta_file': 'dataset/S1/protein.fasta',
                'cache_size': 10000
            },
            'output': {
                'generate_confusion_matrix': True,
                'save_predictions': True,
                'output_dir': None  # Will be auto-generated
            }
        }
    
    def generate_training_config(self, args, output_dir: str) -> Dict[str, Any]:
        """Generate training configuration from command line arguments"""
        config = self.create_default_training_config()
        
        # Update from command line arguments (only if not None)
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['training']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config['training']['learning_rate'] = args.learning_rate
        if args.pooling_type != 'avg':  # 只在非默认值时覆盖
            config['model']['pooling_type'] = args.pooling_type
        if args.interaction_type != 'hadamard':  # 只在非默认值时覆盖
            config['model']['interaction_type'] = args.interaction_type
        config['logging']['level'] = 'DEBUG' if args.debug else 'INFO'
        config['logging']['verbose'] = not args.quiet
        config['data']['use_sequence_data'] = not args.use_pooled_data
        config['output']['output_dir'] = output_dir
        
        # 使用分桶配置文件的默认路径
        config['data']['embedding_file'] = "emb/esm1b_S1_all.NOcls_NOeos.lmdb.buck/bucketed_embeddings.lmdb"
        
        # Save configuration
        config_path_str = str(Path(output_dir) / "training_config.yaml")
        self.save_yaml_config(config, config_path_str)
        
        self.training_config = config
        return config
    
    def generate_inference_config(self, training_config: Dict[str, Any], 
                                  best_model_path: str, best_epoch: int, 
                                  output_dir: str) -> Dict[str, Any]:
        """Generate inference configuration from training configuration"""
        config = self.create_default_inference_config(best_model_path, best_epoch)
        
        # Copy relevant settings from training config
        config['model'].update({
            'pooling_type': training_config['model']['pooling_type'],
            'interaction_type': training_config['model']['interaction_type'],
            'classifier_type': training_config['model']['classifier_type'],
            'embedding_dim': training_config['model']['embedding_dim']
        })
        
        config['data'].update({
            'use_sequence_data': training_config['data']['use_sequence_data'],
            'embedding_file': training_config['data']['embedding_file'],
            'fasta_file': training_config['data']['fasta_file'],
            'cache_size': training_config['data']['cache_size']
        })
        
        config['inference']['batch_size'] = training_config['training']['batch_size']
        config['output']['output_dir'] = output_dir
        
        # Save configuration
        config_path_str = str(Path(output_dir) / "inference_config.yaml")
        self.save_yaml_config(config, config_path_str)
        
        self.inference_config = config
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override_config taking precedence"""
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, override_config)