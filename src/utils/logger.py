"""
Enhanced logging system for SEPAL-PPI
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SEPALLogger:
    """
    Enhanced logger for SEPAL-PPI with structured output
    """
    
    def __init__(self, 
                 name: str = "sepal_ppi",
                 log_level: int = logging.INFO,
                 output_dir: Optional[str] = None,
                 enable_file_logging: bool = True):
        """
        Initialize SEPAL logger
        
        Args:
            name (str): Logger name
            log_level (int): Logging level
            output_dir (str): Output directory for logs
            enable_file_logging (bool): Whether to write logs to file
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        info_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        
    # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
    # In DEBUG show timestamp; INFO+ use compact format
        console_handler.setFormatter(debug_formatter if log_level == logging.DEBUG else info_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file_logging and output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = self.output_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            # Always keep full timestamp and details in file
            file_handler.setFormatter(debug_formatter)
            self.logger.addHandler(file_handler)
            
            self.info(f"Log file: {log_file}")
        
        self.debug("Enhanced logging enabled")
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_config(self, config: dict, title: str = "Configuration"):
        """Log configuration in a structured way"""
        self.info(f"=== {title} ===")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"  {sub_key}: {sub_value}")
            else:
                self.info(f"{key}: {value}")
        self.info("=" * (len(title) + 8))
    
    def log_epoch_results(self, 
                         epoch: int, 
                         train_loss: float,
                         val_metrics: dict,
                         test_metrics: dict,
                         learning_rate: float,
                         best_metric: float,
                         best_epoch: int,
                         epoch_time: float):
        """Log epoch results in a structured format"""
        val_auc = val_metrics.get('roc_auc', 0.0)
        val_ap = val_metrics.get('pr_auc', 0.0)
        test_auc = test_metrics.get('roc_auc', 0.0)
        test_ap = test_metrics.get('pr_auc', 0.0)
        
        message = (f"Ep {epoch} | "
                  f"Val-AUC:{val_auc:.3f} | "
                  f"Val-AP:{val_ap:.3f} | "
                  f"Tes-AUC:{test_auc:.3f} | "
                  f"Tes-AP:{test_ap:.3f} | "
                  f"TrLoss:{train_loss:.3f} | "
                  f"LR:{learning_rate:.6f} | "
                  f"Best@Ep{best_epoch}:{best_metric:.3f} | "
                  f"Time:{epoch_time:.1f}s")
        
        self.info(message)
    
    def log_model_info(self, model, model_name: str = "Model"):
        """Log model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        self.info(f"{model_name}: {model.__class__.__name__} "
            f"({total_params:,} params, {model_size_mb:.1f} MB)")
    
    def log_dataset_info(self, datasets_info: dict):
        """Log dataset information"""
        # Calculate total samples, handling non-numeric values
        total_samples = 0
        for info in datasets_info.values():
            n_samples = info.get('n_samples', 0)
            if isinstance(n_samples, (int, float)):
                total_samples += n_samples
        
        dataset_parts = []
        for name, info in datasets_info.items():
            n_samples = info.get('n_samples', 0)
            if isinstance(n_samples, (int, float)):
                dataset_parts.append(f"{name}={n_samples:,}")
            else:
                dataset_parts.append(f"{name}={n_samples}")
        
        if total_samples > 0:
            self.debug(f"=== Dataset info ===")
            self.debug(f"Total samples: {total_samples:,}")
            self.debug(f"Distribution: {', '.join(dataset_parts)}")
            self.debug("===================")
        else:
            self.debug(f"=== Dataset info ===")
            self.debug(f"Distribution: {', '.join(dataset_parts)}")
            self.debug("===================")


def create_output_directory(base_dir: str = "results", prefix: str = "training") -> str:
    """
    Create output directory with timestamp
    
    Args:
        base_dir (str): Base directory for results
        prefix (str): Prefix for directory name (training, ensemble_training, etc.)
        
    Returns:
        str: Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"sepal_ppi_{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def setup_logger(output_dir: str, name: str = "sepal_ppi", log_level: int = logging.INFO) -> SEPALLogger:
    """
    Setup logger with output directory
    
    Args:
        output_dir (str): Output directory
        name (str): Logger name
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        SEPALLogger: Configured logger
    """
    return SEPALLogger(name=name, output_dir=output_dir, log_level=log_level)


def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes_amount: float) -> str:
    """Format memory amount"""
    if bytes_amount < 1024**3:
        mb = bytes_amount / (1024**2)
        return f"{mb:.1f}MB"
    else:
        gb = bytes_amount / (1024**3)
        return f"{gb:.1f}GB"