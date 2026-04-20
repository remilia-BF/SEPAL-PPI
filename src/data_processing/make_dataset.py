"""
Dataset creation utilities for SEPAL-PPI
"""

import torch
import torch.utils.data as Data
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from .preprocessing import (
    load_embeddings, 
    load_interaction_data, 
    create_protein_pair_embeddings,
    validate_data_consistency
)

logger = logging.getLogger(__name__)


def create_single_dataset(data_file: str, 
                         embedding_dict: Dict[str, torch.Tensor],
                         dataset_name: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a single dataset from interaction data file
    
    Args:
        data_file (str): Path to interaction data file
        embedding_dict (Dict[str, torch.Tensor]): Protein embeddings
        dataset_name (str): Name for logging purposes
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Features and labels
    """
    logger.info(f"Creating {dataset_name} dataset")
    
    # Load protein pairs and labels
    protein_pairs, labels = load_interaction_data(data_file)
    
    # Create embeddings for protein pairs
    embeddings = create_protein_pair_embeddings(protein_pairs, embedding_dict)
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels[:len(embeddings)], dtype=torch.long)
    
    # Validate data consistency
    validate_data_consistency(protein_pairs[:len(embeddings)], 
                            labels[:len(embeddings)], 
                            embeddings)
    
    return embeddings, labels_tensor


def create_data_loader(features: torch.Tensor, 
                      labels: torch.Tensor,
                      batch_size: int = 128,
                      shuffle: bool = True,
                      num_workers: int = 0) -> Data.DataLoader:
    """
    Create PyTorch DataLoader from features and labels
    
    Args:
        features (torch.Tensor): Input features
        labels (torch.Tensor): Target labels
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
    
    Returns:
        Data.DataLoader: PyTorch DataLoader
    """
    dataset = Data.TensorDataset(features, labels)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    logger.info(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}")
    return loader


def create_dataset_loaders(config: Dict) -> Dict[str, Data.DataLoader]:
    """
    Factory function to create all dataset loaders
    
    Args:
        config (Dict): Configuration dictionary with keys:
            - embedding_file: Path to embedding file
            - train_file: Path to training data
            - test_files: Dict of test file paths
            - batch_size: Batch size for training
            - num_workers: Number of workers for data loading
    
    Returns:
        Dict[str, Data.DataLoader]: Dictionary of dataset loaders
    """
    logger.info("Creating Dataset Loaders")
    
    # Load embeddings
    embedding_dict = load_embeddings(config['embedding_file'])
    
    loaders = {}
    
    # Create training dataset and loader
    if 'train_file' in config:
        train_features, train_labels = create_single_dataset(
            config['train_file'], 
            embedding_dict, 
            "Training"
        )
        
        loaders['train'] = create_data_loader(
            train_features, 
            train_labels,
            batch_size=config.get('batch_size', 128),
            shuffle=True,
            num_workers=config.get('num_workers', 0)
        )
    
    # Create test datasets (no DataLoader needed, just tensors)
    if 'test_files' in config:
        for test_name, test_file in config['test_files'].items():
            test_features, test_labels = create_single_dataset(
                test_file, 
                embedding_dict, 
                f"Test ({test_name})"
            )
            
            # Store as tensors for evaluation (no need for DataLoader)
            loaders[f'{test_name}_features'] = test_features
            loaders[f'{test_name}_labels'] = test_labels
    
    logger.info(f"Created {len(loaders)} dataset components")
    return loaders


def get_default_config():
    """
    Get default configuration for dataset creation
    
    Returns:
        Dict: Default configuration
    """
    return {
        'embedding_file': 'emb/esm1b_S1_all.NOcls_NOeos.lmdb',
        'train_file': 'dataset/S1/c1Train.txt',
        'test_files': {
            'c2': 'dataset/S1/c2Validation.txt',
            'c3': 'dataset/S1/c3Test.txt'
        },
        'batch_size': 128,
        'num_workers': 0
    }


def create_default_loaders() -> Dict[str, Data.DataLoader]:
    """
    Create dataset loaders with default configuration
    
    Returns:
        Dict[str, Data.DataLoader]: Dataset loaders
    """
    config = get_default_config()
    return create_dataset_loaders(config)