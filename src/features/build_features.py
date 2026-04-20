"""
Feature engineering for protein-protein interaction prediction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


def hadamard_product_features(protein1_embedding: torch.Tensor, 
                            protein2_embedding: torch.Tensor) -> torch.Tensor:
    """
    Create features using Hadamard (element-wise) product
    
    Args:
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
    
    Returns:
        torch.Tensor: Combined features
    """
    return torch.mul(protein1_embedding, protein2_embedding)


def concatenation_features(protein1_embedding: torch.Tensor, 
                         protein2_embedding: torch.Tensor) -> torch.Tensor:
    """
    Create features by concatenating protein embeddings
    
    Args:
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
    
    Returns:
        torch.Tensor: Concatenated features
    """
    return torch.cat([protein1_embedding, protein2_embedding], dim=-1)


def difference_features(protein1_embedding: torch.Tensor, 
                       protein2_embedding: torch.Tensor) -> torch.Tensor:
    """
    Create features using absolute difference
    
    Args:
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
    
    Returns:
        torch.Tensor: Difference features
    """
    return torch.abs(protein1_embedding - protein2_embedding)


def cosine_similarity_features(protein1_embedding: torch.Tensor, 
                             protein2_embedding: torch.Tensor) -> torch.Tensor:
    """
    Create cosine similarity feature
    
    Args:
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
    
    Returns:
        torch.Tensor: Cosine similarity (single value per pair)
    """
    cosine_sim = torch.nn.functional.cosine_similarity(
        protein1_embedding, protein2_embedding, dim=-1, eps=1e-8
    )
    return cosine_sim.unsqueeze(-1)  # Add dimension for consistency


def combined_features(protein1_embedding: torch.Tensor, 
                     protein2_embedding: torch.Tensor,
                     feature_types: List[str] = None) -> torch.Tensor:
    """
    Create combined features using multiple feature engineering methods
    
    Args:
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
        feature_types (List[str]): List of feature types to include
    
    Returns:
        torch.Tensor: Combined features
    """
    if feature_types is None:
        feature_types = ['hadamard', 'cosine']
    
    features = []
    
    for feature_type in feature_types:
        if feature_type == 'hadamard':
            features.append(hadamard_product_features(protein1_embedding, protein2_embedding))
        elif feature_type == 'concatenation':
            features.append(concatenation_features(protein1_embedding, protein2_embedding))
        elif feature_type == 'difference':
            features.append(difference_features(protein1_embedding, protein2_embedding))
        elif feature_type == 'cosine':
            features.append(cosine_similarity_features(protein1_embedding, protein2_embedding))
        else:
            # Warning: Unknown feature type handled by logging
        pass
    
    if not features:
        raise ValueError("No valid feature types specified")
    
    return torch.cat(features, dim=-1)


def create_protein_pair_features(protein_pairs: torch.Tensor,
                                feature_method: str = 'hadamard',
                                **kwargs) -> torch.Tensor:
    """
    Factory function to create features from protein pairs
    
    Args:
        protein_pairs (torch.Tensor): Tensor of shape (batch_size, 2, embedding_dim)
        feature_method (str): Feature extraction method
        **kwargs: Additional arguments for feature methods
    
    Returns:
        torch.Tensor: Extracted features
    """
    # Extract individual protein embeddings
    protein1 = protein_pairs[:, 0, :]
    protein2 = protein_pairs[:, 1, :]
    
    # Apply feature extraction method
    if feature_method == 'hadamard':
        return hadamard_product_features(protein1, protein2)
    elif feature_method == 'concatenation':
        return concatenation_features(protein1, protein2)
    elif feature_method == 'difference':
        return difference_features(protein1, protein2)
    elif feature_method == 'cosine':
        return cosine_similarity_features(protein1, protein2)
    elif feature_method == 'combined':
        feature_types = kwargs.get('feature_types', ['hadamard', 'cosine'])
        return combined_features(protein1, protein2, feature_types)
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")


# Feature extraction registry for easy extension
FEATURE_METHODS = {
    'hadamard': hadamard_product_features,
    'concatenation': concatenation_features,
    'difference': difference_features,
    'cosine': cosine_similarity_features,
    'combined': combined_features
}


def get_feature_dimension(embedding_dim: int, feature_method: str, **kwargs) -> int:
    """
    Calculate the output dimension for a given feature method
    
    Args:
        embedding_dim (int): Input embedding dimension
        feature_method (str): Feature extraction method
        **kwargs: Additional arguments
    
    Returns:
        int: Output feature dimension
    """
    if feature_method == 'hadamard':
        return embedding_dim
    elif feature_method == 'concatenation':
        return 2 * embedding_dim
    elif feature_method == 'difference':
        return embedding_dim
    elif feature_method == 'cosine':
        return 1
    elif feature_method == 'combined':
        feature_types = kwargs.get('feature_types', ['hadamard', 'cosine'])
        total_dim = 0
        for ft in feature_types:
            total_dim += get_feature_dimension(embedding_dim, ft, **kwargs)
        return total_dim
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")