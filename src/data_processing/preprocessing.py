"""
Data preprocessing utilities for SEPAL-PPI
"""

import pickle
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


def load_embeddings(embedding_file: str) -> Dict[str, torch.Tensor]:
    """
    Load protein embeddings from pickle file
    
    Args:
        embedding_file (str): Path to the embedding pickle file
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping protein IDs to embeddings
    """
    try:
        with open(embedding_file, "rb") as f:
            embedding_dict = pickle.load(f)
        logger.info(f"Successfully loaded embeddings from {embedding_file}")
        logger.info(f"Number of proteins: {len(embedding_dict)}")
        return embedding_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    except Exception as e:
        raise Exception(f"Error loading embeddings: {str(e)}")


def load_interaction_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load protein interaction data from text file
    
    Args:
        file_path (str): Path to the interaction data file
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Protein ID pairs and labels
    """
    try:
        data = np.loadtxt(file_path, dtype=str, usecols=(0, 1, 2))
        
        # Extract protein pairs and labels
        protein_pairs = data[:, :2]  # First two columns are protein IDs
        labels = data[:, 2].astype(int)  # Third column is the interaction label
        
        logger.info(f"Loaded {len(protein_pairs)} protein pairs from {file_path}")
        logger.info(f"Positive interactions: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
        
        return protein_pairs, labels
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Interaction data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading interaction data: {str(e)}")


def create_protein_pair_embeddings(protein_pairs: np.ndarray, 
                                 embedding_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Create paired embeddings for protein pairs
    
    Args:
        protein_pairs (np.ndarray): Array of protein ID pairs
        embedding_dict (Dict[str, torch.Tensor]): Dictionary of protein embeddings
    
    Returns:
        torch.Tensor: Stacked embeddings of shape (n_pairs, 2, embedding_dim)
    """
    embeddings = []
    missing_proteins = set()
    
    for protein1_id, protein2_id in protein_pairs:
        # Check if both proteins have embeddings
        if protein1_id not in embedding_dict:
            missing_proteins.add(protein1_id)
            continue
        if protein2_id not in embedding_dict:
            missing_proteins.add(protein2_id)
            continue
            
        # Stack embeddings for the protein pair
        pair_embedding = torch.stack([
            embedding_dict[protein1_id], 
            embedding_dict[protein2_id]
        ], dim=0)
        embeddings.append(pair_embedding)
    
    if missing_proteins:
        logger.warning(f"{len(missing_proteins)} proteins not found in embeddings")
        logger.debug(f"Missing proteins: {list(missing_proteins)[:10]}...")  # Show first 10
    
    if not embeddings:
        raise ValueError("No valid protein pairs found with embeddings")
    
    return torch.stack(embeddings, dim=0)


def validate_data_consistency(protein_pairs: np.ndarray, 
                            labels: np.ndarray, 
                            embeddings: torch.Tensor) -> bool:
    """
    Validate that data arrays are consistent
    
    Args:
        protein_pairs (np.ndarray): Protein ID pairs
        labels (np.ndarray): Interaction labels
        embeddings (torch.Tensor): Protein pair embeddings
    
    Returns:
        bool: True if data is consistent
    """
    if len(protein_pairs) != len(labels):
        raise ValueError(f"Mismatch: {len(protein_pairs)} pairs vs {len(labels)} labels")
    
    if len(embeddings) != len(labels):
        logger.warning(f"{len(embeddings)} embeddings vs {len(labels)} labels")
        logger.warning("This may indicate missing proteins in embedding dictionary")
    
    logger.info(f"Data validation: {len(embeddings)} valid protein pairs")
    return True