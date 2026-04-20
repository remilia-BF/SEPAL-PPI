"""
Prediction utilities for SEPAL-PPI models
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def predict_interactions(model: nn.Module, 
                        features: torch.Tensor,
                        device: Optional[torch.device] = None,
                        batch_size: int = 1000) -> np.ndarray:
    """
    Predict protein-protein interactions for given features
    
    Args:
        model (nn.Module): Trained model
        features (torch.Tensor): Input features
        device (Optional[torch.device]): Device to run prediction on
        batch_size (int): Batch size for prediction (to avoid memory issues)
    
    Returns:
        np.ndarray: Predicted interaction probabilities
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size].to(device)
            batch_predictions = model(batch_features)
            predictions.extend(batch_predictions.cpu().numpy().flatten())
    
    return np.array(predictions)


def predict_single_pair(model: nn.Module,
                       protein1_embedding: torch.Tensor,
                       protein2_embedding: torch.Tensor,
                       device: Optional[torch.device] = None) -> float:
    """
    Predict interaction probability for a single protein pair
    
    Args:
        model (nn.Module): Trained model
        protein1_embedding (torch.Tensor): First protein embedding
        protein2_embedding (torch.Tensor): Second protein embedding
        device (Optional[torch.device]): Device to run prediction on
    
    Returns:
        float: Interaction probability
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    
    # Create pair tensor
    pair_tensor = torch.stack([protein1_embedding, protein2_embedding], dim=0).unsqueeze(0)
    pair_tensor = pair_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(pair_tensor)
        return prediction.item()


def batch_predict_from_embeddings(model: nn.Module,
                                 protein_pairs: List[Tuple[str, str]],
                                 embedding_dict: Dict[str, torch.Tensor],
                                 device: Optional[torch.device] = None) -> Dict[Tuple[str, str], float]:
    """
    Predict interactions for multiple protein pairs using embedding dictionary
    
    Args:
        model (nn.Module): Trained model
        protein_pairs (List[Tuple[str, str]]): List of protein ID pairs
        embedding_dict (Dict[str, torch.Tensor]): Dictionary of protein embeddings
        device (Optional[torch.device]): Device to run prediction on
    
    Returns:
        Dict[Tuple[str, str], float]: Predictions for each protein pair
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    
    predictions = {}
    valid_pairs = []
    pair_tensors = []
    
    # Prepare valid protein pairs
    for protein1_id, protein2_id in protein_pairs:
        if protein1_id in embedding_dict and protein2_id in embedding_dict:
            valid_pairs.append((protein1_id, protein2_id))
            pair_tensor = torch.stack([
                embedding_dict[protein1_id],
                embedding_dict[protein2_id]
            ], dim=0)
            pair_tensors.append(pair_tensor)
    
    if not pair_tensors:
        logger.warning("No valid protein pairs found for prediction")
        return predictions
    
    # Stack all pairs and predict
    batch_tensor = torch.stack(pair_tensors, dim=0).to(device)
    
    with torch.no_grad():
        batch_predictions = model(batch_tensor)
        batch_predictions = batch_predictions.cpu().numpy().flatten()
    
    # Map predictions back to protein pairs
    for (protein1_id, protein2_id), pred in zip(valid_pairs, batch_predictions):
        predictions[(protein1_id, protein2_id)] = float(pred)
    
    return predictions


def predict_top_interactions(model: nn.Module,
                           target_protein: str,
                           candidate_proteins: List[str],
                           embedding_dict: Dict[str, torch.Tensor],
                           top_k: int = 10,
                           device: Optional[torch.device] = None) -> List[Tuple[str, float]]:
    """
    Predict top-k interaction partners for a target protein
    
    Args:
        model (nn.Module): Trained model
        target_protein (str): Target protein ID
        candidate_proteins (List[str]): List of candidate protein IDs
        embedding_dict (Dict[str, torch.Tensor]): Dictionary of protein embeddings
        top_k (int): Number of top interactions to return
        device (Optional[torch.device]): Device to run prediction on
    
    Returns:
        List[Tuple[str, float]]: Top-k interactions with scores
    """
    if target_protein not in embedding_dict:
        raise ValueError(f"Target protein {target_protein} not found in embeddings")
    
    # Create protein pairs
    protein_pairs = [(target_protein, candidate) for candidate in candidate_proteins
                    if candidate in embedding_dict and candidate != target_protein]
    
    # Get predictions
    predictions = batch_predict_from_embeddings(model, protein_pairs, embedding_dict, device)
    
    # Sort by prediction score and return top-k
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    top_interactions = []
    for (protein1, protein2), score in sorted_predictions[:top_k]:
        partner = protein2 if protein1 == target_protein else protein1
        top_interactions.append((partner, score))
    
    return top_interactions


def create_prediction_report(predictions: np.ndarray,
                           labels: np.ndarray,
                           protein_pairs: Optional[List[Tuple[str, str]]] = None,
                           threshold: float = 0.5) -> Dict:
    """
    Create a detailed prediction report
    
    Args:
        predictions (np.ndarray): Predicted probabilities
        labels (np.ndarray): True labels
        protein_pairs (Optional[List[Tuple[str, str]]]): Protein pair identifiers
        threshold (float): Classification threshold
    
    Returns:
        Dict: Detailed prediction report
    """
    # 检查空数组情况
    if len(predictions) == 0 or len(labels) == 0:
        return {
            'threshold': threshold,
            'total_samples': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'prediction_distribution': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            },
            'individual_predictions': []
        }
    
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Basic statistics
    true_positives = np.sum((binary_predictions == 1) & (labels == 1))
    false_positives = np.sum((binary_predictions == 1) & (labels == 0))
    true_negatives = np.sum((binary_predictions == 0) & (labels == 0))
    false_negatives = np.sum((binary_predictions == 0) & (labels == 1))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(labels) if len(labels) > 0 else 0.0
    
    report = {
        'threshold': threshold,
        'total_samples': len(labels),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'prediction_distribution': {
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions))
        }
    }
    
    # Add individual predictions if protein pairs provided
    if protein_pairs is not None and len(protein_pairs) == len(predictions):
        report['individual_predictions'] = [
            {
                'protein_pair': pair,
                'prediction': float(pred),
                'true_label': int(label),
                'binary_prediction': int(bin_pred)
            }
            for pair, pred, label, bin_pred in zip(protein_pairs, predictions, labels, binary_predictions)
        ]
    
    return report