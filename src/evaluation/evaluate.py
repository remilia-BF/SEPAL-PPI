"""
Evaluation utilities for SEPAL-PPI models
"""

import torch
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from ..models.predict_model import predict_interactions, create_prediction_report

logger = logging.getLogger(__name__)


def evaluate_model(model: torch.nn.Module,
                  features: torch.Tensor,
                  labels: torch.Tensor,
                  device: Optional[torch.device] = None,
                  dataset_name: str = "") -> Dict:
    """
    Comprehensive evaluation of model performance
    
    Args:
        model (torch.nn.Module): Trained model
        features (torch.Tensor): Test features
        labels (torch.Tensor): True labels
        device (Optional[torch.device]): Device to run evaluation on
        dataset_name (str): Name of the dataset for reporting
    
    Returns:
        Dict: Comprehensive evaluation metrics
    """
    if device is None:
        device = torch.device('cpu')
    
    # Get predictions
    predictions = predict_interactions(model, features, device)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Calculate metrics
    try:
        roc_auc = roc_auc_score(labels_np, predictions)
    except ValueError:
        roc_auc = float('nan')
        
    try:
        pr_auc = average_precision_score(labels_np, predictions)
    except ValueError:
        pr_auc = float('nan')
    
    # Calculate precision-recall curve
    try:
        precision, recall, pr_thresholds = precision_recall_curve(labels_np, predictions)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(labels_np, predictions)
    except ValueError:
        precision = recall = pr_thresholds = None
        roc_fpr = roc_tpr = roc_thresholds = None
    
    # Create detailed report at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    threshold_reports = {}
    
    for threshold in thresholds:
        threshold_reports[threshold] = create_prediction_report(
            predictions, labels_np, threshold=threshold
        )
    
    # Compile results
    results = {
        'dataset_name': dataset_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'n_samples': len(labels_np),
        'n_positive': int(np.sum(labels_np)),
        'n_negative': int(len(labels_np) - np.sum(labels_np)),
        'positive_rate': float(np.mean(labels_np)),
        'predictions': predictions,
        'true_labels': labels_np,
        'threshold_reports': threshold_reports
    }
    
    # Add curve data if available
    if precision is not None:
        results['pr_curve'] = {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds
        }
        
    if roc_fpr is not None:
        results['roc_curve'] = {
            'fpr': roc_fpr,
            'tpr': roc_tpr,
            'thresholds': roc_thresholds
        }
    
    return results


def evaluate_multiple_datasets(model: torch.nn.Module,
                              datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                              device: Optional[torch.device] = None) -> Dict[str, Dict]:
    """
    Evaluate model on multiple datasets
    
    Args:
        model (torch.nn.Module): Trained model
        datasets (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): Dictionary of dataset name to (features, labels)
        device (Optional[torch.device]): Device to run evaluation on
    
    Returns:
        Dict[str, Dict]: Evaluation results for each dataset
    """
    results = {}
    
    for dataset_name, (features, labels) in datasets.items():
        logger.info(f"Evaluating on {dataset_name} dataset...")
        results[dataset_name] = evaluate_model(model, features, labels, device, dataset_name)
    
    return results


def print_evaluation_results(results: Dict, detailed: bool = False):
    """
    Print evaluation results in a formatted way
    
    Args:
        results (Dict): Evaluation results from evaluate_model
        detailed (bool): Whether to print detailed metrics
    """
    dataset_name = results.get('dataset_name', 'Unknown')
    
    logger.info(f"=== {dataset_name} Evaluation Results ===")
    logger.info(f"Samples: {results['n_samples']} (Positive: {results['n_positive']}, Negative: {results['n_negative']})")
    logger.info(f"Positive Rate: {results['positive_rate']:.3f}")
    logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
    logger.info(f"PR-AUC: {results['pr_auc']:.4f}")
    
    if detailed and 'threshold_reports' in results:
        logger.info("--- Performance at Different Thresholds ---")
        for threshold, report in results['threshold_reports'].items():
            logger.info(f"Threshold: {threshold}")
            logger.info(f"  Precision: {report['precision']:.4f}")
            logger.info(f"  Recall: {report['recall']:.4f}")
            logger.info(f"  F1-Score: {report['f1_score']:.4f}")
            logger.info(f"  Accuracy: {report['accuracy']:.4f}")


def print_multiple_results(all_results: Dict[str, Dict], summary_only: bool = True):
    """
    Print results for multiple datasets
    
    Args:
        all_results (Dict[str, Dict]): Results from evaluate_multiple_datasets
        summary_only (bool): Whether to print only summary metrics
    """
    logger.info("="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    # Summary table
    logger.info(f"{'Dataset':<15} {'Samples':<8} {'Pos%':<8} {'ROC-AUC':<10} {'PR-AUC':<10}")
    logger.info("-" * 60)
    
    for dataset_name, results in all_results.items():
        pos_rate = results['positive_rate'] * 100
        roc_auc = results['roc_auc']
        pr_auc = results['pr_auc']
        n_samples = results['n_samples']
        
        logger.info(f"{dataset_name:<15} {n_samples:<8} {pos_rate:<8.1f} {roc_auc:<10.4f} {pr_auc:<10.4f}")
    
    if not summary_only:
        for dataset_name, results in all_results.items():
            print_evaluation_results(results, detailed=True)


def compare_models(models: Dict[str, torch.nn.Module],
                  features: torch.Tensor,
                  labels: torch.Tensor,
                  device: Optional[torch.device] = None) -> Dict[str, Dict]:
    """
    Compare multiple models on the same dataset
    
    Args:
        models (Dict[str, torch.nn.Module]): Dictionary of model name to model
        features (torch.Tensor): Test features
        labels (torch.Tensor): True labels
        device (Optional[torch.device]): Device to run evaluation on
    
    Returns:
        Dict[str, Dict]: Evaluation results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        results[model_name] = evaluate_model(model, features, labels, device, model_name)
    
    return results


def find_optimal_threshold(results: Dict, metric: str = 'f1_score') -> Tuple[float, float]:
    """
    Find optimal threshold based on a specific metric
    
    Args:
        results (Dict): Evaluation results from evaluate_model
        metric (str): Metric to optimize ('f1_score', 'precision', 'recall', 'accuracy')
    
    Returns:
        Tuple[float, float]: Optimal threshold and corresponding metric value
    """
    if 'pr_curve' not in results:
        raise ValueError("Precision-recall curve data not available in results")
    
    precision = results['pr_curve']['precision']
    recall = results['pr_curve']['recall']
    thresholds = results['pr_curve']['thresholds']
    
    if metric == 'f1_score':
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element due to threshold length
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = f1_scores[optimal_idx]
    elif metric == 'precision':
        optimal_idx = np.argmax(precision[:-1])
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = precision[optimal_idx]
    elif metric == 'recall':
        optimal_idx = np.argmax(recall[:-1])
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = recall[optimal_idx]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return float(optimal_threshold), float(optimal_value)


def save_evaluation_results(results: Dict, filepath: str):
    """
    Save evaluation results to file
    
    Args:
        results (Dict): Evaluation results
        filepath (str): Path to save results
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_results[key][subkey] = subvalue.tolist()
                else:
                    serializable_results[key][subkey] = subvalue
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {filepath}")


def create_evaluation_summary(all_results: Dict[str, Dict]) -> Dict:
    """
    Create a summary of evaluation results across multiple datasets
    
    Args:
        all_results (Dict[str, Dict]): Results from evaluate_multiple_datasets
    
    Returns:
        Dict: Summary statistics
    """
    summary = {
        'datasets': list(all_results.keys()),
        'average_roc_auc': np.mean([r['roc_auc'] for r in all_results.values() if not np.isnan(r['roc_auc'])]),
        'average_pr_auc': np.mean([r['pr_auc'] for r in all_results.values() if not np.isnan(r['pr_auc'])]),
        'total_samples': sum(r['n_samples'] for r in all_results.values()),
        'total_positive': sum(r['n_positive'] for r in all_results.values()),
        'dataset_performance': {}
    }
    
    for dataset_name, results in all_results.items():
        summary['dataset_performance'][dataset_name] = {
            'roc_auc': results['roc_auc'],
            'pr_auc': results['pr_auc'],
            'n_samples': results['n_samples'],
            'positive_rate': results['positive_rate']
        }
    
    return summary