"""
Result Saver Module for One-Step Prediction

Handles saving prediction results in standardized formats:
1. ensemble_predictions.csv - Prediction results
2. attention_weights.jsonl - Per-protein attention weights
3. prediction_summary.json - Summary with metrics and metadata
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd


class ResultSaver:
    """
    Save prediction results in standardized formats.
    
    Output files:
    - ensemble_predictions.csv: protein1, protein2, ensemble_probability, 
                                ensemble_prediction, model_X_probability, model_X_weight
    - attention_weights.jsonl: Per-protein attention weights in JSONL format
    - prediction_summary.json: Summary with total_pairs, model_info, evaluation_metrics
    """
    
    def __init__(
        self,
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize result saver.
        
        Args:
            output_dir: Output directory path
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        filename: str = "ensemble_predictions.csv"
    ) -> str:
        """
        Save ensemble predictions to CSV.
        
        Expected columns:
        - protein1, protein2: Protein identifiers
        - ensemble_probability: Ensemble prediction probability
        - ensemble_prediction: Binary prediction (0 or 1)
        - {model_name}_probability: Per-model probability
        - {model_name}_weight: Per-model weight in ensemble
        
        Args:
            predictions: DataFrame with prediction results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Validate required columns
        required_columns = ['protein1', 'protein2', 'ensemble_probability', 'ensemble_prediction']
        missing = [col for col in required_columns if col not in predictions.columns]
        if missing:
            self.logger.warning(f"Missing columns in predictions: {missing}")
        
        # Sort by ensemble probability (descending)
        if 'ensemble_probability' in predictions.columns:
            predictions = predictions.sort_values('ensemble_probability', ascending=False)
        
        # Save to CSV
        predictions.to_csv(output_path, index=False, float_format='%.6f')
        
        self.logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        
        return str(output_path)
    
    def save_attention_weights(
        self,
        attention_data: Dict[str, Dict[str, Any]],
        filename: str = "attention_weights.jsonl"
    ) -> str:
        """
        Save attention weights to JSONL format.
        
        Each line is a JSON object with:
        - protein_id: Protein identifier
        - length: Sequence length
        - attention_heads: Number of attention heads
        - attention: Dict mapping head_X to list of normalized weights
        
        Args:
            attention_data: Dict mapping protein_id to attention data dict
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for protein_id, data in attention_data.items():
                if data is not None:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved attention weights for {len(attention_data)} proteins to {output_path}")
        
        return str(output_path)
    
    def save_summary(
        self,
        total_pairs: int,
        model_info: Dict[str, Any],
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        filename: str = "prediction_summary.json"
    ) -> str:
        """
        Save prediction summary to JSON.
        
        Args:
            total_pairs: Total number of protein pairs
            model_info: Model configuration and checkpoint info
            evaluation_metrics: Optional evaluation metrics
            additional_info: Optional additional information
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        summary = {
            "total_pairs": total_pairs,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat(),
            "output_directory": str(self.output_dir)
        }
        
        if evaluation_metrics:
            summary["evaluation_metrics"] = evaluation_metrics
        
        if additional_info:
            summary.update(additional_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved prediction summary to {output_path}")
        
        return str(output_path)
    
    def save_ig_results(
        self,
        ig_results: Dict[str, 'IGResult'],
        filename: str = "ig_attributions.json"
    ) -> str:
        """
        Save IG attribution results to JSON.
        
        Args:
            ig_results: Dict mapping protein_id to IGResult
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        output_data = {}
        
        for protein_id, result in ig_results.items():
            entry = {
                "protein_id": protein_id,
                "sequence": result.sequence if hasattr(result, 'sequence') else "",
                "length": len(result.sequence) if hasattr(result, 'sequence') and result.sequence else 0,
            }
            
            # Add IG components if available
            if hasattr(result, 'internal_projector_ig') and result.internal_projector_ig is not None:
                entry["internal_projector_ig"] = [round(float(x), 6) for x in result.internal_projector_ig]
            
            if hasattr(result, 'pooling_ig') and result.pooling_ig is not None:
                entry["pooling_ig"] = [round(float(x), 6) for x in result.pooling_ig]
            
            if hasattr(result, 'preprocessing_ig') and result.preprocessing_ig is not None:
                entry["preprocessing_ig"] = [round(float(x), 6) for x in result.preprocessing_ig]
            
            if hasattr(result, 'aggregated_ig') and result.aggregated_ig is not None:
                entry["aggregated_ig"] = [round(float(x), 6) for x in result.aggregated_ig]
            
            if hasattr(result, 'normalized_ig') and result.normalized_ig is not None:
                entry["normalized_ig"] = [round(float(x), 2) for x in result.normalized_ig]
            
            output_data[protein_id] = entry
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved IG results for {len(ig_results)} proteins to {output_path}")
        
        return str(output_path)
    
    def save_protein_list(
        self,
        protein_ids: List[str],
        filename: str = "protein_list.txt"
    ) -> str:
        """
        Save list of protein IDs.
        
        Args:
            protein_ids: List of protein identifiers
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for protein_id in protein_ids:
                f.write(f"{protein_id}\n")
        
        self.logger.info(f"Saved {len(protein_ids)} protein IDs to {output_path}")
        
        return str(output_path)
    
    def save_interaction_list(
        self,
        protein_pairs: List[Tuple[str, str]],
        labels: Optional[List[int]] = None,
        filename: str = "interaction_list.txt"
    ) -> str:
        """
        Save interaction list with optional labels.
        
        Args:
            protein_pairs: List of (protein1, protein2) tuples
            labels: Optional list of labels (0 or 1)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (p1, p2) in enumerate(protein_pairs):
                if labels is not None and i < len(labels):
                    f.write(f"{p1},{p2},{labels[i]}\n")
                else:
                    f.write(f"{p1},{p2}\n")
        
        self.logger.info(f"Saved {len(protein_pairs)} interaction pairs to {output_path}")
        
        return str(output_path)
    
    def save_fasta(
        self,
        sequences: Dict[str, str],
        filename: str = "proteins.fasta"
    ) -> str:
        """
        Save protein sequences to FASTA format.
        
        Args:
            sequences: Dict mapping protein_id to sequence
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for protein_id, sequence in sequences.items():
                f.write(f">{protein_id}\n")
                # Write sequence in 60-character lines
                for i in range(0, len(sequence), 60):
                    f.write(f"{sequence[i:i+60]}\n")
        
        self.logger.info(f"Saved {len(sequences)} sequences to {output_path}")
        
        return str(output_path)
    
    def save_all(
        self,
        predictions: pd.DataFrame,
        attention_data: Dict[str, Dict[str, Any]],
        model_info: Dict[str, Any],
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        ig_results: Optional[Dict[str, Any]] = None,
        sequences: Optional[Dict[str, str]] = None,
        protein_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, str]:
        """
        Save all results to output directory.
        
        Args:
            predictions: Ensemble predictions DataFrame
            attention_data: Attention weights dict
            model_info: Model configuration info
            evaluation_metrics: Optional evaluation metrics
            ig_results: Optional IG attribution results
            sequences: Optional protein sequences
            protein_pairs: Optional protein pairs list
            
        Returns:
            Dict mapping result type to output path
        """
        output_paths = {}
        
        # Save predictions
        output_paths['predictions'] = self.save_predictions(predictions)
        
        # Save attention weights
        output_paths['attention'] = self.save_attention_weights(attention_data)
        
        # Save summary
        additional_info = {}
        if ig_results:
            additional_info['ig_analysis'] = {
                'enabled': True,
                'proteins_analyzed': len(ig_results.get('ig_results', {}))
            }
        
        output_paths['summary'] = self.save_summary(
            total_pairs=len(predictions),
            model_info=model_info,
            evaluation_metrics=evaluation_metrics,
            additional_info=additional_info if additional_info else None
        )
        
        # Save IG results if available
        if ig_results and 'ig_results' in ig_results:
            output_paths['ig'] = self.save_ig_results(ig_results['ig_results'])
        
        # Save sequences if provided
        if sequences:
            output_paths['fasta'] = self.save_fasta(sequences)
        
        # Save protein pairs if provided
        if protein_pairs:
            output_paths['interactions'] = self.save_interaction_list(protein_pairs)
        
        self.logger.info(f"All results saved to {self.output_dir}")
        
        return output_paths


def format_evaluation_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format evaluation metrics as a readable report.
    
    Args:
        metrics: Evaluation metrics dictionary
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "EVALUATION METRICS REPORT",
        "=" * 50,
        "",
    ]
    
    # Main metrics
    if 'auroc' in metrics:
        lines.append(f"AUROC:      {metrics['auroc']:.4f}")
    if 'aupr' in metrics:
        lines.append(f"AUPR:       {metrics['aupr']:.4f}")
    if 'accuracy' in metrics:
        lines.append(f"Accuracy:   {metrics['accuracy']:.4f}")
    if 'f1_binary' in metrics:
        lines.append(f"F1 (binary):{metrics['f1_binary']:.4f}")
    if 'f1_macro' in metrics:
        lines.append(f"F1 (macro): {metrics['f1_macro']:.4f}")
    
    lines.append("")
    
    # Sample counts
    if 'total_samples' in metrics:
        lines.append(f"Total samples:    {metrics['total_samples']}")
    if 'positive_samples' in metrics:
        lines.append(f"Positive samples: {metrics['positive_samples']}")
    if 'negative_samples' in metrics:
        lines.append(f"Negative samples: {metrics['negative_samples']}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        lines.append("")
        lines.append("Confusion Matrix:")
        # Handle different confusion matrix shapes (1x1 or 2x2)
        if isinstance(cm, (list, np.ndarray)) and len(cm) >= 2 and len(cm[0]) >= 2:
            lines.append(f"              Predicted")
            lines.append(f"              0      1")
            lines.append(f"Actual 0    {cm[0][0]:5d}  {cm[0][1]:5d}")
            lines.append(f"       1    {cm[1][0]:5d}  {cm[1][1]:5d}")
        else:
            # Single-class case
            lines.append(f"  (Single class in data, full matrix not available)")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)
