#!/usr/bin/env python3
"""
CIS Interpretability Analysis Module
Handles interpretability analysis for CIS models, including Hadamard product calculation and FAISS retrieval
"""

import torch
import torch.nn as nn
import numpy as np
import json
import lmdb
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HadamardProductExtractor:
    """Hadamard Product Extractor - Extracts Hadamard product features for protein pairs from the model"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Hadamard product extractor
        
        Args:
            model: Configured model
            device: Compute device
        """
        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        
        # Register hooks to capture Hadamard products
        self.hadamard_products = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture Hadamard products"""
        def hadamard_hook(module, input, output):
            """Capture output of Hadamard product operation"""
            if len(input) == 2:  # Ensure binary operation
                hadamard_result = torch.mul(input[0], input[1])
                self.hadamard_products.append(hadamard_result.detach().cpu().numpy())
        
        # Find Hadamard product operations in the model
        for name, module in self.model.named_modules():
            if hasattr(module, 'forward') and 'hadamard' in name.lower():
                module.register_forward_hook(hadamard_hook)
            elif hasattr(module, 'forward') and 'interaction' in name.lower():
                # Check if it's a Hadamard product interaction unit
                if hasattr(module, '__class__') and 'Hadamard' in module.__class__.__name__:
                    module.register_forward_hook(hadamard_hook)
    
    def extract_hadamard_products(self, protein1_emb: torch.Tensor, 
                                 protein2_emb: torch.Tensor) -> np.ndarray:
        """
        Extract Hadamard product for protein pair
        
        Args:
            protein1_emb: Protein 1 embedding [1, embedding_dim]
            protein2_emb: Protein 2 embedding [1, embedding_dim]
        
        Returns:
            Hadamard product features [embedding_dim]
        """
        self.hadamard_products.clear()
        
        with torch.no_grad():
            # Ensure inputs are on the correct device
            protein1_emb = protein1_emb.to(self.device)
            protein2_emb = protein2_emb.to(self.device)
            
            # Run model forward pass
            _ = self.model(protein1_emb, protein2_emb)
            
            # If no Hadamard product captured via hooks, compute directly
            if not self.hadamard_products:
                hadamard_product = torch.mul(protein1_emb, protein2_emb)
                return hadamard_product.squeeze().cpu().numpy()
            
            # Return the last Hadamard product (usually the one we need)
            return self.hadamard_products[-1].squeeze()


class CisInterpretabilityEngine:
    """FAISS-based CIS interpretability engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Store FAISS indices for all samples, positive samples, and negative samples for each model
        self.faiss_indices = {}  # model_name -> {'all': index, 'positive': index, 'negative': index}
        self.protein_pair_mappings = {}  # model_name -> {'all': mapping, 'positive': mapping, 'negative': mapping}

    def _pad_with_none(self, samples: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """When retrieval results are empty or less than top_k, pad with None to reach top_k length"""
        padded = list(samples) if samples else []
        for i in range(len(padded), top_k):
            padded.append({
                'training_pair': None,
                'label': None,
                'similarity': None,
                'is_positive': None,
                'rank': i + 1
            })
        # Standardize rank field
        for i, s in enumerate(padded):
            if 'rank' not in s or s['rank'] is None:
                s['rank'] = i + 1
        return padded

    def generate_cis_interpretability(self, model_name: str, model_config: Dict[str, Any], 
                                    test_protein_pairs: List[Tuple[str, str]], 
                                    predictions: np.ndarray, 
                                    global_config: Dict[str, Any] = None,
                                    output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate CIS interpretability for the given model
        
        Args:
            model_name: Model name
            model_config: Model configuration dictionary
            test_protein_pairs: List of protein pairs to predict
            predictions: Model prediction results
            global_config: Global configuration dictionary
            output_dir: Output directory, if provided will save JSONL file
            
        Returns:
            Interpretability results dictionary
        """
        self.logger.info(f"Generating CIS interpretability for model {model_name}...")
        
        if not FAISS_AVAILABLE:
            return {'message': 'FAISS not installed, cannot perform CIS interpretability analysis'}
        
    # Get configuration parameters
        interpretability_lmdb_dir = model_config.get('Interpretability_lmdb_dir')
        
        # Prefer using test_protein_lmdb_path injected by caller (ensemble_predict_engine will inject correct path for each model)
        test_lmdb_dir = model_config.get('test_protein_lmdb_path')
        if not test_lmdb_dir:
            # Compatible with legacy logic: fallback to get from global config (Note: legacy logic defaults to model2 key)
            if global_config and 'test_protein_lmdb_dir' in global_config:
                test_lmdb_dir = global_config['test_protein_lmdb_dir'].get('model2')
            else:
                test_lmdb_dir = model_config.get('test_protein_lmdb_dir', {}).get('model2')
            
        protein_id_file = model_config.get('Interpretability_protein_id')
        top_k = model_config.get('top_k', 10)

        # Additional protection: ensure predictions and test_protein_pairs lengths match
        try:
            required_len = len(test_protein_pairs)
            # Convert predictions to numpy array for easier processing
            if isinstance(predictions, np.ndarray):
                preds_arr = predictions
            else:
                # If iterable object (list, generator, etc.), try converting to list/array first
                try:
                    preds_arr = np.array(list(predictions))
                except Exception:
                    # Final fallback: wrap single scalar as array
                    preds_arr = np.array([predictions])

            if preds_arr.shape[0] < required_len:
                self.logger.warning(f"Model {model_name} provided predictions length ({preds_arr.shape[0]}) less than number of test protein pairs ({required_len}), will use np.nan to pad to target length")
                pad_len = required_len - preds_arr.shape[0]
                preds_arr = np.concatenate([preds_arr, np.full((pad_len,), np.nan)])
            elif preds_arr.shape[0] > required_len:
                self.logger.warning(f"Model {model_name} provided predictions length ({preds_arr.shape[0]}) greater than number of test protein pairs ({required_len}), will truncate excess elements")
                preds_arr = preds_arr[:required_len]

            # Reassign to predictions to ensure safe index access later
            predictions = preds_arr
        except Exception as e:
            self.logger.warning(f"Exception occurred when aligning predictions lengths: {e}, will continue using original predictions (may cause index errors)")
        
        if not all([interpretability_lmdb_dir, test_lmdb_dir, protein_id_file]):
            self.logger.error(f"Model {model_name} missing required interpretability configuration")
            return {'message': 'Missing required interpretability configuration'}
        
        try:
            # Build or load training set FAISS index (including all, positive, negative)
            if model_name not in self.faiss_indices:
                self.logger.info(f"Building FAISS index for model {model_name}...")
                self._build_training_indices(model_name, interpretability_lmdb_dir, protein_id_file)
            
            # Find most similar training samples for each protein pair to predict
            interpretability_results = []
            
            for i, (protein1, protein2) in enumerate(test_protein_pairs):
                prediction_score = predictions[i] if i < len(predictions) else None
                prediction_label = 1 if prediction_score and prediction_score >= 0.5 else 0
                
                # Compute Hadamard product for protein pairs to predict (use test set LMDB)
                query_vector = self._compute_hadamard_product(
                    protein1, protein2, test_lmdb_dir
                )
                
                if query_vector is not None:
                    # Select retrieval strategy based on prediction results
                    if prediction_label == 1:
                        # Positive prediction: retrieve only from positive training samples
                        similar_samples = self._search_similar_training_samples(
                            model_name, query_vector, top_k, index_type='positive'
                        )
                        search_strategy = 'positive_only'
                    else:
                        # Negative prediction: retrieve from all training samples
                        similar_samples = self._search_similar_training_samples(
                            model_name, query_vector, top_k, index_type='all'
                        )
                        search_strategy = 'all_samples'
                    
                    # Pad to top_k to avoid downstream processing errors due to inconsistent lengths
                    similar_samples = self._pad_with_none(similar_samples, top_k)

                    interpretability_results.append({
                        'query_pair': (protein1, protein2),
                        'prediction_score': prediction_score,
                        'prediction_label': prediction_label,
                        'search_strategy': search_strategy,
                        'similar_training_samples': similar_samples
                    })
                else:
                    self.logger.warning(f"Cannot compute protein pair ({protein1}, {protein2}) embedding")
                    # Return top_k None placeholders to avoid triggering downstream shape/type errors
                    interpretability_results.append({
                        'query_pair': (protein1, protein2),
                        'prediction_score': prediction_score,
                        'prediction_label': prediction_label,
                        'search_strategy': 'error',
                        'similar_training_samples': self._pad_with_none([], top_k),
                        'error': 'Cannot get protein embedding'
                    })
            
            # If output directory provided, save as JSONL file
            if output_dir:
                self._save_cis_interpretability_jsonl(model_name, interpretability_results, output_dir)
            
            return {
                'model_name': model_name,
                'interpretability_results': interpretability_results,
                'total_queries': len(test_protein_pairs),
                'successful_queries': len([r for r in interpretability_results if 'error' not in r])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating CIS interpretability: {str(e)}")
            return {'message': f'Error generating interpretability: {str(e)}'}
    
    def _build_training_indices(self, model_name: str, lmdb_dir: str, protein_id_file: str):
        """Build FAISS index for training set (all, positive, negative)"""
        self.logger.info(f"Building training set FAISS index for model {model_name}...")
        
        # Read training set protein pairs and labels
        training_pairs = []
        training_labels = []
        
        with open(protein_id_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    protein1, protein2, label = parts[0], parts[1], int(parts[2])
                    training_pairs.append((protein1, protein2))
                    training_labels.append(label)
        
        self.logger.info(f"Read {len(training_pairs)} training samples")
        
        # Compute Hadamard products for all training samples
        training_vectors = []
        valid_indices = []
        
        for i, (protein1, protein2) in enumerate(tqdm(training_pairs, desc="Computing training sample embeddings")):
            hadamard_product = self._compute_hadamard_product(protein1, protein2, lmdb_dir)
            if hadamard_product is not None:
                training_vectors.append(hadamard_product)
                valid_indices.append(i)
        
        if not training_vectors:
            raise ValueError("Cannot compute any training sample embeddings")
        
        # Convert to numpy array
        training_matrix = np.vstack(training_vectors).astype(np.float32)
        self.logger.info(f"Training matrix shape: {training_matrix.shape}")
        
        # Create mapping of labels and vectors
        valid_pairs = [training_pairs[i] for i in valid_indices]
        valid_labels = [training_labels[i] for i in valid_indices]
        
        # Separate positive and negative samples
        positive_vectors = []
        negative_vectors = []
        positive_pairs = []
        negative_pairs = []
        positive_indices = []
        negative_indices = []
        
        for i, (vector, pair, label) in enumerate(zip(training_matrix, valid_pairs, valid_labels)):
            if label == 1:
                positive_vectors.append(vector)
                positive_pairs.append(pair)
                positive_indices.append(i)
            else:
                negative_vectors.append(vector)
                negative_pairs.append(pair)
                negative_indices.append(i)
        
        # Build three FAISS indices: all, positive, negative
        indices = {}
        mappings = {}
        
        # 1. All samples index
        dimension = training_matrix.shape[1]
        all_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(training_matrix)
        all_index.add(training_matrix)
        indices['all'] = all_index
        mappings['all'] = {
            i: (valid_pairs[i], valid_labels[i])
            for i in range(len(valid_pairs))
        }
        
        # 2. Positive samples index
        if positive_vectors:
            positive_matrix = np.vstack(positive_vectors).astype(np.float32)
            positive_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(positive_matrix)
            positive_index.add(positive_matrix)
            indices['positive'] = positive_index
            mappings['positive'] = {
                i: (positive_pairs[i], 1)
                for i in range(len(positive_pairs))
            }
            self.logger.info(f"Positive samples index contains {positive_index.ntotal} samples")
        else:
            indices['positive'] = None
            mappings['positive'] = {}
            self.logger.warning("No positive training samples")
        
        # 3. Negative samples index
        if negative_vectors:
            negative_matrix = np.vstack(negative_vectors).astype(np.float32)
            negative_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(negative_matrix)
            negative_index.add(negative_matrix)
            indices['negative'] = negative_index
            mappings['negative'] = {
                i: (negative_pairs[i], 0)
                for i in range(len(negative_pairs))
            }
            self.logger.info(f"Negative samples index contains {negative_index.ntotal} samples")
        else:
            indices['negative'] = None
            mappings['negative'] = {}
            self.logger.warning("No negative training samples")
        
        # Store indices and mappings
        self.faiss_indices[model_name] = indices
        self.protein_pair_mappings[model_name] = mappings
        
        self.logger.info(f"Successfully built FAISS index, all samples: {all_index.ntotal}")
    
    def _search_similar_training_samples(self, model_name: str, query_vector: np.ndarray, 
                                       top_k: int, index_type: str = 'all') -> List[Dict[str, Any]]:
        """Search for most similar samples in training set
        
        Args:
            model_name: Model name
            query_vector: Query vector
            top_k: Number of similar samples to return
            index_type: Index type, can be 'all', 'positive', 'negative'
        """
        if model_name not in self.faiss_indices:
            return []
        
        indices = self.faiss_indices[model_name]
        mappings = self.protein_pair_mappings[model_name]
        
        if index_type not in indices or indices[index_type] is None:
            self.logger.warning(f"Model {model_name} does not have {index_type} type index")
            return []
        
        index = indices[index_type]
        pair_mapping = mappings[index_type]
        
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search for most similar samples
        similarities, indices_result = index.search(query_vector, min(top_k, index.ntotal))
        
        similar_samples = []
        for sim, idx in zip(similarities[0], indices_result[0]):
            if idx in pair_mapping:
                (protein1, protein2), label = pair_mapping[idx]
                similar_samples.append({
                    'training_pair': (protein1, protein2),
                    'label': label,
                    'similarity': float(sim),
                    'is_positive': label == 1
                })
        
        return similar_samples
    
    def _compute_hadamard_product(self, protein1: str, protein2: str, lmdb_dir: str) -> Optional[np.ndarray]:
        """Compute Hadamard product of two proteins"""
        try:
            import lmdb
            
            with lmdb.open(lmdb_dir, readonly=True) as env:
                with env.begin() as txn:
                    # Get protein 1 embedding
                    emb1_bytes = txn.get(protein1.encode())
                    if emb1_bytes is None:
                        self.logger.warning(f"Protein {protein1} embedding not found")
                        return None
                    
                    # Get protein 2 embedding
                    emb2_bytes = txn.get(protein2.encode())
                    if emb2_bytes is None:
                        self.logger.warning(f"Protein {protein2} embedding not found")
                        return None
                    
                    # Load embeddings from binary data (LMDB stores raw numpy array binary data)
                    emb1 = np.frombuffer(emb1_bytes, dtype=np.float32)
                    emb2 = np.frombuffer(emb2_bytes, dtype=np.float32)
                    
                    # Compute Hadamard product
                    hadamard_product = emb1 * emb2
                    
                    return hadamard_product
                    
        except Exception as e:
            self.logger.error(f"Error computing Hadamard product: {str(e)}")
            return None 

    def _save_cis_interpretability_jsonl(self, model_name: str, interpretability_results: List[Dict], 
                                       output_dir: Path):
        """Save CIS interpretability results as JSONL file"""
        output_file = output_dir / f"{model_name}_cis_interpretability.jsonl"
        
        self.logger.info(f"Saving CIS interpretability results to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in interpretability_results:
                # Create simplified output format
                jsonl_record = {
                    'protein_pair_id': f"{result['query_pair'][0]}_{result['query_pair'][1]}",
                    'protein1_id': result['query_pair'][0],
                    'protein2_id': result['query_pair'][1],
                    'prediction_score': result.get('prediction_score'),
                    'prediction_label': result.get('prediction_label'),
                    'search_strategy': result.get('search_strategy'),
                    'top_similar_training_samples': result.get('similar_training_samples', [])
                }
                
                # If error exists, record error information
                if 'error' in result:
                    jsonl_record['error'] = result['error']
                
                f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Successfully saved {len(interpretability_results)} CIS interpretability results") 