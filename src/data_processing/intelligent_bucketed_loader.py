#!/usr/bin/env python3
"""
Intelligent Bucketed Data Loader for Single LMDB with Smart Batching and Dynamic Padding

This module provides intelligent data loading from a single bucketed LMDB file,
with smart batch construction based on bucket information, dynamic padding/masking,
and memory-efficient embedding retrieval.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lmdb
import numpy as np
import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from cachetools import LRUCache
from ..utils.helpers import create_generator


class IntelligentBucketedDataset(Dataset):
    """
    Dataset that loads from single bucketed LMDB with smart batch construction
    """
    
    def __init__(self, lmdb_path: str, interaction_pairs: List[Tuple[str, str, int]], 
                 fasta_sequences: Dict[str, str], embedding_dim: int = 1280,
                 max_length: int = 1024, cache_size: int = 10000, 
                 enable_dynamic_loading: bool = True, logger=None):
        """
        Initialize dataset
        
        Args:
            lmdb_path (str): Path to single bucketed LMDB file
            interaction_pairs (List): List of (protein1, protein2, label) tuples
            fasta_sequences (Dict): Mapping of protein_id -> sequence
            embedding_dim (int): Embedding dimension
            max_length (int): Maximum sequence length for padding
            cache_size (int): Number of embeddings to cache
            enable_dynamic_loading (bool): Enable dynamic loading for large embeddings
            logger: Logger instance
        """
        self.lmdb_path = lmdb_path
        self.interaction_pairs = interaction_pairs
        self.fasta_sequences = fasta_sequences
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.cache_size = cache_size
        self.enable_dynamic_loading = enable_dynamic_loading
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize LMDB environment
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        # Load metadata and bucket information
        self._load_metadata()
        
        # Protein to bucket mapping for efficient batch construction
        self.protein_to_bucket = {}
        self.bucket_boundaries = []
        
        # Cache for embeddings
        self.embedding_cache = LRUCache(maxsize=self.cache_size)
        self.cache_lock = threading.RLock()
        
        # Protein key mapping (original_id -> bucketed_key)
        self.protein_key_map = {}
        
        self._build_protein_index()
        
        self.logger.info(f"初始化智能分桶数据集: {len(self.interaction_pairs)} 个交互对")
        self.logger.info(f"分桶边界: {self.bucket_boundaries}")
        self.logger.info(f"缓存大小: {cache_size}")
    
    def _load_metadata(self):
        """Load metadata from LMDB"""
        with self.env.begin() as txn:
            # Load bucket info
            bucket_info = txn.get(b'_bucket_info')
            if bucket_info:
                bucket_data = json.loads(bucket_info.decode())
                self.bucket_boundaries = bucket_data.get('boundaries', [])
                self.num_buckets = bucket_data.get('num_buckets', 1)
            else:
                self.logger.warning("未找到分桶信息，使用默认设置")
                self.bucket_boundaries = [100, 200, 400, 800]
                self.num_buckets = 5
            
            # Load precision info
            precision_info = txn.get(b'_precision_info')
            if precision_info:
                precision_data = json.loads(precision_info.decode())
                self.target_precision = precision_data.get('target_precision', 'fp32')
                self.original_precision = precision_data.get('original_precision', 'fp32')
            else:
                self.target_precision = 'fp32'
                self.original_precision = 'fp32'
    
    def _build_protein_index(self):
        """Build protein index for fast lookup"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            
            while cursor.key():
                key = cursor.key().decode()
                
                # Skip metadata keys
                if key.startswith('_'):
                    cursor.next()
                    continue
                
                # Parse bucket key: bucket_{bucket_id}_len_{max_len}_{protein_id}
                if key.startswith('bucket_'):
                    try:
                        parts = key.split('_')
                        if len(parts) >= 4:
                            bucket_id = int(parts[1])
                            bucket_max_len = int(parts[3])
                            protein_id = '_'.join(parts[4:])  # Handle protein IDs with underscores
                            
                            self.protein_key_map[protein_id] = key
                            self.protein_to_bucket[protein_id] = bucket_id
                    
                    except (ValueError, IndexError):
                        self.logger.warning(f"无法解析键格式: {key}")
                
                cursor.next()
        
        self.logger.info(f"建立蛋白质索引: {len(self.protein_key_map)} 个蛋白质")
    
    def _get_bucket_for_length(self, seq_length: int) -> int:
        """Get bucket ID for given sequence length"""
        for i, boundary in enumerate(self.bucket_boundaries):
            if seq_length <= boundary:
                return i
        return len(self.bucket_boundaries)
    
    def _load_embedding(self, protein_id: str) -> Optional[torch.Tensor]:
        """
        Load embedding for protein with caching
        
        Args:
            protein_id (str): Protein ID
            
        Returns:
            torch.Tensor or None: Embedding tensor [seq_len, embedding_dim]
        """
        # Check cache first
        with self.cache_lock:
            if protein_id in self.embedding_cache:
                return self.embedding_cache[protein_id]
        
        # Get bucketed key
        if protein_id not in self.protein_key_map:
            self.logger.warning(f"蛋白质 {protein_id} 未找到分桶键")
            return None
        
        bucketed_key = self.protein_key_map[protein_id]
        
        # Load from LMDB
        with self.env.begin() as txn:
            data = txn.get(bucketed_key.encode())
            if data is None:
                self.logger.warning(f"蛋白质 {protein_id} 数据未找到")
                return None
            
            try:
                # Parse embedding based on precision
                embedding = self._parse_embedding_data(data)
                
                if embedding is not None:
                    # Cache if space available
                    with self.cache_lock:
                        self.embedding_cache[protein_id] = embedding
                
                return embedding
                
            except Exception as e:
                self.logger.warning(f"解析蛋白质 {protein_id} 嵌入失败: {e}")
                return None
    
    def _parse_embedding_data(self, data: bytes) -> Optional[torch.Tensor]:
        """
        Parse embedding data based on precision
        
        Args:
            data (bytes): Raw embedding data
            
        Returns:
            torch.Tensor or None: Parsed embedding
        """
        try:
            if self.target_precision == 'fp32':
                arr = np.frombuffer(data, dtype=np.float32)
            elif self.target_precision == 'fp16':
                # 保持fp16精度，不转换为fp32
                arr = np.frombuffer(data, dtype=np.float16)
            elif self.target_precision == 'bf16':
                # 保持bf16精度，不转换为fp32
                uint16_data = np.frombuffer(data, dtype=np.uint16)
                tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
                return tensor.view(-1, self.embedding_dim)
            elif self.target_precision == 'int8':
                # First 4 bytes are scale, rest is quantized data
                scale = struct.unpack('f', data[:4])[0]
                quantized = np.frombuffer(data[4:], dtype=np.int8)
                arr = quantized.astype(np.float32) * scale / 127.0
            else:
                # Try pickle fallback
                arr = pickle.loads(data)
                if hasattr(arr, 'shape'):
                    arr = arr.astype(np.float32)
            
            # Reshape if needed
            if hasattr(arr, 'reshape') and len(arr) % self.embedding_dim == 0:
                seq_len = len(arr) // self.embedding_dim
                # Make array writable before converting to tensor
                arr_copy = arr.copy()
                embedding = torch.from_numpy(arr_copy.reshape(seq_len, self.embedding_dim))
                return embedding
            else:
                # Make array writable before converting to tensor
                arr_copy = arr.copy()
                return torch.from_numpy(arr_copy)
                
        except Exception as e:
            self.logger.warning(f"解析嵌入数据失败: {e}")
            return None
    
    def _apply_dynamic_padding_and_mask(self, embedding: torch.Tensor, target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dynamic padding and create attention mask
        
        Args:
            embedding (torch.Tensor): Original embedding [seq_len, embedding_dim]
            target_length (int): Target padded length
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (padded_embedding, attention_mask)
        """
        seq_len = embedding.size(0)
        
        if seq_len >= target_length:
            # Truncate if too long
            padded_embedding = embedding[:target_length]
            attention_mask = torch.ones(target_length)
        else:
            # Pad if too short
            padding = torch.zeros(target_length - seq_len, self.embedding_dim)
            padded_embedding = torch.cat([embedding, padding], dim=0)
            
            attention_mask = torch.zeros(target_length)
            attention_mask[:seq_len] = 1
        
        return padded_embedding, attention_mask
    
    def __len__(self):
        return len(self.interaction_pairs)
    
    def __getitem__(self, idx):
        """
        Get single interaction pair with embeddings
        
        Returns:
            Dict containing protein embeddings, masks, and labels
        """
        protein1, protein2, label = self.interaction_pairs[idx]
        
        # Load embeddings
        embedding1 = self._load_embedding(protein1)
        embedding2 = self._load_embedding(protein2)
        
        if embedding1 is None or embedding2 is None:
            # Return dummy data for missing embeddings
            self.logger.warning(f"缺失嵌入: {protein1} 或 {protein2}")
            dummy_embedding = torch.zeros(self.max_length, self.embedding_dim)
            dummy_mask = torch.zeros(self.max_length)
            
            return {
                'protein1_seq': dummy_embedding,
                'protein1_mask': dummy_mask,
                'protein2_seq': dummy_embedding,  
                'protein2_mask': dummy_mask,
                'label': torch.tensor(label, dtype=torch.float32),
                'protein1_id': protein1,
                'protein2_id': protein2
            }
        
        # Get bucket info for smart batching
        bucket1 = self.protein_to_bucket.get(protein1, -1)
        bucket2 = self.protein_to_bucket.get(protein2, -1)
        
        # Determine target length based on bucket info
        seq_len1 = embedding1.size(0)
        seq_len2 = embedding2.size(0)
        
        # Use adaptive padding based on actual lengths and bucket boundaries
        max_len_in_pair = max(seq_len1, seq_len2)
        
        # Find appropriate bucket boundary for this pair
        target_length = self.max_length
        for boundary in self.bucket_boundaries:
            if max_len_in_pair <= boundary:
                target_length = min(boundary, self.max_length)
                break
        
        # Apply dynamic padding
        padded_embedding1, mask1 = self._apply_dynamic_padding_and_mask(embedding1, target_length)
        padded_embedding2, mask2 = self._apply_dynamic_padding_and_mask(embedding2, target_length)
        
        return {
            'protein1_seq': padded_embedding1,
            'protein1_mask': mask1,
            'protein2_seq': padded_embedding2,
            'protein2_mask': mask2,
            'label': torch.tensor(label, dtype=torch.float32),
            'protein1_id': protein1,
            'protein2_id': protein2,
            'bucket1': bucket1,
            'bucket2': bucket2,
            'target_length': target_length
        }
    
    def get_bucket_info(self, protein_id: str) -> Dict[str, Any]:
        """Get bucket information for a protein"""
        bucket_id = self.protein_to_bucket.get(protein_id, -1)
        seq_length = len(self.fasta_sequences.get(protein_id, ''))
        
        return {
            'bucket_id': bucket_id,
            'sequence_length': seq_length,
            'bucket_boundary': self.bucket_boundaries[bucket_id] if 0 <= bucket_id < len(self.bucket_boundaries) else None
        }
    
    def __del__(self):
        """Clean up LMDB environment"""
        if hasattr(self, 'env'):
            self.env.close()


class SmartBucketBatchSampler:
    """
    Smart batch sampler that groups samples by bucket similarity
    """
    
    def __init__(self, dataset: IntelligentBucketedDataset, batch_size: int = 32,
                 drop_last: bool = False, shuffle: bool = True):
        """
        Initialize smart batch sampler
        
        Args:
            dataset: IntelligentBucketedDataset instance
            batch_size (int): Target batch size
            drop_last (bool): Whether to drop last incomplete batch
            shuffle (bool): Whether to shuffle batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Group samples by bucket similarity
        self.bucket_groups = self._group_by_buckets()
    
    def _group_by_buckets(self) -> Dict[Tuple[int, int], List[int]]:
        """Group sample indices by bucket pairs"""
        bucket_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            protein1, protein2, _ = self.dataset.interaction_pairs[idx]
            bucket1 = self.dataset.protein_to_bucket.get(protein1, -1)
            bucket2 = self.dataset.protein_to_bucket.get(protein2, -1)
            
            # Create bucket pair key (sorted for consistency)
            bucket_pair = tuple(sorted([bucket1, bucket2]))
            bucket_groups[bucket_pair].append(idx)
        
        return bucket_groups
    
    def __iter__(self):
        """Yield batches grouped by bucket similarity"""
        all_batches = []
        
        # Create batches within each bucket group
        for bucket_pair, indices in self.bucket_groups.items():
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Split into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batches if needed
        if self.shuffle:
            np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """Return number of batches"""
        total_batches = 0
        for indices in self.bucket_groups.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches


def smart_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Smart collate function that performs secondary padding based on batch contents
    
    Args:
        batch (List[Dict]): List of sample dictionaries
        
    Returns:
        Dict[str, torch.Tensor]: Batched and padded tensors
    """
    # Find maximum target length in this batch
    max_target_length = max(sample['target_length'] for sample in batch)
    
    # Collect all tensors
    protein1_seqs = []
    protein1_masks = []
    protein2_seqs = []
    protein2_masks = []
    labels = []
    
    for sample in batch:
        # Secondary padding if needed
        if sample['protein1_seq'].size(0) < max_target_length:
            seq1 = sample['protein1_seq']
            mask1 = sample['protein1_mask']
            
            # Pad to max target length in batch
            pad_len = max_target_length - seq1.size(0)
            seq1_padded = F.pad(seq1, (0, 0, 0, pad_len))
            mask1_padded = F.pad(mask1, (0, pad_len))
            
            protein1_seqs.append(seq1_padded)
            protein1_masks.append(mask1_padded)
        else:
            protein1_seqs.append(sample['protein1_seq'][:max_target_length])
            protein1_masks.append(sample['protein1_mask'][:max_target_length])
        
        if sample['protein2_seq'].size(0) < max_target_length:
            seq2 = sample['protein2_seq']
            mask2 = sample['protein2_mask']
            
            pad_len = max_target_length - seq2.size(0)
            seq2_padded = F.pad(seq2, (0, 0, 0, pad_len))
            mask2_padded = F.pad(mask2, (0, pad_len))
            
            protein2_seqs.append(seq2_padded)
            protein2_masks.append(mask2_padded)
        else:
            protein2_seqs.append(sample['protein2_seq'][:max_target_length])
            protein2_masks.append(sample['protein2_mask'][:max_target_length])
        
        labels.append(sample['label'])
    
    # Stack tensors
    return {
        'protein1_seq': torch.stack(protein1_seqs),
        'protein1_mask': torch.stack(protein1_masks),
        'protein2_seq': torch.stack(protein2_seqs),
        'protein2_mask': torch.stack(protein2_masks),
        'label': torch.stack(labels)
    }


def create_intelligent_bucketed_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create data loaders using intelligent bucketed dataset
    
    Args:
        config (Dict): Configuration dictionary containing:
            - embedding_file: Path to bucketed LMDB file
            - train_file: Training pairs file
            - test_files: Dict of test files
            - fasta_file: FASTA sequences file
            - batch_size: Batch size
            - cache_size: Embedding cache size
            - max_length: Maximum sequence length
            
    Returns:
        Dict[str, DataLoader]: Dictionary of data loaders
    """
    from .fasta_parser import parse_fasta_file
    
    logger = logging.getLogger(__name__)
    
    # Load FASTA sequences
    fasta_file = config.get('fasta_file')
    if not fasta_file or not Path(fasta_file).exists():
        raise FileNotFoundError(f"FASTA文件未找到: {fasta_file}")
    
    logger.info(f"加载FASTA文件: {fasta_file}")
    fasta_sequences = parse_fasta_file(fasta_file)
    logger.info(f"加载了 {len(fasta_sequences)} 个序列")
    
    # Load interaction pairs
    def load_pairs(pairs_file: str) -> List[Tuple[str, str, int]]:
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], int(parts[2])))
        return pairs
    
    data_loaders = {}
    
    # Common parameters
    embedding_file = config['embedding_file']
    batch_size = config.get('batch_size', 32)
    cache_size = config.get('cache_size', 10000)
    max_length = config.get('max_length', 1024)
    
    # Create training loader
    if 'train_file' in config:
        train_pairs = load_pairs(config['train_file'])
        logger.info(f"加载训练数据: {len(train_pairs)} 对")
        
        train_dataset = IntelligentBucketedDataset(
            lmdb_path=embedding_file,
            interaction_pairs=train_pairs,
            fasta_sequences=fasta_sequences,
            embedding_dim=config.get('embedding_dim', 1280),
            max_length=max_length,
            cache_size=cache_size,
            logger=logger
        )
        
        # Use smart batch sampler for training
        train_sampler = SmartBucketBatchSampler(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        data_loaders['train'] = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=smart_collate_fn,
            num_workers=0,
            pin_memory=True,
            generator=create_generator(config.get('seed', 42))
        )
    
    # Create test loaders
    test_files = config.get('test_files', {})
    for test_name, test_file in test_files.items():
        test_pairs = load_pairs(test_file)
        logger.info(f"加载测试数据 {test_name}: {len(test_pairs)} 对")
        
        test_dataset = IntelligentBucketedDataset(
            lmdb_path=embedding_file,
            interaction_pairs=test_pairs,
            fasta_sequences=fasta_sequences,
            embedding_dim=config.get('embedding_dim', 1280),
            max_length=max_length,
            cache_size=cache_size // 2,  # Smaller cache for test sets
            logger=logger
        )
        
        # Use smart batch sampler for test data too
        test_sampler = SmartBucketBatchSampler(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        data_loaders[f'{test_name}_dataset'] = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=smart_collate_fn,
            num_workers=1,
            pin_memory=True,
            generator=create_generator(config.get('seed', 42))
        )
    
    logger.info(f"创建了 {len(data_loaders)} 个智能分桶数据加载器")
    return data_loaders
