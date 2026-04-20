"""
Sequence length bucketing and caching system for efficient batch processing
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import logging
from tqdm import tqdm
from .lmdb_loader import ESMLMDBLoader
from .parallel_lmdb_loader import ParallelLMDBLoader, BatchGPUTransfer

logger = logging.getLogger(__name__)


class SequenceBucketing:
    """
    System for grouping proteins by sequence length for efficient batching
    """
    
    def __init__(self, bucket_boundaries: List[int] = None, max_length: int = 2048):
        """
        Initialize bucketing system
        
        Args:
            bucket_boundaries (List[int]): Bucket boundaries for sequence lengths
            max_length (int): Maximum sequence length to handle
        """
        if bucket_boundaries is None:
            # Default buckets: powers of 2 up to max_length
            bucket_boundaries = [32, 64, 128, 256, 512, 1024, max_length]
        
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.max_length = max_length
        
        # Dictionary mapping bucket_id -> list of (protein_id, length)
        self.buckets = defaultdict(list)
        
        # Dictionary mapping protein_id -> bucket_id
        self.protein_to_bucket = {}
        
        logger.debug(f"初始化分桶，边界: {self.bucket_boundaries}")
    
    def _get_bucket_id(self, length: int) -> int:
        """
        Get bucket ID for a given sequence length
        
        Args:
            length (int): Sequence length
            
        Returns:
            int: Bucket ID
        """
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries) - 1  # Last bucket for very long sequences
    
    def add_protein(self, protein_id: str, length: int):
        """
        Add protein to appropriate bucket
        
        Args:
            protein_id (str): Protein identifier
            length (int): Sequence length
        """
        if length > self.max_length:
            logger.warning(f"Protein {protein_id} length {length} exceeds max_length {self.max_length}")
            length = self.max_length
        
        bucket_id = self._get_bucket_id(length)
        self.buckets[bucket_id].append((protein_id, length))
        self.protein_to_bucket[protein_id] = bucket_id
    
    def get_bucket_for_protein(self, protein_id: str) -> Optional[int]:
        """Get bucket ID for a protein"""
        return self.protein_to_bucket.get(protein_id)
    
    def get_bucket_proteins(self, bucket_id: int) -> List[Tuple[str, int]]:
        """Get all proteins in a bucket"""
        return self.buckets.get(bucket_id, [])
    
    def get_compatible_bucket(self, protein1_id: str, protein2_id: str) -> Optional[int]:
        """
        Find a compatible bucket for a protein pair
        
        Args:
            protein1_id (str): First protein ID
            protein2_id (str): Second protein ID
            
        Returns:
            int: Compatible bucket ID or None if proteins not found
        """
        bucket1 = self.get_bucket_for_protein(protein1_id)
        bucket2 = self.get_bucket_for_protein(protein2_id)
        
        if bucket1 is None or bucket2 is None:
            return None
        
        # Use the larger bucket (can accommodate both proteins)
        return max(bucket1, bucket2)
    
    def create_batches(self, protein_pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[List[Tuple[str, str]]]:
        """
        Create batches of protein pairs grouped by compatible buckets
        
        Args:
            protein_pairs (List[Tuple[str, str]]): List of protein pairs
            batch_size (int): Target batch size
            
        Returns:
            List[List[Tuple[str, str]]]: Batched protein pairs
        """
        # Group pairs by compatible bucket
        bucket_pairs = defaultdict(list)
        
        for pair in protein_pairs:
            bucket_id = self.get_compatible_bucket(pair[0], pair[1])
            if bucket_id is not None:
                bucket_pairs[bucket_id].append(pair)
        
        # Create batches within each bucket
        batches = []
        for bucket_id, pairs in bucket_pairs.items():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batches.append(batch)
        
        return batches
    
    def get_stats(self) -> Dict:
        """Get bucketing statistics"""
        stats = {
            'total_proteins': len(self.protein_to_bucket),
            'num_buckets': len(self.buckets),
            'bucket_boundaries': self.bucket_boundaries,
            'bucket_sizes': {}
        }
        
        for bucket_id, proteins in self.buckets.items():
            boundary = self.bucket_boundaries[bucket_id] if bucket_id < len(self.bucket_boundaries) else "inf"
            stats['bucket_sizes'][f'bucket_{bucket_id}_(<={boundary})'] = len(proteins)
        
        return stats


class ProteinEmbeddingCache:
    """
    GPU cache for protein embeddings with bucketing-aware padding
    """
    
    def __init__(self, device: torch.device, max_cache_size: int = 10000):
        """
        Initialize embedding cache
        
        Args:
            device (torch.device): Device to store cached embeddings
            max_cache_size (int): Maximum number of embeddings to cache
        """
        self.device = device
        self.max_cache_size = max_cache_size
        
        # Cache storage: protein_id -> (embedding, original_length)
        self.cache = {}
        self.access_order = []  # For LRU eviction
        
        logger.info(f"在 {device} 上初始化嵌入缓存，最大容量 {max_cache_size}")
    
    def _evict_lru(self):
        """Evict least recently used embedding"""
        if len(self.cache) >= self.max_cache_size and self.access_order:
            lru_protein = self.access_order.pop(0)
            if lru_protein in self.cache:
                del self.cache[lru_protein]
    
    def add_embedding(self, protein_id: str, embedding: torch.Tensor):
        """
        Add embedding to cache
        
        Args:
            protein_id (str): Protein identifier
            embedding (torch.Tensor): Protein embedding [L, embedding_dim]
        """
        # Evict if necessary
        self._evict_lru()
        
        # Store on GPU
        original_length = embedding.shape[0]
        cached_embedding = embedding.to(self.device)
        
        self.cache[protein_id] = (cached_embedding, original_length)
        
        # Update access order
        if protein_id in self.access_order:
            self.access_order.remove(protein_id)
        self.access_order.append(protein_id)
    
    def get_embedding(self, protein_id: str) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Get embedding from cache
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            Tuple[torch.Tensor, int]: (embedding, original_length) or None
        """
        if protein_id in self.cache:
            # Update access order
            self.access_order.remove(protein_id)
            self.access_order.append(protein_id)
            
            return self.cache[protein_id]
        
        return None
    
    def create_padded_batch(self, protein_ids: List[str], max_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create padded batch from cached embeddings
        
        Args:
            protein_ids (List[str]): List of protein IDs
            max_length (int): Maximum length for padding (auto-detect if None)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (padded_embeddings, attention_mask)
                padded_embeddings: [batch_size, max_length, embedding_dim]
                attention_mask: [batch_size, max_length] (1 for real tokens, 0 for padding)
        """
        embeddings = []
        lengths = []
        
        # Collect embeddings and lengths
        for protein_id in protein_ids:
            cached_data = self.get_embedding(protein_id)
            if cached_data is not None:
                embedding, original_length = cached_data
                embeddings.append(embedding)
                lengths.append(original_length)
            else:
                raise ValueError(f"Embedding not found in cache: {protein_id}")
        
        if not embeddings:
            raise ValueError("No valid embeddings found")
        
        # Determine max length
        if max_length is None:
            max_length = max(lengths)
        
        batch_size = len(embeddings)
        embedding_dim = embeddings[0].shape[1]
        
        # Create padded tensor
        padded_embeddings = torch.zeros(batch_size, max_length, embedding_dim, 
                                      device=self.device, dtype=embeddings[0].dtype)
        attention_mask = torch.zeros(batch_size, max_length, device=self.device, dtype=torch.bool)
        
        # Fill with actual embeddings
        for i, (embedding, length) in enumerate(zip(embeddings, lengths)):
            actual_length = min(length, max_length)
            padded_embeddings[i, :actual_length] = embedding[:actual_length]
            attention_mask[i, :actual_length] = True
        
        return padded_embeddings, attention_mask
    
    def preload_proteins(self, protein_ids: List[str], embedding_loader: ESMLMDBLoader, 
                        verbose=True, use_parallel=True, max_workers=4):
        """
        Preload protein embeddings into cache with optional parallel loading
        
        Args:
            protein_ids (List[str]): Protein IDs to preload
            embedding_loader (ESMLMDBLoader): LMDB loader
            verbose (bool): Whether to show progress
            use_parallel (bool): Whether to use parallel loading
            max_workers (int): Maximum number of worker threads for parallel loading
        """
        if verbose:
            logger.info(f"预加载 {len(protein_ids)} 个蛋白质嵌入...")
        
        # Filter out proteins already in cache
        proteins_to_load = [pid for pid in protein_ids if pid not in self.cache]
        
        if not proteins_to_load:
            if verbose:
                logger.info("所有嵌入已在缓存中")
            return
        
        if verbose:
            logger.info(f"需要加载 {len(proteins_to_load)} 个新嵌入")
        
        loaded = 0
        
        if use_parallel and len(proteins_to_load) > 10:  # Use parallel for larger batches
            try:
                # Use parallel LMDB loader
                parallel_loader = ParallelLMDBLoader(
                    str(embedding_loader.lmdb_path), 
                    max_workers=max_workers
                )
                
                # Load embeddings in parallel
                cpu_embeddings = parallel_loader.load_embeddings_parallel(
                    proteins_to_load, verbose=verbose
                )
                
                # Transfer to GPU in batches
                if cpu_embeddings:
                    if verbose:
                        logger.info("批量传输嵌入到GPU缓存...")
                    
                    gpu_embeddings = BatchGPUTransfer.transfer_embeddings_to_gpu(
                        cpu_embeddings, self.device, batch_size=100, verbose=verbose
                    )
                    
                    # Add to cache
                    for protein_id, (gpu_embedding, original_length) in gpu_embeddings.items():
                        self.cache[protein_id] = (gpu_embedding, original_length)
                        
                        # Update access order
                        if protein_id in self.access_order:
                            self.access_order.remove(protein_id)
                        self.access_order.append(protein_id)
                        
                        loaded += 1
                        
                        # Evict if necessary
                        if len(self.cache) > self.max_cache_size:
                            self._evict_lru()
                
            except Exception as e:
                logger.warning(f"并行加载失败，回退到串行模式: {e}")
                use_parallel = False
        
        if not use_parallel:
            # Fallback to sequential loading
            progress_bar = tqdm(proteins_to_load, desc="Loading embeddings", disable=not verbose)
            
            for protein_id in progress_bar:
                embedding = embedding_loader.get_embedding(protein_id)
                if embedding is not None:
                    self.add_embedding(protein_id, embedding)
                    loaded += 1
                    progress_bar.set_postfix({
                        'loaded': loaded, 
                        'cache_size': len(self.cache)
                    })
            
            progress_bar.close()
        
        if verbose:
            logger.info(f"已预加载 {loaded} 个新嵌入。缓存大小: {len(self.cache)}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'device': str(self.device),
            'memory_usage_mb': sum(
                embedding.element_size() * embedding.numel() 
                for embedding, _ in self.cache.values()
            ) / (1024 * 1024)
        }


def create_bucketing_system(embedding_loader: ESMLMDBLoader, 
                          protein_ids: List[str],
                          bucket_boundaries: List[int] = None,
                          fasta_file: str = None,
                          dynamic_bucketing: bool = True,
                          bucket_options: List[int] = [4, 8, 16, 32],
                          verbose: bool = True) -> SequenceBucketing:
    """
    Factory function to create and populate bucketing system
    
    Args:
        embedding_loader (ESMLMDBLoader): LMDB loader
        protein_ids (List[str]): List of protein IDs to bucket
        bucket_boundaries (List[int]): Custom bucket boundaries (if provided, disables dynamic bucketing)
        fasta_file (str): Path to FASTA file for fast length lookup
        dynamic_bucketing (bool): Whether to use dynamic bucketing based on data distribution
        bucket_options (List[int]): Possible bucket counts for dynamic bucketing
        verbose (bool): Whether to show progress
        
    Returns:
        SequenceBucketing: Populated bucketing system
    """
    if verbose:
        logger.info(f"为 {len(protein_ids)} 个蛋白质创建分桶...")
    
    # Collect sequence lengths for dynamic bucketing
    sequence_lengths = []
    
    # Use FASTA file for fast length lookup if available
    if fasta_file:
        from .fasta_parser import FastaLengthParser
        
        if verbose:
            logger.info("使用FASTA文件进行快速序列长度查找")
        
        # Parse FASTA file once
        length_parser = FastaLengthParser(fasta_file)
        
        # Collect lengths for dynamic bucketing
        if verbose:
            logger.info("收集序列长度用于动态分桶...")
        
        for protein_id in protein_ids:
            length = length_parser.get_length(protein_id)
            if length is not None:
                sequence_lengths.append(length)
        
        if verbose:
            logger.debug(f"收集到 {len(sequence_lengths)} 个有效序列长度")
    else:
        # Fallback to LMDB lookup (slower)
        if verbose:
            logger.info("Using LMDB for sequence length lookup (slower)")
        
        progress_bar = tqdm(protein_ids, desc="Collecting lengths", disable=not verbose)
        
        for protein_id in progress_bar:
            length = embedding_loader.get_sequence_length(protein_id)
            if length is not None:
                sequence_lengths.append(length)
        
        progress_bar.close()
    
    # Create dynamic bucket boundaries if requested and not provided
    if dynamic_bucketing and bucket_boundaries is None and sequence_lengths:
        from .dynamic_bucketing import create_dynamic_buckets
        
        bucket_boundaries = create_dynamic_buckets(
            sequence_lengths, 
            bucket_options=bucket_options,
            method='balanced',
            verbose=verbose
        )
    elif bucket_boundaries is None:
        # Fallback to default boundaries
        bucket_boundaries = [32, 64, 128, 256, 512, 1024]
        if verbose:
            logger.warning(f"使用默认分桶边界: {bucket_boundaries}")
    
    # Create bucketing system with determined boundaries
    bucketing = SequenceBucketing(bucket_boundaries)
    
    # Now populate the buckets
    if fasta_file:
        # Use FASTA parser for fast population
        length_parser = FastaLengthParser(fasta_file)
        
        processed = 0
        missing = 0
        progress_bar = tqdm(protein_ids, desc="Populating buckets", disable=not verbose)
        
        for protein_id in progress_bar:
            length = length_parser.get_length(protein_id)
            if length is not None:
                bucketing.add_protein(protein_id, length)
                processed += 1
            else:
                missing += 1
            
            progress_bar.set_postfix({
                'processed': processed,
                'missing': missing,
                'buckets': len(bucketing.buckets)
            })
        
        progress_bar.close()
        
        if missing > 0 and verbose:
            logger.warning(f"{missing} proteins not found in FASTA file")
    else:
        # Use LMDB for population
        processed = 0
        progress_bar = tqdm(protein_ids, desc="Populating buckets", disable=not verbose)
        
        for protein_id in progress_bar:
            length = embedding_loader.get_sequence_length(protein_id)
            if length is not None:
                bucketing.add_protein(protein_id, length)
                processed += 1
                
            progress_bar.set_postfix({
                'processed': processed,
                'buckets': len(bucketing.buckets)
            })
        
        progress_bar.close()
    
    stats = bucketing.get_stats()
    if verbose:
        logger.info(f"分桶完成: {stats}")
    
    return bucketing