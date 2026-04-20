"""
Streaming GPU cache with memory-efficient loading and offline bucketed LMDB preprocessing
"""

import torch
import lmdb
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Iterator
from pathlib import Path
import threading
import queue
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc

logger = logging.getLogger(__name__)


class StreamingGPUCache:
    """
    Memory-efficient streaming GPU cache that loads embeddings on-demand
    with intelligent memory management
    """
    
    def __init__(self, device: torch.device, max_gpu_memory_mb: int = 4096):
        """
        Initialize streaming GPU cache
        
        Args:
            device (torch.device): GPU device
            max_gpu_memory_mb (int): Maximum GPU memory to use for cache (MB)
        """
        self.device = device
        self.max_gpu_memory_bytes = max_gpu_memory_mb * 1024 * 1024
        
        # Cache storage: protein_id -> (embedding, original_length, last_access_time)
        self.cache = {}
        self.current_memory_usage = 0
        
        # Access tracking for smart eviction
        self.access_times = {}
        self.access_counter = 0
        
        # Memory management
        self.memory_threshold = 0.9  # Start evicting at 90% capacity
        
        logger.info(f"初始化流式GPU缓存: 最大内存 {max_gpu_memory_mb}MB")
    
    def _get_tensor_memory_usage(self, tensor: torch.Tensor) -> int:
        """Get memory usage of a tensor in bytes"""
        return tensor.element_size() * tensor.numel()
    
    def _evict_by_memory_pressure(self, required_memory: int):
        """
        Evict embeddings based on memory pressure and access patterns
        
        Args:
            required_memory (int): Memory needed for new embedding
        """
        if self.current_memory_usage + required_memory <= self.max_gpu_memory_bytes:
            return
        
        # Sort by access time (LRU)
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: self.access_times.get(x[0], 0)
        )
        
        freed_memory = 0
        evicted_count = 0
        
        for protein_id, (embedding, length, _) in sorted_items:
            if self.current_memory_usage - freed_memory + required_memory <= self.max_gpu_memory_bytes:
                break
            
            # Free GPU memory
            embedding_memory = self._get_tensor_memory_usage(embedding)
            freed_memory += embedding_memory
            evicted_count += 1
            
            # Remove from cache
            del self.cache[protein_id]
            if protein_id in self.access_times:
                del self.access_times[protein_id]
        
        self.current_memory_usage -= freed_memory
        
        if evicted_count > 0:
            logger.debug(f"驱逐 {evicted_count} 个嵌入，释放 {freed_memory/1024/1024:.1f}MB")
    
    def add_embedding_streaming(self, protein_id: str, embedding: torch.Tensor):
        """
        Add embedding with streaming memory management
        
        Args:
            protein_id (str): Protein identifier
            embedding (torch.Tensor): Protein embedding
        """
        # Calculate memory requirement
        embedding_memory = self._get_tensor_memory_usage(embedding)
        
        # Evict if necessary
        self._evict_by_memory_pressure(embedding_memory)
        
        # Transfer to GPU (non-blocking for better performance)
        gpu_embedding = embedding.to(self.device, non_blocking=True)
        original_length = embedding.shape[0]
        
        # Store in cache
        self.cache[protein_id] = (gpu_embedding, original_length, time.time())
        self.current_memory_usage += embedding_memory
        
        # Update access tracking
        self.access_counter += 1
        self.access_times[protein_id] = self.access_counter
    
    def get_embedding_streaming(self, protein_id: str) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Get embedding with access tracking
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            Tuple[torch.Tensor, int]: (embedding, original_length) or None
        """
        if protein_id in self.cache:
            embedding, length, _ = self.cache[protein_id]
            
            # Update access time
            self.access_counter += 1
            self.access_times[protein_id] = self.access_counter
            
            return embedding, length
        
        return None
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory statistics"""
        return {
            'cache_size': len(self.cache),
            'current_memory_mb': self.current_memory_usage / 1024 / 1024,
            'max_memory_mb': self.max_gpu_memory_bytes / 1024 / 1024,
            'memory_utilization': self.current_memory_usage / self.max_gpu_memory_bytes,
            'device': str(self.device)
        }


class BucketedLMDBPreprocessor:
    """
    Offline preprocessor to create bucketed LMDB for faster loading
    """
    
    def __init__(self, source_lmdb_path: str, output_dir: str):
        """
        Initialize bucketed LMDB preprocessor
        
        Args:
            source_lmdb_path (str): Path to source LMDB
            output_dir (str): Directory for bucketed LMDB files
        """
        self.source_lmdb_path = Path(source_lmdb_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.source_lmdb_path.exists():
            raise FileNotFoundError(f"源LMDB不存在: {source_lmdb_path}")
    
    def create_bucketed_lmdb(self, bucket_boundaries: List[int] = None, 
                           chunk_size: int = 1000) -> Dict[int, str]:
        """
        Create bucketed LMDB files for faster loading
        
        Args:
            bucket_boundaries (List[int]): Bucket boundaries
            chunk_size (int): Processing chunk size
            
        Returns:
            Dict[int, str]: Mapping of bucket_id to LMDB path
        """
        if bucket_boundaries is None:
            bucket_boundaries = [64, 128, 256, 512, 1024, 2048]
        
        logger.info(f"创建分桶LMDB: 边界 {bucket_boundaries}")
        
        # Initialize bucket LMDB environments
        bucket_envs = {}
        bucket_paths = {}
        
        try:
            for i, boundary in enumerate(bucket_boundaries):
                bucket_path = self.output_dir / f"bucket_{i}_max{boundary}.lmdb"
                bucket_paths[i] = str(bucket_path)
                
                # Create LMDB environment for this bucket
                bucket_envs[i] = lmdb.open(
                    str(bucket_path),
                    readonly=False,
                    map_size=10 * 1024 * 1024 * 1024,  # 10GB max
                    max_dbs=0
                )
                
                logger.debug(f"创建桶 {i}: 最大长度 {boundary}, 路径 {bucket_path}")
            
            # Process source LMDB
            self._process_source_lmdb(bucket_envs, bucket_boundaries, chunk_size)
            
            return bucket_paths
            
        finally:
            # Close all bucket environments
            for env in bucket_envs.values():
                env.close()
    
    def _process_source_lmdb(self, bucket_envs: Dict, bucket_boundaries: List[int], 
                           chunk_size: int):
        """
        Process source LMDB and distribute to buckets
        
        Args:
            bucket_envs (Dict): Bucket LMDB environments
            bucket_boundaries (List[int]): Bucket boundaries
            chunk_size (int): Processing chunk size
        """
        # Open source LMDB
        source_env = lmdb.open(str(self.source_lmdb_path), readonly=True)
        
        try:
            with source_env.begin() as source_txn:
                cursor = source_txn.cursor()
                
                # Get total count for progress
                total_entries = source_txn.stat()['entries']
                logger.info(f"处理 {total_entries} 个嵌入...")
                
                # Initialize bucket transactions
                bucket_txns = {}
                for bucket_id, env in bucket_envs.items():
                    bucket_txns[bucket_id] = env.begin(write=True)
                
                processed = 0
                bucket_counts = {i: 0 for i in bucket_envs.keys()}
                
                progress_bar = tqdm(total=total_entries, desc="Processing embeddings")
                
                try:
                    for key, value in cursor:
                        try:
                            protein_id = key.decode('utf-8')
                            
                            # Parse embedding to get length
                            embedding_length = self._get_embedding_length(value)
                            if embedding_length is None:
                                continue
                            
                            # Determine bucket
                            bucket_id = self._get_bucket_id(embedding_length, bucket_boundaries)
                            
                            # Store in appropriate bucket
                            bucket_txns[bucket_id].put(key, value)
                            bucket_counts[bucket_id] += 1
                            
                            processed += 1
                            
                            # Commit in chunks to avoid memory issues
                            if processed % chunk_size == 0:
                                for txn in bucket_txns.values():
                                    txn.commit()
                                
                                # Start new transactions
                                for bucket_id, env in bucket_envs.items():
                                    bucket_txns[bucket_id] = env.begin(write=True)
                                
                                progress_bar.set_postfix({
                                    'processed': processed,
                                    'buckets': len([c for c in bucket_counts.values() if c > 0])
                                })
                            
                            progress_bar.update(1)
                            
                        except Exception as e:
                            logger.warning(f"处理嵌入失败: {e}")
                            continue
                    
                    # Final commit
                    for txn in bucket_txns.values():
                        txn.commit()
                    
                    progress_bar.close()
                    
                    # Log bucket statistics
                    logger.info("分桶统计:")
                    for bucket_id, count in bucket_counts.items():
                        if count > 0:
                            max_len = bucket_boundaries[bucket_id] if bucket_id < len(bucket_boundaries) else "∞"
                            logger.info(f"  桶 {bucket_id} (≤{max_len}): {count} 个嵌入")
                
                except Exception as e:
                    # Rollback all transactions on error
                    for txn in bucket_txns.values():
                        txn.abort()
                    raise e
                
        finally:
            source_env.close()
    
    def _get_embedding_length(self, embedding_data: bytes) -> Optional[int]:
        """
        Get embedding sequence length from binary data
        
        Args:
            embedding_data (bytes): Raw embedding data
            
        Returns:
            int: Sequence length or None if parsing fails
        """
        try:
            # Try as raw binary first
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
            
            # Try different embedding dimensions
            for embedding_dim in [2560, 1280, 768, 1024, 512, 256]:
                if len(embedding) % embedding_dim == 0:
                    return len(embedding) // embedding_dim
            
            # Try pickle format
            import pickle
            try:
                embedding = pickle.loads(embedding_data)
                if hasattr(embedding, 'shape'):
                    return embedding.shape[0]
            except Exception:
                pass
                
        except Exception:
            pass
        
        return None
    
    def _get_bucket_id(self, length: int, bucket_boundaries: List[int]) -> int:
        """Get bucket ID for sequence length"""
        for i, boundary in enumerate(bucket_boundaries):
            if length <= boundary:
                return i
        return len(bucket_boundaries) - 1


class BucketedLMDBLoader:
    """
    Fast loader for bucketed LMDB files
    """
    
    def __init__(self, bucket_lmdb_paths: Dict[int, str]):
        """
        Initialize bucketed LMDB loader
        
        Args:
            bucket_lmdb_paths (Dict[int, str]): Mapping of bucket_id to LMDB path
        """
        self.bucket_paths = bucket_lmdb_paths
        self.bucket_envs = {}
        
        # Open all bucket environments
        for bucket_id, path in bucket_lmdb_paths.items():
            if Path(path).exists():
                self.bucket_envs[bucket_id] = lmdb.open(
                    path, readonly=True, lock=False, readahead=False
                )
        
        logger.info(f"初始化分桶LMDB加载器: {len(self.bucket_envs)} 个桶")
    
    def load_bucket_parallel(self, bucket_id: int, protein_ids: List[str], 
                           max_workers: int = 2) -> Dict[str, torch.Tensor]:
        """
        Load embeddings from specific bucket in parallel
        
        Args:
            bucket_id (int): Bucket ID
            protein_ids (List[str]): Protein IDs to load
            max_workers (int): Number of worker threads
            
        Returns:
            Dict[str, torch.Tensor]: Loaded embeddings
        """
        if bucket_id not in self.bucket_envs:
            return {}
        
        env = self.bucket_envs[bucket_id]
        embeddings = {}
        
        # Use fewer workers for bucketed loading since it's already optimized
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = max(1, len(protein_ids) // max_workers)
            chunks = [protein_ids[i:i + chunk_size] for i in range(0, len(protein_ids), chunk_size)]
            
            futures = []
            for chunk in chunks:
                future = executor.submit(self._load_chunk_from_bucket, env, chunk)
                futures.append(future)
            
            for future in futures:
                chunk_embeddings = future.result()
                embeddings.update(chunk_embeddings)
        
        return embeddings
    
    def _load_chunk_from_bucket(self, env, protein_ids: List[str]) -> Dict[str, torch.Tensor]:
        """Load chunk of embeddings from bucket"""
        embeddings = {}
        
        with env.begin() as txn:
            for protein_id in protein_ids:
                try:
                    key = protein_id.encode()
                    data = txn.get(key)
                    
                    if data is not None:
                        # Parse embedding
                        embedding = self._parse_embedding_data(data)
                        if embedding is not None:
                            embeddings[protein_id] = embedding
                            
                except Exception as e:
                    logger.warning(f"加载嵌入失败 {protein_id}: {e}")
                    continue
        
        return embeddings
    
    def _parse_embedding_data(self, data: bytes) -> Optional[torch.Tensor]:
        """Parse embedding data to tensor"""
        try:
            # Try raw binary first
            embedding = np.frombuffer(data, dtype=np.float32)
            
            # Try different embedding dimensions
            for embedding_dim in [2560, 1280, 768, 1024, 512, 256]:
                if len(embedding) % embedding_dim == 0:
                    seq_length = len(embedding) // embedding_dim
                    embedding = embedding.reshape(seq_length, embedding_dim)
                    return torch.from_numpy(embedding.copy()).float()
            
            # Try pickle
            import pickle
            embedding = pickle.loads(data)
            if isinstance(embedding, np.ndarray):
                return torch.from_numpy(embedding.copy()).float()
            elif isinstance(embedding, torch.Tensor):
                return embedding.float()
                
        except Exception:
            pass
        
        return None
    
    def close(self):
        """Close all bucket environments"""
        for env in self.bucket_envs.values():
            env.close()
        self.bucket_envs.clear()