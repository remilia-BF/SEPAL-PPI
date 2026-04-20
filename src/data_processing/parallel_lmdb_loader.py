"""
Parallel LMDB loader for accelerated embedding loading with multi-threading support
"""

import lmdb
import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParallelLMDBLoader:
    """
    Multi-threaded LMDB loader for accelerated embedding loading
    
    Each thread maintains its own LMDB transaction for thread safety
    """
    
    def __init__(self, lmdb_path: str, max_workers: int = 4, max_readers: int = 126):
        """
        Initialize parallel LMDB loader
        
        Args:
            lmdb_path (str): Path to LMDB database
            max_workers (int): Maximum number of worker threads
            max_readers (int): Maximum number of concurrent LMDB readers
        """
        self.lmdb_path = Path(lmdb_path)
        self.max_workers = max_workers
        self.max_readers = max_readers
        
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB database not found: {self.lmdb_path}")
        
        # Test LMDB connection
        self._test_connection()
        logger.info(f"初始化并行LMDB加载器: {max_workers} 工作线程, 最大读取器: {max_readers}")
    
    def _test_connection(self):
        """Test LMDB connection and get basic info"""
        try:
            env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=self.max_readers
            )
            
            with env.begin() as txn:
                stats = txn.stat()
                logger.debug(f"LMDB统计: {stats['entries']} 条目")
            
            env.close()
            
        except Exception as e:
            raise RuntimeError(f"LMDB连接测试失败: {e}")
    
    def _load_embedding_worker(self, protein_ids: List[str], results_queue: queue.Queue, 
                              progress_lock: Lock, progress_counter: Dict[str, int]):
        """
        Worker function for loading embeddings in a separate thread
        
        Args:
            protein_ids (List[str]): Protein IDs to load
            results_queue (queue.Queue): Queue to store results
            progress_lock (Lock): Lock for progress counter
            progress_counter (Dict[str, int]): Shared progress counter
        """
        # Each worker gets its own LMDB environment and transaction
        env = None
        try:
            env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=self.max_readers
            )
            
            loaded_embeddings = {}
            
            # Use short-lived transactions for each batch
            batch_size = 50  # Process in small batches to avoid long-lived transactions
            
            for i in range(0, len(protein_ids), batch_size):
                batch_ids = protein_ids[i:i + batch_size]
                
                with env.begin() as txn:
                    for protein_id in batch_ids:
                        try:
                            embedding = self._get_embedding_from_txn(txn, protein_id)
                            if embedding is not None:
                                loaded_embeddings[protein_id] = embedding
                                
                                # Update progress counter
                                with progress_lock:
                                    progress_counter['loaded'] += 1
                                    
                        except Exception as e:
                            logger.warning(f"加载嵌入失败 {protein_id}: {e}")
                            continue
            
            # Put results in queue
            results_queue.put(loaded_embeddings)
            
        except Exception as e:
            logger.error(f"工作线程错误: {e}")
            results_queue.put({})  # Put empty dict on error
        finally:
            if env:
                env.close()
    
    def _get_embedding_from_txn(self, txn, protein_id: str) -> Optional[torch.Tensor]:
        """
        Get embedding from LMDB transaction
        
        Args:
            txn: LMDB transaction
            protein_id (str): Protein identifier
            
        Returns:
            torch.Tensor: Embedding tensor or None if not found
        """
        # Try different key formats
        key_variants = [
            protein_id.encode(),
            f"{protein_id}".encode(),
            f"embedding_{protein_id}".encode()
        ]
        
        embedding_data = None
        for key in key_variants:
            embedding_data = txn.get(key)
            if embedding_data is not None:
                break
        
        if embedding_data is None:
            return None
        
        try:
            # Parse binary data
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
            
            # Try different embedding dimensions
            for embedding_dim in [2560, 1280, 768, 1024, 512, 256]:
                if len(embedding) % embedding_dim == 0:
                    seq_length = len(embedding) // embedding_dim
                    embedding = embedding.reshape(seq_length, embedding_dim)
                    break
            else:
                # Try pickle format
                import pickle
                try:
                    embedding = pickle.loads(embedding_data)
                except Exception:
                    return None
            
            # Convert to tensor
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding.copy()).float()
            elif not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"解析嵌入数据失败 {protein_id}: {e}")
            return None
    
    def load_embeddings_parallel(self, protein_ids: List[str], verbose: bool = True) -> Dict[str, torch.Tensor]:
        """
        Load embeddings in parallel using multiple threads
        
        Args:
            protein_ids (List[str]): List of protein IDs to load
            verbose (bool): Whether to show progress
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping protein IDs to embeddings
        """
        if not protein_ids:
            return {}
        
        # Determine optimal number of workers
        actual_workers = min(self.max_workers, len(protein_ids), self.max_readers // 2)
        
        if verbose:
            logger.info(f"使用 {actual_workers} 个线程并行加载 {len(protein_ids)} 个嵌入...")
        
        # Split protein IDs among workers
        chunk_size = max(1, len(protein_ids) // actual_workers)
        protein_chunks = [
            protein_ids[i:i + chunk_size] 
            for i in range(0, len(protein_ids), chunk_size)
        ]
        
        # Shared progress tracking
        progress_lock = Lock()
        progress_counter = {'loaded': 0}
        results_queue = queue.Queue()
        
        # Progress bar
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(protein_ids), desc="Loading embeddings (parallel)", 
                              unit="proteins")
        
        start_time = time.time()
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit tasks
            futures = []
            for chunk in protein_chunks:
                if chunk:  # Only submit non-empty chunks
                    future = executor.submit(
                        self._load_embedding_worker, 
                        chunk, results_queue, progress_lock, progress_counter
                    )
                    futures.append(future)
            
            # Monitor progress
            last_count = 0
            while any(not future.done() for future in futures):
                time.sleep(0.1)  # Check progress every 100ms
                
                if progress_bar:
                    with progress_lock:
                        current_count = progress_counter['loaded']
                        if current_count > last_count:
                            progress_bar.update(current_count - last_count)
                            last_count = current_count
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions from workers
                except Exception as e:
                    logger.error(f"线程执行错误: {e}")
        
        # Final progress update
        if progress_bar:
            with progress_lock:
                final_count = progress_counter['loaded']
                if final_count > last_count:
                    progress_bar.update(final_count - last_count)
            progress_bar.close()
        
        # Collect results from all workers
        all_embeddings = {}
        while not results_queue.empty():
            worker_results = results_queue.get()
            all_embeddings.update(worker_results)
        
        load_time = time.time() - start_time
        
        if verbose:
            logger.info(f"并行加载完成: {len(all_embeddings)}/{len(protein_ids)} 个嵌入, "
                       f"耗时 {load_time:.2f}s, 速度 {len(all_embeddings)/load_time:.1f} proteins/s")
        
        return all_embeddings


class BatchGPUTransfer:
    """
    Optimized batch GPU transfer for embeddings
    """
    
    @staticmethod
    def transfer_embeddings_to_gpu(embeddings_dict: Dict[str, torch.Tensor], 
                                 device: torch.device, 
                                 batch_size: int = 100,
                                 verbose: bool = True) -> Dict[str, Tuple[torch.Tensor, int]]:
        """
        Transfer embeddings to GPU in batches for better efficiency
        
        Args:
            embeddings_dict (Dict[str, torch.Tensor]): CPU embeddings
            device (torch.device): Target GPU device
            batch_size (int): Number of embeddings to transfer per batch
            verbose (bool): Whether to show progress
            
        Returns:
            Dict[str, Tuple[torch.Tensor, int]]: GPU embeddings with original lengths
        """
        if not embeddings_dict:
            return {}
        
        gpu_embeddings = {}
        protein_ids = list(embeddings_dict.keys())
        
        if verbose:
            logger.info(f"批量传输 {len(protein_ids)} 个嵌入到GPU...")
            progress_bar = tqdm(range(0, len(protein_ids), batch_size), 
                              desc="GPU transfer", unit="batch")
        else:
            progress_bar = range(0, len(protein_ids), batch_size)
        
        for i in progress_bar:
            batch_ids = protein_ids[i:i + batch_size]
            
            # Transfer batch to GPU
            for protein_id in batch_ids:
                embedding = embeddings_dict[protein_id]
                original_length = embedding.shape[0]
                gpu_embedding = embedding.to(device, non_blocking=True)
                gpu_embeddings[protein_id] = (gpu_embedding, original_length)
            
            # Synchronize GPU after each batch to prevent memory overflow
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        if verbose and hasattr(progress_bar, 'close'):
            progress_bar.close()
            logger.info(f"GPU传输完成: {len(gpu_embeddings)} 个嵌入")
        
        return gpu_embeddings