"""
Advanced bucketed LMDB loader with CPU padding and async GPU transfer
"""

import os
import json
import logging
import struct
import lmdb
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from collections import defaultdict
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import time

logger = logging.getLogger(__name__)


class BucketedProteinKey:
    """Parse and manage bucketed protein keys"""
    
    def __init__(self, key_str: str):
        """
        Parse bucketed protein key
        
        Format: bucket_{bucket_id}_len_{seq_length}_{protein_id}
        """
        self.original_key = key_str
        
        try:
            parts = key_str.split('_')
            if len(parts) >= 4 and parts[0] == 'bucket' and parts[2] == 'len':
                self.bucket_id = int(parts[1])
                self.seq_length = int(parts[3])
                self.protein_id = '_'.join(parts[4:])  # Handle protein IDs with underscores
                self.is_bucketed = True
            else:
                # Fallback for non-bucketed keys
                self.bucket_id = 0
                self.seq_length = 0
                self.protein_id = key_str
                self.is_bucketed = False
        except (ValueError, IndexError):
            # Fallback for malformed keys
            self.bucket_id = 0
            self.seq_length = 0
            self.protein_id = key_str
            self.is_bucketed = False
    
    def __repr__(self):
        return f"BucketedProteinKey(bucket_id={self.bucket_id}, seq_length={self.seq_length}, protein_id='{self.protein_id}')"


class CPUPaddingProcessor:
    """CPU-based padding processor for efficient batching"""
    
    def __init__(self, embedding_dim: int, target_precision: str = 'fp32'):
        """
        Initialize CPU padding processor
        
        Args:
            embedding_dim (int): Embedding dimension
            target_precision (str): Target precision for tensors
        """
        self.embedding_dim = embedding_dim
        self.target_precision = target_precision
        self.dtype = self._get_torch_dtype(target_precision)
    
    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """Get torch dtype from precision string"""
        if precision == 'fp32':
            return torch.float32
        elif precision == 'fp16':
            return torch.float16
        elif precision == 'bf16':
            return torch.bfloat16
        elif precision == 'int8':
            return torch.int8
        else:
            return torch.float32
    
    def _parse_embedding_data(self, data: bytes, expected_length: int) -> Optional[torch.Tensor]:
        """
        Parse embedding data from bytes with support for various precision formats
        
        Args:
            data (bytes): Raw embedding data
            expected_length (int): Expected sequence length
            
        Returns:
            Optional[torch.Tensor]: Parsed embedding tensor
        """
        try:
            # 首先检查是否是int8格式（带scale前缀）
            if self.target_precision == 'int8' and len(data) >= 4:
                try:
                    import struct
                    # 前4字节是scale
                    scale = struct.unpack('f', data[:4])[0]
                    # 其余是int8数据
                    quantized_data = np.frombuffer(data[4:], dtype=np.int8)
                    if len(quantized_data) % self.embedding_dim == 0:
                        actual_length = len(quantized_data) // self.embedding_dim
                        if abs(actual_length - expected_length) <= 1 or expected_length == 0:
                            # 反量化：int8 -> float32
                            reshaped = quantized_data.reshape(actual_length, self.embedding_dim)
                            dequantized = reshaped.astype(np.float32) * scale / 127.0
                            return torch.from_numpy(dequantized.copy()).to(self.dtype)
                except:
                    pass
            
            # bfloat16格式（存储为uint16）
            if self.target_precision == 'bf16':
                try:
                    arr = np.frombuffer(data, dtype=np.uint16)
                    if len(arr) % self.embedding_dim == 0:
                        actual_length = len(arr) // self.embedding_dim
                        if abs(actual_length - expected_length) <= 1 or expected_length == 0:
                            # 将uint16转换为bfloat16，保持原始精度
                            tensor = torch.from_numpy(arr).view(torch.bfloat16)
                            reshaped = tensor.view(actual_length, self.embedding_dim)
                            return reshaped.to(self.dtype)
                except:
                    pass
            
            # fp16格式
            if self.target_precision == 'fp16':
                try:
                    arr = np.frombuffer(data, dtype=np.float16)
                    if len(arr) % self.embedding_dim == 0:
                        actual_length = len(arr) // self.embedding_dim
                        if abs(actual_length - expected_length) <= 1 or expected_length == 0:
                            # 保持fp16精度，不转换为fp32
                            reshaped = arr.reshape(actual_length, self.embedding_dim)
                            return torch.from_numpy(reshaped.copy()).to(self.dtype)
                except:
                    pass
            
            # fp32格式（默认和兜底）
            try:
                arr = np.frombuffer(data, dtype=np.float32)
                if len(arr) % self.embedding_dim == 0:
                    actual_length = len(arr) // self.embedding_dim
                    if abs(actual_length - expected_length) <= 1 or expected_length == 0:
                        reshaped = arr.reshape(actual_length, self.embedding_dim)
                        return torch.from_numpy(reshaped.copy()).to(self.dtype)
            except:
                pass
            
            # Try pickle format as last resort
            try:
                import pickle
                arr = pickle.loads(data)
                if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                    return torch.from_numpy(arr.copy()).to(self.dtype)
            except:
                pass
        
        except Exception as e:
            logger.debug(f"Failed to parse embedding data: {e}")
        
        return None
    
    def create_padded_batch(self, protein_data: List[Tuple[BucketedProteinKey, bytes]], 
                          target_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Create padded batch from protein data
        
        Args:
            protein_data (List[Tuple[BucketedProteinKey, bytes]]): List of (key, data) pairs
            target_length (Optional[int]): Target padding length (auto-calculated if None)
            
        Returns:
            Dict[str, torch.Tensor]: Padded batch tensors
        """
        if not protein_data:
            return {}
        
        # Parse all embeddings
        embeddings = []
        protein_ids = []
        actual_lengths = []
        
        for key, data in protein_data:
            embedding = self._parse_embedding_data(data, key.seq_length)
            if embedding is not None:
                embeddings.append(embedding)
                protein_ids.append(key.protein_id)
                actual_lengths.append(embedding.shape[0])
        
        if not embeddings:
            logger.warning("No valid embeddings found in batch")
            return {}
        
        # Determine target length
        if target_length is None:
            target_length = max(actual_lengths)
        
        batch_size = len(embeddings)
        
        # Create padded tensors
        padded_embeddings = torch.zeros(batch_size, target_length, self.embedding_dim, dtype=self.dtype)
        attention_masks = torch.zeros(batch_size, target_length, dtype=torch.bool)
        
        # Fill padded tensors
        for i, (embedding, actual_len) in enumerate(zip(embeddings, actual_lengths)):
            seq_len = min(actual_len, target_length)
            padded_embeddings[i, :seq_len] = embedding[:seq_len]
            attention_masks[i, :seq_len] = True
        
        return {
            'embeddings': padded_embeddings,
            'attention_masks': attention_masks,
            'protein_ids': protein_ids,
            'actual_lengths': actual_lengths,
            'target_length': target_length
        }


class AsyncGPUTransfer:
    """Async GPU transfer manager"""
    
    def __init__(self, device: torch.device, max_workers: int = 2):
        """
        Initialize async GPU transfer
        
        Args:
            device (torch.device): Target GPU device
            max_workers (int): Maximum worker threads
        """
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.transfer_queue = []
        self.completed_transfers = {}
    
    def _transfer_to_gpu(self, batch_data: Dict[str, torch.Tensor], batch_id: str) -> Dict[str, torch.Tensor]:
        """Transfer batch data to GPU"""
        gpu_data = {}
        
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor):
                gpu_data[key] = tensor.to(self.device, non_blocking=True)
            else:
                gpu_data[key] = tensor
        
        # Synchronize to ensure transfer is complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        return gpu_data
    
    def submit_transfer(self, batch_data: Dict[str, torch.Tensor], batch_id: str) -> Future:
        """Submit batch for async GPU transfer"""
        future = self.executor.submit(self._transfer_to_gpu, batch_data, batch_id)
        self.transfer_queue.append((batch_id, future))
        return future
    
    def get_completed_transfer(self, batch_id: str, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]:
        """Get completed transfer by batch ID"""
        # Check if already completed
        if batch_id in self.completed_transfers:
            return self.completed_transfers.pop(batch_id)
        
        # Look for the transfer in queue
        for i, (queued_id, future) in enumerate(self.transfer_queue):
            if queued_id == batch_id:
                try:
                    result = future.result(timeout=timeout)
                    self.transfer_queue.pop(i)
                    return result
                except Exception as e:
                    logger.error(f"GPU transfer failed for batch {batch_id}: {e}")
                    self.transfer_queue.pop(i)
                    return None
        
        return None
    
    def close(self):
        """Close the executor"""
        self.executor.shutdown(wait=True)


class BucketedLMDBLoader:
    """
    Advanced bucketed LMDB loader with CPU padding and async GPU transfer
    """
    
    def __init__(self, lmdb_dir: str, device: torch.device, 
                 batch_size: int = 32, max_workers: int = 4):
        """
        Initialize bucketed LMDB loader
        
        Args:
            lmdb_dir (str): Directory containing bucketed LMDB files
            device (torch.device): Target device for tensors
            batch_size (int): Batch size for processing
            max_workers (int): Maximum worker threads
        """
        self.lmdb_dir = Path(lmdb_dir)
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Load metadata and bucket info
        self.bucket_info = self._load_bucket_metadata()
        self.embedding_dim = self.bucket_info.get('embedding_dim', 1280)
        self.target_precision = self.bucket_info.get('target_precision', 'fp32')
        
        # Load protein index if available
        self.protein_index = self._load_protein_index()
        
        # Initialize components
        self.cpu_processor = CPUPaddingProcessor(self.embedding_dim, self.target_precision)
        self.gpu_transfer = AsyncGPUTransfer(device, max_workers=2)
        
        # Bucket environments (lazy loaded)
        self.bucket_envs = {}
        self.bucket_paths = self._discover_bucket_paths()
        
        logger.info(f"初始化分桶LMDB加载器:")
        logger.info(f"  LMDB目录: {self.lmdb_dir}")
        logger.info(f"  发现桶数: {len(self.bucket_paths)}")
        logger.info(f"  嵌入维度: {self.embedding_dim}")
        logger.info(f"  目标精度: {self.target_precision}")
        
        self.dtype = self._get_torch_dtype(self.target_precision)
    
    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """Get torch dtype from precision string"""
        if precision == 'fp32':
            return torch.float32
        elif precision == 'fp16':
            return torch.float16
        elif precision == 'bf16':
            return torch.bfloat16
        elif precision == 'int8':
            return torch.int8
        else:
            return torch.float32
    

    
    def _load_bucket_metadata(self) -> Dict[str, Any]:
        """Load bucket metadata from config file or first available bucket"""
        metadata = {
            'embedding_dim': 1280,
            'target_precision': 'fp32',
            'boundaries': [],
            'num_buckets': 0
        }
        
        try:
            # First try to load from config file (最可靠的方法)
            config_file = self.lmdb_dir / "intelligent_bucketed_lmdb_config.json"
            if config_file.exists():
                try:
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        metadata['embedding_dim'] = config.get('embedding_dim', 1280)
                        metadata['target_precision'] = config.get('target_precision', 'fp32')
                        metadata['original_precision'] = config.get('original_precision', 'fp32')
                        metadata['total_proteins'] = config.get('total_proteins', 0)
                        metadata['bucket_paths'] = config.get('bucket_paths', {})
                        metadata['processor_version'] = config.get('processor_version', '2.0.0')
                        
                        # 配置文件提供基本信息，但需要验证实际存储精度
                        logger.info(f"从配置文件加载元数据:")
                        logger.info(f"  - 嵌入维度: {metadata['embedding_dim']}")
                        logger.info(f"  - 声明精度: {metadata['target_precision']}")
                        logger.info(f"  - 原始精度: {metadata['original_precision']}")
                        logger.info(f"  - 总蛋白质数: {metadata['total_proteins']}")
                        logger.info(f"  - 桶数量: {len(metadata['bucket_paths'])}")
                        
                        # 检查是否有强制指定的精度（通过构造函数参数传入）
                        # 如果配置文件中的target_precision与实际检测的精度不同，优先使用配置文件中的值
                        # 只有在没有强制指定精度时才进行验证
                        forced_precision = getattr(self, 'forced_target_precision', None)
                        if not forced_precision:
                            actual_precision = self._detect_actual_precision(metadata['embedding_dim'])
                            if actual_precision != metadata['target_precision']:
                                logger.warning(f"⚠️ 检测到实际精度 ({actual_precision}) 与声明精度 ({metadata['target_precision']}) 不符")
                                logger.info(f"使用配置文件中指定的精度: {metadata['target_precision']}")
                            else:
                                logger.info(f"✅ 精度验证通过: {actual_precision}")
                        else:
                            logger.info(f"✅ 使用强制指定的精度: {forced_precision}")
                        
                        return metadata
                except Exception as e:
                    logger.warning(f"读取配置文件失败: {e}")
            
            # Fallback: read from LMDB files (如果配置文件不存在或损坏)
            logger.info("配置文件不可用，从LMDB文件读取元数据...")
            
            # 首先检查是否是单文件结构（新版本的分桶格式）
            single_file_path = self.lmdb_dir / "bucketed_embeddings.lmdb"
            if single_file_path.exists():
                logger.info(f"检测到单文件分桶结构: {single_file_path}")
                metadata.update(self._read_metadata_from_lmdb(str(single_file_path)))
            else:
                # 传统多文件结构
                bucket_0_path = self.lmdb_dir / "bucket_0.lmdb"
                if bucket_0_path.exists():
                    metadata.update(self._read_metadata_from_lmdb(str(bucket_0_path)))
                else:
                    # Try any bucket file
                    bucket_files = list(self.lmdb_dir.glob("bucket_*.lmdb"))
                    if bucket_files:
                        metadata.update(self._read_metadata_from_lmdb(str(bucket_files[0])))
        
        except Exception as e:
            logger.warning(f"Failed to load bucket metadata: {e}")
        
        return metadata
    
    def _read_metadata_from_lmdb(self, lmdb_path: str) -> Dict[str, Any]:
        """Read metadata from LMDB file"""
        metadata = {}
        
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                # Read bucket info
                bucket_info = txn.get(b'_bucket_info')
                if bucket_info:
                    bucket_data = json.loads(bucket_info.decode())
                    metadata['boundaries'] = bucket_data.get('boundaries', [])
                    metadata['num_buckets'] = bucket_data.get('num_buckets', 0)
                
                # Read precision info
                precision_info = txn.get(b'_precision_info')
                if precision_info:
                    precision_data = json.loads(precision_info.decode())
                    metadata['embedding_dim'] = precision_data.get('embedding_dim', 1280)
                    metadata['target_precision'] = precision_data.get('target_precision', 'fp32')
        
        finally:
            env.close()
        
        return metadata
    
    def _detect_actual_precision(self, embedding_dim: int) -> str:
        """检测LMDB中实际存储的数据精度"""
        try:
            # 找到第一个可用的桶进行检测
            bucket_files = list(self.lmdb_dir.glob("bucket_*.lmdb"))
            if not bucket_files:
                return 'fp32'  # 默认值
            
            bucket_path = bucket_files[0]
            env = lmdb.open(str(bucket_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                # 找到第一个非元数据键
                while cursor.key():
                    key = cursor.key().decode('utf-8')
                    if not key.startswith('_'):
                        data = cursor.value()
                        
                        # 基于数据大小和嵌入维度推断精度
                        if embedding_dim:
                            # 计算不同精度的预期大小
                            expected_elements = len(data) // 4  # fp32: 4 bytes per element
                            expected_elements_fp16 = len(data) // 2  # fp16: 2 bytes per element
                            
                            # 检查是否能整除嵌入维度
                            if expected_elements % embedding_dim == 0:
                                # 进一步验证：尝试解析数据
                                try:
                                    arr = np.frombuffer(data, dtype=np.float32)
                                    seq_len = len(arr) // embedding_dim
                                    if 10 <= seq_len <= 5000:  # 合理的序列长度
                                        return 'fp32'
                                except:
                                    pass
                            
                            if expected_elements_fp16 % embedding_dim == 0:
                                try:
                                    arr = np.frombuffer(data, dtype=np.float16)
                                    seq_len = len(arr) // embedding_dim
                                    if 10 <= seq_len <= 5000:
                                        return 'fp16'
                                except:
                                    pass
                        
                        # 默认返回fp32
                        return 'fp32'
                    cursor.next()
            
            env.close()
            
        except Exception as e:
            logger.warning(f"精度检测失败: {e}")
        
        return 'fp32'  # 默认值
    
    def _discover_bucket_paths(self) -> Dict[int, str]:
        """Discover all bucket LMDB files using config file or file system"""
        bucket_paths = {}
        
        # First try to get paths from metadata (loaded from config file)
        if hasattr(self, 'bucket_info') and 'bucket_paths' in self.bucket_info:
            config_paths = self.bucket_info['bucket_paths']
            for bucket_id_str, path in config_paths.items():
                try:
                    bucket_id = int(bucket_id_str)
                    
                    # 简化路径处理：直接使用配置文件中的路径
                    path_obj = Path(path)
                    if path_obj.exists():
                        bucket_paths[bucket_id] = str(path_obj)
                    else:
                        logger.warning(f"桶文件不存在: {path}")
                        
                except (ValueError, TypeError):
                    logger.warning(f"Invalid bucket ID in config: {bucket_id_str}")
            
            logger.info(f"从配置文件发现 {len(bucket_paths)} 个有效桶路径")
            if bucket_paths:
                return bucket_paths
        
        # Fallback: scan directory for bucket files
        logger.info("从文件系统扫描桶文件...")
        
        # 首先检查单文件结构（新版本分桶格式）
        single_file_path = self.lmdb_dir / "bucketed_embeddings.lmdb"
        if single_file_path.exists():
            logger.info(f"检测到单文件分桶结构: {single_file_path}")
            # 对于单文件结构，所有数据都在桶0中
            bucket_paths[0] = str(single_file_path)
        else:
            # 传统多文件结构
            for bucket_file in self.lmdb_dir.glob("bucket_*.lmdb"):
                try:
                    bucket_id = int(bucket_file.stem.split('_')[1])
                    bucket_paths[bucket_id] = str(bucket_file)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid bucket file name: {bucket_file}")
        
        logger.info(f"从文件系统发现 {len(bucket_paths)} 个桶文件")
        return bucket_paths
    
    def _get_bucket_env(self, bucket_id: int) -> Optional[lmdb.Environment]:
        """Get or create LMDB environment for bucket"""
        if bucket_id not in self.bucket_envs:
            if bucket_id in self.bucket_paths:
                try:
                    env = lmdb.open(self.bucket_paths[bucket_id], readonly=True, lock=False)
                    self.bucket_envs[bucket_id] = env
                except Exception as e:
                    logger.error(f"Failed to open bucket {bucket_id}: {e}")
                    return None
            else:
                return None
        
        return self.bucket_envs.get(bucket_id)
    
    def _load_protein_index(self) -> Dict[str, Any]:
        """
        Load protein index from LMDB metadata
        
        Returns:
            Dict containing protein index data
        """
        protein_index = {
            'protein_to_bucket': {},
            'bucket_to_proteins': {},
            'prefix_to_proteins': {},
            'length_to_proteins': {},
            'available': False
        }
        
        try:
            # 尝试从第一个可用桶加载索引
            bucket_files = list(self.lmdb_dir.glob("bucket_*.lmdb"))
            if not bucket_files:
                return protein_index
            
            bucket_path = bucket_files[0]
            env = lmdb.open(str(bucket_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                index_data = txn.get(b'_protein_index')
                if index_data:
                    import json
                    index_info = json.loads(index_data.decode('utf-8'))
                    protein_index.update(index_info)
                    protein_index['available'] = True
                    
                    logger.info(f"✅ 加载蛋白质索引: {protein_index.get('total_proteins', 0)} 个蛋白质")
                else:
                    logger.info("⚠️ 未找到蛋白质索引，使用遍历模式")
            
            env.close()
            
        except Exception as e:
            logger.warning(f"加载蛋白质索引失败: {e}")
        
        return protein_index
    
    def find_proteins_by_ids(self, protein_ids: List[str]) -> Dict[str, Tuple[int, str]]:
        """
        Find proteins by their IDs using index
        
        Args:
            protein_ids (List[str]): List of protein IDs to find
            
        Returns:
            Dict[str, Tuple[int, str]]: protein_id -> (bucket_id, bucket_key)
        """
        results = {}
        
        if self.protein_index['available']:
            # 使用索引进行快速查找
            protein_to_bucket = self.protein_index['protein_to_bucket']
            
            for protein_id in protein_ids:
                if protein_id in protein_to_bucket:
                    bucket_id = protein_to_bucket[protein_id]
                    # 构造桶内的完整键名（需要查找实际的键）
                    bucket_key = None
                    
                    # 在指定桶中查找完整键名
                    env = self._get_bucket_env(bucket_id)
                    if env:
                        try:
                            with env.begin() as txn:
                                cursor = txn.cursor()
                                cursor.first()
                                
                                while cursor.key():
                                    key_str = cursor.key().decode('utf-8')
                                    if key_str.endswith(f"_{protein_id}"):
                                        bucket_key = key_str
                                        break
                                    cursor.next()
                        except Exception as e:
                            logger.debug(f"查找桶键失败: {e}")
                    
                    if bucket_key:
                        results[protein_id] = (bucket_id, bucket_key)
                    
        else:
            # Fallback: 遍历所有桶查找
            logger.info("使用遍历模式查找蛋白质...")
            for bucket_id in self.bucket_paths.keys():
                env = self._get_bucket_env(bucket_id)
                if not env:
                    continue
                
                try:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        cursor.first()
                        
                        while cursor.key():
                            key_str = cursor.key().decode('utf-8')
                            if not key_str.startswith('_'):
                                protein_key = BucketedProteinKey(key_str)
                                if protein_key.protein_id in protein_ids:
                                    results[protein_key.protein_id] = (bucket_id, key_str)
                            cursor.next()
                
                except Exception as e:
                    logger.error(f"遍历桶 {bucket_id} 失败: {e}")
        
        return results
    
    def find_proteins_by_prefix(self, prefix: str, limit: int = 100) -> List[str]:
        """
        Find proteins by ID prefix using index
        
        Args:
            prefix (str): Protein ID prefix
            limit (int): Maximum results to return
            
        Returns:
            List[str]: List of matching protein IDs
        """
        results = []
        
        if self.protein_index['available']:
            prefix_to_proteins = self.protein_index['prefix_to_proteins']
            if prefix in prefix_to_proteins:
                results = prefix_to_proteins[prefix][:limit]
            else:
                # 查找更长的前缀
                for indexed_prefix, proteins in prefix_to_proteins.items():
                    if indexed_prefix.startswith(prefix):
                        results.extend(proteins)
                        if len(results) >= limit:
                            break
                
                results = results[:limit]
        
        return results
    
    def get_proteins_by_length_range(self, min_length: int, max_length: int) -> List[str]:
        """
        Get proteins within specified length range
        
        Args:
            min_length (int): Minimum sequence length
            max_length (int): Maximum sequence length
            
        Returns:
            List[str]: List of protein IDs in range
        """
        results = []
        
        if self.protein_index['available']:
            length_to_proteins = self.protein_index['length_to_proteins']
            for length, proteins in length_to_proteins.items():
                if isinstance(length, str):
                    length = int(length)
                if min_length <= length <= max_length:
                    results.extend(proteins)
        
        return results
    
    def load_bucket_data(self, bucket_id: int, protein_ids: Optional[List[str]] = None) -> List[Tuple[BucketedProteinKey, bytes]]:
        """
        Load data from specific bucket
        
        Args:
            bucket_id (int): Bucket ID to load
            protein_ids (Optional[List[str]]): Specific protein IDs to load (None for all)
            
        Returns:
            List[Tuple[BucketedProteinKey, bytes]]: List of (key, data) pairs
        """
        env = self._get_bucket_env(bucket_id)
        if not env:
            return []
        
        bucket_data = []
        target_protein_ids = set(protein_ids) if protein_ids else None
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                while cursor.key():
                    key_bytes = cursor.key()
                    
                    # Skip metadata keys
                    if key_bytes.startswith(b'_'):
                        cursor.next()
                        continue
                    
                    key_str = key_bytes.decode('utf-8')
                    protein_key = BucketedProteinKey(key_str)
                    
                    # Filter by protein IDs if specified
                    if target_protein_ids and protein_key.protein_id not in target_protein_ids:
                        cursor.next()
                        continue
                    
                    data = cursor.value()
                    bucket_data.append((protein_key, data))
                    
                    cursor.next()
        
        except Exception as e:
            logger.error(f"Failed to load bucket {bucket_id}: {e}")
        
        return bucket_data
    
    def create_batched_iterator(self, protein_ids: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Create batched iterator with bucket-based CPU padding and optimized protein lookup
        使用蛋白质索引优化查找：先定位桶，再加载数据
        
        Args:
            protein_ids (List[str]): List of protein IDs to load
            
        Yields:
            Dict[str, torch.Tensor]: Batched and padded tensors on GPU
        """
        total_found = 0
        
        logger.info(f"🔍 查找 {len(protein_ids)} 个蛋白质...")
        
        if self.protein_index['available']:
            # 使用索引进行优化查找
            logger.info("使用蛋白质索引进行快速查找...")
            protein_locations = self.find_proteins_by_ids(protein_ids)
            
            # 按桶分组蛋白质
            bucket_to_proteins = defaultdict(list)
            for protein_id, (bucket_id, bucket_key) in protein_locations.items():
                bucket_to_proteins[bucket_id].append((protein_id, bucket_key))
            
            # 按桶加载和处理数据
            for bucket_id, protein_info in bucket_to_proteins.items():
                env = self._get_bucket_env(bucket_id)
                if not env:
                    continue
                
                bucket_proteins = []
                bucket_found = 0
                
                try:
                    with env.begin() as txn:
                        for protein_id, bucket_key in protein_info:
                            data = txn.get(bucket_key.encode('utf-8'))
                            if data:
                                protein_key = BucketedProteinKey(bucket_key)
                                bucket_proteins.append((protein_key, data))
                                bucket_found += 1
                    
                    if bucket_proteins:
                        logger.info(f"📦 桶 {bucket_id}: 加载 {bucket_found} 个蛋白质")
                        total_found += bucket_found
                        
                        # 基于桶内序列长度进行CPU padding
                        bucket_batch = self._create_bucket_batch(bucket_proteins, bucket_id)
                        
                        if bucket_batch and 'protein_ids' in bucket_batch:
                            logger.debug(f"✅ 桶 {bucket_id} 批次就绪: {len(bucket_batch['protein_ids'])} 个蛋白质")
                            yield bucket_batch
                        else:
                            logger.warning(f"❌ 桶 {bucket_id} 批次创建失败")
                            
                except Exception as e:
                    logger.error(f"处理桶 {bucket_id} 时出错: {e}")
                    continue
        
        else:
            # Fallback: 遍历所有桶查找
            logger.info("使用遍历模式查找蛋白质...")
            target_protein_set = set(protein_ids)
            
            for bucket_id in sorted(self.bucket_paths.keys()):
                env = self._get_bucket_env(bucket_id)
                if not env:
                    continue
                
                bucket_proteins = []
                bucket_found = 0
                
                try:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        cursor.first()
                        
                        while cursor.key():
                            key_bytes = cursor.key()
                            
                            # Skip metadata keys
                            if key_bytes.startswith(b'_'):
                                cursor.next()
                                continue
                            
                            key_str = key_bytes.decode('utf-8')
                            protein_key = BucketedProteinKey(key_str)
                            
                            # 检查蛋白质ID是否在目标列表中
                            if protein_key.protein_id in target_protein_set:
                                data = cursor.value()
                                bucket_proteins.append((protein_key, data))
                                bucket_found += 1
                            
                            cursor.next()
                    
                    if bucket_proteins:
                        logger.info(f"📦 桶 {bucket_id}: 找到 {bucket_found} 个目标蛋白质")
                        total_found += bucket_found
                        
                        # 基于桶内序列长度进行CPU padding
                        bucket_batch = self._create_bucket_batch(bucket_proteins, bucket_id)
                        
                        if bucket_batch and 'protein_ids' in bucket_batch:
                            logger.debug(f"✅ 桶 {bucket_id} 批次就绪: {len(bucket_batch['protein_ids'])} 个蛋白质")
                            yield bucket_batch
                        else:
                            logger.warning(f"❌ 桶 {bucket_id} 批次创建失败")
                            
                except Exception as e:
                    logger.error(f"处理桶 {bucket_id} 时出错: {e}")
                    continue
        
        logger.info(f"✅ 总共找到 {total_found}/{len(protein_ids)} 个蛋白质")
        
        if total_found == 0:
            logger.warning("⚠️ 未找到任何目标蛋白质，请检查蛋白质ID是否存在于分桶LMDB中")
    
    def _create_bucket_batch(self, bucket_proteins: List[Tuple[BucketedProteinKey, bytes]], 
                           bucket_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        创建基于桶的批次，根据桶内最大长度进行CPU padding
        
        Args:
            bucket_proteins: 桶内的蛋白质数据
            bucket_id: 桶ID
            
        Returns:
            Dict包含padded的嵌入和注意力掩码
        """
        if not bucket_proteins:
            return None
        
        # 解析所有嵌入数据
        embeddings = []
        protein_ids = []
        actual_lengths = []
        
        for protein_key, data in bucket_proteins:
            embedding = self.cpu_processor._parse_embedding_data(data, protein_key.seq_length)
            if embedding is not None:
                embeddings.append(embedding)
                protein_ids.append(protein_key.protein_id)
                actual_lengths.append(embedding.shape[0])
        
        if not embeddings:
            logger.warning(f"桶 {bucket_id} 中没有有效的嵌入数据")
            return None
        
        # 确定桶内最大长度进行padding
        max_length = max(actual_lengths)
        batch_size = len(embeddings)
        
        logger.debug(f"桶 {bucket_id}: {batch_size} 个蛋白质, 最大长度: {max_length}")
        
        # 创建padded张量
        padded_embeddings = torch.zeros(batch_size, max_length, self.embedding_dim, dtype=self.dtype)
        attention_masks = torch.zeros(batch_size, max_length, dtype=torch.bool)
        
        # 填充数据
        for i, (embedding, actual_len) in enumerate(zip(embeddings, actual_lengths)):
            seq_len = min(actual_len, max_length)
            padded_embeddings[i, :seq_len] = embedding[:seq_len]
            attention_masks[i, :seq_len] = True
        
        # 转移到目标设备（GPU）
        if self.device.type == 'cuda':
            padded_embeddings = padded_embeddings.to(self.device, non_blocking=True)
            attention_masks = attention_masks.to(self.device, non_blocking=True)
        
        return {
            'embeddings': padded_embeddings,
            'attention_masks': attention_masks,
            'protein_ids': protein_ids,
            'bucket_id': bucket_id,
            'max_length': max_length
        }
    
    def close(self):
        """Close all resources"""
        # Close LMDB environments
        for env in self.bucket_envs.values():
            env.close()
        self.bucket_envs.clear()
        
        # Close GPU transfer
        self.gpu_transfer.close()
        
        logger.info("分桶LMDB加载器已关闭")


def create_bucketed_lmdb_loader(lmdb_dir: str, device: torch.device, 
                              batch_size: int = 32, max_workers: int = 4) -> BucketedLMDBLoader:
    """
    Factory function to create bucketed LMDB loader
    
    Args:
        lmdb_dir (str): Directory containing bucketed LMDB files
        device (torch.device): Target device
        batch_size (int): Batch size
        max_workers (int): Maximum worker threads
        
    Returns:
        BucketedLMDBLoader: Initialized loader
    """
    return BucketedLMDBLoader(lmdb_dir, device, batch_size, max_workers)