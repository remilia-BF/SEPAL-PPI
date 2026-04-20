#!/usr/bin/env python3
"""
智能批次数据加载器 - 支持桶感知的智能批次构建和动态池化

主要特性：
1. 缓存原始嵌入（按桶边界padding，不池化）
2. 智能批次采样（相似桶长度的样本组合）
3. 二次动态padding和masking
4. 支持多种池化策略（avg, max, attention）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import lmdb
import numpy as np
import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict
import logging
import threading
import gc
import random
from tqdm import tqdm
from ..utils.helpers import create_generator


class UnifiedEmbeddingCache:
    """
    统一嵌入缓存系统 - 缓存按桶边界padding的原始嵌入
    """
    
    def __init__(self, lmdb_path: str, cache_size_mb: int = 8000, 
                 embedding_dim: int = 1280, device: str = 'cuda', logger=None,
                 forced_target_precision: Optional[str] = None, cis_type: bool = False):
        self.lmdb_path = lmdb_path
        self.cache_size_mb = cache_size_mb
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        self.cis_type = cis_type  # 是否为cis级别数据
        
        # 缓存状态
        self.gpu_cache = OrderedDict()  # LRU缓存
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 可选：强制指定的目标精度（优先于LMDB元数据）
        self.forced_target_precision = forced_target_precision if forced_target_precision in ['fp32', 'fp16', 'bf16', 'int8'] else None
        
        # LMDB连接
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        if self.cis_type:
            self.logger.info("检测到cis级别数据，跳过分桶检测和padding mask，直接处理长度为1的嵌入")
            # 对于cis数据，不需要加载分桶元数据和索引
            self.bucket_boundaries = [1]
            self.num_buckets = 1
            self.protein_to_bucket = {}
            self.protein_lengths = {}
            self.bucket_proteins = defaultdict(list)
            self.target_precision = forced_target_precision or 'fp32'
        else:
            # 加载元数据和索引
            self._load_metadata()
            self._build_protein_index()
        
        # 计算缓存容量
        self._calculate_cache_capacity()
        
        # 预加载热点数据
        self._preload_cache()
    
    def _load_metadata(self):
        """加载LMDB元数据"""
        with self.env.begin() as txn:
            # 加载分桶信息
            bucket_info = txn.get(b'_bucket_info')
            if bucket_info:
                bucket_data = json.loads(bucket_info.decode())
                self.bucket_boundaries = bucket_data.get('boundaries', [])
                self.num_buckets = bucket_data.get('num_buckets', 1)
            else:
                self.bucket_boundaries = [100, 200, 400, 800]
                self.num_buckets = 5
            
            # 加载精度信息
            precision_info = txn.get(b'_precision_info')
            if precision_info:
                precision_data = json.loads(precision_info.decode())
                detected_precision = precision_data.get('target_precision', 'fp32')
            else:
                detected_precision = 'fp32'
            
            # 如果提供了强制精度，则使用强制精度，否则使用检测到的精度
            if self.forced_target_precision:
                self.target_precision = self.forced_target_precision
                if self.forced_target_precision != detected_precision:
                    self.logger.info(f"使用配置中指定的精度: {self.forced_target_precision} (检测到: {detected_precision})")
                else:
                    self.logger.info(f"使用配置中指定的精度: {self.forced_target_precision}")
            else:
                self.target_precision = detected_precision
                self.logger.info(f"使用LMDB中检测到的精度: {self.target_precision}")
            
            self.logger.info(f"缓存元数据：{self.num_buckets}个桶，边界：{self.bucket_boundaries}")
    
    def _build_protein_index(self):
        """构建蛋白质索引"""
        self.protein_to_bucket = {}
        self.protein_lengths = {}
        self.bucket_proteins = defaultdict(list)
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            
            protein_count = 0
            while cursor.key():
                key = cursor.key().decode()
                
                if key.startswith('_'):
                    cursor.next()
                    continue
                
                # 解析键：bucket_{bucket_id}_len_{max_len}_{protein_id}
                if key.startswith('bucket_'):
                    try:
                        parts = key.split('_')
                        if len(parts) >= 4:
                            bucket_id = int(parts[1])
                            bucket_max_len = int(parts[3])
                            protein_id = '_'.join(parts[4:])
                            
                            self.protein_to_bucket[protein_id] = bucket_id
                            self.bucket_proteins[bucket_id].append(protein_id)
                            
                            # 获取实际序列长度
                            data = cursor.value()
                            seq_len = self._get_sequence_length(data)
                            if seq_len > 0:
                                self.protein_lengths[protein_id] = seq_len
                            
                            protein_count += 1
                    except (ValueError, IndexError):
                        pass
                
                cursor.next()
        
        self.logger.info(f"构建索引：{protein_count}个蛋白质，{len(self.bucket_proteins)}个桶")
        
        # 统计每个桶的蛋白质数量
        for bucket_id, proteins in self.bucket_proteins.items():
            self.logger.debug(f"桶 {bucket_id}: {len(proteins)} 个蛋白质")
    
    def _get_sequence_length(self, data: bytes) -> int:
        """从数据中获取序列长度"""
        try:
            if self.target_precision == 'fp32':
                arr = np.frombuffer(data, dtype=np.float32)
            elif self.target_precision == 'fp16':
                arr = np.frombuffer(data, dtype=np.float16)
            elif self.target_precision == 'bf16':
                # 存储为uint16，每个元素2字节
                arr = np.frombuffer(data, dtype=np.uint16)
            elif self.target_precision == 'int8':
                arr = np.frombuffer(data[4:], dtype=np.int8)
            else:
                return 0
            
            if len(arr) % self.embedding_dim == 0:
                return len(arr) // self.embedding_dim
            else:
                return 0
        except:
            return 0
    
    def _calculate_cache_capacity(self):
        """计算缓存容量"""
        # 每个嵌入占用的字节数（fp32）
        bytes_per_embedding = 4 * self.embedding_dim
        
        # 可用缓存字节数
        cache_bytes = self.cache_size_mb * 1024 * 1024
        
        # 估算平均桶长度
        if self.bucket_boundaries:
            avg_bucket_length = sum(self.bucket_boundaries) / len(self.bucket_boundaries)
        else:
            avg_bucket_length = 300
        
        # 计算可缓存的蛋白质数量
        avg_protein_bytes = avg_bucket_length * bytes_per_embedding
        self.max_cached_proteins = max(1, int(cache_bytes * 0.8 / avg_protein_bytes))
        
        self.logger.info(f"缓存容量：{self.max_cached_proteins}个蛋白质 "
                        f"(平均桶长度{avg_bucket_length:.1f}, {self.cache_size_mb}MB)")
    
    def _preload_cache(self):
        """预加载常用蛋白质到缓存"""
        preload_count = min(self.max_cached_proteins // 4, 500)
        
        if preload_count > 0 and self.protein_lengths:
            # 按序列长度排序，优先加载短序列
            proteins_by_length = [(pid, length) for pid, length in self.protein_lengths.items()]
            proteins_by_length.sort(key=lambda x: x[1])
            
            self.logger.info(f"预加载 {preload_count} 个蛋白质到缓存...")
            
            for i, (protein_id, _) in enumerate(proteins_by_length[:preload_count]):
                if i % 100 == 0:
                    self.logger.debug(f"预加载进度: {i}/{preload_count}")
                self._load_protein_to_cache(protein_id)
    
    def _load_protein_to_cache(self, protein_id: str) -> bool:
        """加载蛋白质到GPU缓存（按桶边界padding，不池化）"""
        if protein_id in self.gpu_cache:
            # 更新LRU顺序
            with self.cache_lock:
                self.gpu_cache.move_to_end(protein_id)
            return True
        
        if self.cis_type:
            # 对于cis数据，直接从LMDB加载，不使用分桶键
            with self.env.begin() as txn:
                data = txn.get(protein_id.encode())
                if data is None:
                    return False
                
                try:
                    # 解析原始嵌入数据
                    embedding = self._parse_embedding_data(data)
                    if embedding is None:
                        return False
                    
                    # 对于cis数据，假设嵌入长度固定为1，不需要padding
                    # 移动到GPU
                    gpu_embedding = embedding.to(self.device, non_blocking=True)
                    # 对于cis数据，attention mask全为1，因为没有padding
                    gpu_mask = torch.ones(gpu_embedding.size(0), device=self.device)
                    
                    with self.cache_lock:
                        # 检查缓存是否已满
                        if len(self.gpu_cache) >= self.max_cached_proteins:
                            # 移除最旧的项目
                            oldest_key = next(iter(self.gpu_cache))
                            del self.gpu_cache[oldest_key]
                            
                            # 释放GPU内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # 添加到缓存（包含mask信息）
                        self.gpu_cache[protein_id] = {
                            'embedding': gpu_embedding,
                            'attention_mask': gpu_mask,
                            'bucket_id': 0,  # cis数据统一使用bucket_id=0
                            'original_length': embedding.size(0),
                            'bucket_length': embedding.size(0)  # cis数据不需要桶边界padding
                        }
                    
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"加载cis数据蛋白质 {protein_id} 到缓存失败: {e}")
                    return False
        else:
            # 原有的分桶数据处理逻辑
            # 从LMDB加载
            bucketed_key = self._get_bucketed_key(protein_id)
            if not bucketed_key:
                return False
            
            with self.env.begin() as txn:
                data = txn.get(bucketed_key.encode())
                if data is None:
                    return False
                
                try:
                    # 解析原始嵌入数据
                    embedding = self._parse_embedding_data(data)
                    if embedding is None:
                        return False
                    
                    # 获取桶信息，按桶边界padding（不池化）
                    bucket_id = self.protein_to_bucket.get(protein_id, 0)
                    bucket_max_len = self._get_bucket_max_length(bucket_id)
                    
                    padded_embedding, attention_mask = self._apply_bucket_padding(
                        embedding, bucket_max_len
                    )
                    
                    # 移动到GPU
                    gpu_embedding = padded_embedding.to(self.device, non_blocking=True)
                    gpu_mask = attention_mask.to(self.device, non_blocking=True)
                    
                    with self.cache_lock:
                        # 检查缓存是否已满
                        if len(self.gpu_cache) >= self.max_cached_proteins:
                            # 移除最旧的项目
                            oldest_key = next(iter(self.gpu_cache))
                            del self.gpu_cache[oldest_key]
                            
                            # 释放GPU内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # 添加到缓存（包含mask信息）
                        self.gpu_cache[protein_id] = {
                            'embedding': gpu_embedding,
                            'attention_mask': gpu_mask,
                            'bucket_id': bucket_id,
                            'original_length': embedding.size(0),
                            'bucket_length': bucket_max_len
                        }
                    
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"加载蛋白质 {protein_id} 到缓存失败: {e}")
                    return False
    
    def _get_bucket_max_length(self, bucket_id: int) -> int:
        """获取桶的最大长度"""
        if bucket_id < len(self.bucket_boundaries):
            return self.bucket_boundaries[bucket_id]
        else:
            return self.bucket_boundaries[-1] * 2 if self.bucket_boundaries else 1000
    
    def _apply_bucket_padding(self, embedding: torch.Tensor, bucket_max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """按桶边界进行padding和创建attention mask"""
        seq_len = embedding.size(0)
        
        if seq_len >= bucket_max_len:
            # 截断到桶边界
            padded_embedding = embedding[:bucket_max_len]
            attention_mask = torch.ones(bucket_max_len)
        else:
            # Padding到桶边界
            padding_size = bucket_max_len - seq_len
            padding = torch.zeros(padding_size, self.embedding_dim)
            padded_embedding = torch.cat([embedding, padding], dim=0)
            
            attention_mask = torch.zeros(bucket_max_len)
            attention_mask[:seq_len] = 1
        
        return padded_embedding, attention_mask
    
    def _get_bucketed_key(self, protein_id: str) -> Optional[str]:
        """获取蛋白质的分桶键"""
        if protein_id not in self.protein_to_bucket:
            return None
        
        bucket_id = self.protein_to_bucket[protein_id]
        bucket_max_len = self._get_bucket_max_length(bucket_id)
        
        return f"bucket_{bucket_id}_len_{bucket_max_len}_{protein_id}"
    
    def _parse_embedding_data(self, data: bytes) -> Optional[torch.Tensor]:
        """解析嵌入数据"""
        try:
            if self.target_precision == 'fp32':
                arr = np.frombuffer(data, dtype=np.float32)
            elif self.target_precision == 'fp16':
                # 保持fp16精度，不转换为fp32
                arr = np.frombuffer(data, dtype=np.float16)
            elif self.target_precision == 'bf16':
                # 保持bf16精度，不转换为fp32
                uint16_data = np.frombuffer(data, dtype=np.uint16)
                # frombuffer 基于 bytes 时通常是只读，需要拷贝为可写以避免 PyTorch 警告
                if not uint16_data.flags.writeable:
                    uint16_data = uint16_data.copy()
                tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
                return tensor.view(-1, self.embedding_dim)
            elif self.target_precision == 'int8':
                scale = struct.unpack('f', data[:4])[0]
                quantized = np.frombuffer(data[4:], dtype=np.int8)
                arr = quantized.astype(np.float32) * scale / 127.0
            else:
                try:
                    arr = pickle.loads(data)
                    if hasattr(arr, 'astype'):
                        arr = arr.astype(np.float32)
                except:
                    return None
            
            if hasattr(arr, 'reshape') and len(arr) % self.embedding_dim == 0:
                seq_len = len(arr) // self.embedding_dim
                if not arr.flags.writeable:
                    arr = arr.copy()
                embedding = torch.from_numpy(arr.reshape(seq_len, self.embedding_dim))
                return embedding
            else:
                return torch.from_numpy(arr.copy() if hasattr(arr, 'copy') else arr)
                
        except Exception as e:
            self.logger.warning(f"解析嵌入数据失败: {e}")
            return None
    
    def get_protein_data(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """获取蛋白质数据（嵌入+mask+元信息）"""
        with self.cache_lock:
            if protein_id in self.gpu_cache:
                self.cache_hits += 1
                # 更新LRU顺序
                self.gpu_cache.move_to_end(protein_id)
                return self.gpu_cache[protein_id]
        
        self.cache_misses += 1
        
        # 尝试加载到缓存
        if self._load_protein_to_cache(protein_id):
            with self.cache_lock:
                if protein_id in self.gpu_cache:
                    return self.gpu_cache[protein_id]
        
        # 缓存失败，直接从LMDB加载
        bucketed_key = self._get_bucketed_key(protein_id)
        if not bucketed_key:
            return None
        
        with self.env.begin() as txn:
            data = txn.get(bucketed_key.encode())
            if data is None:
                return None
            
            embedding = self._parse_embedding_data(data)
            if embedding is not None:
                bucket_id = self.protein_to_bucket.get(protein_id, 0)
                bucket_max_len = self._get_bucket_max_length(bucket_id)
                
                padded_embedding, attention_mask = self._apply_bucket_padding(
                    embedding, bucket_max_len
                )
                
                return {
                    'embedding': padded_embedding.to(self.device, non_blocking=True),
                    'attention_mask': attention_mask.to(self.device, non_blocking=True),
                    'bucket_id': bucket_id,
                    'original_length': embedding.size(0),
                    'bucket_length': bucket_max_len
                }
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'cache_size': len(self.gpu_cache),
            'max_capacity': self.max_cached_proteins,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_usage_mb': self.cache_size_mb * len(self.gpu_cache) / self.max_cached_proteins if self.max_cached_proteins > 0 else 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.gpu_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'env'):
            self.env.close()
        self.clear_cache()


class LengthSortedBatchSampler(Sampler):
    """
    长度排序批次采样器 - 根据序列长度对样本进行排序，并将长度相近的样本组成批次
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = False, drop_last: bool = False):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱批次顺序
            drop_last: 是否丢弃最后不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 计算每个样本的长度（两个蛋白质序列长度之和）
        self.sample_lengths = []
        for idx in range(len(self.dataset)):
            protein1, protein2, _ = self.dataset.interaction_pairs[idx]
            # 从fasta序列中获取实际序列长度
            if protein1 in self.dataset.fasta_sequences and protein2 in self.dataset.fasta_sequences:
                len1 = len(self.dataset.fasta_sequences[protein1])
                len2 = len(self.dataset.fasta_sequences[protein2])
                total_length = len1 + len2
            else:
                # 如果序列不存在，默认长度为200
                total_length = 200
            self.sample_lengths.append((idx, total_length))
        
        # 根据长度排序
        self.sample_lengths.sort(key=lambda x: x[1])
    
    def __iter__(self):
        """生成批次索引"""
        # 按长度顺序分组为批次
        batches = []
        for i in range(0, len(self.sample_lengths), self.batch_size):
            batch = self.sample_lengths[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append([idx for idx, _ in batch])
        
        # 如果需要，打乱批次顺序
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        """计算总批次数"""
        if self.drop_last:
            return len(self.sample_lengths) // self.batch_size
        else:
            return (len(self.sample_lengths) + self.batch_size - 1) // self.batch_size


class SmartBucketBatchSampler(Sampler):
    """
    智能桶感知批次采样器 - 优先选择相似桶长度的样本组成批次
    """
    
    def __init__(self, dataset, batch_size: int, embedding_cache: UnifiedEmbeddingCache,
                 shuffle: bool = True, drop_last: bool = False, bucket_tolerance: int = 1):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            embedding_cache: 嵌入缓存
            shuffle: 是否打乱
            drop_last: 是否丢弃最后不完整的批次
            bucket_tolerance: 桶容差（允许相邻几个桶的样本组合）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.embedding_cache = embedding_cache
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_tolerance = bucket_tolerance
        
        # 构建桶索引
        self._build_bucket_indices()
    
    def _build_bucket_indices(self):
        """构建每个桶的样本索引"""
        self.bucket_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            protein1, protein2, _ = self.dataset.interaction_pairs[idx]
            
            # 获取两个蛋白质的桶ID
            bucket1 = self.embedding_cache.protein_to_bucket.get(protein1, 0)
            bucket2 = self.embedding_cache.protein_to_bucket.get(protein2, 0)
            
            # 使用较大的桶ID作为批次分组依据
            max_bucket = max(bucket1, bucket2)
            self.bucket_indices[max_bucket].append(idx)
        
        # 统计信息
        total_samples = sum(len(indices) for indices in self.bucket_indices.values())
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"智能采样器: {len(self.bucket_indices)} 个桶, {total_samples} 个样本")
        for bucket_id, indices in self.bucket_indices.items():
            self.logger.debug(f"桶 {bucket_id}: {len(indices)} 个样本")
    
    def __iter__(self):
        """生成批次索引"""
        # 收集所有桶的批次
        all_batches = []
        
        for bucket_id, indices in self.bucket_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # 为当前桶生成批次
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
            
            # 考虑桶容差，与相邻桶混合批次
            if self.bucket_tolerance > 0:
                for tolerance in range(1, self.bucket_tolerance + 1):
                    adjacent_bucket = bucket_id + tolerance
                    if adjacent_bucket in self.bucket_indices:
                        adjacent_indices = self.bucket_indices[adjacent_bucket]
                        if self.shuffle:
                            random.shuffle(adjacent_indices)
                        
                        # 创建混合批次
                        mixed_indices = indices + adjacent_indices
                        if self.shuffle:
                            random.shuffle(mixed_indices)
                        
                        for i in range(0, len(mixed_indices), self.batch_size):
                            batch = mixed_indices[i:i + self.batch_size]
                            if len(batch) == self.batch_size or not self.drop_last:
                                all_batches.append(batch)
        
        # 打乱所有批次的顺序
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """计算总批次数"""
        total_batches = 0
        for bucket_id, indices in self.bucket_indices.items():
            # 计算当前桶的批次数量
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
            
            # 考虑桶容差，计算混合批次数量
            if self.bucket_tolerance > 0:
                for tolerance in range(1, self.bucket_tolerance + 1):
                    adjacent_bucket = bucket_id + tolerance
                    if adjacent_bucket in self.bucket_indices:
                        adjacent_indices = self.bucket_indices[adjacent_bucket]
                        mixed_indices = indices + adjacent_indices
                        
                        if self.drop_last:
                            total_batches += len(mixed_indices) // self.batch_size
                        else:
                            total_batches += (len(mixed_indices) + self.batch_size - 1) // self.batch_size
        return total_batches


class SmartBatchPPIDataset(Dataset):
    """
    智能批次PPI数据集 - 支持桶感知和动态池化
    """
    
    def __init__(self, interaction_pairs: List[Tuple[str, str, int]], 
                 embedding_cache: UnifiedEmbeddingCache,
                 fasta_sequences: Dict[str, str], 
                 max_length: int = 1024,
                 pooling_type: str = 'avg',
                 logger=None):
        """
        Args:
            interaction_pairs: 交互对列表 [(protein1, protein2, label), ...]
            embedding_cache: 统一嵌入缓存
            fasta_sequences: FASTA序列字典
            max_length: 全局最大序列长度
            pooling_type: 池化类型 ('avg', 'max', 'attention')
            logger: 日志记录器
        """
        self.interaction_pairs = interaction_pairs
        self.embedding_cache = embedding_cache
        self.fasta_sequences = fasta_sequences
        self.max_length = max_length
        self.pooling_type = pooling_type
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化智能批次PPI数据集: {len(interaction_pairs)} 个交互对, "
                        f"池化类型: {pooling_type}")
    
    def __len__(self):
        return len(self.interaction_pairs)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        protein1, protein2, label = self.interaction_pairs[idx]
        
        # 从缓存获取蛋白质数据（按桶边界padding的原始嵌入）
        protein1_data = self.embedding_cache.get_protein_data(protein1)
        protein2_data = self.embedding_cache.get_protein_data(protein2)
        
        # 处理缺失数据
        if protein1_data is None or protein2_data is None:
            # 返回零向量作为后备方案
            dummy_length = min(self.max_length, 200)  # 使用较短的默认长度
            dummy_embedding = torch.zeros(dummy_length, self.embedding_cache.embedding_dim, 
                                        device=self.embedding_cache.device)
            dummy_mask = torch.zeros(dummy_length, device=self.embedding_cache.device)
            
            return {
                'protein1_embedding': dummy_embedding,
                'protein1_mask': dummy_mask,
                'protein2_embedding': dummy_embedding,
                'protein2_mask': dummy_mask,
                'label': torch.tensor(label, dtype=torch.float32, device=self.embedding_cache.device),
                'protein1_id': protein1,
                'protein2_id': protein2,
                'bucket1_id': 0,
                'bucket2_id': 0,
                'original_length1': 1,
                'original_length2': 1
            }
        
        # 返回原始嵌入（不池化）
        return {
            'protein1_embedding': protein1_data['embedding'],  # [bucket_len, embed_dim]
            'protein1_mask': protein1_data['attention_mask'],   # [bucket_len]
            'protein2_embedding': protein2_data['embedding'],   # [bucket_len, embed_dim]
            'protein2_mask': protein2_data['attention_mask'],   # [bucket_len]
            'label': torch.tensor(label, dtype=torch.float32, device=self.embedding_cache.device),
            'protein1_id': protein1,
            'protein2_id': protein2,
            'bucket1_id': protein1_data['bucket_id'],
            'bucket2_id': protein2_data['bucket_id'],
            'original_length1': protein1_data['original_length'],
            'original_length2': protein2_data['original_length']
        }


def smart_collate_fn(batch: List[Dict[str, torch.Tensor]], pooling_type: str = 'avg',
                    max_length: int = 1024) -> Dict[str, Any]:
    """
    智能批次整理函数 - 进行二次padding并准备池化
    
    Args:
        batch: 批次数据
        pooling_type: 池化类型
        max_length: 最大长度限制
        
    Returns:
        整理后的批次字典
    """
    if not batch:
        return {}
    
    device = batch[0]['protein1_embedding'].device
    embedding_dim = batch[0]['protein1_embedding'].size(1)
    
    # 检查是否为cis数据（所有嵌入长度都为1）
    is_cis_data = all(
        item['protein1_embedding'].size(0) == 1 and item['protein2_embedding'].size(0) == 1 
        for item in batch
    )
    
    if is_cis_data and max_length == 1:
        # 对于cis数据，不需要padding，直接堆叠
        batch_size = len(batch)
        
        protein1_embeddings = torch.stack([item['protein1_embedding'] for item in batch])  # [batch_size, 1, embed_dim]
        protein1_masks = torch.stack([item['protein1_mask'] for item in batch])  # [batch_size, 1]
        protein2_embeddings = torch.stack([item['protein2_embedding'] for item in batch])  # [batch_size, 1, embed_dim]
        protein2_masks = torch.stack([item['protein2_mask'] for item in batch])  # [batch_size, 1]
        # 确保标签形状一致（处理标量张量）
        labels = torch.tensor([item['label'].item() if item['label'].dim() == 0 else item['label'][0].item() for item in batch], 
                             dtype=torch.float32, device=batch[0]['label'].device)  # [batch_size]
        
        # 收集元数据
        metadata = {
            'protein1_ids': [item['protein1_id'] for item in batch],
            'protein2_ids': [item['protein2_id'] for item in batch],
            'bucket1_ids': [item['bucket1_id'] for item in batch],
            'bucket2_ids': [item['bucket2_id'] for item in batch],
            'original_lengths1': [item['original_length1'] for item in batch],
            'original_lengths2': [item['original_length2'] for item in batch],
            'batch_max_len': 1,
            'pooling_type': pooling_type,
            'is_cis_data': True
        }
        
        # 为了支持多模态特征融合，也在顶层添加protein_ids
        protein_ids = [(p1, p2) for p1, p2 in zip(metadata['protein1_ids'], metadata['protein2_ids'])]
        
        return {
            'protein1_seq': protein1_embeddings,  # [batch_size, 1, embed_dim]
            'protein1_mask': protein1_masks,      # [batch_size, 1]
            'protein2_seq': protein2_embeddings,  # [batch_size, 1, embed_dim]
            'protein2_mask': protein2_masks,      # [batch_size, 1]
            'label': labels,                      # [batch_size]
            'protein_ids': protein_ids,           # [(str, str), ...] 用于多模态特征
            'metadata': metadata
        }
    
    # 原有的非cis数据处理逻辑
    # 计算批次中的最大桶长度（用于二次padding）
    max_bucket_len1 = max(item['protein1_embedding'].size(0) for item in batch)
    max_bucket_len2 = max(item['protein2_embedding'].size(0) for item in batch)
    
    # 限制最大长度
    max_bucket_len1 = min(max_bucket_len1, max_length)
    max_bucket_len2 = min(max_bucket_len2, max_length)
    
    # 统一的批次长度
    batch_max_len = max(max_bucket_len1, max_bucket_len2)
    
    batch_size = len(batch)
    
    # 准备批次张量
    protein1_embeddings = torch.zeros(batch_size, batch_max_len, embedding_dim, device=device)
    protein1_masks = torch.zeros(batch_size, batch_max_len, device=device)
    protein2_embeddings = torch.zeros(batch_size, batch_max_len, embedding_dim, device=device)
    protein2_masks = torch.zeros(batch_size, batch_max_len, device=device)
    labels = torch.zeros(batch_size, device=device)
    
    # 填充批次数据
    for i, item in enumerate(batch):
        # Protein 1
        p1_len = min(item['protein1_embedding'].size(0), batch_max_len)
        protein1_embeddings[i, :p1_len] = item['protein1_embedding'][:p1_len]
        protein1_masks[i, :p1_len] = item['protein1_mask'][:p1_len]
        
        # Protein 2
        p2_len = min(item['protein2_embedding'].size(0), batch_max_len)
        protein2_embeddings[i, :p2_len] = item['protein2_embedding'][:p2_len]
        protein2_masks[i, :p2_len] = item['protein2_mask'][:p2_len]
        
        # Labels - 确保从标量或1D张量中提取值
        label_value = item['label'].item() if item['label'].dim() == 0 else item['label'][0].item()
        labels[i] = label_value
    
    # 收集元数据
    metadata = {
        'protein1_ids': [item['protein1_id'] for item in batch],
        'protein2_ids': [item['protein2_id'] for item in batch],
        'bucket1_ids': [item.get('bucket1_id', 0) for item in batch],
        'bucket2_ids': [item.get('bucket2_id', 0) for item in batch],
        'original_lengths1': [item.get('original_length1', 0) for item in batch],
        'original_lengths2': [item.get('original_length2', 0) for item in batch],
        'batch_max_len': batch_max_len,
        'pooling_type': pooling_type,
        'is_cis_data': False
    }
    
    # 为了支持多模态特征融合，也在顶层添加protein_ids
    protein_ids = [(p1, p2) for p1, p2 in zip(metadata['protein1_ids'], metadata['protein2_ids'])]
    
    return {
        'protein1_seq': protein1_embeddings,  # [batch_size, seq_len, embed_dim]
        'protein1_mask': protein1_masks,      # [batch_size, seq_len]
        'protein2_seq': protein2_embeddings,  # [batch_size, seq_len, embed_dim]
        'protein2_mask': protein2_masks,      # [batch_size, seq_len]
        'label': labels,                      # [batch_size]
        'protein_ids': protein_ids,           # [(str, str), ...] 用于多模态特征
        'metadata': metadata
    }


def create_smart_batch_data_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建智能批次数据加载器系统
    
    Args:
        config: 配置字典，包含以下键：
            - embedding_file: 嵌入文件路径
            - train_file: 训练数据文件路径
            - test_files: 测试数据文件路径字典
            - fasta_file: FASTA序列文件路径
            - batch_size: 批次大小
            - cache_size: 缓存大小
            - embedding_dim: 嵌入维度
            - max_length: 最大序列长度
            - pooling_type: 池化类型
            - target_precision: 目标精度
            - cis_type: 是否为cis级别数据 (默认 False)
        
    Returns:
        包含数据加载器和缓存系统的字典
    """
    from .fasta_parser import parse_fasta_file
    
    logger = logging.getLogger(__name__)
    
    # 检查是否为cis级别数据
    cis_type = config.get('cis_type', False)
    max_length = config.get('max_length', 1024)
    
    # 如果是cis数据且max_length为1，启用cis模式
    if cis_type and max_length == 1:
        logger.info("检测到cis级别数据配置 (cis_type=True, max_length=1)")
        use_cis_mode = True
    else:
        use_cis_mode = False
    
    # 加载FASTA序列
    fasta_file = config.get('fasta_file')
    if not fasta_file or not Path(fasta_file).exists():
        raise FileNotFoundError(f"FASTA文件未找到: {fasta_file}")
    
    logger.info(f"加载FASTA文件: {fasta_file}")
    fasta_sequences = parse_fasta_file(fasta_file)
    logger.info(f"加载了 {len(fasta_sequences)} 个序列")
    
    # 初始化统一缓存系统
    # 允许通过config覆盖LMDB中的精度元数据
    embedding_cache = UnifiedEmbeddingCache(
        lmdb_path=config['embedding_file'],
        cache_size_mb=config.get('cache_size', 8000),
        embedding_dim=config.get('embedding_dim', 1280),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        forced_target_precision=config.get('target_precision'),
        cis_type=use_cis_mode  # 传递cis_type参数
    )
    
    # 加载交互对数据的通用函数
    def load_pairs(pairs_file: str) -> List[Tuple[str, str, int]]:
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], int(parts[2])))
        return pairs
    
    # 创建数据加载器
    data_loaders = {}
    datasets = {}
    
    # 通用参数
    batch_size = config.get('batch_size', 32)
    max_length = config.get('max_length', 1024)
    pooling_type = config.get('pooling_type', 'avg')
    
    # 创建自定义collate函数
    def create_collate_fn(pooling_type: str):
        def collate_fn(batch):
            return smart_collate_fn(batch, pooling_type=pooling_type, max_length=max_length)
        return collate_fn
    
    collate_fn = create_collate_fn(pooling_type)
    
    # 创建训练数据集
    if 'train_file' in config:
        train_pairs = load_pairs(config['train_file'])
        logger.info(f"加载训练数据: {len(train_pairs)} 对")
        
        train_dataset = SmartBatchPPIDataset(
            interaction_pairs=train_pairs,
            embedding_cache=embedding_cache,
            fasta_sequences=fasta_sequences,
            max_length=max_length,
            pooling_type=pooling_type,
            logger=logger
        )
        
        # 使用智能批次采样器
        train_sampler = SmartBucketBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            embedding_cache=embedding_cache,
            shuffle=True,
            drop_last=True,
            bucket_tolerance=1
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,  # 由于使用GPU缓存，不使用多进程
            pin_memory=False,  # 数据已在GPU上
            generator=create_generator(config.get('seed', 42))
        )
        
        data_loaders['train'] = train_loader
        datasets['train'] = train_dataset
    
    # 创建验证和测试数据集
    test_files = config.get('test_files', {})
    for test_name, test_file in test_files.items():
        test_pairs = load_pairs(test_file)
        logger.info(f"加载测试数据 {test_name}: {len(test_pairs)} 对")
        
        test_dataset = SmartBatchPPIDataset(
            interaction_pairs=test_pairs,
            embedding_cache=embedding_cache,
            fasta_sequences=fasta_sequences,
            max_length=max_length,
            pooling_type=pooling_type,
            logger=logger
        )
        
        # 使用长度排序采样器
        test_sampler = LengthSortedBatchSampler(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            generator=create_generator(config.get('seed', 42))
        )
        
        data_loaders[test_name] = test_loader
        datasets[test_name] = test_dataset
    
    logger.info(f"创建了 {len(data_loaders)} 个数据加载器")
    logger.info(f"缓存统计: {embedding_cache.get_cache_stats()}")
    
    return {
        'data_loaders': data_loaders,
        'datasets': datasets,
        'embedding_cache': embedding_cache,
        'cache_stats': embedding_cache.get_cache_stats(),
        'pooling_type': pooling_type
    }
