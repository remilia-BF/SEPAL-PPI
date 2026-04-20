#!/usr/bin/env python3
"""
智能缓存数据加载器 - 统一LMDB缓存和动态批次构建

这个模块实现：
1. 统一加载整个分桶LMDB到显存缓存
2. 动态读取训练/验证/测试数据
3. 智能padding和masking
4. 内存管理和GPU缓存控制
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
from collections import defaultdict, OrderedDict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
from tqdm import tqdm
from ..utils.helpers import create_generator


class UnifiedEmbeddingCache:
    """
    统一的嵌入缓存系统，管理GPU内存和动态加载
    """
    
    def __init__(self, lmdb_path: str, cache_size_mb: int = 8000, 
                 embedding_dim: int = 1280, device: str = 'cuda', logger=None):
        """
        初始化统一缓存系统
        
        Args:
            lmdb_path (str): 分桶LMDB路径
            cache_size_mb (int): GPU缓存大小(MB)
            embedding_dim (int): 嵌入维度
            device (str): 设备类型
            logger: 日志记录器
        """
        self.lmdb_path = lmdb_path
        self.cache_size_mb = cache_size_mb
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        
        # 缓存状态
        self.gpu_cache = OrderedDict()  # LRU缓存
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # LMDB连接
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
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
                self.target_precision = precision_data.get('target_precision', 'fp32')
                self.original_precision = precision_data.get('original_precision', 'fp32')
            else:
                self.target_precision = 'fp32'
                self.original_precision = 'fp32'
            
            self.logger.info(f"加载元数据：{self.num_buckets}个桶，精度：{self.target_precision}")
    
    def _build_protein_index(self):
        """构建蛋白质索引"""
        self.protein_to_bucket = {}
        self.protein_lengths = {}
        self.bucket_proteins = defaultdict(list)
        self.protein_to_key = {}  # 保存完整的键信息用于解析
        
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
                            self.protein_to_key[protein_id] = key  # 保存完整键
                            
                            # 从LMDB获取实际长度
                            data = cursor.value()
                            seq_len = self._get_sequence_length(data)
                            if seq_len > 0:
                                self.protein_lengths[protein_id] = seq_len
                            
                            protein_count += 1
                    except (ValueError, IndexError):
                        pass
                
                cursor.next()
        
        self.logger.info(f"构建索引：{protein_count}个蛋白质，{len(self.bucket_proteins)}个桶")
    
    def _get_sequence_length(self, data: bytes) -> int:
        """从数据中获取序列长度"""
        try:
            if self.target_precision == 'fp32':
                arr = np.frombuffer(data, dtype=np.float32)
            elif self.target_precision == 'fp16':
                arr = np.frombuffer(data, dtype=np.float16)
            elif self.target_precision == 'int8':
                # 跳过scale信息
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
        
        # 估算平均序列长度
        if self.protein_lengths:
            avg_length = sum(self.protein_lengths.values()) / len(self.protein_lengths)
        else:
            avg_length = 200  # 默认估算
        
        # 计算可缓存的蛋白质数量
        avg_protein_bytes = avg_length * bytes_per_embedding
        self.max_cached_proteins = max(1, int(cache_bytes * 0.8 / avg_protein_bytes))
        
        self.logger.info(f"缓存容量：{self.max_cached_proteins}个蛋白质 "
                        f"(平均长度{avg_length:.1f}, {self.cache_size_mb}MB)")
    
    def _preload_cache(self):
        """预加载常用蛋白质到缓存"""
        # 这里可以根据使用频率预加载
        # 现在简单地预加载一些短序列蛋白质
        preload_count = min(self.max_cached_proteins // 4, 1000)
        
        if preload_count > 0:
            # 按序列长度排序，优先加载短序列
            proteins_by_length = [(pid, length) for pid, length in self.protein_lengths.items()]
            proteins_by_length.sort(key=lambda x: x[1])
            
            self.logger.info(f"预加载 {preload_count} 个蛋白质到缓存...")
            
            for i, (protein_id, _) in enumerate(proteins_by_length[:preload_count]):
                if i % 200 == 0:
                    self.logger.debug(f"预加载进度: {i}/{preload_count}")
                self._load_protein_to_cache(protein_id)
    
    def _load_protein_to_cache(self, protein_id: str) -> bool:
        """加载蛋白质到GPU缓存"""
        if protein_id in self.gpu_cache:
            # 更新LRU顺序
            self.gpu_cache.move_to_end(protein_id)
            return True
        
        # 从LMDB加载
        bucketed_key = self._get_bucketed_key(protein_id)
        if not bucketed_key:
            return False
        
        with self.env.begin() as txn:
            data = txn.get(bucketed_key.encode())
            if data is None:
                return False
            
            try:
                # 解析数据
                embedding = self._parse_embedding_data(data)
                if embedding is None:
                    return False
                
                # 移动到GPU
                gpu_embedding = embedding.to(self.device, non_blocking=True)
                
                with self.cache_lock:
                    # 检查缓存是否已满
                    if len(self.gpu_cache) >= self.max_cached_proteins:
                        # 移除最旧的项目
                        oldest_key = next(iter(self.gpu_cache))
                        del self.gpu_cache[oldest_key]
                        
                        # 释放GPU内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 添加到缓存
                    self.gpu_cache[protein_id] = gpu_embedding
                
                return True
                
            except Exception as e:
                self.logger.warning(f"加载蛋白质 {protein_id} 到缓存失败: {e}")
                return False
    
    def _get_bucketed_key(self, protein_id: str) -> Optional[str]:
        """获取蛋白质的分桶键"""
        if protein_id not in self.protein_to_bucket:
            return None
        
        bucket_id = self.protein_to_bucket[protein_id]
        
        # 构造可能的键格式
        if bucket_id < len(self.bucket_boundaries):
            bucket_max_len = self.bucket_boundaries[bucket_id]
        else:
            bucket_max_len = self.bucket_boundaries[-1] * 2 if self.bucket_boundaries else 1000
        
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
                # 确保数组可写
                if not arr.flags.writeable:
                    arr = arr.copy()
                embedding = torch.from_numpy(arr.reshape(seq_len, self.embedding_dim))
                return embedding
            else:
                return torch.from_numpy(arr.copy() if hasattr(arr, 'copy') else arr)
                
        except Exception as e:
            self.logger.warning(f"解析嵌入数据失败: {e}")
            return None
    
    def get_embedding(self, protein_id: str) -> Optional[torch.Tensor]:
        """获取蛋白质嵌入（优先从缓存，否则动态加载）"""
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
        
        # 缓存失败，直接从LMDB加载到CPU然后移动到GPU
        bucketed_key = self._get_bucketed_key(protein_id)
        if not bucketed_key:
            return None
        
        with self.env.begin() as txn:
            data = txn.get(bucketed_key.encode())
            if data is None:
                return None
            
            embedding = self._parse_embedding_data(data)
            if embedding is not None:
                return embedding.to(self.device, non_blocking=True)
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


class UnifiedPPIDataset(Dataset):
    """
    统一的PPI数据集，使用统一缓存系统
    """
    
    def __init__(self, interaction_pairs: List[Tuple[str, str, int]], 
                 embedding_cache: UnifiedEmbeddingCache,
                 fasta_sequences: Dict[str, str], 
                 max_length: int = 1024,
                 logger=None):
        """
        初始化数据集
        
        Args:
            interaction_pairs: 交互对列表
            embedding_cache: 统一嵌入缓存
            fasta_sequences: FASTA序列
            max_length: 最大序列长度
            logger: 日志记录器
        """
        self.interaction_pairs = interaction_pairs
        self.embedding_cache = embedding_cache
        self.fasta_sequences = fasta_sequences
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化PPI数据集: {len(interaction_pairs)} 个交互对")
    
    def __len__(self):
        return len(self.interaction_pairs)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        protein1, protein2, label = self.interaction_pairs[idx]
        
        # 从缓存获取嵌入
        embedding1 = self.embedding_cache.get_embedding(protein1)
        embedding2 = self.embedding_cache.get_embedding(protein2)
        
        # 处理缺失嵌入
        if embedding1 is None or embedding2 is None:
            dummy_embedding = torch.zeros(self.max_length, self.embedding_cache.embedding_dim, 
                                        device=self.embedding_cache.device)
            dummy_mask = torch.zeros(self.max_length, device=self.embedding_cache.device)
            
            return {
                'protein1_embedding': dummy_embedding,  # 使用embedding键名以触发smart_collate_fn
                'protein1_mask': dummy_mask,
                'protein2_embedding': dummy_embedding,  # 使用embedding键名以触发smart_collate_fn
                'protein2_mask': dummy_mask,
                'label': torch.tensor(label, dtype=torch.float32, device=self.embedding_cache.device),
                'protein1_id': protein1,
                'protein2_id': protein2
            }
        
        # 获取序列长度并进行智能padding
        seq_len1 = embedding1.size(0)
        seq_len2 = embedding2.size(0)
        
        # 根据桶边界确定target_length
        max_len_in_pair = max(seq_len1, seq_len2)
        target_length = self.max_length
        
        for boundary in self.embedding_cache.bucket_boundaries:
            if max_len_in_pair <= boundary:
                target_length = min(boundary, self.max_length)
                break
        
        # 应用动态padding
        padded_embedding1, mask1 = self._apply_padding(embedding1, target_length)
        padded_embedding2, mask2 = self._apply_padding(embedding2, target_length)
        
        # 获取桶信息
        bucket1_id = self.embedding_cache.protein_to_bucket.get(protein1, 0)
        bucket2_id = self.embedding_cache.protein_to_bucket.get(protein2, 0)
        
        # 获取原始长度（从LMDB中存储的实际长度）
        original_length1 = self.embedding_cache.protein_lengths.get(protein1, seq_len1)
        original_length2 = self.embedding_cache.protein_lengths.get(protein2, seq_len2)
        
        return {
            'protein1_embedding': padded_embedding1,  # 使用embedding键名以触发smart_collate_fn
            'protein1_mask': mask1,
            'protein2_embedding': padded_embedding2,  # 使用embedding键名以触发smart_collate_fn
            'protein2_mask': mask2,
            'label': torch.tensor(label, dtype=torch.float32, device=self.embedding_cache.device),
            'protein1_id': protein1,
            'protein2_id': protein2,
            'bucket1_id': bucket1_id,
            'bucket2_id': bucket2_id,
            'original_length1': original_length1,
            'original_length2': original_length2,
            'target_length': target_length
        }
    
    def _apply_padding(self, embedding: torch.Tensor, target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用动态padding和创建mask"""
        seq_len = embedding.size(0)
        device = embedding.device
        
        if seq_len >= target_length:
            # 截断
            padded_embedding = embedding[:target_length]
            attention_mask = torch.ones(target_length, device=device)
        else:
            # Padding
            padding_size = target_length - seq_len
            padding = torch.zeros(padding_size, self.embedding_cache.embedding_dim, device=device)
            padded_embedding = torch.cat([embedding, padding], dim=0)
            
            attention_mask = torch.zeros(target_length, device=device)
            attention_mask[:seq_len] = 1
        
        return padded_embedding, attention_mask


def create_unified_data_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建统一的数据加载器系统
    
    Args:
        config: 配置字典
        
    Returns:
        包含数据加载器和缓存系统的字典
    """
    from .fasta_parser import parse_fasta_file
    
    logger = logging.getLogger(__name__)
    
    # 加载FASTA序列
    fasta_file = config.get('fasta_file')
    if not fasta_file or not Path(fasta_file).exists():
        raise FileNotFoundError(f"FASTA文件未找到: {fasta_file}")
    
    logger.info(f"加载FASTA文件: {fasta_file}")
    fasta_sequences = parse_fasta_file(fasta_file)
    logger.info(f"加载了 {len(fasta_sequences)} 个序列")
    
    # 初始化统一缓存系统
    embedding_cache = UnifiedEmbeddingCache(
        lmdb_path=config['embedding_file'],
        cache_size_mb=config.get('cache_size', 8000),
        embedding_dim=config.get('embedding_dim', 1280),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger
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
    
    # 定义collate函数以确保正确的数据格式
    def unified_collate_fn(batch):
        """统一数据集的collate函数"""
        # 收集所有数据
        protein1_seqs = []
        protein1_masks = []
        protein2_seqs = []
        protein2_masks = []
        labels = []
        protein1_ids = []
        protein2_ids = []
        bucket1_ids = []
        bucket2_ids = []
        original_lengths1 = []
        original_lengths2 = []
        
        for item in batch:
            protein1_seqs.append(item['protein1_embedding'])
            protein1_masks.append(item['protein1_mask'])
            protein2_seqs.append(item['protein2_embedding'])
            protein2_masks.append(item['protein2_mask'])
            labels.append(item['label'])
            protein1_ids.append(item['protein1_id'])
            protein2_ids.append(item['protein2_id'])
            bucket1_ids.append(item['bucket1_id'])
            bucket2_ids.append(item['bucket2_id'])
            original_lengths1.append(item['original_length1'])
            original_lengths2.append(item['original_length2'])
        
        # 堆叠张量 - 使用智能批次格式
        result = {
            'protein1_embedding': torch.stack(protein1_seqs),  # 智能批次格式
            'protein1_mask': torch.stack(protein1_masks),
            'protein2_embedding': torch.stack(protein2_seqs),  # 智能批次格式
            'protein2_mask': torch.stack(protein2_masks),
            'label': torch.stack(labels),
            'metadata': {
                'protein1_ids': protein1_ids,
                'protein2_ids': protein2_ids,
                'bucket1_ids': bucket1_ids,
                'bucket2_ids': bucket2_ids,
                'original_lengths1': original_lengths1,
                'original_lengths2': original_lengths2,
                'batch_max_len': max([seq.size(0) for seq in protein1_seqs + protein2_seqs]),
                'pooling_type': 'avg',  # 默认池化类型
                'is_cis_data': False
            }
        }
        
        # 调试信息
        logger.debug(f"Collate函数返回的键: {list(result.keys())}")
        logger.debug(f"metadata中的protein1_ids数量: {len(result['metadata']['protein1_ids'])}")
        
        return result
    
    # 创建训练数据集
    if 'train_file' in config:
        train_pairs = load_pairs(config['train_file'])
        logger.info(f"加载训练数据: {len(train_pairs)} 对")
        
        train_dataset = UnifiedPPIDataset(
            interaction_pairs=train_pairs,
            embedding_cache=embedding_cache,
            fasta_sequences=fasta_sequences,
            max_length=max_length,
            logger=logger
        )
        
        # 简单的DataLoader，不使用复杂的sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 由于使用GPU缓存，不使用多进程
            pin_memory=False,  # 数据已在GPU上
            drop_last=True,
            collate_fn=unified_collate_fn,
            generator=create_generator(config.get('seed', 42))
        )
        
        data_loaders['train'] = train_loader
        datasets['train'] = train_dataset
    
    # 创建验证和测试数据集
    test_files = config.get('test_files', {})
    for test_name, test_file in test_files.items():
        test_pairs = load_pairs(test_file)
        logger.info(f"加载测试数据 {test_name}: {len(test_pairs)} 对")
        
        test_dataset = UnifiedPPIDataset(
            interaction_pairs=test_pairs,
            embedding_cache=embedding_cache,
            fasta_sequences=fasta_sequences,
            max_length=max_length,
            logger=logger
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=unified_collate_fn,
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
        'cache_stats': embedding_cache.get_cache_stats()
    }
