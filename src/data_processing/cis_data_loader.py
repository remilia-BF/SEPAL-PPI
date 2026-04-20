#!/usr/bin/env python3
"""
CIS数据加载器 - 专门为cis级别数据优化的简化数据加载器

主要特性：
1. 简化的LMDB读取（直接使用蛋白质ID作为键）
2. 固定长度处理（长度为1，无需复杂padding）
3. 简化的批次构建
4. 高效的GPU缓存
5. 改进的LMDB连接和错误处理
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
from collections import OrderedDict
import logging
import threading
import gc
from tqdm import tqdm
from ..utils.helpers import create_generator


class CisEmbeddingCache:
    """
    专门为cis数据优化的嵌入缓存系统
    """
    
    def __init__(self, lmdb_path: str, cache_size_mb: int = 8000, 
                 embedding_dim: int = 1280, device: str = 'cuda', logger=None,
                 target_precision: str = 'fp32'):
        """
        初始化cis嵌入缓存
        
        Args:
            lmdb_path (str): LMDB文件路径
            cache_size_mb (int): GPU缓存大小(MB)
            embedding_dim (int): 嵌入维度
            device (str): 设备类型
            logger: 日志记录器
            target_precision (str): 目标精度
        """
        self.lmdb_path = lmdb_path
        self.cache_size_mb = cache_size_mb
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        self.target_precision = target_precision
        
        # 缓存状态
        self.gpu_cache = OrderedDict()  # LRU缓存
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # LMDB连接 - 添加异常处理
        try:
            # 检查LMDB路径是否存在
            lmdb_path_obj = Path(lmdb_path)
            if not lmdb_path_obj.exists():
                raise FileNotFoundError(f"LMDB路径不存在: {lmdb_path}")
            
            # 尝试打开LMDB
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
            
            # 验证LMDB是否可读
            with self.env.begin() as txn:
                # 尝试获取一个键来验证连接
                cursor = txn.cursor()
                if not cursor.first():
                    self.logger.warning(f"LMDB文件为空: {lmdb_path}")
                else:
                    self.logger.debug(f"LMDB连接成功: {lmdb_path}")
                    
        except Exception as e:
            self.logger.error(f"LMDB连接失败: {lmdb_path}, 错误: {e}")
            raise RuntimeError(f"无法连接到LMDB文件: {lmdb_path}") from e
        
        # 计算缓存容量
        self._calculate_cache_capacity()
        
        self.logger.info(f"初始化CIS嵌入缓存: {lmdb_path}, 缓存容量: {self.max_cached_proteins} 个蛋白质")
    
    def _calculate_cache_capacity(self):
        """计算缓存容量"""
        # 每个cis嵌入的内存占用：1 * embedding_dim * 4 bytes (fp32)
        bytes_per_protein = self.embedding_dim * 4
        
        # 加上attention mask: 1 * 4 bytes
        bytes_per_protein += 4
        
        # 总可用字节数
        total_bytes = self.cache_size_mb * 1024 * 1024
        
        # 计算可缓存的蛋白质数量（保留20%余量）
        self.max_cached_proteins = int(total_bytes * 0.8 / bytes_per_protein)
        
        self.logger.info(f"CIS缓存容量: {self.max_cached_proteins} 个蛋白质 "
                        f"(每个占用 {bytes_per_protein} 字节)")
    
    def get_protein_data(self, protein_id: str) -> Optional[torch.Tensor]:
        """
        获取蛋白质的cis嵌入数据
        
        Args:
            protein_id (str): 蛋白质ID
            
        Returns:
            torch.Tensor or None: 嵌入张量 [1, embedding_dim]
        """
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
        try:
            with self.env.begin() as txn:
                data = txn.get(protein_id.encode())
                if data is None:
                    self.logger.warning(f"蛋白质 {protein_id} 在LMDB中未找到")
                    return None
                
                embedding = self._parse_embedding_data(data)
                if embedding is not None:
                    # 确保是正确的形状 [1, embedding_dim]
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)
                    return embedding.to(self.device, non_blocking=True)
                return None
        except Exception as e:
            self.logger.error(f"从LMDB读取蛋白质 {protein_id} 失败: {e}")
            return None
    
    def _load_protein_to_cache(self, protein_id: str) -> bool:
        """加载蛋白质到GPU缓存"""
        try:
            with self.env.begin() as txn:
                data = txn.get(protein_id.encode())
                if data is None:
                    return False
                
                # 解析原始嵌入数据
                embedding = self._parse_embedding_data(data)
                if embedding is None:
                    return False
                
                # 确保是正确的形状 [1, embedding_dim]
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
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
    
    def _parse_embedding_data(self, data: bytes) -> Optional[torch.Tensor]:
        """解析嵌入数据"""
        try:
            if self.target_precision == 'fp32':
                arr = np.frombuffer(data, dtype=np.float32)
            elif self.target_precision == 'fp16':
                arr = np.frombuffer(data, dtype=np.float16)
                arr = arr.astype(np.float32)  # 转换为fp32以保持精度
            elif self.target_precision == 'bf16':
                uint16_data = np.frombuffer(data, dtype=np.uint16)
                tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
                return tensor.view(-1, self.embedding_dim).float()
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
            
            # 对于cis数据，应该是一维数组 [embedding_dim]
            if len(arr) == self.embedding_dim:
                return torch.from_numpy(arr.copy() if hasattr(arr, 'copy') else arr)
            else:
                self.logger.warning(f"嵌入数据长度不匹配: {len(arr)} vs {self.embedding_dim}")
                return None
                
        except Exception as e:
            self.logger.warning(f"解析嵌入数据失败: {e}")
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
            'memory_usage_mb': len(self.gpu_cache) * self.embedding_dim * 4 / (1024 * 1024)
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.gpu_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def close(self):
        """显式关闭LMDB连接"""
        if hasattr(self, 'env'):
            try:
                self.env.close()
                self.logger.debug("LMDB连接已关闭")
            except Exception as e:
                self.logger.warning(f"关闭LMDB连接时出错: {e}")
    
    def __del__(self):
        """清理资源"""
        try:
            self.close()
            self.clear_cache()
        except Exception:
            pass  # 忽略析构函数中的异常


class CisPPIDataset(Dataset):
    """
    CIS数据专用的PPI数据集
    """
    
    def __init__(self, interaction_pairs: List[Tuple[str, str, int]], 
                 embedding_cache: CisEmbeddingCache,
                 logger=None):
        """
        Args:
            interaction_pairs: 交互对列表 [(protein1, protein2, label), ...]
            embedding_cache: CIS嵌入缓存
            logger: 日志记录器
        """
        self.interaction_pairs = interaction_pairs
        self.embedding_cache = embedding_cache
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化CIS PPI数据集: {len(interaction_pairs)} 个交互对")
    
    def __len__(self):
        return len(self.interaction_pairs)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        protein1, protein2, label = self.interaction_pairs[idx]
        
        # 从缓存获取蛋白质数据
        protein1_embedding = self.embedding_cache.get_protein_data(protein1)
        protein2_embedding = self.embedding_cache.get_protein_data(protein2)
        
        # 处理缺失数据
        if protein1_embedding is None or protein2_embedding is None:
            # 返回零向量作为后备方案
            dummy_embedding = torch.zeros(1, self.embedding_cache.embedding_dim, 
                                        device=self.embedding_cache.device)
            
            if protein1_embedding is None:
                protein1_embedding = dummy_embedding
                self.logger.warning(f"蛋白质 {protein1} cis数据缺失，使用零向量")
            if protein2_embedding is None:
                protein2_embedding = dummy_embedding
                self.logger.warning(f"蛋白质 {protein2} cis数据缺失，使用零向量")
        
        # 获取桶信息（CIS数据通常没有桶，设为默认值）
        bucket1_id = getattr(self.embedding_cache, 'protein_to_bucket', {}).get(protein1, 0)
        bucket2_id = getattr(self.embedding_cache, 'protein_to_bucket', {}).get(protein2, 0)
        
        # 获取原始长度（CIS数据通常是1）
        original_length1 = getattr(self.embedding_cache, 'protein_lengths', {}).get(protein1, 1)
        original_length2 = getattr(self.embedding_cache, 'protein_lengths', {}).get(protein2, 1)
        
        return {
            'protein1_seq': protein1_embedding,  # [1, embed_dim]
            'protein1_mask': torch.ones(1, device=self.embedding_cache.device),  # [1]
            'protein2_seq': protein2_embedding,  # [1, embed_dim]
            'protein2_mask': torch.ones(1, device=self.embedding_cache.device),  # [1]
            'label': torch.tensor(label, dtype=torch.float32, device=self.embedding_cache.device),
            'protein1_id': protein1,
            'protein2_id': protein2,
            'bucket1_id': bucket1_id,
            'bucket2_id': bucket2_id,
            'original_length1': original_length1,
            'original_length2': original_length2,
            'target_length': 1,  # CIS数据目标长度为1
            'protein_ids': (protein1, protein2)  # 保持向后兼容
        }


def cis_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    CIS数据专用的批次整理函数
    
    Args:
        batch: 批次数据
        
    Returns:
        整理后的批次字典
    """
    if not batch:
        return {}
    
    batch_size = len(batch)
    device = batch[0]['protein1_seq'].device
    embedding_dim = batch[0]['protein1_seq'].size(1)
    
    # 对于cis数据，所有序列长度都是1，直接堆叠即可
    protein1_embeddings = torch.stack([item['protein1_seq'] for item in batch])  # [batch_size, 1, embed_dim]
    protein1_masks = torch.stack([item['protein1_mask'] for item in batch])      # [batch_size, 1]
    protein2_embeddings = torch.stack([item['protein2_seq'] for item in batch])  # [batch_size, 1, embed_dim]
    protein2_masks = torch.stack([item['protein2_mask'] for item in batch])      # [batch_size, 1]
    labels = torch.stack([item['label'] for item in batch])                     # [batch_size]
    
    # 收集元数据
    protein1_ids = [item['protein1_id'] for item in batch]
    protein2_ids = [item['protein2_id'] for item in batch]
    bucket1_ids = [item['bucket1_id'] for item in batch]
    bucket2_ids = [item['bucket2_id'] for item in batch]
    original_lengths1 = [item['original_length1'] for item in batch]
    original_lengths2 = [item['original_length2'] for item in batch]
    
    return {
        'protein1_seq': protein1_embeddings,
        'protein1_mask': protein1_masks,
        'protein2_seq': protein2_embeddings,
        'protein2_mask': protein2_masks,
        'label': labels,
        'metadata': {
            'protein1_ids': protein1_ids,
            'protein2_ids': protein2_ids,
            'bucket1_ids': bucket1_ids,
            'bucket2_ids': bucket2_ids,
            'original_lengths1': original_lengths1,
            'original_lengths2': original_lengths2,
            'batch_max_len': 1,  # CIS数据长度固定为1
            'pooling_type': 'avg',  # 默认池化类型
            'is_cis_data': True
        },
        'protein_ids': [item['protein_ids'] for item in batch]  # 保持向后兼容
    }


def create_cis_data_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建CIS数据专用的数据加载器系统
    
    Args:
        config: 配置字典，包含以下键：
            - embedding_file: 嵌入文件路径
            - train_file: 训练数据文件路径
            - test_files: 测试数据文件路径字典
            - batch_size: 批次大小
            - cache_size: 缓存大小
            - embedding_dim: 嵌入维度
            - target_precision: 目标精度
        
    Returns:
        包含数据加载器和缓存系统的字典
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=== 创建CIS数据加载器系统 ===")
    
    # 验证配置
    required_keys = ['embedding_file', 'embedding_dim']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"CIS数据加载器配置缺少必需参数: {missing_keys}")
    
    # 检查LMDB文件是否存在
    embedding_file = config['embedding_file']
    if not Path(embedding_file).exists():
        raise FileNotFoundError(f"LMDB文件不存在: {embedding_file}")
    
    # 初始化CIS嵌入缓存
    try:
        embedding_cache = CisEmbeddingCache(
            lmdb_path=embedding_file,
            cache_size_mb=config.get('cache_size', 8000),
            embedding_dim=config.get('embedding_dim', 1280),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            logger=logger,
            target_precision=config.get('target_precision', 'fp32')
        )
    except Exception as e:
        logger.error(f"初始化CIS嵌入缓存失败: {e}")
        raise
    
    # 加载交互对数据的通用函数
    def load_pairs(pairs_file: str) -> List[Tuple[str, str, int]]:
        pairs = []
        try:
            with open(pairs_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            protein1, protein2, label = parts[0], parts[1], int(parts[2])
                            pairs.append((protein1, protein2, label))
                        except ValueError as e:
                            logger.warning(f"第{line_num}行数据格式错误: {line}, 错误: {e}")
                    else:
                        logger.warning(f"第{line_num}行数据格式错误: {line}")
        except Exception as e:
            logger.error(f"读取文件 {pairs_file} 失败: {e}")
            raise
        return pairs
    
    # 创建数据加载器
    data_loaders = {}
    datasets = {}
    
    # 通用参数
    batch_size = config.get('batch_size', 32)
    
    # 创建训练数据集
    if 'train_file' in config:
        train_file = config['train_file']
        if not Path(train_file).exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        train_pairs = load_pairs(train_file)
        logger.info(f"加载训练数据: {len(train_pairs)} 对")
        
        train_dataset = CisPPIDataset(
            interaction_pairs=train_pairs,
            embedding_cache=embedding_cache,
            logger=logger
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=cis_collate_fn,
            num_workers=0,  # CIS数据已在GPU缓存中，不需要多进程
            pin_memory=False,  # 数据已在GPU上
            drop_last=True,
            generator=create_generator(config.get('seed', 42))
        )
        
        data_loaders['train'] = train_loader
        datasets['train'] = train_dataset
    
    # 创建验证和测试数据集
    test_files = config.get('test_files', {})
    for test_name, test_file in test_files.items():
        if test_file and Path(test_file).exists():
            test_pairs = load_pairs(test_file)
            logger.info(f"加载{test_name}数据: {len(test_pairs)} 对")
            
            test_dataset = CisPPIDataset(
                interaction_pairs=test_pairs,
                embedding_cache=embedding_cache,
                logger=logger
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=cis_collate_fn,
                num_workers=0,
                pin_memory=False,
                generator=create_generator(config.get('seed', 42))
            )
            
            data_loaders[test_name] = test_loader
            datasets[test_name] = test_dataset
        else:
            logger.warning(f"测试文件不存在: {test_file}")
    
    # 获取缓存统计
    cache_stats = embedding_cache.get_cache_stats()
    
    logger.info(f"CIS数据加载器创建完成:")
    logger.debug(f"  - 数据集: {list(data_loaders.keys())}")
    logger.info(f"  - 批次大小: {batch_size}")
    logger.info(f"  - 缓存统计: {cache_stats['cache_size']}/{cache_stats['max_capacity']} "
               f"(命中率: {cache_stats['hit_rate']:.1%})")
    
    return {
        'data_loaders': data_loaders,
        'datasets': datasets,
        'embedding_cache': embedding_cache,
        'cache_stats': cache_stats,
        'pooling_type': 'avg',  # CIS数据使用简单平均
        'is_cis_data': True
    } 