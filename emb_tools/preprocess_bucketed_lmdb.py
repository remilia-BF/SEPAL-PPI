#!/usr/bin/env python3
"""
Advanced offline preprocessing script to create intelligently bucketed LMDB with optimizations

特殊处理说明：
- 当嵌入维度为1时（即每个蛋白质对应一个标量或单一向量），脚本会：
  1. 自动检测并使用合理的默认序列长度分布进行分桶
  2. 保持分桶前缀格式以适应后续分析流程
  3. 建议提供FASTA文件以获得更准确的序列长度信息
  4. 或者手动指定正确的嵌入维度（如果1不是真实维度）

使用建议：
- 对于维度1的嵌入：python preprocess_bucketed_lmdb.py --override-embedding-dim 1 --fasta-file your_sequences.fasta
- 如果没有FASTA文件：脚本会使用默认的合理长度分布
"""

import sys
import os
import argparse
import logging
import json
import hashlib
import time
import struct
import numpy as np
import torch
import lmdb
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

# Add src to Python path
# Assuming src is in the parent directory of the script's directory.
# Adjust if your project structure is different.
# sys.path.append(str(Path(__file__).parent.parent / "src"))
# from src.data_processing.dynamic_bucketing import create_dynamic_buckets
# Mocking the function since the original is not provided.
def create_dynamic_buckets(lengths, bucket_options, method, verbose):
    logger = logging.getLogger(__name__)
    if not lengths:
        logger.warning("No sequence lengths provided for bucketing. Using default buckets.")
        return [256, 512, 768, 1024]
    
    # This is a placeholder for a real bucketing algorithm.
    # It creates 'k' buckets with roughly equal numbers of items.
    k = min(bucket_options, key=lambda x: abs(x-16)) # Choose a reasonable default number of buckets
    
    sorted_lengths = sorted(lengths)
    num_lengths = len(sorted_lengths)
    
    boundaries = []
    for i in range(1, k):
        index = int(num_lengths * i / k)
        boundaries.append(sorted_lengths[index])
        
    boundaries = sorted(list(set(boundaries))) # Remove duplicates
    
    if verbose:
        logger.info(f"[Mock] Selected {len(boundaries)+1} buckets with boundaries: {boundaries}")
    return boundaries


def setup_logging(level=logging.INFO):
    """Setup logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class IntelligentBucketedLMDBProcessor:
    """
    Advanced LMDB processor with intelligent bucketing, precision optimization, and streaming writes
    """
    
    def __init__(self, source_lmdb: str, output_dir: str, logger=None):
        """
        Initialize processor
        
        Args:
            source_lmdb (str): Path to source LMDB
            output_dir (str): Output directory for bucketed LMDBs
            logger: Logger instance
        """
        self.source_lmdb = Path(source_lmdb)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Metadata
        self.embedding_dim = None
        self.original_precision = None
        self.total_proteins = 0
        self.sequence_lengths = []
        
    def _detect_embedding_info(self, sample_size: int = 50) -> Tuple[int, str]:
        """
        Detect embedding dimension and precision from LMDB samples
        
        Args:
            sample_size (int): Number of samples to analyze
            
        Returns:
            Tuple[int, str]: (embedding_dim, precision)
        """
        self.logger.info(f"检测嵌入维度和精度 (采样 {sample_size} 个样本)...")
        
        env = lmdb.open(str(self.source_lmdb), readonly=True, lock=False)
        dimension_votes = {}
        precision_votes = {}
        samples_analyzed = 0
        
        with env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            
            while cursor.key() and samples_analyzed < sample_size:
                key = cursor.key()
                
                # Skip metadata keys
                if key.startswith(b'_'):
                    cursor.next()
                    continue
                
                data = cursor.value()
                
                try:
                    # Try raw binary first
                    arr = np.frombuffer(data, dtype=np.float32)
                    
                    # Test common dimensions
                    for dim in [2560, 1280, 768, 1024, 512, 256]:
                        if len(arr) % dim == 0:
                            seq_length = len(arr) // dim
                            if 10 <= seq_length <= 5000:  # Reasonable sequence length
                                dimension_votes[dim] = dimension_votes.get(dim, 0) + 1
                                precision_votes['fp32'] = precision_votes.get('fp32', 0) + 1
                                break
                    
                    samples_analyzed += 1
                    
                except Exception:
                    # Try pickle format
                    try:
                        import pickle
                        arr = pickle.loads(data)
                        if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                            dim = arr.shape[1]
                            dimension_votes[dim] = dimension_votes.get(dim, 0) + 1
                            
                            # Detect precision
                            if hasattr(arr, 'dtype'):
                                if 'float32' in str(arr.dtype):
                                    precision_votes['fp32'] = precision_votes.get('fp32', 0) + 1
                                elif 'float16' in str(arr.dtype):
                                    precision_votes['fp16'] = precision_votes.get('fp16', 0) + 1
                                elif 'bfloat16' in str(arr.dtype):
                                    precision_votes['bf16'] = precision_votes.get('bf16', 0) + 1
                            
                            samples_analyzed += 1
                    except Exception:
                        pass
                
                cursor.next()
        
        env.close()
        
        # Determine best dimension and precision
        if dimension_votes:
            best_dim = max(dimension_votes.items(), key=lambda x: x[1])[0]
            confidence = dimension_votes[best_dim] / samples_analyzed if samples_analyzed > 0 else 0
            self.logger.info(f"检测到嵌入维度: {best_dim} (置信度: {confidence:.1%})")
        else:
            best_dim = 1280  # Fallback
            self.logger.warning(f"无法检测嵌入维度，使用默认值: {best_dim}")
        
        if precision_votes:
            best_precision = max(precision_votes.items(), key=lambda x: x[1])[0]
            self.logger.info(f"检测到原始精度: {best_precision}")
        else:
            best_precision = 'fp32'  # Fallback
            self.logger.warning(f"无法检测精度，使用默认值: {best_precision}")
        
        return best_dim, best_precision
    
    def _detect_precision_only(self, sample_size: int = 10) -> str:
        """
        Only detect precision from LMDB samples (when dimension is manually specified)
        
        Args:
            sample_size (int): Number of samples to analyze
            
        Returns:
            str: Detected precision
        """
        self.logger.info(f"🔍 检测数据精度 (采样 {sample_size} 个样本，维度已手动指定为 {self.embedding_dim})...")
        
        env = lmdb.open(str(self.source_lmdb), readonly=True, lock=False)
        precision_votes = {}
        samples_analyzed = 0
        
        with env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            
            while cursor.key() and samples_analyzed < sample_size:
                key = cursor.key()
                
                # Skip metadata keys
                if key.startswith(b'_'):
                    cursor.next()
                    continue
                
                data = cursor.value()
                
                try:
                    # Try raw binary first - assume fp32 by default
                    arr = np.frombuffer(data, dtype=np.float32)
                    if self.embedding_dim and len(arr) % self.embedding_dim == 0:
                        precision_votes['fp32'] = precision_votes.get('fp32', 0) + 1
                        samples_analyzed += 1
                        
                except Exception:
                    # Try pickle format
                    try:
                        import pickle
                        arr = pickle.loads(data)
                        if hasattr(arr, 'dtype'):
                            if 'float32' in str(arr.dtype):
                                precision_votes['fp32'] = precision_votes.get('fp32', 0) + 1
                            elif 'float16' in str(arr.dtype):
                                precision_votes['fp16'] = precision_votes.get('fp16', 0) + 1
                            elif 'bfloat16' in str(arr.dtype):
                                precision_votes['bf16'] = precision_votes.get('bf16', 0) + 1
                        
                        samples_analyzed += 1
                    except Exception:
                        pass
                
                cursor.next()
        
        env.close()
        
        # Determine precision
        if precision_votes:
            best_precision = max(precision_votes.items(), key=lambda x: x[1])[0]
            self.logger.info(f"✅ 检测到原始精度: {best_precision}")
        else:
            best_precision = 'fp32'  # Fallback
            self.logger.warning(f"⚠️ 无法检测精度，使用默认值: {best_precision}")
        
        return best_precision
    
    def _collect_sequence_lengths(self, max_samples: int = 1000, fasta_file: Optional[str] = None, force_single_bucket: bool = False) -> List[int]:
        """
        Collect sequence lengths for intelligent bucketing
        
        Args:
            max_samples (int): Maximum samples to analyze. If 0, use all sequences.
            fasta_file (Optional[str]): If provided, read sequence lengths from a FASTA file
            force_single_bucket (bool): If True, force all embeddings into a single bucket (for CIS tasks)
            
        Returns:
            List[int]: Sequence lengths
        """
        mode_desc = "全部序列" if max_samples == 0 else f"最多 {max_samples} 个样本"
        
        # 如果强制单桶模式，直接返回统一长度
        if force_single_bucket:
            self.logger.info("🔒 强制单桶模式：所有嵌入将被分到同一个桶中")
            # 返回统一的长度值，用于创建单一桶
            return [1] * 100  # 返回100个长度为1的值，确保创建单一桶
        
        if fasta_file:
            self.logger.info(f"从FASTA文件收集序列长度信息 ({mode_desc})... 文件: {fasta_file}")
            lengths: List[int] = []
            samples_processed = 0
            path = Path(fasta_file)
            if not path.exists():
                self.logger.error(f"FASTA文件不存在: {path}")
                return lengths
            # 支持.gz压缩的FASTA
            open_fn = gzip.open if str(path).endswith('.gz') else open
            try:
                with open_fn(path, 'rt') as f:
                    current_len = 0
                    for line in f:
                        if not line:
                            continue
                        if line.startswith('>'):
                            if current_len > 0:
                                if 10 <= current_len <= 5000:
                                    lengths.append(current_len)
                                    samples_processed += 1
                                    if max_samples > 0 and samples_processed >= max_samples:
                                        break
                            current_len = 0
                        else:
                            current_len += len(line.strip())
                    # flush last record
                    if (max_samples == 0 or samples_processed < max_samples) and current_len > 0:
                        if 10 <= current_len <= 5000:
                            lengths.append(current_len)
                            samples_processed += 1
            except Exception as e:
                self.logger.error(f"解析FASTA文件失败: {e}")
                lengths = []
            
            self.logger.info(f"收集到 {len(lengths)} 个序列长度")
            if lengths:
                self.logger.info(f"长度范围: {min(lengths)} - {max(lengths)}")
                self.logger.info(f"平均长度: {np.mean(lengths):.1f}")
            return lengths
        else:
            self.logger.info(f"从LMDB收集序列长度信息 ({mode_desc})...")
            
            # 特殊处理：当嵌入维度为1时，使用原始嵌入的实际长度
            if self.embedding_dim == 1:
                self.logger.warning("⚠️ 检测到嵌入维度为1，将使用原始嵌入的实际长度进行分桶")
                self.logger.info("💡 建议：")
                self.logger.info("   - 对于CIS任务（所有嵌入长度相同），使用 --force-single-bucket 参数")
                self.logger.info("   - 为了更准确的分桶，请提供FASTA文件")
                self.logger.info("   - 或手动指定正确的嵌入维度（如果1不是真实维度）")
                
                # 当嵌入维度为1时，我们假设每个嵌入向量对应一个完整的蛋白质
                # 使用固定的合理长度范围来创建分桶
                env = lmdb.open(str(self.source_lmdb), readonly=True, lock=False)
                lengths = []
                samples_processed = 0
                
                try:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        cursor.first()
                        
                        while cursor.key() and (max_samples == 0 or samples_processed < max_samples):
                            key = cursor.key()
                            
                            # Skip metadata keys
                            if key.startswith(b'_'):
                                cursor.next()
                                continue
                            
                            data = cursor.value()
                            
                            try:
                                # 对于维度1的情况，每个嵌入代表整个蛋白质，使用实际的数据大小作为"长度"
                                arr = np.frombuffer(data, dtype=np.float32)
                                # 使用数据元素个数作为序列特征长度
                                feature_length = len(arr)
                                
                                # 为了保持分桶的合理性，我们将特征长度映射到合理的序列长度范围
                                # 这里假设特征长度与序列长度存在某种对应关系
                                if 1 <= feature_length <= 10000:  # 扩大接受范围
                                    # 如果特征长度就是1，我们使用一个默认的合理长度
                                    if feature_length == 1:
                                        # 对于长度为1的嵌入，我们假设对应中等长度的蛋白质序列
                                        mapped_length = 200  # 使用默认长度200作为代表
                                    else:
                                        mapped_length = feature_length
                                    
                                    lengths.append(mapped_length)
                                    samples_processed += 1
                            
                            except Exception:
                                try:
                                    import pickle
                                    arr = pickle.loads(data)
                                    if hasattr(arr, 'shape'):
                                        if len(arr.shape) == 1:
                                            # 一维数组，整个数组长度作为特征
                                            feature_length = arr.shape[0]
                                            mapped_length = feature_length if feature_length > 1 else 200
                                        else:
                                            # 多维数组，使用第一个维度
                                            mapped_length = arr.shape[0]
                                        
                                        if 1 <= mapped_length <= 10000:
                                            lengths.append(mapped_length)
                                            samples_processed += 1
                                except Exception:
                                    pass
                            
                            cursor.next()
                finally:
                    env.close()
                
                # 如果没有收集到任何长度，使用默认的合理分桶
                if not lengths:
                    self.logger.warning("⚠️ 无法从嵌入数据推断序列长度，使用默认长度分布")
                    # 创建一个合理的默认长度分布
                    import random
                    lengths = []
                    for _ in range(min(1000, max_samples) if max_samples > 0 else 1000):
                        # 模拟常见蛋白质长度分布
                        if random.random() < 0.3:
                            lengths.append(random.randint(50, 150))   # 短蛋白质
                        elif random.random() < 0.7:
                            lengths.append(random.randint(150, 400))  # 中等蛋白质
                        else:
                            lengths.append(random.randint(400, 800))  # 长蛋白质
                
            else:
                # 原有的正常处理逻辑（嵌入维度 > 1）
                env = lmdb.open(str(self.source_lmdb), readonly=True, lock=False)
                lengths = []
                samples_processed = 0
                try:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        cursor.first()
                        
                        while cursor.key() and (max_samples == 0 or samples_processed < max_samples):
                            key = cursor.key()
                            
                            # Skip metadata keys
                            if key.startswith(b'_'):
                                cursor.next()
                                continue
                            
                            data = cursor.value()
                            
                            try:
                                # Try to get sequence length
                                arr = np.frombuffer(data, dtype=np.float32)
                                if self.embedding_dim and len(arr) % self.embedding_dim == 0:
                                    seq_length = len(arr) // self.embedding_dim
                                    if 10 <= seq_length <= 5000:  # Filter reasonable lengths
                                        lengths.append(seq_length)
                                        samples_processed += 1
                            
                            except Exception:
                                try:
                                    import pickle
                                    arr = pickle.loads(data)
                                    if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                                        lengths.append(arr.shape[0])
                                        samples_processed += 1
                                except Exception:
                                    pass
                            
                            cursor.next()
                finally:
                    env.close()
            
            self.logger.info(f"收集到 {len(lengths)} 个序列长度")
            if lengths:
                self.logger.info(f"长度范围: {min(lengths)} - {max(lengths)}")
                self.logger.info(f"平均长度: {np.mean(lengths):.1f}")
            
            return lengths
    
    def _convert_precision(self, data: np.ndarray, target_precision: str) -> Tuple[bytes, Optional[float]]:
        """
        Convert embedding precision and return serialized data with shape preservation
        
        Args:
            data (np.ndarray): Original data
            target_precision (str): Target precision ('fp32', 'fp16', 'bf16', 'int8')
            
        Returns:
            Tuple[bytes, Optional[float]]: (serialized_data, scale_factor)
            scale_factor is only used for int8 quantization
        """
        if target_precision == 'fp32':
            converted = data.astype(np.float32)
            # 使用原始二进制格式存储，保持与训练代码的兼容性
            return converted.tobytes(), None
        elif target_precision == 'fp16':
            # 安全转换到fp16，避免溢出
            clipped_data = np.clip(data, -65504, 65504)
            converted = clipped_data.astype(np.float16)
            # 使用原始二进制格式存储
            return converted.tobytes(), None
        elif target_precision == 'bf16':
            # Convert to bfloat16 via torch，然后转换为可序列化格式
            tensor = torch.from_numpy(data.astype(np.float32)).to(torch.bfloat16)
            # 将bfloat16作为uint16保存（2字节表示）
            bf16_as_uint16 = tensor.view(torch.uint16).numpy()
            # 使用原始二进制格式存储
            return bf16_as_uint16.tobytes(), None
        elif target_precision == 'int8':
            # 改进的int8量化，按通道计算scale
            # 计算全局scale
            scale = np.max(np.abs(data))
            if scale == 0:
                scale = 1.0  # 避免除零
            
            # 量化到int8范围 [-127, 127] 保留-128作为特殊值
            quantized = np.round(data * 127.0 / scale).astype(np.int8)
            
            # 对于int8，我们需要保存scale信息，但使用更紧凑的格式
            # 将scale作为4字节float32前缀，然后是量化数据
            scale_bytes = struct.pack('f', scale)
            quantized_bytes = quantized.tobytes()
            return scale_bytes + quantized_bytes, scale
        else:
            raise ValueError(f"不支持的精度: {target_precision}")
    
    def _create_protein_index(self, bucket_data: Dict[int, Dict[str, bytes]]) -> Dict[str, bytes]:
        """
        Create protein index for fast lookups
        
        Args:
            bucket_data (Dict[int, Dict[str, bytes]]): All bucket data
            
        Returns:
            Dict[str, bytes]: Index metadata to store in LMDB
        """
        # 创建多级索引
        protein_to_bucket = {}      # protein_id -> bucket_id
        bucket_to_proteins = defaultdict(list)  # bucket_id -> [protein_ids]
        prefix_to_proteins = defaultdict(list)  # prefix -> [protein_ids]
        length_to_proteins = defaultdict(list)  # seq_length -> [protein_ids]
        
        for bucket_id, proteins in bucket_data.items():
            for key in proteins.keys():
                # 解析桶格式的键: bucket_{bucket_id}_len_{seq_length}_{protein_id}
                parts = key.split('_')
                if len(parts) >= 4 and parts[0] == 'bucket' and parts[2] == 'len':
                    try:
                        seq_length = int(parts[3])
                        protein_id = '_'.join(parts[4:])  # 处理包含下划线的蛋白质ID
                        
                        # 建立索引
                        protein_to_bucket[protein_id] = bucket_id
                        bucket_to_proteins[bucket_id].append(protein_id)
                        length_to_proteins[seq_length].append(protein_id)
                        
                        # 前缀索引（支持前缀搜索）
                        for i in range(1, min(len(protein_id) + 1, 8)):  # 最多7字符前缀
                            prefix = protein_id[:i]
                            if protein_id not in prefix_to_proteins[prefix]:
                                prefix_to_proteins[prefix].append(protein_id)
                    except (ValueError, IndexError):
                        continue
        
        # 序列化索引
        index_data = {
            '_protein_index': {
                'protein_to_bucket': protein_to_bucket,
                'bucket_to_proteins': dict(bucket_to_proteins),
                'prefix_to_proteins': {k: v for k, v in prefix_to_proteins.items() if len(v) < 1000},  # 限制前缀索引大小
                'length_to_proteins': dict(length_to_proteins),
                'total_proteins': len(protein_to_bucket),
                'index_version': '1.0.0'
            }
        }
        
        # 序列化为字节
        index_bytes = {}
        for key, value in index_data.items():
            index_bytes[key.encode()] = json.dumps(value, ensure_ascii=False).encode()
        
        return index_bytes
    
    def _create_metadata_keys(self, bucket_boundaries: List[int], 
                              target_precision: str) -> Dict[str, bytes]:
        """
        Create metadata keys for the bucketed LMDB
        
        Args:
            bucket_boundaries (List[int]): Bucket boundaries
            target_precision (str): Target precision
            
        Returns:
            Dict[str, bytes]: Metadata key-value pairs
        """
        # Calculate source hash
        source_hash = hashlib.md5(str(self.source_lmdb).encode()).hexdigest()[:16]
        
        metadata = {
            '_bucket_info': {
                'boundaries': bucket_boundaries,
                'num_buckets': len(bucket_boundaries) + 1,
                'strategy': 'intelligent_dynamic',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            '_precision_info': {
                'original_precision': self.original_precision,
                'target_precision': target_precision,
                'embedding_dim': self.embedding_dim
            },
            '_creation_info': {
                'source_lmdb': str(self.source_lmdb),
                'source_hash': source_hash,
                'processor_version': '2.0.0',
                'total_proteins': self.total_proteins
            }
        }
        
        # Serialize metadata
        metadata_bytes = {}
        for key, value in metadata.items():
            metadata_bytes[key.encode()] = json.dumps(value, ensure_ascii=False).encode()
        
        return metadata_bytes
    
    def _get_bucket_for_length(self, seq_length: int, boundaries: List[int]) -> int:
        """Get bucket ID for given sequence length"""
        for i, boundary in enumerate(boundaries):
            if seq_length <= boundary:
                return i
        return len(boundaries)  # Last bucket for sequences longer than all boundaries
    
    def _stream_write_single_lmdb(self, all_bucket_data: Dict[int, Dict[str, bytes]], 
                                  output_file: Path, metadata: Dict[str, bytes]) -> int:
        """
        Stream write all bucketed data to a single LMDB file with dynamic map size expansion
        
        Args:
            all_bucket_data (Dict[int, Dict[str, bytes]]): All bucket data combined
            output_file (Path): Output path for single LMDB file
            metadata (Dict[str, bytes]): Metadata to include
            
        Returns:
            int: Total number of proteins written
        """
        total_proteins = sum(len(bucket_data) for bucket_data in all_bucket_data.values())
        self.logger.info(f"流式写入单个LMDB文件到 {output_file} ({total_proteins} 个蛋白质)...")
        
        # Calculate initial map size based on all data
        total_size = 0
        for bucket_data in all_bucket_data.values():
            total_size += sum(len(v) for v in bucket_data.values())
        
        estimated_size = total_size * 3  # 3x buffer for safety
        estimated_size = max(estimated_size, 1024 * 1024 * 1024)  # At least 1GB
        
        # Check if file already exists and get current size
        if output_file.exists():
            try:
                current_size = sum(f.stat().st_size for f in output_file.rglob('*') if f.is_file())
                estimated_size = max(estimated_size, current_size * 2)
            except:
                pass
        
        written = 0
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                self.logger.info(f"尝试打开LMDB (重试 {retry+1}/{max_retries})，映射大小: {estimated_size / 1024 / 1024:.1f}MB")
                env = lmdb.open(str(output_file), map_size=estimated_size)
                
                try:
                    with env.begin(write=True) as txn:
                        # Write metadata first
                        for key, value in metadata.items():
                            txn.put(key, value)
                        self.logger.info("元数据写入完成")
                        
                        # Flatten all bucket data
                        all_items = []
                        for bucket_id in sorted(all_bucket_data.keys()):
                            bucket_data = all_bucket_data[bucket_id]
                            for protein_key, protein_data in bucket_data.items():
                                all_items.append((protein_key, protein_data))
                        
                        self.logger.info(f"开始写入 {len(all_items)} 个蛋白质...")
                        
                        # Write all protein data in batches
                        batch_size = 1000
                        for i in range(0, len(all_items), batch_size):
                            batch = all_items[i:i + batch_size]
                            
                            for protein_key, protein_data in batch:
                                txn.put(protein_key.encode(), protein_data)
                                written += 1
                            
                            if written > 0 and (i + len(batch)) % 5000 < batch_size :
                                self.logger.info(f"已写入 {written}/{len(all_items)} 个蛋白质 ({written/len(all_items)*100:.1f}%)")
                
                finally:
                    env.close()
                
                # If we get here, writing was successful
                break
                
            except lmdb.MapFullError:
                if retry < max_retries - 1:
                    estimated_size = int(estimated_size * 2)
                    self.logger.warning(f"LMDB映射空间不足，增加到 {estimated_size / 1024 / 1024:.1f}MB 并重试...")
                    if output_file.exists():
                        try: env.close()
                        except: pass
                else:
                    self.logger.error(f"LMDB映射空间不足，已重试 {max_retries} 次，放弃写入")
                    raise
            
            except Exception as e:
                self.logger.error(f"写入LMDB时发生错误: {e}")
                if retry == max_retries - 1:
                    raise
                else:
                    self.logger.warning(f"重试写入LMDB (重试 {retry+1}/{max_retries})")
        
        self.logger.info(f"单个LMDB文件写入完成: {written} 个蛋白质")
        return written
    
    def process_intelligent_bucketing(self, target_precision: str = 'fp32', 
                                      bucket_options: List[int] = [4, 8, 16, 32],
                                      chunk_size: int = 1000,
                                      max_length_samples: int = 1000,
                                      override_embedding_dim: Optional[int] = None,
                                      fasta_file: Optional[str] = None,
                                      force_single_bucket: bool = False) -> str:
        """
        Main processing method with intelligent bucketing and single LMDB write
        
        Args:
            target_precision (str): Target precision for embeddings
            bucket_options (List[int]): Options for number of buckets
            chunk_size (int): Processing chunk size for memory management
            max_length_samples (int): Maximum samples for length analysis. If 0, use all sequences.
            override_embedding_dim (Optional[int]): Manual override for embedding dimension
            fasta_file (Optional[str]): If provided, use FASTA sequence lengths for bucketing analysis
            force_single_bucket (bool): If True, force all embeddings into a single bucket (for CIS tasks)
            
        Returns:
            str: Path to the single output LMDB file
        """
        self.logger.info("=== 开始智能分桶预处理 (单个LMDB文件) ===")
        
        # Step 1: Detect or use override embedding info
        # ================== FIX STARTS HERE ==================
        if override_embedding_dim:
            self.logger.info(f"✅ 使用手动指定的嵌入维度: {override_embedding_dim}")
            self.embedding_dim = override_embedding_dim
            # 即使维度是手动的，仍然需要检测原始精度以确保转换逻辑被正确触发
            self.original_precision = self._detect_precision_only()
        else:
            self.logger.info("🔍 自动检测嵌入维度和精度...")
            self.embedding_dim, self.original_precision = self._detect_embedding_info()
        # =================== FIX ENDS HERE ===================
        
        # Step 2: Collect sequence lengths for intelligent bucketing
        self.sequence_lengths = self._collect_sequence_lengths(max_length_samples, fasta_file, force_single_bucket)
        
        # Step 3: Create optimal bucket boundaries using dynamic bucketing
        if force_single_bucket:
            self.logger.info("🔒 强制单桶模式：创建单一桶边界")
            bucket_boundaries = []  # 空边界列表表示只有一个桶
        else:
            self.logger.info("使用动态分桶算法确定最优边界...")
            bucket_boundaries = create_dynamic_buckets(
                lengths=self.sequence_lengths,
                bucket_options=bucket_options,
                method='balanced',
                verbose=True
            )
        
        self.logger.info(f"确定的分桶边界: {bucket_boundaries}")
        
        # Step 4: Create metadata
        metadata = self._create_metadata_keys(bucket_boundaries, target_precision)
        
        # Step 5: Process and bucket data with index generation
        all_bucket_data = defaultdict(dict)  # bucket_id -> {protein_key: protein_data}
        bucket_counts = defaultdict(int)
        
        self.logger.info("开始处理和分桶数据...")
        
        env = lmdb.open(str(self.source_lmdb), readonly=True, lock=False)
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                processed = 0
                
                while cursor.key():
                    key = cursor.key()
                    
                    if key.startswith(b'_'):
                        cursor.next()
                        continue
                    
                    data = cursor.value()
                    protein_id = key.decode('utf-8')
                    
                    try:
                        arr = None
                        seq_length = 0
                        
                        # 在强制单桶模式下，简化处理逻辑
                        if force_single_bucket:
                            try:
                                arr = np.frombuffer(data, dtype=np.float32)
                                # 对于单桶模式，我们不关心实际的序列长度，使用固定值
                                seq_length = 1  # 统一长度
                                
                                # 关键修复：确保CIS数据被正确reshape为(1, embedding_dim)格式
                                if len(arr) == self.embedding_dim:
                                    # 这是CIS数据：(embedding_dim,) -> (1, embedding_dim)
                                    arr = arr.reshape(1, self.embedding_dim)
                                    self.logger.debug(f"CIS数据reshape: {protein_id} -> (1, {self.embedding_dim})")
                                elif len(arr) % self.embedding_dim == 0:
                                    # 常规数据：按嵌入维度reshape
                                    actual_seq_length = len(arr) // self.embedding_dim
                                    arr = arr.reshape(actual_seq_length, self.embedding_dim)
                                    # 对于强制单桶，我们仍然使用seq_length=1作为标识
                                    self.logger.debug(f"常规数据reshape: {protein_id} -> ({actual_seq_length}, {self.embedding_dim})")
                                else:
                                    # 无法按嵌入维度整除，保持一维但记录警告
                                    self.logger.warning(f"无法按维度{self.embedding_dim}整除的数据: {protein_id}, 长度={len(arr)}")
                                    arr = arr.reshape(-1, 1)  # 保守处理
                                    
                            except Exception:
                                try:
                                    import pickle
                                    arr = pickle.loads(data)
                                    seq_length = 1  # 统一长度
                                    
                                    # 确保pickle数据也是2D格式
                                    if hasattr(arr, 'shape'):
                                        if len(arr.shape) == 0:  # 标量
                                            arr = np.array([[arr.item()]])
                                        elif len(arr.shape) == 1:  # 一维
                                            if len(arr) == self.embedding_dim:
                                                # CIS格式：(embedding_dim,) -> (1, embedding_dim)
                                                arr = arr.reshape(1, self.embedding_dim)
                                            else:
                                                # 其他一维数据
                                                arr = arr.reshape(-1, 1)
                                        # 多维保持不变，但确保是2D
                                        elif len(arr.shape) > 2:
                                            arr = arr.reshape(arr.shape[0], -1)
                                except Exception:
                                    self.logger.warning(f"无法解析蛋白质 {protein_id} 的嵌入数据")
                                    cursor.next()
                                    continue
                        else:
                            # 原有的复杂处理逻辑（非单桶模式）
                            # 特殊处理嵌入维度为1的情况
                            if self.embedding_dim == 1:
                                try:
                                    arr = np.frombuffer(data, dtype=np.float32)
                                    # 对于维度1的嵌入，每个蛋白质的嵌入就是一个标量或向量
                                    # 我们需要保持原始形状或者转换为统一格式
                                    if len(arr) == 1:
                                        # 单个标量值，保持为1D数组
                                        seq_length = 1
                                        arr = arr.reshape(1, 1)  # 转换为 (1, 1) 形状
                                    else:
                                        # 多个值，假设每个值代表一个位置的特征
                                        seq_length = len(arr)
                                        arr = arr.reshape(seq_length, 1)  # 转换为 (seq_length, 1) 形状
                                    
                                    # 对于维度1的情况，我们使用特殊的长度映射策略
                                    # 如果实际数据长度过大，映射到合理范围
                                    if seq_length > 5000:
                                        # 超长序列映射到最大桶
                                        mapped_seq_length = max(self.sequence_lengths) if self.sequence_lengths else 1000
                                    elif seq_length < 10:
                                        # 短序列映射到默认长度
                                        mapped_seq_length = 200
                                    else:
                                        mapped_seq_length = seq_length
                                    
                                    seq_length = mapped_seq_length
                                    
                                except Exception:
                                    try:
                                        import pickle
                                        arr = pickle.loads(data)
                                        if hasattr(arr, 'shape'):
                                            if len(arr.shape) == 0:  # 标量
                                                seq_length = 1
                                                arr = np.array([[arr.item()]])  # 转换为 (1, 1)
                                            elif len(arr.shape) == 1:  # 一维数组
                                                seq_length = len(arr)
                                                arr = arr.reshape(-1, 1)  # 转换为 (seq_length, 1)
                                            else:
                                                seq_length = arr.shape[0]
                                                # 保持现有形状，但确保第二维度为1
                                                if arr.shape[1] != 1:
                                                    self.logger.warning(f"蛋白质 {protein_id} 的嵌入维度不为1 ({arr.shape})，将使用第一列")
                                                    arr = arr[:, :1]
                                            
                                            # 应用相同的长度映射策略
                                            if seq_length > 5000:
                                                mapped_seq_length = max(self.sequence_lengths) if self.sequence_lengths else 1000
                                            elif seq_length < 10:
                                                mapped_seq_length = 200
                                            else:
                                                mapped_seq_length = seq_length
                                            
                                            seq_length = mapped_seq_length
                                    except Exception:
                                        self.logger.warning(f"无法解析蛋白质 {protein_id} 的嵌入数据")
                                        cursor.next()
                                        continue
                            else:
                                # 原有的处理逻辑（嵌入维度 > 1）
                                try:
                                    arr = np.frombuffer(data, dtype=np.float32)
                                    if self.embedding_dim and len(arr) % self.embedding_dim == 0:
                                        seq_length = len(arr) // self.embedding_dim
                                        arr = arr.reshape(seq_length, self.embedding_dim)
                                except:
                                    import pickle
                                    arr = pickle.loads(data)
                                    seq_length = arr.shape[0]
                        
                        # 过滤不合理的序列长度（单桶模式下跳过过滤）
                        if arr is None:
                            cursor.next()
                            continue
                        
                        if not force_single_bucket:
                            # 对于维度1的情况，我们放宽长度限制
                            if self.embedding_dim == 1:
                                if seq_length < 1 or seq_length > 10000:
                                    cursor.next()
                                    continue
                            else:
                                if seq_length < 10 or seq_length > 5000:
                                    cursor.next()
                                    continue
                        
                        # Convert precision if needed
                        if target_precision != self.original_precision:
                            converted_data, scale = self._convert_precision(arr, target_precision)
                        else:
                            # 保持二进制格式以兼容训练代码，数据加载器会自动reshape
                            converted_data = arr.astype(np.float32).tobytes()
                        
                        # Determine bucket
                        if force_single_bucket:
                            bucket_id = 0  # 所有数据都放在桶0中
                        else:
                            bucket_id = self._get_bucket_for_length(seq_length, bucket_boundaries)
                        
                        # Create new protein key with bucket info
                        if force_single_bucket:
                            # 单桶模式：使用简化的键格式，保持分桶前缀但所有都在桶0
                            new_key = f"bucket_0_len_1_{protein_id}"
                        elif self.embedding_dim == 1:
                            # 对于维度1的情况，使用实际的序列长度而不是桶的最大长度
                            new_key = f"bucket_{bucket_id}_len_{seq_length}_{protein_id}"
                        else:
                            bucket_max_len = bucket_boundaries[bucket_id] if bucket_id < len(bucket_boundaries) else bucket_boundaries[-1] * 2
                            new_key = f"bucket_{bucket_id}_len_{bucket_max_len}_{protein_id}"
                        
                        # Add to bucket data
                        all_bucket_data[bucket_id][new_key] = converted_data
                        bucket_counts[bucket_id] += 1
                        processed += 1
                        
                        if processed % 10000 == 0:
                            self.logger.info(f"已处理 {processed} 个蛋白质...")
                    
                    except Exception as e:
                        self.logger.warning(f"处理蛋白质 {protein_id} 失败: {e}")
                    
                    cursor.next()
        
        finally:
            env.close()
        
        # Step 6: Generate protein index
        self.logger.info("生成蛋白质索引...")
        protein_index = self._create_protein_index(all_bucket_data)
        
        # Step 7: Combine metadata and protein index
        combined_metadata = {**metadata, **protein_index}
        
        # Step 8: Write all data to single LMDB file
        output_file = self.output_dir / "bucketed_embeddings.lmdb"
        
        try:
            self._stream_write_single_lmdb(all_bucket_data, output_file, combined_metadata)
        except Exception as e:
            self.logger.error(f"写入单个LMDB文件失败: {e}")
            raise
        
        self.total_proteins = processed
        
        # Log results
        self.logger.info("=== 分桶处理完成 ===")
        self.logger.info(f"总处理蛋白质数: {processed}")
        self.logger.info(f"输出文件: {output_file}")
        self.logger.info(f"创建桶数: {len(all_bucket_data)}")
        
        for bucket_id in sorted(all_bucket_data.keys()):
            count = bucket_counts[bucket_id]
            max_len_str = str(bucket_boundaries[bucket_id]) if bucket_id < len(bucket_boundaries) else '∞'
            self.logger.info(f"  桶 {bucket_id} (≤{max_len_str}): {count} 个蛋白质")
        
        return str(output_file)


def main():
    """Main preprocessing function with intelligent bucketing"""
    parser = argparse.ArgumentParser(description='智能LMDB分桶预处理器')
    
    parser.add_argument('--source-lmdb', type=str, required=True,
                        help='源LMDB文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录路径')
    
    # Precision and dimension options
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16', 'int8'],
                        help='目标嵌入精度 (默认: fp32)')
    parser.add_argument('--override-embedding-dim', type=int, default=None,
                        help='手动覆盖嵌入维度，跳过自动检测 (默认: None)')

    # Intelligent bucketing options
    parser.add_argument('--bucket-options', type=int, nargs='+',
                        default=[4, 8, 16, 32],
                        help='分桶数量选项，算法会选择最优的 (默认: 4 8 16 32)')
    parser.add_argument('--max-length-samples', type=int, default=1000,
                        help='用于长度分析的最大样本数，设为0表示使用所有序列 (默认: 1000)')
    parser.add_argument('--fasta-file', type=str, default=None,
                        help='可选：提供FASTA序列文件以根据真实序列长度进行分桶')
    
    # Special handling for CIS tasks
    parser.add_argument('--force-single-bucket', action='store_true',
                        help='强制单桶模式：适用于CIS任务，所有嵌入都分到同一个桶中')

    # Logging option
    parser.add_argument('--verbose', action='store_true',
                        help='启用详细日志记录')
    
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)

    # Check source path
    if not Path(args.source_lmdb).exists():
        logger.error(f"源LMDB路径不存在: {args.source_lmdb}")
        sys.exit(1)

    # Initialize and run processor
    processor = IntelligentBucketedLMDBProcessor(
        source_lmdb=args.source_lmdb,
        output_dir=args.output_dir,
        logger=logger
    )

    try:
        output_path = processor.process_intelligent_bucketing(
            target_precision=args.precision,
            bucket_options=args.bucket_options,
            max_length_samples=args.max_length_samples,
            override_embedding_dim=args.override_embedding_dim,
            fasta_file=args.fasta_file,
            force_single_bucket=args.force_single_bucket
        )
        logger.info(f"🎉 预处理成功完成！输出文件位于: {output_path}")
    except Exception as e:
        logger.error(f"❌ 处理过程中发生严重错误: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()