"""
Automatic bucketed LMDB management with intelligent detection and creation
"""

import lmdb
import torch
import numpy as np
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AutoBucketedLMDBManager:
    """
    Automatic LMDB management with intelligent bucketing detection and creation
    """
    
    # Special keys for metadata (prefixed with _ to ensure they come first)
    BUCKET_INFO_KEY = b'_bucket_info'
    PRECISION_INFO_KEY = b'_precision_info'
    CREATION_INFO_KEY = b'_creation_info'
    
    def __init__(self, base_lmdb_path: str, bucketed_dir: str = None):
        """
        Initialize auto bucketed LMDB manager
        
        Args:
            base_lmdb_path (str): Path to base/original LMDB
            bucketed_dir (str): Directory for bucketed LMDB files (auto-generated if None)
        """
        self.base_lmdb_path = Path(base_lmdb_path)
        
        if bucketed_dir is None:
            # Auto-generate bucketed directory based on original LMDB path
            bucketed_dir = self.base_lmdb_path.parent / f"bucketed_{self.base_lmdb_path.stem}"
        
        self.bucketed_dir = Path(bucketed_dir)
        self.bucketed_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for LMDB environments
        self._bucket_envs = {}
        self._base_env = None
        
        logger.info(f"初始化自动分桶LMDB管理器")
        logger.info(f"  原始LMDB: {self.base_lmdb_path}")
        logger.info(f"  分桶目录: {self.bucketed_dir}")
    
    def _get_lmdb_hash(self, lmdb_path: Path) -> str:
        """
        Get hash of LMDB file for version tracking
        
        Args:
            lmdb_path (Path): Path to LMDB
            
        Returns:
            str: MD5 hash of LMDB data file
        """
        data_file = lmdb_path / "data.mdb"
        if not data_file.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(data_file, "rb") as f:
            # Read first 1MB for hash (enough to detect changes)
            chunk = f.read(1024 * 1024)
            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()[:16]  # First 16 chars
    
    def _detect_embedding_dimension(self, lmdb_path: Path, sample_size: int = 10) -> int:
        """
        Intelligently detect embedding dimension using statistical analysis
        
        Args:
            lmdb_path (Path): Path to LMDB
            sample_size (int): Number of samples to analyze
            
        Returns:
            int: Most likely embedding dimension
        """
        try:
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            
            # Common embedding dimensions with their typical sources
            common_dimensions = {
                2560: 'ESM2-t36-3B',
                1280: 'ESM-1b', 
                768: 'ProtBERT/ESM-small',
                1024: 'ESM-medium',
                512: 'Custom/Small',
                256: 'Lightweight'
            }
            
            dimension_votes = {}
            samples_analyzed = 0
            
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                # Skip metadata keys and analyze multiple samples
                while cursor.key() and samples_analyzed < sample_size:
                    if cursor.key().startswith(b'_'):
                        cursor.next()
                        continue
                    
                    data = cursor.value()
                    detected_dim = None
                    
                    try:
                        # Try as raw binary first
                        arr = np.frombuffer(data, dtype=np.float32)
                        
                        # Test each dimension and track success
                        for dim in common_dimensions.keys():
                            if len(arr) % dim == 0:
                                seq_length = len(arr) // dim
                                # Reasonable sequence length check (10-5000 amino acids)
                                if 10 <= seq_length <= 5000:
                                    detected_dim = dim
                                    break
                        
                        if detected_dim is None:
                            # Try pickle format
                            import pickle
                            try:
                                arr = pickle.loads(data)
                                if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                                    detected_dim = arr.shape[1]
                            except:
                                pass
                        
                        if detected_dim:
                            dimension_votes[detected_dim] = dimension_votes.get(detected_dim, 0) + 1
                            samples_analyzed += 1
                    
                    except Exception:
                        pass
                    
                    cursor.next()
            
            env.close()
            
            if dimension_votes:
                # Find most voted dimension
                best_dim = max(dimension_votes.items(), key=lambda x: x[1])
                detected_dimension = best_dim[0]
                confidence = best_dim[1] / samples_analyzed
                
                model_name = common_dimensions.get(detected_dimension, 'Unknown')
                logger.info(f"检测到嵌入维度: {detected_dimension} ({model_name})")
                logger.info(f"检测置信度: {confidence:.1%} ({best_dim[1]}/{samples_analyzed} 样本)")
                
                if confidence < 0.8:
                    logger.warning(f"检测置信度较低 ({confidence:.1%})，请验证结果")
                
                return detected_dimension
            
        except Exception as e:
            logger.warning(f"智能嵌入维度检测失败: {e}")
        
        # Fallback: try simple detection on first sample
        try:
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                while cursor.key() and cursor.key().startswith(b'_'):
                    cursor.next()
                
                if cursor.key():
                    data = cursor.value()
                    arr = np.frombuffer(data, dtype=np.float32)
                    
                    # Try most common dimensions first
                    for dim in [2560, 1280, 768, 1024]:
                        if len(arr) % dim == 0:
                            env.close()
                            logger.warning(f"回退检测到嵌入维度: {dim}")
                            return dim
            env.close()
        except:
            pass
        
        # Final fallback
        logger.warning("无法检测嵌入维度，使用默认值1280")
        return 1280
    
    def _detect_lmdb_precision(self, lmdb_path: Path) -> str:
        """
        Detect precision of embeddings in LMDB
        
        Args:
            lmdb_path (Path): Path to LMDB
            
        Returns:
            str: Precision type ('fp32', 'fp16', 'bf16', etc.)
        """
        try:
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                
                # Skip metadata keys
                while cursor.key() and cursor.key().startswith(b'_'):
                    cursor.next()
                
                if cursor.key():
                    data = cursor.value()
                    
                    # Try to detect data type
                    try:
                        # Try as raw binary
                        arr = np.frombuffer(data, dtype=np.float32)
                        if len(arr) > 0:
                            return 'fp32'
                    except:
                        pass
                    
                    try:
                        # Try pickle
                        import pickle
                        arr = pickle.loads(data)
                        if hasattr(arr, 'dtype'):
                            dtype_str = str(arr.dtype)
                            if 'float32' in dtype_str:
                                return 'fp32'
                            elif 'float16' in dtype_str:
                                return 'fp16'
                            elif 'bfloat16' in dtype_str:
                                return 'bf16'
                    except:
                        pass
            
            env.close()
            
        except Exception as e:
            logger.warning(f"检测精度失败: {e}")
        
        return 'unknown'
    
    def _is_bucketed_lmdb(self, lmdb_path: Path) -> Tuple[bool, Optional[Dict]]:
        """
        Check if LMDB is already bucketed by looking for metadata keys
        
        Args:
            lmdb_path (Path): Path to LMDB
            
        Returns:
            Tuple[bool, Optional[Dict]]: (is_bucketed, bucket_info)
        """
        if not lmdb_path.exists():
            return False, None
        
        try:
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                bucket_info_data = txn.get(self.BUCKET_INFO_KEY)
                
                if bucket_info_data:
                    bucket_info = json.loads(bucket_info_data.decode('utf-8'))
                    env.close()
                    return True, bucket_info
            
            env.close()
            
        except Exception as e:
            logger.warning(f"检查分桶状态失败: {e}")
        
        return False, None
    
    def _get_optimal_bucket_boundaries(self, protein_ids: List[str], 
                                     bucket_options: List[int] = [4, 8, 16, 32]) -> List[int]:
        """
        Determine optimal bucket boundaries using advanced dynamic bucketing algorithm
        
        Args:
            protein_ids (List[str]): List of protein IDs
            bucket_options (List[int]): Possible bucket counts to choose from
            
        Returns:
            List[int]: Optimal bucket boundaries
        """
        logger.info("使用动态分桶算法分析序列长度分布...")
        
        # Collect sequence lengths (sample if too many)
        sample_size = min(5000, len(protein_ids))  # Sample up to 5000 for analysis
        sample_ids = protein_ids[:sample_size] if len(protein_ids) <= sample_size else \
                    np.random.choice(protein_ids, sample_size, replace=False).tolist()
        
        lengths = []
        env = lmdb.open(str(self.base_lmdb_path), readonly=True, lock=False)
        
        try:
            with env.begin() as txn:
                progress_desc = f"Analyzing {sample_size} protein lengths"
                for protein_id in tqdm(sample_ids, desc=progress_desc):
                    key = protein_id.encode()
                    data = txn.get(key)
                    
                    if data:
                        length = self._get_embedding_length_from_data(data)
                        if length:
                            lengths.append(length)
        finally:
            env.close()
        
        if not lengths:
            logger.warning("无法获取序列长度，使用默认分桶策略")
            return [64, 128, 256, 512, 1024, 2048]
        
        # Use the existing dynamic bucketing implementation
        from .dynamic_bucketing import create_dynamic_buckets
        
        boundaries = create_dynamic_buckets(
            lengths=lengths,
            bucket_options=bucket_options,
            method='balanced',  # Use balanced method for better distribution
            verbose=True
        )
        
        logger.info(f"动态分桶算法确定的最优边界: {boundaries}")
        return boundaries
    

    
    def _get_embedding_length_from_data(self, data: bytes) -> Optional[int]:
        """Get embedding length from raw data"""
        try:
            # Try raw binary first
            arr = np.frombuffer(data, dtype=np.float32)
            
            # Try different embedding dimensions in order of likelihood
            for dim in [2560, 1280, 768, 1024, 512, 256]:  # Common embedding dimensions
                if len(arr) % dim == 0:
                    return len(arr) // dim
            
            # Try pickle
            import pickle
            arr = pickle.loads(data)
            if hasattr(arr, 'shape'):
                return arr.shape[0]
                
        except Exception:
            pass
        
        return None
    
    def _create_bucketed_lmdb(self, bucket_boundaries: List[int]) -> Dict[int, str]:
        """
        Create bucketed LMDB files with metadata
        
        Args:
            bucket_boundaries (List[int]): Bucket boundaries
            
        Returns:
            Dict[int, str]: Mapping of bucket_id to LMDB path
        """
        logger.info(f"创建分桶LMDB，边界: {bucket_boundaries}")
        
        # Create bucket LMDB environments
        bucket_paths = {}
        bucket_envs = {}
        
        try:
            for i, boundary in enumerate(bucket_boundaries):
                bucket_path = self.bucketed_dir / f"bucket_{i}_max{boundary}.lmdb"
                bucket_paths[i] = str(bucket_path)
                
                bucket_envs[i] = lmdb.open(
                    str(bucket_path),
                    readonly=False,
                    map_size=10 * 1024 * 1024 * 1024,  # 10GB
                    max_dbs=0
                )
            
            # Process source LMDB
            self._distribute_to_buckets(bucket_envs, bucket_boundaries)
            
            # Add metadata to each bucket
            precision = self._detect_lmdb_precision(self.base_lmdb_path)
            embedding_dim = self._detect_embedding_dimension(self.base_lmdb_path)
            lmdb_hash = self._get_lmdb_hash(self.base_lmdb_path)
            
            creation_info = {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_lmdb': str(self.base_lmdb_path),
                'source_hash': lmdb_hash,
                'bucket_boundaries': bucket_boundaries,
                'total_buckets': len(bucket_boundaries),
                'embedding_dim': embedding_dim
            }
            
            for bucket_id, env in bucket_envs.items():
                with env.begin(write=True) as txn:
                    # Bucket info
                    bucket_info = {
                        'bucket_id': bucket_id,
                        'max_length': bucket_boundaries[bucket_id],
                        'is_bucketed': True,
                        'embedding_dim': embedding_dim
                    }
                    txn.put(self.BUCKET_INFO_KEY, json.dumps(bucket_info).encode('utf-8'))
                    
                    # Precision info
                    precision_info = {
                        'precision': precision,
                        'embedding_dim': embedding_dim
                    }
                    txn.put(self.PRECISION_INFO_KEY, json.dumps(precision_info).encode('utf-8'))
                    
                    # Creation info
                    txn.put(self.CREATION_INFO_KEY, json.dumps(creation_info).encode('utf-8'))
            
            return bucket_paths
            
        finally:
            for env in bucket_envs.values():
                env.close()
    
    def _distribute_to_buckets(self, bucket_envs: Dict, bucket_boundaries: List[int]):
        """Distribute embeddings to appropriate buckets"""
        source_env = lmdb.open(str(self.base_lmdb_path), readonly=True, lock=False)
        
        try:
            with source_env.begin() as source_txn:
                total_entries = source_txn.stat()['entries']
                
                # Initialize bucket transactions
                bucket_txns = {}
                for bucket_id, env in bucket_envs.items():
                    bucket_txns[bucket_id] = env.begin(write=True)
                
                bucket_counts = {i: 0 for i in bucket_envs.keys()}
                processed = 0
                chunk_size = 1000
                
                progress_bar = tqdm(total=total_entries, desc="Distributing to buckets")
                
                try:
                    cursor = source_txn.cursor()
                    for key, value in cursor:
                        # Skip if key starts with _ (metadata)
                        if key.startswith(b'_'):
                            continue
                        
                        # Get embedding length
                        length = self._get_embedding_length_from_data(value)
                        if length is None:
                            continue
                        
                        # Determine bucket
                        bucket_id = self._get_bucket_id(length, bucket_boundaries)
                        
                        # Store in bucket
                        bucket_txns[bucket_id].put(key, value)
                        bucket_counts[bucket_id] += 1
                        processed += 1
                        
                        # Commit in chunks
                        if processed % chunk_size == 0:
                            for txn in bucket_txns.values():
                                txn.commit()
                            
                            # Restart transactions
                            for bucket_id, env in bucket_envs.items():
                                bucket_txns[bucket_id] = env.begin(write=True)
                        
                        progress_bar.update(1)
                    
                    # Final commit
                    for txn in bucket_txns.values():
                        txn.commit()
                    
                    progress_bar.close()
                    
                    # Log statistics
                    logger.info("分桶分布统计:")
                    for bucket_id, count in bucket_counts.items():
                        if count > 0:
                            max_len = bucket_boundaries[bucket_id]
                            logger.info(f"  桶 {bucket_id} (≤{max_len}): {count} 个嵌入")
                
                except Exception as e:
                    for txn in bucket_txns.values():
                        txn.abort()
                    raise e
                
        finally:
            source_env.close()
    
    def _get_bucket_id(self, length: int, boundaries: List[int]) -> int:
        """Get bucket ID for sequence length"""
        for i, boundary in enumerate(boundaries):
            if length <= boundary:
                return i
        return len(boundaries) - 1
    
    def get_or_create_bucketed_lmdb(self, force_recreate: bool = False) -> Dict[int, str]:
        """
        Get existing bucketed LMDB or create new one if needed
        
        Args:
            force_recreate (bool): Force recreation even if bucketed LMDB exists
            
        Returns:
            Dict[int, str]: Mapping of bucket_id to LMDB path
        """
        # Check if base LMDB is already bucketed
        is_bucketed, bucket_info = self._is_bucketed_lmdb(self.base_lmdb_path)
        
        if is_bucketed and not force_recreate:
            logger.info("检测到原始LMDB已经分桶，直接使用")
            return {0: str(self.base_lmdb_path)}  # Single bucket
        
        # Look for existing bucketed LMDBs
        existing_buckets = {}
        source_hash = self._get_lmdb_hash(self.base_lmdb_path)
        
        if not force_recreate:
            for bucket_file in self.bucketed_dir.glob("bucket_*.lmdb"):
                try:
                    env = lmdb.open(str(bucket_file), readonly=True, lock=False)
                    with env.begin() as txn:
                        creation_data = txn.get(self.CREATION_INFO_KEY)
                        if creation_data:
                            creation_info = json.loads(creation_data.decode('utf-8'))
                            
                            # Check if this bucket matches our source LMDB
                            if creation_info.get('source_hash') == source_hash:
                                bucket_data = txn.get(self.BUCKET_INFO_KEY)
                                if bucket_data:
                                    bucket_info = json.loads(bucket_data.decode('utf-8'))
                                    bucket_id = bucket_info['bucket_id']
                                    existing_buckets[bucket_id] = str(bucket_file)
                    env.close()
                except Exception:
                    continue
        
        if existing_buckets and not force_recreate:
            logger.info(f"找到现有分桶LMDB: {len(existing_buckets)} 个桶")
            return existing_buckets
        
        # Need to create new bucketed LMDB
        logger.info("未找到匹配的分桶LMDB，开始创建...")
        
        # Get all protein IDs for analysis
        protein_ids = self._get_all_protein_ids()
        
        # Determine optimal bucket boundaries
        boundaries = self._get_optimal_bucket_boundaries(protein_ids)
        
        # Create bucketed LMDB
        bucket_paths = self._create_bucketed_lmdb(boundaries)
        
        logger.info(f"分桶LMDB创建完成: {len(bucket_paths)} 个桶")
        return bucket_paths
    
    def _get_all_protein_ids(self) -> List[str]:
        """Get all protein IDs from base LMDB"""
        protein_ids = []
        env = lmdb.open(str(self.base_lmdb_path), readonly=True, lock=False)
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    # Skip metadata keys
                    if not key.startswith(b'_'):
                        try:
                            protein_ids.append(key.decode('utf-8'))
                        except UnicodeDecodeError:
                            continue
        finally:
            env.close()
        
        return protein_ids
    
    def load_embeddings_auto(self, protein_ids: List[str], device: torch.device,
                           max_workers: int = 4) -> Dict[str, torch.Tensor]:
        """
        Automatically load embeddings using optimal strategy
        
        Args:
            protein_ids (List[str]): Protein IDs to load
            device (torch.device): Target device
            max_workers (int): Number of worker threads
            
        Returns:
            Dict[str, torch.Tensor]: Loaded embeddings
        """
        # Get or create bucketed LMDB
        bucket_paths = self.get_or_create_bucketed_lmdb()
        
        # Load embeddings from buckets
        all_embeddings = {}
        
        if len(bucket_paths) == 1:
            # Single bucket (original LMDB is already bucketed)
            single_path = list(bucket_paths.values())[0]
            all_embeddings = self._load_from_single_lmdb(single_path, protein_ids, max_workers)
        else:
            # Multiple buckets
            from .streaming_gpu_cache import BucketedLMDBLoader
            loader = BucketedLMDBLoader(bucket_paths)
            
            try:
                for bucket_id in bucket_paths.keys():
                    bucket_embeddings = loader.load_bucket_parallel(
                        bucket_id, protein_ids, max_workers=max_workers
                    )
                    all_embeddings.update(bucket_embeddings)
            finally:
                loader.close()
        
        return all_embeddings
    
    def _load_from_single_lmdb(self, lmdb_path: str, protein_ids: List[str],
                              max_workers: int) -> Dict[str, torch.Tensor]:
        """Load embeddings from single LMDB file"""
        from .parallel_lmdb_loader import ParallelLMDBLoader
        
        loader = ParallelLMDBLoader(lmdb_path, max_workers=max_workers)
        return loader.load_embeddings_parallel(protein_ids, verbose=True)
    
    def get_lmdb_info(self) -> Dict[str, Any]:
        """Get information about LMDB status and configuration"""
        info = {
            'base_lmdb': str(self.base_lmdb_path),
            'bucketed_dir': str(self.bucketed_dir),
            'base_lmdb_exists': self.base_lmdb_path.exists(),
            'precision': self._detect_lmdb_precision(self.base_lmdb_path),
            'embedding_dim': self._detect_embedding_dimension(self.base_lmdb_path),
            'source_hash': self._get_lmdb_hash(self.base_lmdb_path)
        }
        
        # Check bucketing status
        is_bucketed, bucket_info = self._is_bucketed_lmdb(self.base_lmdb_path)
        info['base_is_bucketed'] = is_bucketed
        if bucket_info:
            info['base_bucket_info'] = bucket_info
        
        # Check for existing bucketed files
        bucket_files = list(self.bucketed_dir.glob("bucket_*.lmdb"))
        info['existing_buckets'] = len(bucket_files)
        
        return info
    
    def close(self):
        """Close all LMDB environments"""
        for env in self._bucket_envs.values():
            env.close()
        self._bucket_envs.clear()
        
        if self._base_env:
            self._base_env.close()
            self._base_env = None