"""
Hybrid bucketing manager combining online and offline strategies
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from .auto_bucketed_lmdb import AutoBucketedLMDBManager

logger = logging.getLogger(__name__)


class HybridBucketingManager:
    """
    Hybrid bucketing manager that combines the flexibility of online bucketing
    with the speed of offline caching
    """
    
    def __init__(self, lmdb_path: str, cache_dir: str = None):
        """
        Initialize hybrid bucketing manager
        
        Args:
            lmdb_path (str): Path to LMDB file
            cache_dir (str): Directory for bucketing cache (auto-generated if None)
        """
        self.lmdb_path = Path(lmdb_path)
        self.auto_manager = AutoBucketedLMDBManager(str(self.lmdb_path))
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.lmdb_path.parent / f"bucketing_cache_{self.lmdb_path.stem}"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"初始化混合分桶管理器: {self.lmdb_path}")
        logger.debug(f"缓存目录: {self.cache_dir}")
    
    def _get_data_fingerprint(self) -> str:
        """
        Generate a fingerprint for the current data state
        
        Returns:
            str: Data fingerprint for cache validation
        """
        # Use file modification time and size for fingerprint
        if not self.lmdb_path.exists():
            return "missing"
        
        stat = self.lmdb_path.stat()
        fingerprint_data = f"{stat.st_mtime}_{stat.st_size}"
        
        # Add LMDB content hash for more robust validation
        try:
            lmdb_hash = self.auto_manager._get_lmdb_hash(self.lmdb_path)
            fingerprint_data += f"_{lmdb_hash}"
        except Exception:
            pass
        
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
    
    def _get_cache_file_path(self, fingerprint: str) -> Path:
        """Get cache file path for given fingerprint"""
        return self.cache_dir / f"bucketing_{fingerprint}.json"
    
    def _save_bucketing_cache(self, cache_file: Path, bucketing_info: Dict):
        """Save bucketing information to cache"""
        try:
            cache_data = {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lmdb_path': str(self.lmdb_path),
                'fingerprint': self._get_data_fingerprint(),
                'bucketing_info': bucketing_info,
                'version': '1.0'
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info(f"分桶信息已缓存: {cache_file}")
            
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def _load_bucketing_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load bucketing information from cache"""
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache
            if cache_data.get('fingerprint') != self._get_data_fingerprint():
                logger.warning("缓存指纹不匹配，需要重新生成")
                return None
            
            bucketing_info = cache_data.get('bucketing_info')
            if not bucketing_info:
                logger.warning("缓存数据不完整")
                return None
            
            logger.info(f"成功加载缓存的分桶信息: {cache_file}")
            logger.info(f"缓存创建时间: {cache_data.get('created_at', 'Unknown')}")
            
            return bucketing_info
            
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None
    
    def _cleanup_old_caches(self, keep_recent: int = 3):
        """Clean up old cache files, keeping only recent ones"""
        try:
            cache_files = list(self.cache_dir.glob("bucketing_*.json"))
            if len(cache_files) <= keep_recent:
                return
            
            # Sort by modification time, keep recent ones
            cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_cache in cache_files[keep_recent:]:
                old_cache.unlink()
                logger.debug(f"清理旧缓存: {old_cache}")
                
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")
    
    def get_or_create_bucketing(self, force_refresh: bool = False) -> Dict[int, str]:
        """
        Get bucketing information using hybrid strategy
        
        Args:
            force_refresh (bool): Force refresh even if cache exists
            
        Returns:
            Dict[int, str]: Bucket paths mapping
        """
        fingerprint = self._get_data_fingerprint()
        cache_file = self._get_cache_file_path(fingerprint)
        
        # Try to load from cache first
        if not force_refresh and cache_file.exists():
            logger.info("尝试从缓存加载分桶信息...")
            cached_info = self._load_bucketing_cache(cache_file)
            
            if cached_info:
                # Validate that bucket files still exist
                bucket_paths = cached_info.get('bucket_paths', {})
                if self._validate_bucket_paths(bucket_paths):
                    logger.info("✅ 使用缓存的分桶信息")
                    return bucket_paths
                else:
                    logger.warning("缓存的分桶文件不存在，重新创建")
        
        # Create new bucketing information
        logger.info("创建新的分桶信息...")
        start_time = time.time()
        
        bucket_paths = self.auto_manager.get_or_create_bucketed_lmdb(force_recreate=force_refresh)
        
        creation_time = time.time() - start_time
        logger.info(f"分桶创建完成，耗时: {creation_time:.1f}秒")
        
        # Cache the result
        bucketing_info = {
            'bucket_paths': bucket_paths,
            'creation_time': creation_time,
            'lmdb_info': self.auto_manager.get_lmdb_info()
        }
        
        self._save_bucketing_cache(cache_file, bucketing_info)
        
        # Cleanup old caches
        self._cleanup_old_caches()
        
        return bucket_paths
    
    def _validate_bucket_paths(self, bucket_paths: Dict) -> bool:
        """Validate that bucket files exist"""
        if not bucket_paths:
            return False
        
        for bucket_id, path in bucket_paths.items():
            if not Path(path).exists():
                logger.warning(f"分桶文件不存在: {path}")
                return False
        
        return True
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache status"""
        fingerprint = self._get_data_fingerprint()
        cache_file = self._get_cache_file_path(fingerprint)
        
        cache_files = list(self.cache_dir.glob("bucketing_*.json"))
        
        info = {
            'cache_dir': str(self.cache_dir),
            'current_fingerprint': fingerprint,
            'current_cache_exists': cache_file.exists(),
            'total_cache_files': len(cache_files),
            'cache_files': [f.name for f in cache_files]
        }
        
        if cache_file.exists():
            try:
                stat = cache_file.stat()
                info['current_cache_size'] = stat.st_size
                info['current_cache_modified'] = time.ctime(stat.st_mtime)
            except Exception:
                pass
        
        return info
    
    def clear_cache(self, confirm: bool = False):
        """Clear all cache files"""
        if not confirm:
            logger.warning("需要确认才能清理缓存，使用 confirm=True")
            return
        
        try:
            cache_files = list(self.cache_dir.glob("bucketing_*.json"))
            
            for cache_file in cache_files:
                cache_file.unlink()
                logger.info(f"已删除缓存文件: {cache_file}")
            
            logger.info(f"缓存清理完成，删除了 {len(cache_files)} 个文件")
            
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def close(self):
        """Close the manager and cleanup resources"""
        if hasattr(self, 'auto_manager'):
            self.auto_manager.close()


def create_hybrid_bucketing_manager(lmdb_path: str, cache_dir: str = None) -> HybridBucketingManager:
    """
    Factory function to create hybrid bucketing manager
    
    Args:
        lmdb_path (str): Path to LMDB file
        cache_dir (str): Cache directory (optional)
        
    Returns:
        HybridBucketingManager: Initialized manager
    """
    return HybridBucketingManager(lmdb_path, cache_dir)