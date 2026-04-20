"""
LMDB data loader for ESM embeddings
"""

import lmdb
import pickle
import torch
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ESMLMDBLoader:
    """
    Loader for ESM embeddings stored in LMDB format
    
    Loads sequence-level embeddings of shape [L, embedding_dim] where L is sequence length
    """
    
    def __init__(self, lmdb_path: str, readonly: bool = True):
        """
        Initialize LMDB loader
        
        Args:
            lmdb_path (str): Path to LMDB database
            readonly (bool): Whether to open in readonly mode
        """
        self.lmdb_path = Path(lmdb_path)
        self.readonly = readonly
        self.env = None
        self._open_connection()
        
        # Cache for frequently accessed embeddings
        self._cache = {}
        self._cache_size = 1000  # Maximum number of embeddings to cache
        
    def _open_connection(self):
        """Open LMDB connection"""
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB database not found: {self.lmdb_path}")
        
        try:
            self.env = lmdb.open(
                str(self.lmdb_path), 
                readonly=self.readonly,
                lock=False,
                readahead=False,
                meminit=False
            )
            logger.info(f"已打开LMDB数据库: {self.lmdb_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to open LMDB database: {e}")
    
    def close(self):
        """Close LMDB connection"""
        if self.env:
            self.env.close()
            self.env = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()
    
    def get_embedding(self, protein_id: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a single protein
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            torch.Tensor: Embedding of shape [L, embedding_dim] or None if not found
        """
        # Check cache first
        if protein_id in self._cache:
            return self._cache[protein_id]
        
        if self.env is None:
            self._open_connection()
        
        try:
            with self.env.begin() as txn:
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
                    logger.warning(f"Embedding not found for protein: {protein_id}")
                    return None
                
                # Load from raw binary data (assuming float32 format)
                try:
                    # First try as raw binary data
                    embedding = np.frombuffer(embedding_data, dtype=np.float32)
                    
                    # Try different embedding dimensions
                    embedding_reshaped = False
                    for embedding_dim in [2560, 1280, 768, 1024, 512, 256]:
                        if len(embedding) % embedding_dim == 0:
                            seq_length = len(embedding) // embedding_dim
                            embedding = embedding.reshape(seq_length, embedding_dim)
                            embedding_reshaped = True
                            break
                    
                    if not embedding_reshaped:
                        # Try as pickle if raw binary doesn't work
                        try:
                            embedding = pickle.loads(embedding_data)
                        except Exception:
                            logger.warning(f"Cannot process embedding for {protein_id}, size: {len(embedding)}")
                            return None
                            
                except Exception as e:
                    logger.error(f"Error processing embedding for {protein_id}: {e}")
                    return None
                
                # Convert to tensor if numpy array
                if isinstance(embedding, np.ndarray):
                    # Make a copy to ensure the array is writable
                    embedding = torch.from_numpy(embedding.copy()).float()
                elif not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                # Cache if space available
                if len(self._cache) < self._cache_size:
                    self._cache[protein_id] = embedding
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error loading embedding for {protein_id}: {e}")
            return None
    
    def get_batch_embeddings(self, protein_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for multiple proteins
        
        Args:
            protein_ids (List[str]): List of protein identifiers
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping protein IDs to embeddings
        """
        embeddings = {}
        
        for protein_id in protein_ids:
            embedding = self.get_embedding(protein_id)
            if embedding is not None:
                embeddings[protein_id] = embedding
        
        return embeddings
    
    def get_sequence_length(self, protein_id: str) -> Optional[int]:
        """
        Get sequence length for a protein
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            int: Sequence length or None if not found
        """
        embedding = self.get_embedding(protein_id)
        if embedding is not None:
            return embedding.shape[0]
        return None
    
    def get_embedding_dim(self, protein_id: str) -> Optional[int]:
        """
        Get embedding dimension
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            int: Embedding dimension or None if not found
        """
        embedding = self.get_embedding(protein_id)
        if embedding is not None:
            return embedding.shape[1]
        return None
    
    def list_proteins(self, limit: int = 10) -> List[str]:
        """
        List available protein IDs (for debugging)
        
        Args:
            limit (int): Maximum number of IDs to return
            
        Returns:
            List[str]: List of protein IDs
        """
        protein_ids = []
        
        if self.env is None:
            self._open_connection()
        
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for i, (key, _) in enumerate(cursor):
                    if i >= limit:
                        break
                    try:
                        protein_id = key.decode('utf-8')
                        protein_ids.append(protein_id)
                    except UnicodeDecodeError:
                        # Skip non-text keys
                        continue
        except Exception as e:
            logger.error(f"Error listing proteins: {e}")
        
        return protein_ids
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dict: Database statistics
        """
        stats = {
            'total_proteins': 0,
            'sample_embedding_shape': None,
            'cache_size': len(self._cache)
        }
        
        if self.env is None:
            self._open_connection()
        
        try:
            with self.env.begin() as txn:
                stats['total_proteins'] = txn.stat()['entries']
                
                # Get sample embedding shape
                cursor = txn.cursor()
                cursor.first()
                if cursor.key():
                    sample_data = cursor.value()
                    try:
                        # Try as raw binary first
                        sample_embedding = np.frombuffer(sample_data, dtype=np.float32)
                        
                        # Try different embedding dimensions
                        embedding_detected = False
                        for embedding_dim in [2560, 1280, 768, 1024, 512, 256]:
                            if len(sample_embedding) % embedding_dim == 0:
                                seq_length = len(sample_embedding) // embedding_dim
                                stats['sample_embedding_shape'] = (seq_length, embedding_dim)
                                embedding_detected = True
                                break
                        
                        if not embedding_detected:
                            # Try pickle
                            sample_embedding = pickle.loads(sample_data)
                            if isinstance(sample_embedding, np.ndarray):
                                stats['sample_embedding_shape'] = sample_embedding.shape
                            elif isinstance(sample_embedding, torch.Tensor):
                                stats['sample_embedding_shape'] = tuple(sample_embedding.shape)
                    except Exception:
                        stats['sample_embedding_shape'] = 'unknown'
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return stats


def load_lmdb_embeddings(lmdb_path: str) -> ESMLMDBLoader:
    """
    Factory function to create LMDB loader
    
    Args:
        lmdb_path (str): Path to LMDB database
        
    Returns:
        ESMLMDBLoader: Initialized loader
    """
    return ESMLMDBLoader(lmdb_path)


def test_lmdb_loader(lmdb_path: str, test_proteins: List[str] = None) -> bool:
    """
    Test LMDB loader functionality
    
    Args:
        lmdb_path (str): Path to LMDB database
        test_proteins (List[str]): Proteins to test (uses random sample if None)
        
    Returns:
        bool: True if test passes
    """
    try:
        loader = load_lmdb_embeddings(lmdb_path)
        
        # Get stats
        stats = loader.get_stats()
        logger.info(f"数据库统计: {stats}")
        
        # Get sample proteins if not provided
        if test_proteins is None:
            test_proteins = loader.list_proteins(5)
        
        logger.debug(f"Testing with proteins: {test_proteins}")
        
        # Test individual loading
        for protein_id in test_proteins:
            embedding = loader.get_embedding(protein_id)
            if embedding is not None:
                logger.debug(f"{protein_id}: shape {embedding.shape}")
            else:
                logger.warning(f"{protein_id}: not found")
        
        # Test batch loading
        batch_embeddings = loader.get_batch_embeddings(test_proteins)
        logger.debug(f"批量加载 {len(batch_embeddings)} 个嵌入")
        
        loader.close()
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False