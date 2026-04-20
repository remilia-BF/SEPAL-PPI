"""
Sequence-level dataset loader with bucketing and caching support
"""

import torch
import torch.utils.data as Data
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from .lmdb_loader import ESMLMDBLoader
from .bucketing import SequenceBucketing, ProteinEmbeddingCache, create_bucketing_system
from .preprocessing import load_interaction_data
from .auto_bucketed_lmdb import AutoBucketedLMDBManager
from .hybrid_bucketing_manager import HybridBucketingManager
from .bucketed_lmdb_loader import BucketedLMDBLoader, create_bucketed_lmdb_loader
from ..utils.helpers import get_device

logger = logging.getLogger(__name__)


def _is_new_bucketed_format(embedding_file: str) -> bool:
    """
    Check if embedding file uses new bucketed format
    
    Args:
        embedding_file (str): Path to embedding file or directory
        
    Returns:
        bool: True if new bucketed format, False otherwise
    """
    try:
        import json
        
        file_path = Path(embedding_file)
        
        # If it's a directory, look for config file first (most reliable)
        if file_path.is_dir():
            config_file = file_path / "intelligent_bucketed_lmdb_config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        processor_version = config.get('processor_version', '1.0')
                        logger.debug(f"配置文件检测: processor_version={processor_version}")
                        if processor_version.startswith('2.'):
                            logger.info(f"✅ 检测到新版分桶格式 (v{processor_version})")
                            return True
                except Exception as e:
                    logger.debug(f"配置文件读取失败: {e}")
            
            # Fallback: check if there are bucket files
            bucket_files = list(file_path.glob("bucket_*.lmdb"))
            if bucket_files:
                logger.debug(f"发现 {len(bucket_files)} 个桶文件，检查LMDB元数据...")
                # Try to check first bucket file for metadata
                try:
                    import lmdb
                    env = lmdb.open(str(bucket_files[0]), readonly=True, lock=False)
                    try:
                        with env.begin() as txn:
                            creation_info = txn.get(b'_creation_info')
                            if creation_info:
                                creation_data = json.loads(creation_info.decode())
                                processor_version = creation_data.get('processor_version', '1.0')
                                return processor_version.startswith('2.')
                    finally:
                        env.close()
                except Exception as e:
                    logger.debug(f"LMDB元数据检查失败: {e}")
            
            return False
        
        else:
            # Single file path - check if it's a new format LMDB
            try:
                import lmdb
                env = lmdb.open(str(file_path), readonly=True, lock=False)
                try:
                    with env.begin() as txn:
                        creation_info = txn.get(b'_creation_info')
                        if creation_info:
                            creation_data = json.loads(creation_info.decode())
                            processor_version = creation_data.get('processor_version', '1.0')
                            return processor_version.startswith('2.')
                finally:
                    env.close()
            except Exception as e:
                logger.debug(f"单文件LMDB检查失败: {e}")
            
            return False
    
    except Exception as e:
        logger.debug(f"检测新分桶格式时出错: {e}")
        return False


class NewBucketedSequenceDataset(Data.Dataset):
    """
    Dataset using new bucketed LMDB loader with CPU padding and async GPU transfer
    """
    
    def __init__(self, interactions: List[Tuple[str, str, int]], 
                 bucketed_loader: BucketedLMDBLoader, all_proteins: List[str]):
        """
        Initialize dataset
        
        Args:
            interactions: List of (protein1, protein2, label) tuples
            bucketed_loader: Bucketed LMDB loader instance
            all_proteins: List of all unique proteins
        """
        self.interactions = interactions
        self.bucketed_loader = bucketed_loader
        self.all_proteins = all_proteins
        
        # Pre-load all protein embeddings using batched iterator
        logger.info(f"🔄 开始预加载 {len(all_proteins)} 个蛋白质嵌入...")
        self.protein_cache = {}
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm未安装，将使用基本进度显示")
        
        batch_count = 0
        processed_proteins = 0
        
        # 估算总批次数（用于进度条）
        estimated_batches = (len(all_proteins) + 31) // 32  # 假设批次大小32
        
        iterator = bucketed_loader.create_batched_iterator(all_proteins)
        if use_tqdm:
            iterator = tqdm(iterator, total=estimated_batches, desc="加载嵌入", unit="batch")
        
        for batch_data in iterator:
            if not batch_data or 'protein_ids' not in batch_data:
                continue
                
            protein_ids = batch_data['protein_ids']
            embeddings = batch_data['embeddings']
            attention_masks = batch_data['attention_masks']
            
            # Store individual proteins (remove padding for each protein)
            batch_size = len(protein_ids)
            for i, protein_id in enumerate(protein_ids):
                # Get the actual sequence length for this protein (remove padding)
                attention_mask = attention_masks[i]  # shape: [max_length]
                actual_length = attention_mask.sum().item()  # Count True values
                
                # Extract non-padded sequence
                protein_embedding = embeddings[i, :actual_length, :]  # shape: [actual_length, embedding_dim]
                protein_mask = attention_mask[:actual_length]  # shape: [actual_length]
                
                self.protein_cache[protein_id] = {
                    'embedding': protein_embedding,
                    'attention_mask': protein_mask
                }
            
            processed_proteins += batch_size
            batch_count += 1
            
            # 更新进度（如果没有tqdm）
            if not use_tqdm and batch_count % 5 == 0:
                logger.info(f"📦 已处理 {batch_count} 个批次，加载 {processed_proteins}/{len(all_proteins)} 个蛋白质")
        
        logger.info(f"✅ 蛋白质嵌入预加载完成: {len(self.protein_cache)}/{len(all_proteins)} 个")
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        protein1, protein2, label = self.interactions[idx]
        
        # Get cached embeddings
        protein1_data = self.protein_cache.get(protein1)
        protein2_data = self.protein_cache.get(protein2)
        
        if protein1_data is None or protein2_data is None:
            # Fallback: skip this sample or use dummy data
            logger.warning(f"Missing protein data: {protein1} or {protein2}")
            # Return dummy data to avoid breaking training
            dummy_embedding = torch.zeros(100, self.bucketed_loader.embedding_dim, 
                                        dtype=torch.float32, device=self.bucketed_loader.device)
            dummy_mask = torch.ones(100, dtype=torch.bool, device=self.bucketed_loader.device)
            
            return {
                'protein1_seq': dummy_embedding,
                'protein1_mask': dummy_mask,
                'protein2_seq': dummy_embedding,
                'protein2_mask': dummy_mask,
                'label': torch.tensor(label, dtype=torch.float32, device=self.bucketed_loader.device),
                'protein_ids': (protein1, protein2)
            }
        
        return {
            'protein1_seq': protein1_data['embedding'],
            'protein1_mask': protein1_data['attention_mask'],
            'protein2_seq': protein2_data['embedding'],
            'protein2_mask': protein2_data['attention_mask'],
            'label': torch.tensor(label, dtype=torch.float32, device=self.bucketed_loader.device),
            'protein_ids': (protein1, protein2)
        }


class SequenceInteractionDataset(Data.Dataset):
    """
    Dataset for protein-protein interactions using sequence-level embeddings
    """
    
    def __init__(self,
                 protein_pairs: List[Tuple[str, str]],
                 labels: np.ndarray,
                 embedding_loader: ESMLMDBLoader,
                 embedding_cache: ProteinEmbeddingCache,
                 bucketing_system: SequenceBucketing,
                 max_length: int = 1024,
                 preload_embeddings: bool = True):
        """
        Initialize sequence interaction dataset
        
        Args:
            protein_pairs (List[Tuple[str, str]]): List of protein ID pairs
            labels (np.ndarray): Interaction labels
            embedding_loader (ESMLMDBLoader): LMDB embedding loader
            embedding_cache (ProteinEmbeddingCache): GPU cache for embeddings
            bucketing_system (SequenceBucketing): Bucketing system
            max_length (int): Maximum sequence length
            preload_embeddings (bool): Whether to preload all embeddings
        """
        self.protein_pairs = protein_pairs
        self.labels = labels
        self.embedding_loader = embedding_loader
        self.embedding_cache = embedding_cache
        self.bucketing_system = bucketing_system
        self.max_length = max_length
        
        # Get unique proteins
        all_proteins = set()
        for p1, p2 in protein_pairs:
            all_proteins.add(p1)
            all_proteins.add(p2)
        
        logger.info(f"数据集初始化: {len(protein_pairs)} 对, {len(all_proteins)} 独特蛋白质")
        
        if preload_embeddings:
            # Smart preloading strategy: only preload if cache can fit most proteins
            cache_capacity = embedding_cache.max_cache_size
            if len(all_proteins) <= cache_capacity * 1.2:  # Allow 20% overflow
                logger.info(f"预加载所有 {len(all_proteins)} 个嵌入...")
                self.embedding_cache.preload_proteins(list(all_proteins), embedding_loader, verbose=True)
            else:
                logger.warning(f"蛋白质数量 ({len(all_proteins)}) 超过缓存容量 ({cache_capacity})")
                logger.info("使用懒加载策略")
        else:
            logger.info("使用懒加载 (嵌入按需加载)")
        
        logger.debug(f"数据集初始化完成: {len(protein_pairs)} 个蛋白质对")
    
    def __len__(self):
        return len(self.protein_pairs)
    
    def __getitem__(self, idx):
        """
        Get a single protein pair interaction sample
        
        Returns:
            Dict: Sample containing protein sequences, masks, and label
        """
        protein1_id, protein2_id = self.protein_pairs[idx]
        label = self.labels[idx]
        
        # Try to get from cache first
        protein1_data = self.embedding_cache.get_embedding(protein1_id)
        protein2_data = self.embedding_cache.get_embedding(protein2_id)
        
        # If not in cache, load from LMDB and cache
        if protein1_data is None:
            protein1_embedding = self.embedding_loader.get_embedding(protein1_id)
            if protein1_embedding is not None:
                self.embedding_cache.add_embedding(protein1_id, protein1_embedding)
                protein1_data = (protein1_embedding.to(self.embedding_cache.device), protein1_embedding.shape[0])
        
        if protein2_data is None:
            protein2_embedding = self.embedding_loader.get_embedding(protein2_id)
            if protein2_embedding is not None:
                self.embedding_cache.add_embedding(protein2_id, protein2_embedding)
                protein2_data = (protein2_embedding.to(self.embedding_cache.device), protein2_embedding.shape[0])
        
        if protein1_data is None or protein2_data is None:
            # Return dummy data if embeddings not found
            dummy_seq = torch.zeros(1, 1280, device=self.embedding_cache.device)
            dummy_mask = torch.ones(1, device=self.embedding_cache.device, dtype=torch.bool)
            return {
                'protein1_seq': dummy_seq,
                'protein1_mask': dummy_mask,
                'protein2_seq': dummy_seq,
                'protein2_mask': dummy_mask,
                'label': torch.tensor(label, dtype=torch.float32),
                'protein_ids': (protein1_id, protein2_id)
            }
        
        protein1_seq, protein1_length = protein1_data
        protein2_seq, protein2_length = protein2_data
        
        # Truncate if too long
        if protein1_length > self.max_length:
            protein1_seq = protein1_seq[:self.max_length]
            protein1_length = self.max_length
        
        if protein2_length > self.max_length:
            protein2_seq = protein2_seq[:self.max_length]
            protein2_length = self.max_length
        
        # Create attention masks
        protein1_mask = torch.ones(protein1_length, device=self.embedding_cache.device, dtype=torch.bool)
        protein2_mask = torch.ones(protein2_length, device=self.embedding_cache.device, dtype=torch.bool)
        
        return {
            'protein1_seq': protein1_seq,
            'protein1_mask': protein1_mask,
            'protein2_seq': protein2_seq,
            'protein2_mask': protein2_mask,
            'label': torch.tensor(label, dtype=torch.float32),
            'protein_ids': (protein1_id, protein2_id)
        }


class BucketedCollator:
    """
    Custom collator for bucketed batching with padding
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding to maximum length in batch
        
        Args:
            batch (List[Dict]): List of samples
            
        Returns:
            Dict[str, torch.Tensor]: Batched and padded data
        """
        batch_size = len(batch)
        
        # Get maximum lengths in batch
        max_len1 = max(sample['protein1_seq'].shape[0] for sample in batch)
        max_len2 = max(sample['protein2_seq'].shape[0] for sample in batch)
        
        embedding_dim = batch[0]['protein1_seq'].shape[1]
        
        # Initialize padded tensors
        protein1_seqs = torch.zeros(batch_size, max_len1, embedding_dim, device=self.device)
        protein1_masks = torch.zeros(batch_size, max_len1, device=self.device, dtype=torch.bool)
        protein2_seqs = torch.zeros(batch_size, max_len2, embedding_dim, device=self.device)
        protein2_masks = torch.zeros(batch_size, max_len2, device=self.device, dtype=torch.bool)
        labels = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        protein_ids = []
        
        # Fill padded tensors
        for i, sample in enumerate(batch):
            # Protein 1
            seq1_len = sample['protein1_seq'].shape[0]
            protein1_seqs[i, :seq1_len] = sample['protein1_seq']
            protein1_masks[i, :seq1_len] = sample['protein1_mask']
            
            # Protein 2
            seq2_len = sample['protein2_seq'].shape[0]
            protein2_seqs[i, :seq2_len] = sample['protein2_seq']
            protein2_masks[i, :seq2_len] = sample['protein2_mask']
            
            # Label
            labels[i] = sample['label']
            
            # Protein IDs
            protein_ids.append(sample['protein_ids'])
        
        return {
            'protein1_seq': protein1_seqs,
            'protein1_mask': protein1_masks,
            'protein2_seq': protein2_seqs,
            'protein2_mask': protein2_masks,
            'label': labels,
            'protein_ids': protein_ids
        }


def create_sequence_dataset_loaders(config: Dict) -> Dict[str, Data.DataLoader]:
    """
    Create sequence-level dataset loaders with intelligent bucketing support
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict[str, Data.DataLoader]: Dataset loaders
    """
    embedding_file = config['embedding_file']
    
    # Check if using new bucketed format
    is_new_format = _is_new_bucketed_format(embedding_file)
    logger.debug(f"新格式检测结果: {is_new_format} for {embedding_file}")
    
    # 临时强制使用新格式（如果目录存在配置文件）
    from pathlib import Path
    config_file = Path(embedding_file) / "intelligent_bucketed_lmdb_config.json"
    if config_file.exists():
        logger.info("✅ 发现智能分桶配置文件，强制使用新版加载器...")
        return _create_new_bucketed_loaders(config)
    elif is_new_format:
        logger.info("检测到新版智能分桶格式，使用优化加载器...")
        return _create_new_bucketed_loaders(config)
    else:
        logger.info("使用传统自动分桶模式...")
        return _create_legacy_bucketed_loaders(config)


def _create_new_bucketed_loaders(config: Dict) -> Dict[str, Data.DataLoader]:
    """Create loaders for new bucketed format"""
    logger.info("创建新版智能分桶数据集加载器...")
    
    embedding_file = config['embedding_file']
    device = get_device()
    
    # Initialize new bucketed loader
    bucketed_loader = create_bucketed_lmdb_loader(
        lmdb_dir=embedding_file,
        device=device,
        batch_size=config.get('batch_size', 32),
        max_workers=config.get('parallel_workers', 4)
    )
    
    loaders = {}
    
    try:
        # Load interaction data
        protein_pairs, labels = load_interaction_data(config['train_file'])
        
        # Convert to expected format: list of (protein1, protein2, label) tuples
        train_interactions = []
        for i in range(len(protein_pairs)):
            train_interactions.append((protein_pairs[i][0], protein_pairs[i][1], labels[i]))
        
        logger.info(f"加载训练数据: {len(train_interactions)} 个交互")
        
        # Get all unique proteins
        all_proteins = set()
        for p1, p2, _ in train_interactions:
            all_proteins.add(p1)
            all_proteins.add(p2)
        
        # Create training dataset with new loader
        train_dataset = NewBucketedSequenceDataset(
            interactions=train_interactions,
            bucketed_loader=bucketed_loader,
            all_proteins=list(all_proteins)
        )
        
        # Create DataLoader
        train_loader = Data.DataLoader(
            train_dataset,
            batch_size=1,  # Dataset handles internal batching
            shuffle=True,
            num_workers=0,  # Don't use additional workers
            collate_fn=lambda x: x[0]  # Pass through the pre-batched data
        )
        
        loaders['train'] = train_loader
        
        # Handle test files
        test_files = config.get('test_files', {})
        for test_name, test_file in test_files.items():
            test_protein_pairs, test_labels = load_interaction_data(test_file)
            
            # Convert to expected format
            test_interactions = []
            for i in range(len(test_protein_pairs)):
                test_interactions.append((test_protein_pairs[i][0], test_protein_pairs[i][1], test_labels[i]))
            
            logger.info(f"加载测试数据 {test_name}: {len(test_interactions)} 个交互")
            
            test_dataset = NewBucketedSequenceDataset(
                interactions=test_interactions,
                bucketed_loader=bucketed_loader,
                all_proteins=list(all_proteins)
            )
            
            loaders[f'{test_name}_dataset'] = test_dataset
        
        # Store bucketed loader for cleanup
        loaders['bucketed_loader'] = bucketed_loader
        
        logger.info(f"新版分桶加载器创建完成: {len(loaders)} 个组件")
        return loaders
        
    except Exception as e:
        logger.error(f"新版分桶加载器创建失败: {e}")
        bucketed_loader.close()
        raise


def _create_legacy_bucketed_loaders(config: Dict) -> Dict[str, Data.DataLoader]:
    """Create loaders using legacy auto-bucketing approach"""
    logger.info("创建传统自动分桶数据集加载器...")
    
    # Initialize hybrid bucketing manager (combines online + offline strategies)
    embedding_file = config['embedding_file']
    bucketing_strategy = config.get('bucketing_strategy', 'hybrid')
    
    if bucketing_strategy == 'hybrid':
        cache_dir = config.get('bucketing_cache_dir')
        hybrid_manager = HybridBucketingManager(embedding_file, cache_dir)
        
        # Get cache info
        cache_info = hybrid_manager.get_cache_info()
        logger.info(f"混合分桶管理器状态:")
        logger.info(f"  缓存目录: {cache_info['cache_dir']}")
        logger.info(f"  当前缓存: {'存在' if cache_info['current_cache_exists'] else '不存在'}")
        logger.info(f"  缓存文件数: {cache_info['total_cache_files']}")
        
        # For compatibility, also get LMDB info
        auto_lmdb_manager = hybrid_manager.auto_manager
        lmdb_info = auto_lmdb_manager.get_lmdb_info()
        logger.info(f"LMDB信息:")
        logger.info(f"  嵌入维度: {lmdb_info['embedding_dim']}")
        logger.info(f"  精度: {lmdb_info['precision']}")
    else:
        # Fallback to original auto manager
        auto_lmdb_manager = AutoBucketedLMDBManager(embedding_file)
        hybrid_manager = None
        
        lmdb_info = auto_lmdb_manager.get_lmdb_info()
        logger.info(f"LMDB状态检查:")
        logger.info(f"  原始文件: {lmdb_info['base_lmdb']}")
        logger.info(f"  嵌入维度: {lmdb_info['embedding_dim']}")
        logger.info(f"  精度: {lmdb_info['precision']}")
        logger.info(f"  已分桶: {lmdb_info['base_is_bucketed']}")
        logger.info(f"  现有桶数: {lmdb_info['existing_buckets']}")
    
    # Initialize traditional LMDB loader for compatibility
    embedding_loader = ESMLMDBLoader(embedding_file)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize cache (use streaming cache if enabled)
    cache_size = config.get('cache_size', 10000)
    use_streaming_cache = config.get('use_streaming_cache', False)
    
    if use_streaming_cache:
        from .streaming_gpu_cache import StreamingGPUCache
        max_gpu_memory = config.get('max_gpu_memory_mb', 4096)
        embedding_cache = StreamingGPUCache(device, max_gpu_memory_mb=max_gpu_memory)
        logger.info(f"使用流式GPU缓存，最大内存: {max_gpu_memory}MB")
    else:
        embedding_cache = ProteinEmbeddingCache(device, cache_size)
        logger.info(f"使用传统GPU缓存，容量: {cache_size}")
    
    loaders = {}
    
    # 首先，从所有数据集中收集所有蛋白质以便优化分桶和缓存
    logger.info("收集所有数据集中的蛋白质...")
    all_proteins = set()
    
    # Collect proteins from training data
    train_pairs, train_labels = None, None
    if 'train_file' in config:
        logger.info(f"加载训练数据: {config['train_file']}")
        train_pairs, train_labels = load_interaction_data(config['train_file'])
        for p1, p2 in train_pairs:
            all_proteins.add(p1)
            all_proteins.add(p2)
        logger.debug(f"训练集蛋白质数: {len(set(p for pair in train_pairs for p in pair))}")
    
    # Collect proteins from test datasets
    test_data = {}
    if 'test_files' in config:
        for test_name, test_file in config['test_files'].items():
            logger.info(f"加载 {test_name} 数据: {test_file}")
            test_pairs, test_labels = load_interaction_data(test_file)
            test_data[test_name] = (test_pairs, test_labels)
            
            test_proteins = set()
            for p1, p2 in test_pairs:
                all_proteins.add(p1)
                all_proteins.add(p2)
                test_proteins.add(p1)
                test_proteins.add(p2)
            logger.debug(f"{test_name}蛋白质数: {len(test_proteins)}")
    
    logger.info(f"总计蛋白质数: {len(all_proteins)}")
    
    # Create bucketing system using ALL proteins
    if train_pairs is not None:
        logger.info("基于所有蛋白质创建分桶系统...")
        fasta_file = config.get('fasta_file', 'dataset/S1/protein.fasta')
        bucketing_system = create_bucketing_system(
            embedding_loader, 
            list(all_proteins),  # Use ALL proteins for bucketing
            config.get('bucket_boundaries'),
            fasta_file=fasta_file,
            dynamic_bucketing=config.get('dynamic_bucketing', True),
            bucket_options=config.get('bucket_options', [4, 8, 16, 32]),
            verbose=True
        )
        
        # Create dataset with smart preloading for ALL proteins
        logger.info("创建训练数据集...")
        
        # Auto-load embeddings using optimal strategy
        if config.get('preload_embeddings', True):
            logger.info(f"自动加载 {len(all_proteins)} 个蛋白质嵌入...")
            
            # Use auto bucketed LMDB manager for optimal loading
            max_workers = config.get('parallel_workers', 4)
            
            try:
                # Load embeddings using hybrid or auto bucketed manager
                if hybrid_manager:
                    # Use hybrid manager for optimal performance
                    force_refresh = config.get('force_bucketing_refresh', False)
                    bucket_paths = hybrid_manager.get_or_create_bucketing(force_refresh=force_refresh)
                    
                    # Load using the bucket paths
                    auto_embeddings = auto_lmdb_manager.load_embeddings_auto(
                        list(all_proteins), device, max_workers=max_workers
                    )
                else:
                    # Fallback to auto manager
                    auto_embeddings = auto_lmdb_manager.load_embeddings_auto(
                        list(all_proteins), device, max_workers=max_workers
                    )
                
                # Add to cache
                if use_streaming_cache:
                    for protein_id, embedding in auto_embeddings.items():
                        embedding_cache.add_embedding_streaming(protein_id, embedding)
                else:
                    for protein_id, embedding in auto_embeddings.items():
                        embedding_cache.add_embedding(protein_id, embedding)
                
                logger.info(f"自动加载完成: {len(auto_embeddings)} 个嵌入已缓存")
                
            except Exception as e:
                logger.warning(f"自动加载失败，回退到传统方法: {e}")
                
                # Fallback to traditional loading
                cache_capacity = getattr(embedding_cache, 'max_cache_size', cache_size)
                if len(all_proteins) <= cache_capacity * 1.2:
                    use_parallel = config.get('use_parallel_loading', True)
                    
                    if hasattr(embedding_cache, 'preload_proteins'):
                        embedding_cache.preload_proteins(
                            list(all_proteins), 
                            embedding_loader, 
                            verbose=True,
                            use_parallel=use_parallel,
                            max_workers=max_workers
                        )
                    else:
                        # Streaming cache fallback
                        for protein_id in all_proteins:
                            embedding = embedding_loader.get_embedding(protein_id)
                            if embedding is not None:
                                embedding_cache.add_embedding_streaming(protein_id, embedding)
                    
                    logger.info("传统方法加载完成")
                else:
                    logger.warning(f"蛋白质数量 ({len(all_proteins)}) 过多，使用懒加载策略")
        
        train_dataset = SequenceInteractionDataset(
            train_pairs, train_labels, embedding_loader, 
            embedding_cache, bucketing_system,
            config.get('max_length', 1024),
            preload_embeddings=False  # Already pre-loaded above
        )
        
        # Create collator
        collator = BucketedCollator(device)
        
        # Create data loader
        loaders['train'] = Data.DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collator,
            num_workers=0,  # Must be 0 for GPU tensors
            pin_memory=False
        )
    
    # Create test datasets using pre-loaded data and shared cache
    if test_data:
        for test_name, (test_pairs, test_labels) in test_data.items():
            logger.info(f"创建 {test_name} 测试数据集...")
            
            # Use same bucketing system as training, NO additional preloading (already done)
            test_dataset = SequenceInteractionDataset(
                test_pairs, test_labels, embedding_loader,
                embedding_cache, bucketing_system,
                config.get('max_length', 1024),
                preload_embeddings=False  # Don't preload again - already cached
            )
            
            # Store dataset (not DataLoader) for evaluation
            loaders[f'{test_name}_dataset'] = test_dataset
    
    # Store components for access
    loaders['embedding_loader'] = embedding_loader
    loaders['embedding_cache'] = embedding_cache
    loaders['bucketing_system'] = bucketing_system
    
    # Store managers for cleanup
    if hybrid_manager:
        loaders['hybrid_manager'] = hybrid_manager
        loaders['auto_lmdb_manager'] = auto_lmdb_manager
    else:
        loaders['auto_lmdb_manager'] = auto_lmdb_manager
    
    # Log final statistics
    total_datasets = len([k for k in loaders.keys() if 'dataset' in k or 'train' in k])
    logger.info(f"已创建 {total_datasets} 个数据集")
    
    if 'embedding_cache' in loaders:
        if hasattr(loaders['embedding_cache'], 'get_stats'):
            cache_stats = loaders['embedding_cache'].get_stats()
            if 'memory_usage_mb' in cache_stats:
                logger.info(f"缓存状态: {cache_stats['cache_size']} 蛋白质, {cache_stats['memory_usage_mb']:.1f}MB")
            else:
                logger.info(f"缓存状态: {cache_stats}")
        elif hasattr(loaders['embedding_cache'], 'get_memory_stats'):
            cache_stats = loaders['embedding_cache'].get_memory_stats()
            logger.info(f"流式缓存状态: {cache_stats['cache_size']} 蛋白质, {cache_stats['current_memory_mb']:.1f}MB")
    
    return loaders


def get_sequence_default_config():
    """
    Get default configuration for sequence-level data loading
    
    Returns:
        Dict: Default configuration
    """
    return {
        'embedding_file': 'emb/esm1b_S1_all.NOcls_NOeos.lmdb',
        'fasta_file': 'dataset/S1/protein.fasta',
        'train_file': 'dataset/S1/c1Train.txt',
        'test_files': {
            'c2': 'dataset/S1/c2Validation.txt',
            'c3': 'dataset/S1/c3Test.txt'
        },
        'batch_size': 32,
        'max_length': 1024,
        'cache_size': 10000,
        'dynamic_bucketing': True,
        'bucket_options': [4, 8, 16, 32],
        'bucket_boundaries': None,  # Will be auto-generated if dynamic_bucketing=True
        'preload_embeddings': True,
        # Auto-bucketing and parallel loading (simplified config)
        'use_parallel_loading': True,  # Enable parallel LMDB loading
        'parallel_workers': 4,  # Number of worker threads for parallel loading
        
        # Bucketing strategy
        'bucketing_strategy': 'hybrid',  # 'online', 'offline', 'hybrid'
        'bucketing_cache_dir': None,  # Auto-generated if None
        'force_bucketing_refresh': False,  # Force refresh cached bucketing
        
        # Advanced caching options
        'use_streaming_cache': False,  # Enable streaming GPU cache for memory efficiency
        'max_gpu_memory_mb': 4096,  # Maximum GPU memory for streaming cache (MB)
        'num_workers': 0
    }