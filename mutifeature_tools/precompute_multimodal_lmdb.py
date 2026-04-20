#!/usr/bin/env python3
"""
预计算多模态特征并存储到 LMDB
消除逐样本循环瓶颈，实现批量化GPU加速

用法:
    python mutifeature_tools/precompute_multimodal_lmdb.py \
        --config config/model_unit/preprocessing/feature_concat_S1.yaml \
        --output mutifeature/S1_precomputed.lmdb
"""

import argparse
import json
import logging
import lmdb
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFeaturePreprocessor:
    """多模态特征预处理器：JSON → LMDB"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_folder = self.config['data_processing']['all_feature_folder']
        self.feature_files = self.config['feature_files']
        
        # 只处理启用的特征
        self.enabled_features = {
            name: cfg for name, cfg in self.feature_files.items()
            if cfg.get('enabled', False)
        }
        
        logger.info(f"启用的特征: {list(self.enabled_features.keys())}")
        
        # 计算特征维度和顺序
        self.feature_order = sorted(self.enabled_features.keys())
        self.feature_dims = []
        self.total_dim = 0
        
        for name in self.feature_order:
            cfg = self.enabled_features[name]
            if cfg.get('feature_type') == 'embedding':
                # embedding特征存储ID，维度为1
                dim = 1
            else:
                dim = cfg.get('raw_dim', cfg.get('input_dim', cfg.get('feature_dim', 1)))
            self.feature_dims.append(dim)
            self.total_dim += dim
        
        logger.info(f"特征维度: {dict(zip(self.feature_order, self.feature_dims))}")
        logger.info(f"总维度: {self.total_dim}")
    
    def load_feature_data(self, feature_name: str) -> Dict[str, Any]:
        """加载单个特征文件"""
        cfg = self.enabled_features[feature_name]
        file_path = cfg.get('file_path')
        
        if not file_path:
            return {}
        
        full_path = Path(self.feature_folder) / file_path
        if not full_path.exists():
            logger.warning(f"特征文件不存在: {full_path}")
            return {}
        
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # 转换为 {protein_id: data} 格式
        feature_dict = {}
        for item in data:
            protein_id = item.get('protein_id') or item.get('Protein_ID')
            if protein_id:
                feature_dict[protein_id] = item
        
        logger.info(f"加载 {feature_name}: {len(feature_dict)} 条")
        return feature_dict
    
    def extract_feature_array(self, item: Dict, feature_name: str, config: Dict) -> np.ndarray:
        """从JSON项提取特征数组"""
        
        # embedding特征：只存储ID
        if config.get('feature_type') == 'embedding':
            if feature_name == 'protein_length':
                # 蛋白长度需要从序列长度计算
                return None  # 稍后处理
            
            # 其他embedding特征：提取ID序列
            data_field = feature_name.replace('_features', '').replace('_', '')
            if feature_name == 's1ssttoken':
                data_field = 'prosst_structure_tokens'
            elif feature_name == 'hmm_pfam_id':
                data_field = 'hmm_pfam_id'
            
            id_str = item.get(data_field, '')
            if not id_str:
                return np.array([], dtype=np.int32)
            
            # embedding特征：只存储ID
            ids = [int(x) for x in id_str.split(',') if x.strip()]
            return np.array(ids, dtype=np.float32).reshape(-1, 1)  # 统一为float32
        
        # 普通特征：提取浮点数
        if feature_name == 'secondary_structure_features':
            # 独热编码特征
            ss_str = item.get('secondary_structure', '')
            if not ss_str:
                return np.array([], dtype=np.float32)
            
            # 转换为独热编码 [H, E, C]
            ss_map = {'H': [1, 0, 0], 'E': [0, 1, 0], 'C': [0, 0, 1]}
            one_hot = [ss_map.get(c, [0, 0, 0]) for c in ss_str]
            return np.array(one_hot, dtype=np.float32)
        
        # 其他数值特征
        data_field = feature_name.replace('_features', '').replace('_', '')
        if 'hydrophobicity' in feature_name:
            data_field = 'hydrophobicity'
        elif 'sasa' in feature_name:
            data_field = 'sasa'
        elif 'pqr' in feature_name:
            data_field = 'pqr'
        
        value_str = item.get(data_field, item.get(feature_name, ''))
        if not value_str:
            return np.array([], dtype=np.float32)
        
        values = [float(x) for x in str(value_str).split(',') if x.strip()]
        feature_dim = config.get('raw_dim', config.get('feature_dim', 1))
        
        if feature_dim == 1:
            arr = np.array(values, dtype=np.float32).reshape(-1, 1)
        else:
            usable = (len(values) // feature_dim) * feature_dim
            arr = np.array(values[:usable], dtype=np.float32).reshape(-1, feature_dim)
        
        return arr
    
    def get_all_protein_ids(self) -> List[str]:
        """获取所有蛋白ID（从第一个特征文件）"""
        first_feature = self.feature_order[0]
        data = self.load_feature_data(first_feature)
        return list(data.keys())
    
    def process_protein(self, protein_id: str, all_features: Dict[str, Dict]) -> Tuple[np.ndarray, Dict]:
        """处理单个蛋白的所有特征"""
        features_list = []
        metadata = {'protein_id': protein_id, 'feature_order': self.feature_order}
        
        for feature_name in self.feature_order:
            cfg = self.enabled_features[feature_name]
            
            if feature_name not in all_features:
                # 特征文件未加载
                dim = self.feature_dims[self.feature_order.index(feature_name)]
                arr = np.zeros((0, dim), dtype=np.float32)
            elif protein_id not in all_features[feature_name]:
                # 蛋白在该特征文件中缺失
                dim = self.feature_dims[self.feature_order.index(feature_name)]
                arr = np.zeros((0, dim), dtype=np.float32)
            else:
                item = all_features[feature_name][protein_id]
                arr = self.extract_feature_array(item, feature_name, cfg)
                
                if arr is None or len(arr) == 0:
                    dim = self.feature_dims[self.feature_order.index(feature_name)]
                    arr = np.zeros((0, dim), dtype=np.float32)
            
            features_list.append(arr)
        
        # 找到最大序列长度
        max_len = max((arr.shape[0] for arr in features_list if arr.shape[0] > 0), default=0)
        metadata['seq_len'] = max_len
        
        # Padding 所有特征到相同长度
        padded_features = []
        for idx, (arr, dim) in enumerate(zip(features_list, self.feature_dims)):
            feature_name = self.feature_order[idx]
            cfg = self.enabled_features[feature_name]
            is_embedding = cfg.get('feature_type') == 'embedding'
            
            if arr.shape[0] == 0:
                # 空特征：填充零
                dtype = np.int32 if is_embedding else np.float32
                padded = np.zeros((max_len, dim), dtype=dtype)
            elif arr.shape[0] < max_len:
                # Padding
                pad_len = max_len - arr.shape[0]
                padded = np.vstack([arr, np.zeros((pad_len, dim), dtype=arr.dtype)])
            else:
                # Truncate
                padded = arr[:max_len]
            
            # 统一转换为float32（包括embedding ID）
            padded_features.append(padded.astype(np.float32))
        
        # 拼接: [seq_len, total_dim]
        combined = np.concatenate(padded_features, axis=-1)
        return combined, metadata
    
    def build_lmdb(self, output_path: str, map_size: int = 50 * 1024**3):
        """构建LMDB数据库"""
        
        # 加载所有特征文件
        logger.info("加载所有特征文件...")
        all_features = {}
        for feature_name in tqdm(self.feature_order, desc="Loading features"):
            all_features[feature_name] = self.load_feature_data(feature_name)
        
        # 获取所有蛋白ID
        protein_ids = self.get_all_protein_ids()
        logger.info(f"总共 {len(protein_ids)} 个蛋白")
        
        # 创建LMDB
        env = lmdb.open(output_path, map_size=map_size)
        
        with env.begin(write=True) as txn:
            # 存储全局元数据
            global_meta = {
                'feature_order': self.feature_order,
                'feature_dims': self.feature_dims,
                'total_dim': self.total_dim,
                'num_proteins': len(protein_ids)
            }
            txn.put(b'__global_meta__', json.dumps(global_meta).encode())
            
            # 处理每个蛋白
            for protein_id in tqdm(protein_ids, desc="Processing proteins"):
                try:
                    combined, metadata = self.process_protein(protein_id, all_features)
                    
                    # 存储特征数组
                    txn.put(protein_id.encode(), combined.tobytes())
                    
                    # 存储元数据
                    meta_key = f'__meta_{protein_id}'.encode()
                    txn.put(meta_key, json.dumps(metadata).encode())
                    
                except Exception as e:
                    logger.error(f"处理 {protein_id} 失败: {e}")
        
        env.close()
        logger.info(f"✓ LMDB 已保存到: {output_path}")
        logger.info(f"✓ 总大小: {Path(output_path).stat().st_size / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="预计算多模态特征到LMDB")
    parser.add_argument('--config', required=True, help='特征配置文件路径')
    parser.add_argument('--output', default=None, help='输出LMDB路径（默认：特征文件夹下的multimodal_features.lmdb）')
    parser.add_argument('--map-size', type=int, default=50, help='LMDB大小(GB)')
    parser.add_argument('--force', action='store_true', help='强制重新生成LMDB（即使已存在）')
    
    args = parser.parse_args()
    
    preprocessor = MultimodalFeaturePreprocessor(args.config)
    
    # 自动确定输出路径
    if args.output is None:
        output_path = str(Path(preprocessor.feature_folder) / 'multimodal_features.lmdb')
    else:
        output_path = args.output
    
    # 检查是否已存在
    if Path(output_path).exists() and not args.force:
        logger.warning(f"LMDB已存在: {output_path}")
        logger.warning("使用 --force 强制重新生成")
        return
    
    preprocessor.build_lmdb(output_path, map_size=args.map_size * 1024**3)


if __name__ == '__main__':
    main()
