"""
批量化多模态特征加载器

从预计算的 LMDB 批量加载特征，消除逐样本循环瓶颈
"""

import lmdb
import torch
import numpy as np
import json
from typing import List, Tuple, Optional
from pathlib import Path


class BatchedMultimodalLoader:
    """批量加载多模态特征的辅助类"""
    
    def __init__(self, lmdb_path: str, device: str = 'cuda'):
        """
        Args:
            lmdb_path: 预计算的LMDB路径
            device: 目标设备
        """
        self.lmdb_path = lmdb_path
        self.device = device
        
        # 打开LMDB（只读）
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                            readahead=False, meminit=False)
        
        # 加载全局元数据
        with self.env.begin() as txn:
            meta_bytes = txn.get(b'__global_meta__')
            self.global_meta = json.loads(meta_bytes.decode())
        
        self.feature_order = self.global_meta['feature_order']
        self.feature_dims = self.global_meta['feature_dims']
        self.total_dim = self.global_meta['total_dim']
        
        print(f"✓ 加载多模态LMDB: {lmdb_path}")
        print(f"  特征数: {len(self.feature_order)}")
        print(f"  总维度: {self.total_dim}")
    
    def load_batch(self, protein_ids: List[str], 
                   max_length: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量加载特征
        
        Args:
            protein_ids: 蛋白ID列表
            max_length: 最大序列长度（padding/truncate）
        
        Returns:
            features: [batch, seq_len, total_dim] 张量
            mask: [batch, seq_len] padding mask (1=valid, 0=padding)
        """
        batch_size = len(protein_ids)
        
        # 预分配张量
        features = torch.zeros((batch_size, max_length, self.total_dim), 
                              dtype=torch.float32, device='cpu')
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device='cpu')
        
        with self.env.begin() as txn:
            for i, protein_id in enumerate(protein_ids):
                # 读取特征
                feat_bytes = txn.get(protein_id.encode())
                if feat_bytes is None:
                    # 缺失特征：保持零填充
                    continue
                
                # 读取元数据
                meta_bytes = txn.get(f'__meta_{protein_id}'.encode())
                metadata = json.loads(meta_bytes.decode())
                seq_len = metadata['seq_len']
                
                # 解析特征数组
                feat_array = np.frombuffer(feat_bytes, dtype=np.float32)
                feat_array = feat_array.reshape(seq_len, self.total_dim)
                
                # Padding/Truncate
                actual_len = min(seq_len, max_length)
                features[i, :actual_len] = torch.from_numpy(feat_array[:actual_len])
                mask[i, :actual_len] = True
        
        # 移动到目标设备
        features = features.to(self.device)
        mask = mask.to(self.device)
        
        return features, mask
    
    def get_feature_splits(self) -> List[int]:
        """返回特征分割维度列表（用于 torch.split）"""
        return self.feature_dims
    
    def close(self):
        """关闭LMDB"""
        if self.env is not None:
            self.env.close()
    
    def __del__(self):
        self.close()


def test_loader():
    """测试批量加载器"""
    loader = BatchedMultimodalLoader('mutifeature/S1_precomputed.lmdb')
    
    # 测试加载
    protein_ids = ['P12345', 'P67890']  # 示例ID
    features, mask = loader.load_batch(protein_ids, max_length=512)
    
    print(f"✓ 批量特征形状: {features.shape}")
    print(f"✓ Mask 形状: {mask.shape}")
    
    # 分割特征
    splits = loader.get_feature_splits()
    feature_list = features.split(splits, dim=-1)
    print(f"✓ 分割后特征数: {len(feature_list)}")
    for i, feat in enumerate(feature_list):
        print(f"  特征 {i}: {feat.shape}")
    
    loader.close()


if __name__ == '__main__':
    test_loader()
