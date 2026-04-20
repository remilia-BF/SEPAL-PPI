"""
交互单元模块
包含各种蛋白质交互方法的可配置实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HadamardInteractionUnit(nn.Module):
    """
    哈达玛积交互单元
    
    对两个蛋白质嵌入进行元素级乘法操作
    输出维度与输入维度相同
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化哈达玛积交互单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(HadamardInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim  # 哈达玛积保持维度不变
        
        logger.debug(f"Create HadamardInteractionUnit: embedding_dim={embedding_dim}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 交互特征 [batch_size, embedding_dim]
        """
        return torch.mul(protein1_emb, protein2_emb)


class OuterProductInteractionUnit(nn.Module):
    """
    外积交互单元
    
    对两个蛋白质嵌入进行外积操作，然后展平
    输出维度为 embedding_dim^2
    """
    
    def __init__(self, 
                 embedding_dim: int, 
                 use_flattened: bool = True,
                 reduction_dim: Optional[int] = None,
                 **kwargs):
        """
        初始化外积交互单元
        
        Args:
            embedding_dim (int): 嵌入维度
            use_flattened (bool): 是否将外积结果展平，默认为True
                                如果为False，则保持矩阵形式 [batch_size, embedding_dim, embedding_dim]
            reduction_dim (Optional[int]): 可选的降维目标维度，如果指定则添加线性层进行降维
                               用于控制输出特征数量，避免维度爆炸
            **kwargs: 额外参数
        """
        super(OuterProductInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_flattened = use_flattened
        self.reduction_dim = reduction_dim
        
        if use_flattened:
            self.output_dim = embedding_dim * embedding_dim
        else:
            self.output_dim = (embedding_dim, embedding_dim)  # 矩阵形式
        
        # 可选的降维层
        if reduction_dim is not None and use_flattened:
            self.reduction_layer = nn.Linear(embedding_dim * embedding_dim, reduction_dim)
            self.output_dim = reduction_dim
        else:
            self.reduction_layer = None
        
        logger.debug(f"Create OuterProductInteractionUnit: embedding_dim={embedding_dim}, "
                f"use_flattened={use_flattened}, reduction_dim={reduction_dim}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 交互特征 [batch_size, embedding_dim^2] 或 [batch_size, embedding_dim, embedding_dim]
        """
        # 计算外积: [batch_size, embedding_dim, embedding_dim]
        outer_product = torch.bmm(
            protein1_emb.unsqueeze(2),  # [batch_size, embedding_dim, 1]
            protein2_emb.unsqueeze(1)   # [batch_size, 1, embedding_dim]
        )
        
        if self.use_flattened:
            # 展平外积矩阵
            flattened = outer_product.view(outer_product.size(0), -1)  # [batch_size, embedding_dim^2]
            
            # 可选的降维
            if self.reduction_layer is not None:
                return self.reduction_layer(flattened)
            else:
                return flattened
        else:
            return outer_product


class ConcatenationInteractionUnit(nn.Module):
    """
    拼接交互单元
    
    将两个蛋白质嵌入简单拼接
    输出维度为 2 * embedding_dim
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化拼接交互单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(ConcatenationInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = 2 * embedding_dim  # 拼接后维度翻倍
        
        logger.debug(f"Create ConcatenationInteractionUnit: embedding_dim={embedding_dim}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 交互特征 [batch_size, 2 * embedding_dim]
        """
        return torch.cat([protein1_emb, protein2_emb], dim=1)


class DifferenceInteractionUnit(nn.Module):
    """
    差值交互单元
    
    计算两个蛋白质嵌入的差值，支持绝对差值和向量差值两种模式
    输出维度与输入维度相同
    """
    
    def __init__(self, embedding_dim: int, absolute_difference: bool = True, **kwargs):
        """
        初始化差值交互单元
        
        Args:
            embedding_dim (int): 嵌入维度
            absolute_difference (bool): 是否计算绝对差值。True为绝对差值|a-b|，False为向量差值a-b
            **kwargs: 额外参数 (为扩展性保留)
        """
        super(DifferenceInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.absolute_difference = absolute_difference
        self.output_dim = embedding_dim  # 差值保持维度不变
        
        logger.debug(f"Create DifferenceInteractionUnit: embedding_dim={embedding_dim}, "
                f"absolute_difference={absolute_difference}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 交互特征 [batch_size, embedding_dim]
        """
        difference = protein1_emb - protein2_emb
        
        if self.absolute_difference:
            return torch.abs(difference)
        else:
            return difference


class CosineInteractionUnit(nn.Module):
    """
    余弦相似度交互单元
    
    计算两个蛋白质嵌入的余弦相似度
    输出维度为1
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化余弦相似度交互单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(CosineInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = 1  # 余弦相似度是标量
        
        logger.debug(f"Create CosineInteractionUnit: embedding_dim={embedding_dim}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 交互特征 [batch_size, 1]
        """
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(protein1_emb, protein2_emb, dim=1, eps=1e-8)
        return cosine_sim.unsqueeze(1)  # [batch_size, 1]


class FastCompactBilinearInteractionUnit(nn.Module):
    """
    标准低秩双线性池化交互单元

    使用低秩分解近似双线性映射：
        z = W_o((W_1 x) ⊙ (W_2 y))

    特性:
    1. 低秩维度可控，计算与显存开销显著低于完整双线性
    2. 支持Dropout与可选归一化
    3. 线性层使用正交初始化

    说明:
    - 本模块不在内部重置随机种子，初始化随机性直接继承主程序
      的全局随机种子（例如通过 set_random_seed 设置）。
    """
    
    def __init__(self, 
                 embedding_dim: int, 
                 rank_dim: int = 128,
                 output_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_normalization: bool = False,
                 **kwargs):
        """
        初始化标准低秩双线性池化交互单元
        
        Args:
            embedding_dim (int): 输入嵌入维度
            rank_dim (int): 低秩维度，默认为128
            output_dim (Optional[int]): 输出特征维度。None时默认等于embedding_dim
            dropout (float): 低秩交互后的Dropout概率
            use_normalization (bool): 是否对输出进行LayerNorm归一化
            **kwargs: 额外参数
        """
        super(FastCompactBilinearInteractionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.rank_dim = rank_dim
        self.output_dim = output_dim if output_dim is not None else embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.use_normalization = use_normalization

        # 低秩双线性分解层
        self.left_projection = nn.Linear(embedding_dim, rank_dim, bias=False)
        self.right_projection = nn.Linear(embedding_dim, rank_dim, bias=False)
        self.output_projection = nn.Linear(rank_dim, self.output_dim, bias=True)

        # 可选输出归一化
        self.output_norm = nn.LayerNorm(self.output_dim) if use_normalization else nn.Identity()

        self._reset_parameters()
        
        logger.debug(f"Create FastCompactBilinearInteractionUnit: "
                f"embedding_dim={embedding_dim}, rank_dim={rank_dim}, output_dim={self.output_dim}, "
                f"dropout={dropout}, use_normalization={use_normalization}")

    def _reset_parameters(self):
        """参数初始化（正交初始化）"""
        nn.init.orthogonal_(self.left_projection.weight)
        nn.init.orthogonal_(self.right_projection.weight)
        nn.init.orthogonal_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_emb (torch.Tensor): 第一个蛋白质嵌入 [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): 第二个蛋白质嵌入 [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: 低秩双线性池化特征 [batch_size, output_dim]
        """
        left_features = self.left_projection(protein1_emb)     # [batch_size, rank_dim]
        right_features = self.right_projection(protein2_emb)   # [batch_size, rank_dim]

        bilinear_features = left_features * right_features     # [batch_size, rank_dim]
        bilinear_features = self.dropout(bilinear_features)
        bilinear_features = self.output_projection(bilinear_features)
        bilinear_features = self.output_norm(bilinear_features)

        return bilinear_features


def create_interaction_unit(interaction_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建交互单元
    
    Args:
        interaction_config (Dict): 交互配置，包含operation和其他参数
                                 必须包含: operation (str) - 交互操作名称
                                 可选包含: embedding_dim, use_flattened, reduction_dim等
    
    Returns:
        nn.Module: 交互单元实例
    
    Raises:
        ValueError: 当交互操作不支持时
    
    Examples:
        >>> config = {
        ...     'operation': 'hadamard_product',
        ...     'embedding_dim': 1280
        ... }
        >>> interaction = create_interaction_unit(config)
        
        >>> config = {
        ...     'operation': 'outer_product',
        ...     'embedding_dim': 1280,
        ...     'use_flattened': True,
        ...     'reduction_dim': 512
        ... }
        >>> interaction = create_interaction_unit(config)
        
        >>> config = {
        ...     'operation': 'fast_compact_bilinear',
        ...     'embedding_dim': 1280,
        ...     'output_dim': 1024,
        ...     'use_fft': True
        ... }
        >>> interaction = create_interaction_unit(config)
    """
    operation = interaction_config.get('operation', 'hadamard_product')
    embedding_dim = interaction_config.get('embedding_dim', None)
    
    if embedding_dim is None:
        raise ValueError("Interaction unit requires 'embedding_dim' parameter")
    
    # 移除operation和embedding_dim，将其余参数传递给具体的交互单元
    kwargs = {k: v for k, v in interaction_config.items() if k not in ['operation', 'embedding_dim']}
    
    if operation == 'hadamard_product':
        return HadamardInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    elif operation == 'outer_product':
        return OuterProductInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    elif operation == 'concatenation':
        return ConcatenationInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    elif operation == 'difference':
        return DifferenceInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    elif operation == 'cosine':
        return CosineInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    elif operation == 'fast_compact_bilinear':
        return FastCompactBilinearInteractionUnit(embedding_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported interaction operation: {operation}. "
                        f"Supported: hadamard_product, outer_product, concatenation, "
                        f"difference, cosine, fast_compact_bilinear")


def test_fast_compact_bilinear_interaction():
    """
    测试低秩双线性池化交互单元
    
    Returns:
        bool: 测试是否通过
    """
    try:
        # 测试参数
        batch_size = 4
        embedding_dim = 640
        output_dim = embedding_dim
        
        # 创建低秩双线性池化交互单元
        interaction_config = {
            'operation': 'fast_compact_bilinear',
            'embedding_dim': embedding_dim,
            'output_dim': output_dim,
            'rank_dim': 128,
            'dropout': 0.1,
            'use_normalization': True
        }
        
        interaction_unit = create_interaction_unit(interaction_config)
        
        # 创建测试数据
        protein1_emb = torch.randn(batch_size, embedding_dim)
        protein2_emb = torch.randn(batch_size, embedding_dim)
        
        # 前向传播
        output = interaction_unit(protein1_emb, protein2_emb)
        
        # 检查输出形状
        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        
        # 检查输出值是否合理
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        logger.info(f"Low-rank bilinear pooling test passed: input shape {protein1_emb.shape}, output shape {output.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Low-rank bilinear pooling test failed: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    test_fast_compact_bilinear_interaction()