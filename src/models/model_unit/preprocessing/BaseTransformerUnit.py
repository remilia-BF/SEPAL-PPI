import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Mapping, cast, Union, Tuple
import math
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

class BaseTransformerUnit(nn.Module):
    """
    基础Transformer预处理单元
    
    训练一个自注意力Transformer模型，将变长蛋白嵌入向量转换为等长的向量
    进行一些转换以提高模型性能（和整体模型共同训练）
    """
    
    def __init__(self,
                 embedding_dim: int,
                 transformer_layers: Optional[List[int]] = None,
                 transformer_heads: int = 8,
                 transformer_layer_num: int = 2,
                 transformer_dropout: float = 0.1,
                 transformer_attention_dropout: float = 0.1,
                 transformer_activation: str = "ReLU",
                 transformer_norm: str = "LayerNorm",
                 transformer_residual: bool = True,
                 **kwargs):
        """
        初始化基础Transformer预处理单元
        
        Args:
            embedding_dim (int): 嵌入维度
            transformer_layers (list): 隐藏层维度列表，默认为[1024, 512]
                                     用于前馈网络的维度配置
            transformer_heads (int): 多头注意力机制的注意力头数，默认为8
                                   更多头可以关注不同的表示子空间
            transformer_layer_num (int): transformer的层数，默认为2
                                       设置为2代表进行两次transformer关注核心序列
            transformer_dropout (float): dropout率，默认为0.1
                                       用于防止过拟合
            transformer_attention_dropout (float): 注意力机制的dropout率，默认为0.1
                                                  专门用于注意力权重的正则化
            transformer_activation (str): 激活函数，默认为"ReLU"
                                        前馈网络中使用的激活函数
            transformer_norm (str): 归一化层类型，默认为"LayerNorm"
                                  用于稳定训练过程
            transformer_residual (bool): 是否使用残差连接，默认为True
                                       有助于深层网络的训练
            **kwargs: 额外参数
        """
        super(BaseTransformerUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.transformer_layer_num = transformer_layer_num
        self.transformer_residual = transformer_residual
        
        if transformer_layers is None:
            transformer_layers = [1024, 512]
        
        self.transformer_layers = transformer_layers
        self.output_dim = embedding_dim  # 输出维度保持与输入相同
        
        # 选择激活函数
        if transformer_activation == "ReLU":
            activation_fn = nn.ReLU()
        elif transformer_activation == "GELU":
            activation_fn = nn.GELU()
        elif transformer_activation == "SiLU":
            activation_fn = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {transformer_activation}")
        
        # 选择归一化层
        if transformer_norm == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif transformer_norm == "BatchNorm":
            norm_layer = lambda dim: nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"不支持的归一化层: {transformer_norm}")
        
        # 创建多层Transformer编码器
        self.transformer_layers_list = nn.ModuleList()
        
        for layer_idx in range(transformer_layer_num):
            # 多头自注意力层
            attention_layer = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=transformer_heads,
                dropout=transformer_attention_dropout,
                batch_first=True  # 使用batch_first格式: [batch_size, seq_len, embed_dim]
            )
            
            # 前馈网络
            ffn_layers = []
            input_dim = embedding_dim
            
            for hidden_dim in transformer_layers:
                ffn_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    activation_fn,
                    nn.Dropout(transformer_dropout)
                ])
                input_dim = hidden_dim
            
            # 输出层，将维度映射回embedding_dim
            ffn_layers.append(nn.Linear(input_dim, embedding_dim))
            ffn = nn.Sequential(*ffn_layers)
            
            # 归一化层
            norm1 = norm_layer(embedding_dim)
            norm2 = norm_layer(embedding_dim)
            
            # 将所有组件打包为一个字典
            transformer_layer = nn.ModuleDict({
                'attention': attention_layer,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'dropout': nn.Dropout(transformer_dropout)
            })
            
            self.transformer_layers_list.append(transformer_layer)
        
        logger.debug(f"创建基础Transformer预处理单元: embedding_dim={embedding_dim}, "
                    f"layers={transformer_layer_num}, heads={transformer_heads}, "
                    f"ffn_dims={transformer_layers}")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings (torch.Tensor): 输入嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
                                         (1表示真实token，0表示padding)
        
        Returns:
            torch.Tensor: 变换后的嵌入 [batch_size, seq_len, embedding_dim]
        """
        x = embeddings
        
        # 如果有注意力掩码，需要转换格式用于MultiheadAttention
        # MultiheadAttention需要的掩码格式: [seq_len, seq_len] 或 [batch_size*num_heads, seq_len, seq_len]
        attn_mask = None
        key_padding_mask = None
        
        if attention_mask is not None:
            # key_padding_mask: [batch_size, seq_len], True表示应该被忽略的位置
            key_padding_mask = (attention_mask == 0)
        
        # 逐层处理
        for layer_dict in self.transformer_layers_list:
            # 直接通过getattr访问模块，避免类型检查问题
            norm_layer1 = getattr(layer_dict, 'norm1')
            attention_layer = getattr(layer_dict, 'attention')
            ffn = getattr(layer_dict, 'ffn')
            norm_layer2 = getattr(layer_dict, 'norm2')
            dropout_layer = getattr(layer_dict, 'dropout')
            
            # 保存残差连接的输入
            residual = x
            
            # 层归一化1 + 自注意力
            x_norm1 = norm_layer1(x)
            attn_output, _ = attention_layer(
                query=x_norm1,
                key=x_norm1,
                value=x_norm1,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
            
            # 残差连接 + dropout
            if self.transformer_residual:
                x = residual + dropout_layer(attn_output)
            else:
                x = dropout_layer(attn_output)
            
            # 保存残差连接的输入
            residual = x
            
            # 层归一化2 + 前馈网络
            x_norm2 = norm_layer2(x)
            ffn_output = ffn(x_norm2)
            
            # 残差连接 + dropout
            if self.transformer_residual:
                x = residual + dropout_layer(ffn_output)
            else:
                x = dropout_layer(ffn_output)
        
        return x
