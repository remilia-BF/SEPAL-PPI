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

class FlashAttentionTransformerUnit(nn.Module):
    """
    使用Flash Attention的Transformer预处理单元
    
    训练一个自注意力Transformer模型，使用Flash Attention优化注意力计算，
    将变长蛋白嵌入向量转换为等长的向量
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
        初始化Flash Attention Transformer预处理单元
        
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
        super(FlashAttentionTransformerUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.transformer_layer_num = transformer_layer_num
        self.transformer_residual = transformer_residual
        self.transformer_attention_dropout = transformer_attention_dropout
        
        if transformer_layers is None:
            transformer_layers = [1024, 512]
        
        self.transformer_layers = transformer_layers
        self.output_dim = embedding_dim  # 输出维度保持与输入相同
        
        # 尝试导入flash attention
        self.use_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
            logger.info("Flash Attention可用，将使用Flash Attention优化")
        except ImportError:
            logger.warning("Flash Attention不可用，将使用标准注意力机制")
        
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
            # 注意力层（不初始化，运行时根据use_flash_attn决定使用哪种）
            attention_layer = None
            
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
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'dropout': nn.Dropout(transformer_dropout)
            })
            
            self.transformer_layers_list.append(transformer_layer)
        
        logger.debug(f"创建Flash Attention Transformer预处理单元: embedding_dim={embedding_dim}, "
                    f"layers={transformer_layer_num}, heads={transformer_heads}, "
                    f"ffn_dims={transformer_layers}, use_flash_attn={self.use_flash_attn}")

    def _convert_to_flash_attn_dtype(self, tensor: torch.Tensor):
        """
        将张量转换为适合Flash Attention的数据类型
        
        Args:
            tensor: 输入张量
            
        Returns:
            tuple: (转换后的张量, 原始数据类型)
        """
        original_dtype = tensor.dtype
        
        # Flash Attention只支持fp16和bf16
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            # 如果CUDA可用且支持bfloat16，优先使用bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                # 否则使用float16
                target_dtype = torch.float16
            
            tensor = tensor.to(dtype=target_dtype)
        
        return tensor, original_dtype

    def _convert_back_from_flash_attn_dtype(self, tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        """
        将张量从Flash Attention数据类型转换回原始数据类型
        
        Args:
            tensor: Flash Attention计算结果张量
            original_dtype: 原始数据类型
            
        Returns:
            转换回原始数据类型的张量
        """
        # 如果数据类型不一致，则转换回原始类型
        if tensor.dtype != original_dtype:
            tensor = tensor.to(dtype=original_dtype)
        
        return tensor

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
        
        # 如果有注意力掩码，需要转换格式
        key_padding_mask = None
        if attention_mask is not None:
            # key_padding_mask: [batch_size, seq_len], True表示应该被忽略的位置
            key_padding_mask = (attention_mask == 0)
        
        # 逐层处理
        for layer_dict in self.transformer_layers_list:
            # 直接通过getattr访问模块，避免类型检查问题
            norm_layer1 = getattr(layer_dict, 'norm1')
            ffn = getattr(layer_dict, 'ffn')
            norm_layer2 = getattr(layer_dict, 'norm2')
            dropout_layer = getattr(layer_dict, 'dropout')
            
            # 保存残差连接的输入
            residual = x
            
            # 层归一化1 + 自注意力
            x_norm1 = norm_layer1(x)
            
            if self.use_flash_attn:
                # 使用Flash Attention实现
                # 获取输入的维度信息
                batch_size, seq_len, embed_dim = x_norm1.size()
                head_dim = embed_dim // self.transformer_heads
                
                # 确保嵌入维度可以被头数整除
                assert embed_dim % self.transformer_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除"
                
                # 转换数据类型以满足Flash Attention要求
                x_norm1, orig_dtype = self._convert_to_flash_attn_dtype(x_norm1)
                
                # 重塑为Flash Attention格式: [batch_size, seq_len, num_heads, head_dim]
                x_reshaped = x_norm1.reshape(batch_size, seq_len, self.transformer_heads, head_dim)
                
                # 调用Flash Attention
                # flash_attn_func期望输入为[q, k, v]，每个都是[batch_size, seq_len, num_heads, head_dim]
                attn_output = self.flash_attn_func(
                    x_reshaped, x_reshaped, x_reshaped, 
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0, 
                    causal=False
                )
                
                # 确保attn_output不是None
                if attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式: [batch_size, seq_len, embed_dim]
                attn_output = attn_output.view(batch_size, seq_len, embed_dim)
                
                # 转换回原始数据类型
                attn_output = self._convert_back_from_flash_attn_dtype(attn_output, orig_dtype)
            else:
                # 使用标准注意力机制
                # 创建标准注意力层（如果尚未创建）
                if not hasattr(layer_dict, 'attention') or layer_dict.attention is None:
                    attention_layer = nn.MultiheadAttention(
                        embed_dim=self.embedding_dim,
                        num_heads=self.transformer_heads,
                        dropout=self.transformer_attention_dropout,
                        batch_first=True
                    )
                    # 使用setattr添加到ModuleDict中
                    setattr(layer_dict, 'attention', attention_layer)
                else:
                    attention_layer = getattr(layer_dict, 'attention')
                
                attn_output, _ = attention_layer(
                    query=x_norm1,
                    key=x_norm1,
                    value=x_norm1,
                    key_padding_mask=key_padding_mask
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
