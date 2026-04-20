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

class CrossAttentionEnhancedTransformerUnit(nn.Module):
    """
    交叉注意力增强Transformer预处理单元
    
    使用交叉注意力机制增强蛋白对的嵌入表示，对于蛋白A，使用蛋白B作为KV来增强A的表示，
    对蛋白B也同样处理。使用Flash Attention优化计算效率。
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
        初始化交叉注意力增强Transformer预处理单元
        
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
        super(CrossAttentionEnhancedTransformerUnit, self).__init__()
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
            logger.info("Flash Attention可用，将使用Flash Attention优化交叉注意力计算")
        except ImportError:
            logger.warning("Flash Attention不可用，将使用标准注意力机制实现交叉注意力")
        
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
        
        # 创建多层Transformer编码器（用于自注意力）
        self.self_attention_layers = nn.ModuleList()
        # 创建交叉注意力层
        self.cross_attention_layers = nn.ModuleList()
        
        for layer_idx in range(transformer_layer_num):
            # 自注意力层组件
            self_attn_ffn_layers = []
            input_dim = embedding_dim
            
            for hidden_dim in transformer_layers:
                self_attn_ffn_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    activation_fn,
                    nn.Dropout(transformer_dropout)
                ])
                input_dim = hidden_dim
            
            # 输出层，将维度映射回embedding_dim
            self_attn_ffn_layers.append(nn.Linear(input_dim, embedding_dim))
            self_attn_ffn = nn.Sequential(*self_attn_ffn_layers)
            
            # 交叉注意力层组件
            cross_attn_ffn_layers = []
            input_dim = embedding_dim
            
            for hidden_dim in transformer_layers:
                cross_attn_ffn_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    activation_fn,
                    nn.Dropout(transformer_dropout)
                ])
                input_dim = hidden_dim
            
            # 输出层，将维度映射回embedding_dim
            cross_attn_ffn_layers.append(nn.Linear(input_dim, embedding_dim))
            cross_attn_ffn = nn.Sequential(*cross_attn_ffn_layers)
            
            # 归一化层
            norm1 = norm_layer(embedding_dim)  # 自注意力后归一化
            norm2 = norm_layer(embedding_dim)  # 交叉注意力后归一化
            norm3 = norm_layer(embedding_dim)  # FFN后归一化
            
            # 自注意力层打包
            self_attn_layer = nn.ModuleDict({
                'ffn': self_attn_ffn,
                'norm1': norm1,
                'norm2': norm2,  # 用于交叉注意力后归一化
                'norm3': norm3,  # 用于FFN后归一化
                'dropout': nn.Dropout(transformer_dropout)
            })
            
            # 交叉注意力层打包
            cross_attn_layer = nn.ModuleDict({
                'ffn': cross_attn_ffn,
                'dropout': nn.Dropout(transformer_dropout)
            })
            
            self.self_attention_layers.append(self_attn_layer)
            self.cross_attention_layers.append(cross_attn_layer)
        
        logger.debug(f"创建交叉注意力增强Transformer预处理单元: embedding_dim={embedding_dim}, "
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

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor,
                attention_mask_a: Optional[torch.Tensor] = None,
                                         attention_mask_b: Optional[torch.Tensor] = None):
        """
        前向传播，对蛋白对进行交叉注意力增强
        
        Args:
            embeddings_a (torch.Tensor): 蛋白A的输入嵌入 [batch_size, seq_len_a, embedding_dim]
            embeddings_b (torch.Tensor): 蛋白B的输入嵌入 [batch_size, seq_len_b, embedding_dim]
            attention_mask_a (torch.Tensor): 蛋白A的注意力掩码 [batch_size, seq_len_a]
            attention_mask_b (torch.Tensor): 蛋白B的注意力掩码 [batch_size, seq_len_b]
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 增强后的嵌入 (enhanced_a, enhanced_b)
        """
        x_a = embeddings_a
        x_b = embeddings_b
        
        # 处理注意力掩码
        key_padding_mask_a = None
        key_padding_mask_b = None
        
        if attention_mask_a is not None:
            key_padding_mask_a = (attention_mask_a == 0)
            
        if attention_mask_b is not None:
            key_padding_mask_b = (attention_mask_b == 0)
        
        # 逐层处理
        for self_attn_dict, cross_attn_dict in zip(self.self_attention_layers, self.cross_attention_layers):
            # 获取层组件
            self_norm1 = getattr(self_attn_dict, 'norm1')
            self_norm2 = getattr(self_attn_dict, 'norm2')
            self_norm3 = getattr(self_attn_dict, 'norm3')
            self_ffn = getattr(self_attn_dict, 'ffn')
            self_dropout = getattr(self_attn_dict, 'dropout')
            
            cross_ffn = getattr(cross_attn_dict, 'ffn')
            cross_dropout = getattr(cross_attn_dict, 'dropout')
            
            # 1. 蛋白A自注意力
            residual_a = x_a
            x_a_norm = self_norm1(x_a)
            
            if self.use_flash_attn:
                # 使用Flash Attention实现自注意力
                batch_size, seq_len_a, embed_dim = x_a_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                assert embed_dim % self.transformer_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除"
                
                # 转换数据类型以满足Flash Attention要求
                x_a_norm, orig_dtype_a = self._convert_to_flash_attn_dtype(x_a_norm)
                
                # 重塑为Flash Attention格式
                x_a_reshaped = x_a_norm.reshape(batch_size, seq_len_a, self.transformer_heads, head_dim)
                
                # 自注意力计算
                self_attn_output = self.flash_attn_func(
                    x_a_reshaped, x_a_reshaped, x_a_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                self_attn_output = self_attn_output.view(batch_size, seq_len_a, embed_dim)
                
                # 转换回原始数据类型
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype_a)
            else:
                # 创建标准自注意力层（如果尚未创建）
                if not hasattr(self_attn_dict, 'self_attention') or self_attn_dict.self_attention is None:
                    self_attention_layer = nn.MultiheadAttention(
                        embed_dim=self.embedding_dim,
                        num_heads=self.transformer_heads,
                        dropout=self.transformer_attention_dropout,
                        batch_first=True
                    )
                    setattr(self_attn_dict, 'self_attention', self_attention_layer)
                else:
                    self_attention_layer = getattr(self_attn_dict, 'self_attention')
                
                self_attn_output, _ = self_attention_layer(
                    query=x_a_norm,
                    key=x_a_norm,
                    value=x_a_norm,
                    key_padding_mask=key_padding_mask_a
                )
            
            # 蛋白A自注意力残差连接
            if self.transformer_residual:
                x_a = residual_a + self_dropout(self_attn_output)
            else:
                x_a = self_dropout(self_attn_output)
            
            # 2. 蛋白B自注意力
            residual_b = x_b
            x_b_norm = self_norm1(x_b)
            
            if self.use_flash_attn:
                # 使用Flash Attention实现自注意力
                batch_size, seq_len_b, embed_dim = x_b_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                assert embed_dim % self.transformer_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除"
                
                # 转换数据类型以满足Flash Attention要求
                x_b_norm, orig_dtype_b = self._convert_to_flash_attn_dtype(x_b_norm)
                
                # 重塑为Flash Attention格式
                x_b_reshaped = x_b_norm.reshape(batch_size, seq_len_b, self.transformer_heads, head_dim)
                
                # 自注意力计算
                self_attn_output = self.flash_attn_func(
                    x_b_reshaped, x_b_reshaped, x_b_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                self_attn_output = self_attn_output.view(batch_size, seq_len_b, embed_dim)
                
                # 转换回原始数据类型
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype_b)
            else:
                # 使用已创建的标准自注意力层
                self_attn_output, _ = self_attention_layer(
                    query=x_b_norm,
                    key=x_b_norm,
                    value=x_b_norm,
                    key_padding_mask=key_padding_mask_b
                )
            
            # 蛋白B自注意力残差连接
            if self.transformer_residual:
                x_b = residual_b + self_dropout(self_attn_output)
            else:
                x_b = self_dropout(self_attn_output)
            
            # 3. 蛋白A对蛋白B的交叉注意力
            residual_a = x_a
            x_a_norm = self_norm2(x_a)  # 使用第二个归一化层
            x_b_norm = self_norm2(x_b)  # 蛋白B也需要归一化
            
            if self.use_flash_attn:
                # 使用Flash Attention实现交叉注意力
                batch_size, seq_len_a, embed_dim = x_a_norm.size()
                _, seq_len_b, _ = x_b_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                # 转换数据类型以满足Flash Attention要求
                q_a, orig_dtype_qa = self._convert_to_flash_attn_dtype(x_a_norm)
                k_b, orig_dtype_kb = self._convert_to_flash_attn_dtype(x_b_norm)
                v_b, orig_dtype_vb = self._convert_to_flash_attn_dtype(x_b_norm)
                
                # 重塑为Flash Attention格式
                q_a = q_a.reshape(batch_size, seq_len_a, self.transformer_heads, head_dim)
                k_b = k_b.reshape(batch_size, seq_len_b, self.transformer_heads, head_dim)
                v_b = v_b.reshape(batch_size, seq_len_b, self.transformer_heads, head_dim)
                
                # 交叉注意力计算 (A查询，B的KV)
                cross_attn_output = self.flash_attn_func(
                    q_a, k_b, v_b,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                cross_attn_output = cross_attn_output.view(batch_size, seq_len_a, embed_dim)
                
                # 转换回原始数据类型 (使用查询的原始类型)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_qa)
            else:
                # 创建交叉注意力层（如果尚未创建）
                if not hasattr(cross_attn_dict, 'cross_attention') or cross_attn_dict.cross_attention is None:
                    cross_attention_layer = nn.MultiheadAttention(
                        embed_dim=self.embedding_dim,
                        num_heads=self.transformer_heads,
                        dropout=self.transformer_attention_dropout,
                        batch_first=True
                    )
                    setattr(cross_attn_dict, 'cross_attention', cross_attention_layer)
                else:
                    cross_attention_layer = getattr(cross_attn_dict, 'cross_attention')
                
                cross_attn_output, _ = cross_attention_layer(
                    query=x_a_norm,
                    key=x_b_norm,
                    value=x_b_norm,
                    key_padding_mask=key_padding_mask_b
                )
            
            # 蛋白A交叉注意力残差连接
            if self.transformer_residual:
                x_a = residual_a + cross_dropout(cross_attn_output)
            else:
                x_a = cross_dropout(cross_attn_output)
            
            # 4. 蛋白B对蛋白A的交叉注意力
            residual_b = x_b
            x_a_norm = self_norm2(x_a)  # 蛋白A再次归一化
            x_b_norm = self_norm2(x_b)  # 蛋白B再次归一化
            
            if self.use_flash_attn:
                # 使用Flash Attention实现交叉注意力
                batch_size, seq_len_b, embed_dim = x_b_norm.size()
                _, seq_len_a, _ = x_a_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                # 转换数据类型以满足Flash Attention要求
                q_b, orig_dtype_qb = self._convert_to_flash_attn_dtype(x_b_norm)
                k_a, orig_dtype_ka = self._convert_to_flash_attn_dtype(x_a_norm)
                v_a, orig_dtype_va = self._convert_to_flash_attn_dtype(x_a_norm)
                
                # 重塑为Flash Attention格式
                q_b = q_b.reshape(batch_size, seq_len_b, self.transformer_heads, head_dim)
                k_a = k_a.reshape(batch_size, seq_len_a, self.transformer_heads, head_dim)
                v_a = v_a.reshape(batch_size, seq_len_a, self.transformer_heads, head_dim)
                
                # 交叉注意力计算 (B查询，A的KV)
                cross_attn_output = self.flash_attn_func(
                    q_b, k_a, v_a,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                cross_attn_output = cross_attn_output.view(batch_size, seq_len_b, embed_dim)
                
                # 转换回原始数据类型 (使用查询的原始类型)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_qb)
            else:
                # 使用已创建的交叉注意力层
                cross_attn_output, _ = cross_attention_layer(
                    query=x_b_norm,
                    key=x_a_norm,
                    value=x_a_norm,
                    key_padding_mask=key_padding_mask_a
                )
            
            # 蛋白B交叉注意力残差连接
            if self.transformer_residual:
                x_b = residual_b + cross_dropout(cross_attn_output)
            else:
                x_b = cross_dropout(cross_attn_output)
            
            # 5. 蛋白A前馈网络
            residual_a = x_a
            x_a_norm = self_norm3(x_a)  # 使用第三个归一化层
            ffn_output = self_ffn(x_a_norm)
            
            if self.transformer_residual:
                x_a = residual_a + self_dropout(ffn_output)
            else:
                x_a = self_dropout(ffn_output)
            
            # 6. 蛋白B前馈网络
            residual_b = x_b
            x_b_norm = self_norm3(x_b)  # 使用第三个归一化层
            ffn_output = self_ffn(x_b_norm)  # 使用相同的FFN结构
            
            if self.transformer_residual:
                x_b = residual_b + self_dropout(ffn_output)
            else:
                x_b = self_dropout(ffn_output)
        
        return x_a, x_b
