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

class FlashTransformerDecoderLayer(nn.Module):
    """使用Flash Attention的Transformer解码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        super(FlashTransformerDecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, f"d_model {d_model} 必须能被 nhead {nhead} 整除"
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # Flash Attention相关
        self.flash_attn_available = False
        self.flash_attn_func = None
        self._init_flash_attention()
    
    def _init_flash_attention(self):
        """初始化Flash Attention"""
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_available = True
            logger.debug("FlashTransformerDecoderLayer: Flash Attention可用")
        except ImportError:
            logger.warning("FlashTransformerDecoderLayer: Flash Attention不可用，将使用标准注意力")
            self.flash_attn_available = False
    
    def _convert_to_flash_attn_dtype(self, x: torch.Tensor):
        """转换为Flash Attention支持的数据类型"""
        orig_dtype = x.dtype
        if x.dtype == torch.float32:
            return x.to(torch.bfloat16), orig_dtype
        elif x.dtype in [torch.float16, torch.bfloat16]:
            return x, orig_dtype
        else:
            return x.to(torch.bfloat16), orig_dtype
    
    def _convert_back_from_flash_attn_dtype(self, x: torch.Tensor, orig_dtype: torch.dtype):
        """转换回原始数据类型"""
        return x.to(orig_dtype)
    
    def _flash_self_attention(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """使用Flash Attention实现自注意力（解码器版本）"""
        batch_size, seq_len, embed_dim = x.shape
        
        # 转换数据类型
        x_flash, orig_dtype = self._convert_to_flash_attn_dtype(x)
        
        # 重塑为Flash Attention格式: [batch, seq_len, num_heads, head_dim]
        x_reshaped = x_flash.reshape(batch_size, seq_len, self.nhead, self.head_dim)
        
        # 准备Flash Attention参数
        flash_kwargs = {
            'dropout_p': self.dropout.p if self.training else 0.0,
            'causal': False  # 非因果注意力
        }
        
        # 处理padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len], True表示需要被忽略的位置
            # Flash Attention需要cu_seqlens格式：每个序列的有效长度
            try:
                # 计算每个batch的有效序列长度
                seq_lens = (~key_padding_mask).sum(dim=1).int()  # [batch_size]
                
                # 转换为cu_seqlens格式: [0, len1, len1+len2, ...]
                cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
                cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
                
                # 检查是否所有序列长度都相同（没有padding）
                if seq_lens.min() == seq_lens.max() == seq_len:
                    # 没有实际的padding，使用标准API
                    pass  # 使用标准的flash_attn_func
                else:
                    # 有padding，需要使用varlen版本
                    try:
                        from flash_attn import flash_attn_varlen_func
                        
                        # 重塑输入为varlen格式
                        total_len = cu_seqlens[-1].item()
                        
                        # 创建varlen格式的输入
                        x_varlen = x_reshaped.view(-1, self.nhead, self.head_dim)[:total_len]
                        
                        attn_output = flash_attn_varlen_func(
                            x_varlen, x_varlen, x_varlen,  # q, k, v
                            cu_seqlens, cu_seqlens,  # cu_seqlens_q, cu_seqlens_k
                            seq_len, seq_len,  # max_seqlen_q, max_seqlen_k
                            **flash_kwargs
                        )
                        
                        # 重塑回原始格式并处理padding
                        output_reshaped = torch.zeros_like(x_reshaped)
                        start_idx = 0
                        for i, length in enumerate(seq_lens):
                            end_idx = start_idx + length
                            if length > 0:
                                output_reshaped[i, :length] = attn_output[start_idx:end_idx]
                            start_idx = end_idx
                        
                        attn_output = output_reshaped
                        
                    except ImportError:
                        # flash_attn_varlen_func不可用，回退到标准实现
                        raise RuntimeError("Variable length Flash Attention not available")
                        
            except Exception as e:
                # Flash Attention mask处理失败，回退到标准实现
                raise RuntimeError(f"Flash Attention mask处理失败: {e}")
        else:
            # 没有mask，使用标准Flash Attention
            attn_output = self.flash_attn_func(
                x_reshaped, x_reshaped, x_reshaped,  # q, k, v
                **flash_kwargs
            )
        
        # 如果没有使用varlen版本，直接调用标准版本
        if 'attn_output' not in locals():
            attn_output = self.flash_attn_func(
                x_reshaped, x_reshaped, x_reshaped,  # q, k, v
                **flash_kwargs
            )
        
        if attn_output is None:
            raise RuntimeError("Flash Attention returned None")
        
        # 重塑回原始格式
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        
        # 转换回原始数据类型
        attn_output = self._convert_back_from_flash_attn_dtype(attn_output, orig_dtype)
        
        return attn_output
    
    def _flash_cross_attention(self, tgt: torch.Tensor, memory: torch.Tensor,
                              memory_key_padding_mask: Optional[torch.Tensor] = None):
        """使用Flash Attention实现交叉注意力"""
        batch_size, tgt_len, embed_dim = tgt.shape
        _, mem_len, _ = memory.shape
        
        # 转换数据类型
        tgt_flash, orig_dtype_tgt = self._convert_to_flash_attn_dtype(tgt)
        memory_flash, orig_dtype_mem = self._convert_to_flash_attn_dtype(memory)
        
        # 重塑为Flash Attention格式
        q_reshaped = tgt_flash.reshape(batch_size, tgt_len, self.nhead, self.head_dim)
        k_reshaped = memory_flash.reshape(batch_size, mem_len, self.nhead, self.head_dim)
        v_reshaped = memory_flash.reshape(batch_size, mem_len, self.nhead, self.head_dim)
        
        # 准备Flash Attention参数
        flash_kwargs = {
            'dropout_p': self.dropout.p if self.training else 0.0,
            'causal': False
        }
        
        # 处理memory的padding mask
        if memory_key_padding_mask is not None:
            # memory_key_padding_mask: [batch_size, mem_len], True表示需要被忽略的位置
            try:
                # 计算每个batch的memory有效序列长度
                mem_seq_lens = (~memory_key_padding_mask).sum(dim=1).int()  # [batch_size]
                
                # 对于交叉注意力，查询长度通常是固定的，但key/value可能有padding
                # 检查是否有实际的padding
                if mem_seq_lens.min() == mem_seq_lens.max() == mem_len:
                    # 没有实际的memory padding，使用标准API
                    pass
                else:
                    # 有memory padding，需要特殊处理
                    # 注意：Flash Attention的交叉注意力变长支持有限
                    # 这里使用mask post-processing的方式
                    
                    # 先使用标准Flash Attention
                    attn_output = self.flash_attn_func(
                        q_reshaped, k_reshaped, v_reshaped,
                        **flash_kwargs
                    )
                    
                    if attn_output is None:
                        raise RuntimeError("Flash Attention returned None")
                    
                    # 重塑回原始格式
                    attn_output = attn_output.view(batch_size, tgt_len, embed_dim)
                    
                    # 后处理：对于被mask的位置，将输出设为0
                    # 这不是最优的，但保证了正确性
                    for batch_idx in range(batch_size):
                        valid_len = mem_seq_lens[batch_idx].item()
                        if valid_len < mem_len:
                            # 需要重新计算这个batch的注意力，但Flash Attention不直接支持
                            # 回退到标准实现
                            raise RuntimeError("Cross attention with variable memory length requires standard attention")
                    
                    # 转换回原始数据类型
                    attn_output = self._convert_back_from_flash_attn_dtype(attn_output, orig_dtype_tgt)
                    return attn_output
                    
            except Exception as e:
                # Flash Attention mask处理失败，回退到标准实现
                raise RuntimeError(f"Flash Attention cross mask处理失败: {e}")
        
        # 没有mask或没有实际padding，使用标准Flash Attention
        attn_output = self.flash_attn_func(
            q_reshaped, k_reshaped, v_reshaped,
            **flash_kwargs
        )
        
        if attn_output is None:
            raise RuntimeError("Flash Attention returned None")
        
        # 重塑回原始格式
        attn_output = attn_output.view(batch_size, tgt_len, embed_dim)
        
        # 转换回原始数据类型
        attn_output = self._convert_back_from_flash_attn_dtype(attn_output, orig_dtype_tgt)
        
        return attn_output
    
    def _standard_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           key_padding_mask: Optional[torch.Tensor] = None, is_self_attn: bool = False):
        """标准PyTorch多头注意力（回退方案）"""
        
        # 为自注意力和交叉注意力创建不同的注意力层
        attn_key = 'self_attn' if is_self_attn else 'cross_attn'
        
        if not hasattr(self, f'_standard_{attn_key}'):
            setattr(self, f'_standard_{attn_key}', nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=self.dropout.p,
                batch_first=True
            ).to(query.device))
        
        attn_layer = getattr(self, f'_standard_{attn_key}')
        attn_output, _ = attn_layer(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask
        )
        
        return attn_output
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        """前向传播"""
        
        # === 自注意力子层 ===
        residual = tgt
        tgt = self.norm1(tgt)
        
        try:
            if self.flash_attn_available:
                attn_output = self._flash_self_attention(tgt, tgt_key_padding_mask)
            else:
                attn_output = self._standard_attention(tgt, tgt, tgt, tgt_key_padding_mask, is_self_attn=True)
        except Exception as e:
            logger.warning(f"解码器自注意力Flash Attention失败，回退到标准注意力: {e}")
            attn_output = self._standard_attention(tgt, tgt, tgt, tgt_key_padding_mask, is_self_attn=True)
        
        tgt = residual + self.dropout1(attn_output)
        
        # === 交叉注意力子层 ===
        residual = tgt
        tgt = self.norm2(tgt)
        
        try:
            if self.flash_attn_available:
                attn_output = self._flash_cross_attention(tgt, memory, memory_key_padding_mask)
            else:
                attn_output = self._standard_attention(tgt, memory, memory, memory_key_padding_mask, is_self_attn=False)
        except Exception as e:
            logger.warning(f"解码器交叉注意力Flash Attention失败，回退到标准注意力: {e}")
            attn_output = self._standard_attention(tgt, memory, memory, memory_key_padding_mask, is_self_attn=False)
        
        tgt = residual + self.dropout2(attn_output)
        
        # === 前馈网络子层 ===
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        
        return tgt
