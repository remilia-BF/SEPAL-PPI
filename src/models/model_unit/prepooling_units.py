"""
预池化单元模块
包含各种预池化方法的可配置实现，用于在标准池化之前对蛋白对进行交互增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple

# 获取根logger以确保日志能正确输出
logger = logging.getLogger("sepal_ppi.prepooling_units")


class NonePrepoolingUnit(nn.Module):
    """
    无预池化单元（恒等变换）
    
    不对输入进行任何修改，直接返回原始序列嵌入
    适用于不需要预池化的场景
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化无预池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(NonePrepoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        
        logger.debug("=" * 50)
        logger.debug("创建无预池化单元（恒等变换）")
        logger.debug(f"嵌入维度: {embedding_dim}")
        logger.debug("=" * 50)
    
    def forward(self, 
                protein1_emb: torch.Tensor, 
                protein2_emb: torch.Tensor,
                protein1_mask: Optional[torch.Tensor] = None,
                protein2_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 恒等变换
        
        Args:
            protein1_emb (torch.Tensor): 蛋白1嵌入 [batch_size, seq_len1, embedding_dim]
            protein2_emb (torch.Tensor): 蛋白2嵌入 [batch_size, seq_len2, embedding_dim]
            protein1_mask (torch.Tensor): 蛋白1注意力掩码 [batch_size, seq_len1]
            protein2_mask (torch.Tensor): 蛋白2注意力掩码 [batch_size, seq_len2]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 未修改的嵌入 (protein1_emb, protein2_emb)
        """
        # 直接返回输入嵌入，不进行任何修改
        return protein1_emb, protein2_emb


class CrossAttentionPoolingUnit(nn.Module):
    """
    交叉注意力池化单元
    
    使用交叉注意力机制增强蛋白对的嵌入表示，对于蛋白A，使用蛋白B作为KV来增强A的表示，
    对蛋白B也同样处理。使用Flash Attention优化计算效率。
    """
    
    def __init__(self,
                 embedding_dim: int,
                 transformer_layers: Optional[list] = None,
                 transformer_heads: int = 8,
                 transformer_layer_num: int = 1,
                 transformer_dropout: float = 0.1,
                 transformer_attention_dropout: float = 0.1,
                 transformer_activation: str = "ReLU",
                 transformer_norm: str = "LayerNorm",
                 transformer_residual: bool = True,
                 **kwargs):
        """
        初始化交叉注意力池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            transformer_layers (list): 隐藏层维度列表，默认为[1280, 512]
                                     用于前馈网络的维度配置
            transformer_heads (int): 多头注意力机制的注意力头数，默认为8
                                   更多头可以关注不同的表示子空间
            transformer_layer_num (int): transformer的层数，默认为1
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
        super(CrossAttentionPoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.transformer_layer_num = transformer_layer_num
        self.transformer_residual = transformer_residual
        self.transformer_attention_dropout = transformer_attention_dropout
        
        if transformer_layers is None:
            transformer_layers = [1280, 512]
        
        self.transformer_layers = transformer_layers
        
        logger.debug("=" * 50)
        logger.debug("开始初始化交叉注意力池化单元")
        logger.debug(f"配置参数: embedding_dim={embedding_dim}, transformer_heads={transformer_heads}, "
                   f"transformer_layer_num={transformer_layer_num}")
        logger.debug("=" * 50)
        
        # 尝试导入flash attention
        self.use_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
            logger.debug("Flash Attention可用，将使用Flash Attention优化交叉注意力计算")
        except ImportError as e:
            logger.warning(f"Flash Attention不可用，将使用标准注意力机制实现交叉注意力: {e}")
        except Exception as e:
            logger.warning(f"Flash Attention导入时发生未知错误，将使用标准注意力机制实现交叉注意力: {e}")
        
        # 添加调试信息
        if self.use_flash_attn:
            logger.debug("Flash Attention成功初始化")
        else:
            logger.warning("使用标准PyTorch注意力机制")
        
        logger.debug("交叉注意力池化单元初始化完成")
        
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
            
            self.transformer_layers_list.append(self_attn_layer)
            self.transformer_layers_list.append(cross_attn_layer)
        
        logger.debug(f"创建交叉注意力池化单元: embedding_dim={embedding_dim}, "
                    f"layers={transformer_layer_num}, heads={transformer_heads}, "
                    f"ffn_dims={transformer_layers}, use_flash_attn={self.use_flash_attn}")

    def _convert_to_flash_attn_dtype(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
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

    def forward(self, 
                protein1_emb: torch.Tensor, 
                protein2_emb: torch.Tensor,
                protein1_mask: Optional[torch.Tensor] = None,
                protein2_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，对蛋白对进行交叉注意力增强
        
        Args:
            protein1_emb (torch.Tensor): 蛋白1嵌入 [batch_size, seq_len1, embedding_dim]
            protein2_emb (torch.Tensor): 蛋白2嵌入 [batch_size, seq_len2, embedding_dim]
            protein1_mask (torch.Tensor): 蛋白1注意力掩码 [batch_size, seq_len1]
            protein2_mask (torch.Tensor): 蛋白2注意力掩码 [batch_size, seq_len2]
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 增强后的嵌入 (enhanced_protein1, enhanced_protein2)
        """
        x1 = protein1_emb
        x2 = protein2_emb
        
        # 处理注意力掩码
        key_padding_mask1 = None
        key_padding_mask2 = None
        
        if protein1_mask is not None:
            key_padding_mask1 = (protein1_mask == 0)
            
        if protein2_mask is not None:
            key_padding_mask2 = (protein2_mask == 0)
        
        # 逐层处理
        for i in range(0, len(self.transformer_layers_list), 2):
            self_attn_dict = self.transformer_layers_list[i]
            cross_attn_dict = self.transformer_layers_list[i+1]
            
            # 获取层组件
            self_norm1 = getattr(self_attn_dict, 'norm1')
            self_norm2 = getattr(self_attn_dict, 'norm2')
            self_norm3 = getattr(self_attn_dict, 'norm3')
            self_ffn = getattr(self_attn_dict, 'ffn')
            self_dropout = getattr(self_attn_dict, 'dropout')
            
            cross_ffn = getattr(cross_attn_dict, 'ffn')
            cross_dropout = getattr(cross_attn_dict, 'dropout')
            
            # 1. 蛋白1自注意力
            residual1 = x1
            x1_norm = self_norm1(x1)
            
            if self.use_flash_attn:
                # 使用Flash Attention实现自注意力
                batch_size, seq_len1, embed_dim = x1_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                assert embed_dim % self.transformer_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除"
                
                # 转换数据类型以满足Flash Attention要求
                x1_norm, orig_dtype1 = self._convert_to_flash_attn_dtype(x1_norm)
                
                # 重塑为Flash Attention格式
                x1_reshaped = x1_norm.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                
                # 自注意力计算
                self_attn_output = self.flash_attn_func(
                    x1_reshaped, x1_reshaped, x1_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                self_attn_output = self_attn_output.view(batch_size, seq_len1, embed_dim)
                
                # 转换回原始数据类型
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype1)
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
                    query=x1_norm,
                    key=x1_norm,
                    value=x1_norm,
                    key_padding_mask=key_padding_mask1
                )
            
            # 蛋白1自注意力残差连接
            if self.transformer_residual:
                x1 = residual1 + self_dropout(self_attn_output)
            else:
                x1 = self_dropout(self_attn_output)
            
            # 2. 蛋白2自注意力
            residual2 = x2
            x2_norm = self_norm1(x2)
            
            if self.use_flash_attn:
                # 使用Flash Attention实现自注意力
                batch_size, seq_len2, embed_dim = x2_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                assert embed_dim % self.transformer_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除"
                
                # 转换数据类型以满足Flash Attention要求
                x2_norm, orig_dtype2 = self._convert_to_flash_attn_dtype(x2_norm)
                
                # 重塑为Flash Attention格式
                x2_reshaped = x2_norm.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                
                # 自注意力计算
                self_attn_output = self.flash_attn_func(
                    x2_reshaped, x2_reshaped, x2_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                self_attn_output = self_attn_output.view(batch_size, seq_len2, embed_dim)
                
                # 转换回原始数据类型
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype2)
            else:
                # 使用已创建的标准自注意力层
                self_attn_output, _ = self_attention_layer(
                    query=x2_norm,
                    key=x2_norm,
                    value=x2_norm,
                    key_padding_mask=key_padding_mask2
                )
            
            # 蛋白2自注意力残差连接
            if self.transformer_residual:
                x2 = residual2 + self_dropout(self_attn_output)
            else:
                x2 = self_dropout(self_attn_output)
            
            # 3. 蛋白1对蛋白2的交叉注意力
            residual1 = x1
            x1_norm = self_norm2(x1)  # 使用第二个归一化层
            x2_norm = self_norm2(x2)  # 蛋白2也需要归一化
            
            if self.use_flash_attn:
                # 使用Flash Attention实现交叉注意力
                batch_size, seq_len1, embed_dim = x1_norm.size()
                _, seq_len2, _ = x2_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                # 转换数据类型以满足Flash Attention要求
                q1, orig_dtype_q1 = self._convert_to_flash_attn_dtype(x1_norm)
                k2, orig_dtype_k2 = self._convert_to_flash_attn_dtype(x2_norm)
                v2, orig_dtype_v2 = self._convert_to_flash_attn_dtype(x2_norm)
                
                # 重塑为Flash Attention格式
                q1 = q1.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                k2 = k2.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                v2 = v2.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                
                # 交叉注意力计算 (1查询，2的KV)
                cross_attn_output = self.flash_attn_func(
                    q1, k2, v2,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                cross_attn_output = cross_attn_output.view(batch_size, seq_len1, embed_dim)
                
                # 转换回原始数据类型 (使用查询的原始类型)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_q1)
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
                    query=x1_norm,
                    key=x2_norm,
                    value=x2_norm,
                    key_padding_mask=key_padding_mask2
                )
            
            # 蛋白1交叉注意力残差连接
            if self.transformer_residual:
                x1 = residual1 + cross_dropout(cross_attn_output)
            else:
                x1 = cross_dropout(cross_attn_output)
            
            # 4. 蛋白2对蛋白1的交叉注意力
            residual2 = x2
            x1_norm = self_norm2(x1)  # 蛋白1再次归一化
            x2_norm = self_norm2(x2)  # 蛋白2再次归一化
            
            if self.use_flash_attn:
                # 使用Flash Attention实现交叉注意力
                batch_size, seq_len2, embed_dim = x2_norm.size()
                _, seq_len1, _ = x1_norm.size()
                head_dim = embed_dim // self.transformer_heads
                
                # 转换数据类型以满足Flash Attention要求
                q2, orig_dtype_q2 = self._convert_to_flash_attn_dtype(x2_norm)
                k1, orig_dtype_k1 = self._convert_to_flash_attn_dtype(x1_norm)
                v1, orig_dtype_v1 = self._convert_to_flash_attn_dtype(x1_norm)
                
                # 重塑为Flash Attention格式
                q2 = q2.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                k1 = k1.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                v1 = v1.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                
                # 交叉注意力计算 (2查询，1的KV)
                cross_attn_output = self.flash_attn_func(
                    q2, k1, v1,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                cross_attn_output = cross_attn_output.view(batch_size, seq_len2, embed_dim)
                
                # 转换回原始数据类型 (使用查询的原始类型)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_q2)
            else:
                # 使用已创建的交叉注意力层
                cross_attn_output, _ = cross_attention_layer(
                    query=x2_norm,
                    key=x1_norm,
                    value=x1_norm,
                    key_padding_mask=key_padding_mask1
                )
            
            # 蛋白2交叉注意力残差连接
            if self.transformer_residual:
                x2 = residual2 + cross_dropout(cross_attn_output)
            else:
                x2 = cross_dropout(cross_attn_output)
            
            # 5. 蛋白1前馈网络
            residual1 = x1
            x1_norm = self_norm3(x1)  # 使用第三个归一化层
            ffn_output = self_ffn(x1_norm)
            
            if self.transformer_residual:
                x1 = residual1 + self_dropout(ffn_output)
            else:
                x1 = self_dropout(ffn_output)
            
            # 6. 蛋白2前馈网络
            residual2 = x2
            x2_norm = self_norm3(x2)  # 使用第三个归一化层
            ffn_output = self_ffn(x2_norm)  # 使用相同的FFN结构
            
            if self.transformer_residual:
                x2 = residual2 + self_dropout(ffn_output)
            else:
                x2 = self_dropout(ffn_output)
        
        return x1, x2


class CrossAttentionPoolingOnlyAvgUnit(CrossAttentionPoolingUnit):
    """
    交叉注意力池化（仅平均KV）

    使用对方蛋白的平均池化向量作为Key/Value，对每个残基token进行交叉注意力增强：
    - 蛋白1的token以蛋白2的平均池化作为K/V
    - 蛋白2的token以蛋白1的平均池化作为K/V
    其他部分（自注意力、FFN、残差、归一化、dropout）与基类一致。
    """

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        对序列维度做掩码平均池化。
        Args:
            x: [batch, seq_len, embedding_dim]
            mask: [batch, seq_len]，1为有效，0为padding；若为None，等价于全1
        Returns:
            avg: [batch, 1, embedding_dim]
        """
        if mask is None:
            avg = x.mean(dim=1, keepdim=True)
            return avg
        # 转为浮点权重 [batch, seq_len, 1]
        weights = mask.float().unsqueeze(-1)
        summed = (x * weights).sum(dim=1, keepdim=True)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        avg = summed / denom
        return avg

    def forward(self,
                protein1_emb: torch.Tensor,
                protein2_emb: torch.Tensor,
                protein1_mask: Optional[torch.Tensor] = None,
                protein2_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = protein1_emb
        x2 = protein2_emb

        key_padding_mask1 = None if protein1_mask is None else (protein1_mask == 0)
        key_padding_mask2 = None if protein2_mask is None else (protein2_mask == 0)

        for i in range(0, len(self.transformer_layers_list), 2):
            self_attn_dict = self.transformer_layers_list[i]
            cross_attn_dict = self.transformer_layers_list[i+1]

            self_norm1 = getattr(self_attn_dict, 'norm1')
            self_norm2 = getattr(self_attn_dict, 'norm2')
            self_norm3 = getattr(self_attn_dict, 'norm3')
            self_ffn = getattr(self_attn_dict, 'ffn')
            self_dropout = getattr(self_attn_dict, 'dropout')

            cross_ffn = getattr(cross_attn_dict, 'ffn')
            cross_dropout = getattr(cross_attn_dict, 'dropout')

            # 1. 蛋白1自注意力
            residual1 = x1
            x1_norm = self_norm1(x1)
            if self.use_flash_attn:
                batch_size, seq_len1, embed_dim = x1_norm.size()
                head_dim = embed_dim // self.transformer_heads
                assert embed_dim % self.transformer_heads == 0, (
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除")
                x1_norm, orig_dtype1 = self._convert_to_flash_attn_dtype(x1_norm)
                x1_reshaped = x1_norm.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                self_attn_output = self.flash_attn_func(
                    x1_reshaped, x1_reshaped, x1_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                self_attn_output = self_attn_output.view(batch_size, seq_len1, embed_dim)
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype1)
            else:
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
                    query=x1_norm,
                    key=x1_norm,
                    value=x1_norm,
                    key_padding_mask=key_padding_mask1
                )
            x1 = residual1 + self_dropout(self_attn_output) if self.transformer_residual else self_dropout(self_attn_output)

            # 2. 蛋白2自注意力
            residual2 = x2
            x2_norm = self_norm1(x2)
            if self.use_flash_attn:
                batch_size, seq_len2, embed_dim = x2_norm.size()
                head_dim = embed_dim // self.transformer_heads
                assert embed_dim % self.transformer_heads == 0, (
                    f"嵌入维度{embed_dim}不能被头数{self.transformer_heads}整除")
                x2_norm, orig_dtype2 = self._convert_to_flash_attn_dtype(x2_norm)
                x2_reshaped = x2_norm.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                self_attn_output = self.flash_attn_func(
                    x2_reshaped, x2_reshaped, x2_reshaped,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                if self_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                self_attn_output = self_attn_output.view(batch_size, seq_len2, embed_dim)
                self_attn_output = self._convert_back_from_flash_attn_dtype(self_attn_output, orig_dtype2)
            else:
                self_attn_output, _ = self_attention_layer(
                    query=x2_norm,
                    key=x2_norm,
                    value=x2_norm,
                    key_padding_mask=key_padding_mask2
                )
            x2 = residual2 + self_dropout(self_attn_output) if self.transformer_residual else self_dropout(self_attn_output)

            # 3. 平均池化得到对方蛋白的K/V
            x1_norm = self_norm2(x1)
            x2_norm = self_norm2(x2)
            avg2 = self._masked_mean(x2_norm, protein2_mask)  # [B,1,E]
            avg1 = self._masked_mean(x1_norm, protein1_mask)  # [B,1,E]

            # 4. 蛋白1 <- 蛋白2平均的交叉注意力
            residual1 = x1
            if self.use_flash_attn:
                batch_size, seq_len1, embed_dim = x1_norm.size()
                head_dim = embed_dim // self.transformer_heads
                q1, orig_dtype_q1 = self._convert_to_flash_attn_dtype(x1_norm)
                k2, _ = self._convert_to_flash_attn_dtype(avg2)
                v2, _ = self._convert_to_flash_attn_dtype(avg2)
                q1 = q1.reshape(batch_size, seq_len1, self.transformer_heads, head_dim)
                k2 = k2.reshape(batch_size, 1, self.transformer_heads, head_dim)
                v2 = v2.reshape(batch_size, 1, self.transformer_heads, head_dim)
                cross_attn_output = self.flash_attn_func(
                    q1, k2, v2,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                cross_attn_output = cross_attn_output.view(batch_size, seq_len1, embed_dim)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_q1)
            else:
                if not hasattr(cross_attn_dict, 'cross_attention_avg') or cross_attn_dict.cross_attention_avg is None:
                    cross_attention_layer = nn.MultiheadAttention(
                        embed_dim=self.embedding_dim,
                        num_heads=self.transformer_heads,
                        dropout=self.transformer_attention_dropout,
                        batch_first=True
                    )
                    setattr(cross_attn_dict, 'cross_attention_avg', cross_attention_layer)
                else:
                    cross_attention_layer = getattr(cross_attn_dict, 'cross_attention_avg')
                cross_attn_output, _ = cross_attention_layer(
                    query=x1_norm,
                    key=avg2,
                    value=avg2,
                    key_padding_mask=None
                )
            x1 = residual1 + cross_dropout(cross_attn_output) if self.transformer_residual else cross_dropout(cross_attn_output)

            # 5. 蛋白2 <- 蛋白1平均的交叉注意力
            residual2 = x2
            # 经过上一步更新的x1/x2，再规范化一次
            x1_norm = self_norm2(x1)
            x2_norm = self_norm2(x2)
            avg1 = self._masked_mean(x1_norm, protein1_mask)
            if self.use_flash_attn:
                batch_size, seq_len2, embed_dim = x2_norm.size()
                head_dim = embed_dim // self.transformer_heads
                q2, orig_dtype_q2 = self._convert_to_flash_attn_dtype(x2_norm)
                k1, _ = self._convert_to_flash_attn_dtype(avg1)
                v1, _ = self._convert_to_flash_attn_dtype(avg1)
                q2 = q2.reshape(batch_size, seq_len2, self.transformer_heads, head_dim)
                k1 = k1.reshape(batch_size, 1, self.transformer_heads, head_dim)
                v1 = v1.reshape(batch_size, 1, self.transformer_heads, head_dim)
                cross_attn_output = self.flash_attn_func(
                    q2, k1, v1,
                    dropout_p=self.transformer_attention_dropout if self.training else 0.0,
                    causal=False
                )
                if cross_attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                cross_attn_output = cross_attn_output.view(batch_size, seq_len2, embed_dim)
                cross_attn_output = self._convert_back_from_flash_attn_dtype(cross_attn_output, orig_dtype_q2)
            else:
                cross_attn_output, _ = cross_attention_layer(
                    query=x2_norm,
                    key=avg1,
                    value=avg1,
                    key_padding_mask=None
                )
            x2 = residual2 + cross_dropout(cross_attn_output) if self.transformer_residual else cross_dropout(cross_attn_output)

            # 6. FFN
            residual1 = x1
            x1_norm = self_norm3(x1)
            ffn_output = self_ffn(x1_norm)
            x1 = residual1 + self_dropout(ffn_output) if self.transformer_residual else self_dropout(ffn_output)

            residual2 = x2
            x2_norm = self_norm3(x2)
            ffn_output = self_ffn(x2_norm)
            x2 = residual2 + self_dropout(ffn_output) if self.transformer_residual else self_dropout(ffn_output)

        return x1, x2


def create_prepairing_unit(prepairing_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建预池化单元
    
    Args:
        prepairing_config (Dict): 预池化配置，包含method和其他参数
                                 必须包含: method (str) - 预池化方法名称
                                 可选包含: embedding_dim, transformer_layers等
    
    Returns:
        nn.Module: 预池化单元实例
    
    Raises:
        ValueError: 当预池化方法不支持时
    
    Examples:
        >>> config = {
        ...     'method': 'none',
        ...     'embedding_dim': 1280
        ... }
        >>> prepairing = create_prepairing_unit(config)
        
        >>> config = {
        ...     'method': 'cross_attention_pooling', 
        ...     'embedding_dim': 1280,
        ...     'transformer_layers': [1280, 512],
        ...     'transformer_heads': 10,
        ...     'transformer_layer_num': 1
        ... }
        >>> prepairing = create_prepairing_unit(config)
    """
    method = prepairing_config.get('method', 'none')
    embedding_dim = prepairing_config.get('embedding_dim', None)
    
    if embedding_dim is None:
        raise ValueError("预池化单元需要指定embedding_dim参数")
    
    logger.debug("=" * 60)
    logger.debug("创建预池化单元")
    logger.debug(f"方法: {method}")
    logger.debug(f"嵌入维度: {embedding_dim}")
    logger.debug(f"配置: {prepairing_config}")
    logger.debug("=" * 60)
    
    # 移除method和embedding_dim，将其余参数传递给具体的预池化单元
    kwargs = {k: v for k, v in prepairing_config.items() if k not in ['method', 'embedding_dim']}
    
    if method == 'none':
        unit = NonePrepoolingUnit(embedding_dim=embedding_dim, **kwargs)
        logger.debug("创建无预池化单元完成")
        return unit
    elif method == 'cross_attention_pooling':
        unit = CrossAttentionPoolingUnit(embedding_dim=embedding_dim, **kwargs)
        logger.debug("创建交叉注意力预池化单元完成")
        return unit
    elif method == 'cross_attention_pooling_onlyavg':
        unit = CrossAttentionPoolingOnlyAvgUnit(embedding_dim=embedding_dim, **kwargs)
        logger.debug("创建交叉注意力（仅平均KV）预池化单元完成")
        return unit
    else:
        error_msg = f"不支持的预池化方法: {method}. 支持的方法: none, cross_attention_pooling, cross_attention_pooling_onlyavg"
        logger.error(error_msg)
        raise ValueError(error_msg)