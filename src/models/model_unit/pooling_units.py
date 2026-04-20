"""
池化单元模块
包含各种池化方法的可配置实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any
import math
import json
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AveragePoolingUnit(nn.Module):
    """
    平均池化单元
    
    将序列级嵌入 [batch_size, seq_len, embedding_dim] 转换为
    固定大小表示 [batch_size, embedding_dim]
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化平均池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(AveragePoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        
        logger.debug(f"Create AveragePoolingUnit: embedding_dim={embedding_dim}")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings (torch.Tensor): 输入嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len] 
                                         (1表示真实token，0表示padding)
        
        Returns:
            torch.Tensor: 池化后的嵌入 [batch_size, embedding_dim]
        """
        if attention_mask is None:
            # 简单平均，如果没有掩码
            return embeddings.mean(dim=1)
        
        # 带掩码的平均池化
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        masked_embeddings = embeddings * mask
        
        # 求和并除以实际长度
        summed = masked_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        lengths = mask.sum(dim=1)  # [batch_size, 1]
        
        # 避免除零
        lengths = torch.clamp(lengths, min=1.0)
        
        return summed / lengths


class MaxPoolingUnit(nn.Module):
    """
    最大池化单元
    
    将序列级嵌入进行最大池化
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化最大池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(MaxPoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        
        logger.debug(f"Create MaxPoolingUnit: embedding_dim={embedding_dim}")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings (torch.Tensor): 输入嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
        
        Returns:
            torch.Tensor: 池化后的嵌入 [batch_size, embedding_dim]
        """
        if attention_mask is None:
            # 简单最大池化
            pooled, _ = embeddings.max(dim=1)
            return pooled
        
        # 带掩码的最大池化
        # 将padding位置设为-inf以在max操作中被忽略
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        masked_embeddings = embeddings + (1.0 - mask) * (-1e9)
        
        pooled, _ = masked_embeddings.max(dim=1)
        return pooled


class AttentionPoolingUnit(nn.Module):
    """
    注意力池化单元
    
    使用学习到的注意力权重对序列进行加权平均
    注意力矩阵可以保存用于后续可解释性分析
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 attention_num: int = 1,
                 dropout: float = 0.1,
                 save_attention: bool = True,
                 **kwargs):
        """
        初始化注意力池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            attention_num (int): 注意力头数，默认为1
            dropout (float): dropout率，默认为0.1，用于防止过拟合
            save_attention (bool): 是否保存注意力权重，默认为True，用于可解释性分析
            **kwargs: 额外参数
        """
        super(AttentionPoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_num = attention_num
        self.save_attention = save_attention

        # 
        # :
        #   type: temperature_original | temperature_rel | top_k
        #   temperature: float
        #   alpha, min_temperature, max_temperature:  temperature_rel
        #   top_k, topk_sharpness:  top_k
        attention_config = kwargs.pop('attention_config', {}) or {}
        self.attention_type = attention_config.get('type', 'temperature_original')

        # 
        self.base_temperature = float(attention_config.get('temperature', 1.0))

        # temperature_rel 
        self.alpha = float(attention_config.get('alpha', 0.5))
        self.min_temperature = float(attention_config.get('min_temperature', 0.5))
        self.max_temperature = float(attention_config.get('max_temperature', 2.0))

        # top_k 
        self.top_k = int(attention_config.get('top_k', 32))
        self.topk_sharpness = float(attention_config.get('topk_sharpness', 10.0))

        #  (, )
        self.use_layer_norm = bool(attention_config.get('use_layer_norm', False))
        self.residual_connection = bool(attention_config.get('residual_connection', False))

        # 
        # 
        self.attention_weights = nn.Linear(embedding_dim, attention_num)

        # 
        self.layer_norm = nn.LayerNorm(attention_num) if self.use_layer_norm else None

        # dropout
        self.dropout = nn.Dropout(dropout)

        #  ( )
        self.protein1_attention_weights = None
        self.protein2_attention_weights = None
        self._call_count = 0  # 

        logger.debug(
            f"Create AttentionPoolingUnit: embedding_dim={embedding_dim}, "
            f"attention_num={attention_num}, dropout={dropout}"
        )
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings (torch.Tensor): 输入嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
        
        Returns:
            torch.Tensor: 池化后的嵌入 [batch_size, embedding_dim]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # 计算注意力分数 [batch_size, seq_len, attention_num]
        attention_scores = self.attention_weights(embeddings)

        # 可选的层归一化
        if self.layer_norm is not None:
            attention_scores = self.layer_norm(attention_scores)
        
        # 应用dropout
        attention_scores = self.dropout(attention_scores)
        
        # 如果有注意力掩码，则掩盖padding位置
        if attention_mask is not None:
            # 将padding位置设为-inf，使softmax后为0
            mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            attention_scores = attention_scores + (1.0 - mask) * (-1e9)

        # 计算温度 (temperature_original / temperature_rel)
        if self.attention_type == 'temperature_rel':
            # 根据序列长度 L 计算 T(L)=max(min_temperature, min(max_temperature, alpha * log(L)))
            # 这里使用非 padding token 数量作为 L
            if attention_mask is not None:
                lengths = attention_mask.float().sum(dim=1)  # [batch_size]
            else:
                lengths = torch.full((batch_size,), float(seq_len), device=embeddings.device)

            # 避免 log(0)
            lengths = torch.clamp(lengths, min=1.0)
            dynamic_temp = self.alpha * torch.log(lengths)
            dynamic_temp = torch.clamp(dynamic_temp,
                                       min=self.min_temperature,
                                       max=self.max_temperature)
            # 形状调整为 [batch_size, 1, 1]
            temperature = dynamic_temp.view(batch_size, 1, 1)
        else:
            # 固定温度
            temperature = torch.tensor(self.base_temperature,
                                       device=embeddings.device,
                                       dtype=embeddings.dtype).view(1, 1, 1)

        # 防止温度过小导致数值不稳定
        temperature = torch.clamp(temperature, min=1e-4)

        # temperature_original / temperature_rel：用温度缩放 logits
        scaled_scores = attention_scores / temperature

        if self.attention_type == 'top_k':
            # 使用可微分 Top-K 近似构造稀疏注意力权重
            # 这里使用 Gumbel-softmax 风格的“温度+sharpness+hard mask”近似
            # 计算 softmax 权重
            soft_weights = F.softmax(self.topk_sharpness * scaled_scores, dim=1)  # [B, L, H]

            # 计算 per-head 的 Top-K 阈值（在概率空间中）
            # 先在 logits 上做一个近似：根据 soft_weights 的排序来构造 mask
            # 为避免排序算子不可微，我们在前向中仍然使用真实排序，但梯度主要来自 soft_weights
            with torch.no_grad():
                # 找到每个 head 的 top_k 索引
                k = min(self.top_k, seq_len)
                # soft_weights: [B, L, H] -> [B, H, L]
                sw = soft_weights.permute(0, 2, 1)
                # topk 返回 [values, indices]，形状 [B, H, k]
                _, topk_indices = torch.topk(sw, k=k, dim=-1)

                # 构造 one-hot mask: [B, H, L]
                mask_topk = torch.zeros_like(sw)
                mask_topk.scatter_(dim=-1, index=topk_indices, value=1.0)
                # 转回 [B, L, H]
                mask_topk = mask_topk.permute(0, 2, 1)

            # 将 mask 视作常数，结合 soft_weights 得到近似的 top-k 权重
            attention_weights = soft_weights * mask_topk

            # 归一化，以确保每个 head 的权重和为 1
            attention_weights_sum = attention_weights.sum(dim=1, keepdim=True) + 1e-8
            attention_weights = attention_weights / attention_weights_sum
        else:
            # 计算注意力权重 [batch_size, seq_len, attention_num]
            attention_weights = F.softmax(scaled_scores, dim=1)
        
        # 保存注意力权重用于可解释性分析
        if self.save_attention:
            # 分别保存蛋白质1和蛋白质2的注意力权重
            self._call_count += 1
            if self._call_count % 2 == 1:
                # 奇数次调用 - protein1
                self.protein1_attention_weights = attention_weights.detach().cpu()
            else:
                # 偶数次调用 - protein2
                self.protein2_attention_weights = attention_weights.detach().cpu()
        
        # 如果有多个注意力头，对每个头分别计算加权平均，然后平均
        if self.attention_num == 1:
            # 单头注意力，直接加权平均
            weighted_embeddings = embeddings * attention_weights  # [batch_size, seq_len, embedding_dim]
            pooled = weighted_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        else:
            # 多头注意力，对每个头分别计算
            pooled_heads = []
            for head_idx in range(self.attention_num):
                head_weights = attention_weights[:, :, head_idx:head_idx+1]  # [batch_size, seq_len, 1]
                weighted_embeddings = embeddings * head_weights
                head_pooled = weighted_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
                pooled_heads.append(head_pooled)
            
            # 平均所有头的结果
            pooled = torch.stack(pooled_heads, dim=0).mean(dim=0)  # [batch_size, embedding_dim]
        
        return pooled
    
    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取最后一次前向传播的注意力权重
        
        Returns:
            Dict[str, torch.Tensor]: 包含protein1和protein2注意力权重的字典，或None
        """
        if not self.save_attention:
            return None
        
        result = {}
        if self.protein1_attention_weights is not None:
            result['protein1'] = self.protein1_attention_weights
        if self.protein2_attention_weights is not None:
            result['protein2'] = self.protein2_attention_weights
        
        return result if result else None
    
    def reset_attention_tracking(self):
        """
        重置注意力权重跟踪，用于新的样本
        """
        self._call_count = 0
        self.protein1_attention_weights = None
        self.protein2_attention_weights = None


class NonePoolingUnit(nn.Module):
    """
    无池化单元（恒等变换）
    
    不对输入进行任何修改，直接返回原始序列嵌入
    适用于需要保留完整序列信息的场景
    """
    
    def __init__(self, embedding_dim: int, **kwargs):
        """
        初始化无池化单元
        
        Args:
            embedding_dim (int): 嵌入维度
            **kwargs: 额外参数 (当前未使用，为扩展性保留)
        """
        super(NonePoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        
        logger.debug(f"Create NonePoolingUnit: embedding_dim={embedding_dim}")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播 - 恒等变换
        
        Args:
            embeddings (torch.Tensor): 输入嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len] (未使用)
        
        Returns:
            torch.Tensor: 未修改的嵌入 [batch_size, seq_len, embedding_dim]
        """
        # 直接返回输入嵌入，不进行任何修改
        return embeddings


class AdvancedAttentionPoolingUnit(nn.Module):
    """
    高级注意力池化单元

    支持：
    1) Soft prior 注入: e_ij = Linear(h_j) + lambda * Prior_i(j)
    2) Hard mask: zero_mask / none_zero_mask
    3) none_Mask 语义: how2mask=none -> 该头使用平均池化；how2mask=attention -> 标准注意力
    4) 导出 protein1/protein2 的注意力权重和每头池化表示
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_num: int = 8,
        dropout: float = 0.1,
        save_attention: bool = True,
        **kwargs,
    ):
        super(AdvancedAttentionPoolingUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_num = int(attention_num)
        self.save_attention = save_attention

        attention_config = kwargs.pop('attention_config', {}) or {}
        self.attention_type = attention_config.get('type', 'temperature_original')
        self.base_temperature = float(attention_config.get('temperature', 1.0))
        self.alpha = float(attention_config.get('alpha', 0.5))
        self.min_temperature = float(attention_config.get('min_temperature', 0.5))
        self.max_temperature = float(attention_config.get('max_temperature', 2.0))
        self.top_k = int(attention_config.get('top_k', 32))
        self.topk_sharpness = float(attention_config.get('topk_sharpness', 10.0))
        self.use_layer_norm = bool(attention_config.get('use_layer_norm', False))
        self.prior_lambda = float(attention_config.get('prior_lambda', 1.0))

        self.all_feature_folder = kwargs.get('all_feature_folder')
        self.head_cfg_map = kwargs.get('attention_head', {}) or {}

        self.attention_weights_layer = nn.Linear(embedding_dim, self.attention_num)
        self.layer_norm = nn.LayerNorm(self.attention_num) if self.use_layer_norm else None
        self.dropout = nn.Dropout(dropout)

        self._prior_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._head_specs = self._build_head_specs()

        self.protein1_attention_weights = None
        self.protein2_attention_weights = None
        self.protein1_head_pooled = None
        self.protein2_head_pooled = None
        self._live_protein1_head_pooled = None
        self._live_protein2_head_pooled = None
        self._call_count = 0

        logger.debug(
            f"Create AdvancedAttentionPoolingUnit: embedding_dim={embedding_dim}, attention_num={self.attention_num}, "
            f"dropout={dropout}, prior_lambda={self.prior_lambda}, feature_root={self.all_feature_folder}"
        )

    def _build_head_specs(self) -> list:
        specs = []
        for idx in range(self.attention_num):
            key = f'head{idx + 1}'
            cfg = self.head_cfg_map.get(key, {}) if isinstance(self.head_cfg_map, dict) else {}
            enabled = bool(cfg.get('enabled', True))
            head_type = str(cfg.get('type', 'none_Mask'))
            if head_type == 'none':
                head_type = 'none_Mask'
            how2mask = str(cfg.get('how2mask', 'attention'))
            file_path = cfg.get('file_path', None)
            raw_dim = int(cfg.get('raw_dim', 1))
            specs.append(
                {
                    'enabled': enabled,
                    'type': head_type,
                    'how2mask': how2mask,
                    'file_path': file_path,
                    'raw_dim': raw_dim,
                }
            )
        return specs

    def _resolve_feature_path(self, file_path: Any) -> Optional[Path]:
        if file_path is None:
            return None
        path_str = str(file_path).strip()
        if path_str.lower() in ('none', 'null', ''):
            return None

        p = Path(path_str)
        if p.exists():
            return p

        if self.all_feature_folder:
            candidate = Path(self.all_feature_folder) / p
            if candidate.exists():
                return candidate

        return p if p.exists() else None

    def _parse_numeric_list(self, raw: Any) -> np.ndarray:
        if raw is None:
            return np.array([], dtype=np.float32)
        if isinstance(raw, list):
            try:
                return np.array(raw, dtype=np.float32)
            except Exception:
                return np.array([], dtype=np.float32)
        if isinstance(raw, (int, float)):
            return np.array([float(raw)], dtype=np.float32)
        if isinstance(raw, str):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if not nums:
                return np.array([], dtype=np.float32)
            return np.array([float(v) for v in nums], dtype=np.float32)
        return np.array([], dtype=np.float32)

    def _parse_secondary_structure(self, raw_value: Any) -> np.ndarray:
        if raw_value is None:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(raw_value, list):
            arr = np.array(raw_value, dtype=np.float32)
            if arr.ndim == 1:
                usable = (arr.shape[0] // 3) * 3
                arr = arr[:usable].reshape(-1, 3)
            elif arr.ndim == 2 and arr.shape[1] != 3:
                usable = (arr.shape[1] // 3) * 3
                arr = arr[:, :usable].reshape(arr.shape[0], -1, 3)[:, 0, :]
            return arr.astype(np.float32)

        text = str(raw_value).replace(' ', '').replace('\n', '')
        segs = text.split('],[')
        if not segs:
            return np.zeros((0, 3), dtype=np.float32)
        segs[0] = segs[0].lstrip('[')
        segs[-1] = segs[-1].rstrip(']')
        rows = []
        for seg in segs:
            vals = [float(x) for x in seg.split(',') if x != '']
            if len(vals) != 3:
                vals = [0.0, 0.0, 1.0]
            rows.append(vals)
        return np.array(rows, dtype=np.float32)

    def _parse_feature_item(self, item: Dict[str, Any], raw_dim: int) -> np.ndarray:
        if 'secondary_structure' in item:
            return self._parse_secondary_structure(item.get('secondary_structure'))

        preferred_fields = ['sasa', 'hydrophobicity', 'embeddings', 'values', 'score']
        values = None
        for field in preferred_fields:
            if field in item:
                values = self._parse_numeric_list(item.get(field))
                break
        if values is None:
            values = np.array([], dtype=np.float32)

        if raw_dim <= 1:
            return values.reshape(-1, 1).astype(np.float32)

        usable = (len(values) // raw_dim) * raw_dim
        if usable <= 0:
            return np.zeros((0, raw_dim), dtype=np.float32)
        return values[:usable].reshape(-1, raw_dim).astype(np.float32)

    def _load_prior_cache(self, head_idx: int, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if head_idx in self._prior_cache:
            return self._prior_cache[head_idx]

        cache: Dict[str, np.ndarray] = {}
        resolved_path = self._resolve_feature_path(spec.get('file_path'))
        if resolved_path is None:
            self._prior_cache[head_idx] = cache
            return cache

        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw_dim = int(spec.get('raw_dim', 1))
            for item in data:
                if not isinstance(item, dict):
                    continue
                pid = item.get('protein_id')
                if pid is None:
                    continue
                cache[str(pid)] = self._parse_feature_item(item, raw_dim)
            logger.info(f"Loaded prior feature head{head_idx + 1} from {resolved_path}, proteins={len(cache)}")
        except Exception as e:
            logger.warning(f"Failed loading prior feature for head{head_idx + 1} from {resolved_path}: {e}")

        self._prior_cache[head_idx] = cache
        return cache

    def _prepare_batch_prior(
        self,
        head_idx: int,
        spec: Dict[str, Any],
        protein_ids: Optional[list],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if protein_ids is None:
            return None

        cache = self._load_prior_cache(head_idx, spec)
        raw_dim = int(spec.get('raw_dim', 1))
        head_type = str(spec.get('type', 'none_Mask'))
        batch_size = len(protein_ids)
        if batch_size <= 0:
            return None

        effective_dim = 1 if (head_type == 'Loop_Mask' and raw_dim == 3) else max(raw_dim, 1)
        prior = torch.zeros((batch_size, seq_len, effective_dim), device=device, dtype=dtype)
        for i, pid in enumerate(protein_ids):
            arr = cache.get(str(pid))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if raw_dim > 1 and arr.shape[1] != raw_dim:
                usable = (arr.shape[1] // raw_dim) * raw_dim
                if usable <= 0:
                    continue
                arr = arr[:, :usable].reshape(arr.shape[0], -1, raw_dim)[:, 0, :]

            # Loop_Mask: 将 [helix, sheet, loop] 映射为二值 [0/1]
            if head_type == 'Loop_Mask' and raw_dim == 3:
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    arr = (arr[:, 2:3] > 0.5).astype(np.float32)
                elif arr.ndim == 1 and arr.shape[0] >= 3:
                    arr = (arr.reshape(-1, 3)[:, 2:3] > 0.5).astype(np.float32)
                else:
                    arr = np.zeros((arr.shape[0], 1), dtype=np.float32)

            actual_len = min(seq_len, arr.shape[0])
            if actual_len <= 0:
                continue
            prior[i, :actual_len, :arr.shape[1]] = torch.from_numpy(arr[:actual_len]).to(device=device, dtype=dtype)

        return prior

    def _build_soft_prior(self, prior_tensor: torch.Tensor, how2mask: str, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if prior_tensor.dim() == 3 and prior_tensor.size(-1) > 1:
            scalar = prior_tensor.mean(dim=-1)
        else:
            scalar = prior_tensor.squeeze(-1)

        if how2mask == 'lower_mask':
            scalar = -scalar

        if attention_mask is not None:
            valid = attention_mask.float()
            scalar = scalar * valid
        else:
            valid = torch.ones_like(scalar)

        # 样本内归一化到 [0, 1]
        min_val = torch.where(valid > 0, scalar, torch.full_like(scalar, float('inf'))).min(dim=1, keepdim=True).values
        max_val = torch.where(valid > 0, scalar, torch.full_like(scalar, float('-inf'))).max(dim=1, keepdim=True).values
        denom = torch.clamp(max_val - min_val, min=1e-8)
        normalized = (scalar - min_val) / denom
        normalized = torch.where(valid > 0, normalized, torch.zeros_like(normalized))
        return normalized

    def _build_hard_invalid_mask(self, prior_tensor: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
        how2mask = str(spec.get('how2mask', 'zero_mask'))
        raw_dim = int(spec.get('raw_dim', 1))
        head_type = str(spec.get('type', 'Hard_Mask'))

        # Loop_Mask 已在 _prepare_batch_prior 映射为 [0,1]，统一按 zero/none_zero 处理
        if head_type == 'Loop_Mask':
            scalar = prior_tensor.squeeze(-1)
            if how2mask == 'none_zero_mask':
                # 屏蔽 loop（值为1）
                return scalar > 0.5
            # zero_mask：屏蔽非loop（值为0）
            return scalar <= 0.5

        if raw_dim == 3 and prior_tensor.size(-1) >= 3 and how2mask == 'zero_mask':
            # 二级结构特征语义：[helix, sheet, loop]，仅屏蔽 loop（最后一位=1）
            invalid = prior_tensor[..., 2] > 0.5
            return invalid

        if prior_tensor.dim() == 3 and prior_tensor.size(-1) > 1:
            scalar = prior_tensor.mean(dim=-1)
        else:
            scalar = prior_tensor.squeeze(-1)

        if how2mask == 'none_zero_mask':
            # 屏蔽非零位置
            return torch.abs(scalar) > 1e-8
        # 默认 zero_mask：屏蔽零值位置
        return torch.abs(scalar) <= 1e-8

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protein_ids: Optional[list] = None,
        protein_tag: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape

        attention_scores = self.attention_weights_layer(embeddings)
        if self.layer_norm is not None:
            attention_scores = self.layer_norm(attention_scores)
        attention_scores = self.dropout(attention_scores)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            attention_scores = attention_scores + (1.0 - mask) * (-1e9)

        avg_pool_heads = set()
        for h in range(self.attention_num):
            spec = self._head_specs[h]
            if not spec.get('enabled', True):
                attention_scores[:, :, h] = -1e9
                continue

            head_type = str(spec.get('type', 'none_Mask'))
            how2mask = str(spec.get('how2mask', 'attention'))
            prior_tensor = self._prepare_batch_prior(
                head_idx=h,
                spec=spec,
                protein_ids=protein_ids,
                seq_len=seq_len,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

            if head_type == 'none_Mask' and how2mask == 'none':
                avg_pool_heads.add(h)
                continue

            if head_type == 'Soft_prior_weighting' and prior_tensor is not None:
                prior_scalar = self._build_soft_prior(prior_tensor, how2mask, attention_mask)
                attention_scores[:, :, h] = attention_scores[:, :, h] + self.prior_lambda * prior_scalar

            if head_type in ('Hard_Mask', 'Loop_Mask') and prior_tensor is not None:
                invalid = self._build_hard_invalid_mask(prior_tensor, spec)
                attention_scores[:, :, h] = attention_scores[:, :, h].masked_fill(invalid, -1e9)

        if self.attention_type == 'temperature_rel':
            if attention_mask is not None:
                lengths = attention_mask.float().sum(dim=1)
            else:
                lengths = torch.full((batch_size,), float(seq_len), device=embeddings.device)
            lengths = torch.clamp(lengths, min=1.0)
            dynamic_temp = self.alpha * torch.log(lengths)
            dynamic_temp = torch.clamp(dynamic_temp, min=self.min_temperature, max=self.max_temperature)
            temperature = dynamic_temp.view(batch_size, 1, 1)
        else:
            temperature = torch.tensor(self.base_temperature, device=embeddings.device, dtype=embeddings.dtype).view(1, 1, 1)

        temperature = torch.clamp(temperature, min=1e-4)
        scaled_scores = attention_scores / temperature

        if self.attention_type == 'top_k':
            soft_weights = F.softmax(self.topk_sharpness * scaled_scores, dim=1)
            with torch.no_grad():
                k = min(self.top_k, seq_len)
                sw = soft_weights.permute(0, 2, 1)
                _, topk_indices = torch.topk(sw, k=k, dim=-1)
                mask_topk = torch.zeros_like(sw)
                mask_topk.scatter_(dim=-1, index=topk_indices, value=1.0)
                mask_topk = mask_topk.permute(0, 2, 1)
            attention_weights = soft_weights * mask_topk
            attention_weights_sum = attention_weights.sum(dim=1, keepdim=True) + 1e-8
            attention_weights = attention_weights / attention_weights_sum
        else:
            attention_weights = F.softmax(scaled_scores, dim=1)

        if avg_pool_heads:
            if attention_mask is not None:
                valid = attention_mask.float()
                denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
                uniform = (valid / denom).unsqueeze(-1)
            else:
                uniform = torch.full(
                    (batch_size, seq_len, 1),
                    1.0 / max(seq_len, 1),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
            avg_head_mask = torch.zeros(
                (1, 1, self.attention_num),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            for h in avg_pool_heads:
                avg_head_mask[..., h] = 1.0
            avg_head_mask = avg_head_mask.expand(batch_size, seq_len, -1)
            uniform_all_heads = uniform.expand(-1, -1, self.attention_num)
            attention_weights = attention_weights * (1.0 - avg_head_mask) + uniform_all_heads * avg_head_mask

        # [B, H, D]
        pooled_heads = torch.einsum('blh,bld->bhd', attention_weights, embeddings)
        pooled = pooled_heads.mean(dim=1)

        if self.save_attention:
            self._call_count += 1
            tag = protein_tag
            if tag is None:
                tag = 'protein1' if self._call_count % 2 == 1 else 'protein2'

            if tag == 'protein1':
                self.protein1_attention_weights = attention_weights.detach().cpu()
                self.protein1_head_pooled = pooled_heads.detach().cpu()
                self._live_protein1_head_pooled = pooled_heads
            else:
                self.protein2_attention_weights = attention_weights.detach().cpu()
                self.protein2_head_pooled = pooled_heads.detach().cpu()
                self._live_protein2_head_pooled = pooled_heads

        return pooled

    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.save_attention:
            return None
        result = {}
        if self.protein1_attention_weights is not None:
            result['protein1'] = self.protein1_attention_weights
        if self.protein2_attention_weights is not None:
            result['protein2'] = self.protein2_attention_weights
        return result if result else None

    def get_head_pooled_outputs(self, live: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        result = {}
        if live:
            if self._live_protein1_head_pooled is not None:
                result['protein1'] = self._live_protein1_head_pooled
            if self._live_protein2_head_pooled is not None:
                result['protein2'] = self._live_protein2_head_pooled
        else:
            if self.protein1_head_pooled is not None:
                result['protein1'] = self.protein1_head_pooled
            if self.protein2_head_pooled is not None:
                result['protein2'] = self.protein2_head_pooled
        return result if result else None

    def reset_attention_tracking(self):
        self._call_count = 0
        self.protein1_attention_weights = None
        self.protein2_attention_weights = None
        self.protein1_head_pooled = None
        self.protein2_head_pooled = None
        self._live_protein1_head_pooled = None
        self._live_protein2_head_pooled = None


def create_pooling_unit(pooling_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建池化单元
    
    Args:
        pooling_config (Dict): 池化配置，包含method和其他参数
                               必须包含: method (str) - 池化方法名称
                               可选包含: embedding_dim, attention_num, dropout等
    
    Returns:
        nn.Module: 池化单元实例
    
    Raises:
        ValueError: 当池化方法不支持时
    
    Examples:
        >>> config = {
        ...     'method': 'average_pooling',
        ...     'embedding_dim': 1280
        ... }
        >>> pooling = create_pooling_unit(config)
        
        >>> config = {
        ...     'method': 'attention_pooling', 
        ...     'embedding_dim': 1280,
        ...     'attention_num': 8,
        ...     'dropout': 0.1
        ... }
        >>> pooling = create_pooling_unit(config)
    """
    method = pooling_config.get('method', 'average_pooling')
    embedding_dim = pooling_config.get('embedding_dim', None)
    
    if embedding_dim is None:
        raise ValueError("Pooling unit requires 'embedding_dim' parameter")
    
    # 移除method和embedding_dim，将其余参数传递给具体的池化单元
    kwargs = {k: v for k, v in pooling_config.items() if k not in ['method', 'embedding_dim']}
    
    if method == 'average_pooling':
        return AveragePoolingUnit(embedding_dim=embedding_dim, **kwargs)
    elif method == 'max_pooling':
        return MaxPoolingUnit(embedding_dim=embedding_dim, **kwargs)
    elif method == 'attention_pooling':
        return AttentionPoolingUnit(embedding_dim=embedding_dim, **kwargs)
    elif method == 'advanced_attention_pooling':
        return AdvancedAttentionPoolingUnit(embedding_dim=embedding_dim, **kwargs)
    elif method == 'none':
        return NonePoolingUnit(embedding_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported pooling method: {method}. "
                        f"Supported: average_pooling, max_pooling, attention_pooling, advanced_attention_pooling, none")
