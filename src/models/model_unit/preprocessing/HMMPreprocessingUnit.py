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

class HMMPreprocessingUnit(nn.Module):
    """
    基于HMM结构域识别的预处理单元
    
    - 使用HMM结构域信息为序列残基分配权重，提升关键区域的表示
    - 支持将每条序列的残基权重缓存至 `cache/hmm/weights`，加速后续训练/推理
    - 依赖的结构域扫描缓存由 HMMCacheManager 管理，存储在 `cache/hmm/domains`
    
    注意:
    - forward 可选接收 `sequence_ids`（List[str]），用于从缓存中获取结构域信息
    - 若未提供 sequence_ids，则退化为恒等映射（安全降级）
    """

    def __init__(self,
                 embedding_dim: int,
                 hmm_db_path: Optional[str] = None,
                 hmm_data_dir: str = "./cache/hmm/domains",
                 hmm_weights_dir: str = "./cache/hmm/weights",
                 fasta_file: Optional[str] = None,
                 dropout: float = 0.1,
                 use_boundary_enhancer: bool = True,
                 create_cache_if_missing: bool = True,
                 cache_mode: str = "weights",
                 cache_file_suffix: str = "weights.npy",
                 batch_size: int = 200,
                 num_processes: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.dropout_rate = dropout
        self.use_boundary_enhancer = use_boundary_enhancer
        self.cache_mode = cache_mode
        self.cache_file_suffix = cache_file_suffix

        # 路径准备
        self.hmm_data_dir = Path(hmm_data_dir)
        self.hmm_weights_dir = Path(hmm_weights_dir)
        self.hmm_data_dir.mkdir(parents=True, exist_ok=True)
        self.hmm_weights_dir.mkdir(parents=True, exist_ok=True)

        # 依赖与可用性
        self.hmm_cache_manager = None
        self.hmm_cache: Optional[dict] = None
        self.hmm_available: bool = False
        self.fasta_file = fasta_file
        self.hmm_db_path = hmm_db_path
        self.batch_size = batch_size
        self.num_processes = num_processes

        # 轻量边界增强网络（可选）
        if self.use_boundary_enhancer:
            self.boundary_enhancer = nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2, groups=embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.boundary_enhancer = None

        # 置信度投影（与现有增强逻辑保持一致的轻量版本）
        self.confidence_projector = nn.Sequential(
            nn.Linear(embedding_dim, max(1, embedding_dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, embedding_dim // 4), 1),
            nn.Sigmoid()
        )
        self.domain_enhancer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # 尝试加载 HMMCacheManager
        try:
            from hmmtools.hmm_cache_manager import HMMCacheManager as ExternalHMMCacheManager  # type: ignore
            self.hmm_cache_manager = ExternalHMMCacheManager(
                hmm_data_dir=str(self.hmm_data_dir),
                hmm_db_path=str(self.hmm_db_path) if self.hmm_db_path is not None else ""
            )
            self.hmm_available = True
            logger.info("HMMCacheManager 已加载")
        except Exception as e:  # pragma: no cover - 防御式加载
            logger.warning(f"无法加载HMMCacheManager，HMM增强将被禁用: {e}")
            self.hmm_available = False

        # 若可用且需要，预加载/创建结构域缓存
        if self.hmm_available and create_cache_if_missing and self.fasta_file:
            try:
                # 使用提供的便捷方法（若存在）
                if hasattr(self.hmm_cache_manager, "load_or_create_cache"):
                    self.hmm_cache = self.hmm_cache_manager.load_or_create_cache(  # type: ignore[attr-defined]
                        self.fasta_file,
                        batch_size=self.batch_size,
                        num_processes=self.num_processes
                    )
                else:
                    # 回退：直接创建（内部包含存在检测）
                    self.hmm_cache = self.hmm_cache_manager.create_hmm_database(  # type: ignore[union-attr]
                        self.fasta_file,
                        batch_size=self.batch_size,
                        num_processes=self.num_processes
                    )
                logger.info(f"HMM缓存已准备，条目数: {len(self.hmm_cache) if self.hmm_cache is not None else 0}")
            except Exception as e:
                logger.warning(f"预加载/创建HMM缓存失败，将在运行时尽力使用已有缓存: {e}")
                self.hmm_cache = None

    @staticmethod
    def _is_catalytic_domain(domain_name: str) -> bool:
        catalytic_keywords = [
            'protein-protein interaction', 'ppi', 'binding', 'interface',
            'dimerization', 'multimerization', 'complex', 'association',
            'recognition', 'docking', 'molecular recognition', 'interaction domain',
            'sh2', 'sh3', 'pdz', 'btb', 'ww', 'bromodomain', 'chromodomain',
            'pleckstrin homology', 'src homology 2', 'src homology 3',
            'ptb domain', 'fha domain'
        ]
        dn = domain_name.lower()
        return any(k in dn for k in catalytic_keywords)

    def _generate_residue_weights(self, seq_length: int, domains: List[tuple]) -> torch.Tensor:
        """根据结构域信息创建长度为 L 的残基权重向量 (CPU 上构造，随后迁移至目标设备)。"""
        weights = torch.ones(seq_length, dtype=torch.float32)
        for start, end, domain_name, _evalue, _score, confidence in domains:
            start = max(0, min(int(start), seq_length - 1))
            end = max(start, min(int(end), seq_length - 1))

            is_catalytic = self._is_catalytic_domain(domain_name)
            base_weight = (1.0 + confidence * 0.8) if is_catalytic else (1.0 + confidence * 0.5)
            weights[start:end + 1] *= float(base_weight)

            boundary_margin = min(5, (end - start + 1) // 4)
            if boundary_margin > 0:
                boundary_weight = (1.0 + confidence * 0.5) if is_catalytic else (1.0 + confidence * 0.3)
                b_s = max(0, start - boundary_margin)
                b_e = min(seq_length, start + boundary_margin)
                weights[b_s:b_e] *= float(boundary_weight)
                b_s = max(0, end - boundary_margin)
                b_e = min(seq_length, end + boundary_margin)
                weights[b_s:b_e] *= float(boundary_weight)

            if is_catalytic:
                center = (start + end) // 2
                center_margin = min(10, (end - start + 1) // 3)
                c_s = max(start, center - center_margin)
                c_e = min(end + 1, center + center_margin)
                weights[c_s:c_e] *= float(1.0 + confidence * 1.3)

        return weights

    def _weight_cache_path(self, seq_id: str, seq_len: int) -> Path:
        # 使用 seq_id 和 长度 生成稳定文件名（避免特殊字符）
        import hashlib
        safe_key = hashlib.md5(f"{seq_id}|{seq_len}".encode("utf-8")).hexdigest()
        return self.hmm_weights_dir / f"{safe_key}.{self.cache_file_suffix}"

    def _get_domains(self, seq_id: str) -> List[tuple]:
        if not (self.hmm_available and self.hmm_cache_manager):
            return []
        try:
            # 优先使用已加载缓存
            if self.hmm_cache is not None and seq_id in self.hmm_cache:
                return cast(List[tuple], self.hmm_cache[seq_id])
            # 回退：通过管理器查询（允许空映射）
            return cast(List[tuple], self.hmm_cache_manager.get_domains(seq_id, self.hmm_cache))  # type: ignore[union-attr]
        except Exception as e:
            logger.debug(f"获取结构域信息失败(seq_id={seq_id}): {e}")
            return []

    def _maybe_load_or_build_weights(self, seq_id: str, seq_len: int, device: torch.device) -> torch.Tensor:
        path = self._weight_cache_path(seq_id, seq_len)
        if path.exists():
            try:
                npy = torch.from_numpy(__import__('numpy').load(path))  # lazy import numpy
                return npy.to(device=device, dtype=torch.float32)
            except Exception as e:
                logger.debug(f"读取权重缓存失败，重新计算: {e}")

        domains = self._get_domains(seq_id)
        if len(domains) == 0:
            # 无结构域信息，返回全1权重
            return torch.ones(seq_len, device=device, dtype=torch.float32)

        weights = self._generate_residue_weights(seq_len, domains).to(device)
        # 保存缓存
        try:
            __import__('numpy').save(path, weights.detach().cpu().numpy())
        except Exception as e:
            logger.debug(f"保存权重缓存失败: {e}")
        return weights

    def forward(self,
                embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                sequence_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq_len, embed_dim]
            attention_mask: [batch, seq_len]，1为有效，0为padding
            sequence_ids: 可选，[batch] 序列ID列表，用于查询HMM结构域信息与权重缓存
        Returns:
            torch.Tensor: 同形状增强后的嵌入
        """
        if embeddings.dim() != 3:
            raise ValueError("HMMPreprocessingUnit 期望输入形状为 [batch, seq_len, embed_dim]")

        batch_size, seq_len, embed_dim = embeddings.shape
        if embed_dim != self.embedding_dim:
            raise ValueError(f"嵌入维度不匹配: got {embed_dim}, expected {self.embedding_dim}")

        # 如果无可用HMM或未提供ID，安全降级为恒等
        if (not self.hmm_available) or sequence_ids is None or len(sequence_ids) != batch_size:
            logger.debug("HMM不可用或未提供sequence_ids，返回原始嵌入")
            return embeddings

        device = embeddings.device
        # 根据 attention_mask 估计每条序列的真实长度
        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1).long().tolist()
        else:
            valid_lengths = [seq_len] * batch_size

        # 逐样本应用残基权重
        weighted_list: List[torch.Tensor] = []
        for i in range(batch_size):
            sid = sequence_ids[i]
            real_len = int(valid_lengths[i]) if valid_lengths[i] > 0 else seq_len
            weights = self._maybe_load_or_build_weights(sid, real_len, device=device)
            # 对齐至 seq_len
            if weights.shape[0] < seq_len:
                pad = torch.ones(seq_len - weights.shape[0], device=device, dtype=weights.dtype)
                weights = torch.cat([weights, pad], dim=0)
            elif weights.shape[0] > seq_len:
                weights = weights[:seq_len]

            weighted = embeddings[i] * weights.unsqueeze(-1)
            weighted_list.append(weighted)

        x = torch.stack(weighted_list, dim=0)  # [B, L, D]

        # 可选的边界增强（逐样本以兼容 BatchNorm1d）
        if self.boundary_enhancer is not None:
            x_t = x.transpose(1, 2)  # [B, D, L]
            enhanced_chunks = []
            for i in range(batch_size):
                enhanced_chunks.append(self.boundary_enhancer(x_t[i:i+1]))
            x = torch.cat(enhanced_chunks, dim=0).transpose(1, 2)

        # 置信度加权 + 结构域特征增强
        attn_like = self.norm1(x)
        conf = self.confidence_projector(attn_like)
        x = x * conf
        x = self.domain_enhancer(x)
        x = self.norm2(x + embeddings)
        return x
