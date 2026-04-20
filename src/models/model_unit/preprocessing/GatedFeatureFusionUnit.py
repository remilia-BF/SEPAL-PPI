import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)


class GatedFeatureFusionUnit(nn.Module):
    """
    简化的门控特征融合预处理单元（无注意力、无Transformer）

    目标：以最小复杂度融合多源残基级辅助特征到ESM嵌入。

    管道：
    1) 对每种特征，得到每个残基的表示：
       - ID/离散类（如 Pfam、s1ssttoken）：小型Embedding (dim<=256)，高dropout
       - 浮点/one-hot：单层线性 -> per_feature_dim（默认32）
    2) 将所有特征在特征维度拼接，得到 concat_aux ∈ R^{sum_dim}
    3) 线性融合到统一维度 d_f（默认128）：a = GELU( W_fuse · concat_aux + b )
    4) 将 a 投影到 d_e（与ESM相同维度）：a' = GELU( W_a · a + b )
    5) 门控：
       - 简化门：g = sigmoid( W_g · a + b )   （默认，更稳定）
       - 拼接门：g = sigmoid( W_g · [e; a'] + b )
    6) 输出：h = e + g ⊙ a'

    训练正则：对门控输出 g 施加 L1（稀疏化），通过三元返回值把损失传回上层：
        return enhanced, g, gate_l1_mean
    上层会按 reconstruction_loss_weight 进行缩放后加入总损失。
    """

    def __init__(self,
                 embedding_dim: int,
                 feature_files: Dict[str, Dict[str, Any]],
                 gating: Optional[Dict[str, Any]] = None,
                 data_processing: Optional[Dict[str, Any]] = None,
                 embedding: Optional[Dict[str, Any]] = None,
                 projection: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()

        # 基本维度
        self.embedding_dim = embedding_dim  # d_e
        self.output_dim = embedding_dim

        # 配置
        self.feature_files = feature_files or {}
        self.gating_cfg = gating or {}
        self.dp_cfg = data_processing or {}
        self.emb_cfg = embedding or {}
        self.proj_cfg = projection or {}

        # 数据处理参数
        self.protein_fasta_path = self.dp_cfg.get('protein_fasta_path', 'dataset/S1/protein.fasta')
        self.all_feature_folder = self.dp_cfg.get('all_feature_folder', None)
        self.cache_features = self.dp_cfg.get('cache_features', True)
        self.normalization_method = self.dp_cfg.get('normalization_method', 'z_score')
        self.handle_missing = self.dp_cfg.get('handle_missing', 'zero')
        self.log_missing_as_error = self.dp_cfg.get('log_missing_as_error', False)

        # 统一维度与子维度
        self.df = int(self.gating_cfg.get('df', 128))  # 融合后的统一维度
        self.per_feature_dim = int(self.gating_cfg.get('per_feature_dim', 32))  # 每种特征局部维度
        self.gate_type = str(self.gating_cfg.get('gate_type', 'simple'))  # simple | concat
        self.id_emb_dim = int(self.emb_cfg.get('id_emb_dim', min(128, self.df)))  # ID嵌入维度 <=256
        self.id_emb_dropout = float(self.emb_cfg.get('dropout', 0.4))  # 高dropout抑制过拟合
        self.float_feature_dropout = float(self.proj_cfg.get('dropout', 0.1))

        # 门控正则（通过三元返回传回上层）
        self.enable_reconstruction = bool(self.gating_cfg.get('enable_l1', True))
        self.reconstruction_loss_weight = float(self.gating_cfg.get('l1_lambda', 0.0))

        # 统计启用的特征与类型
        self.enabled_features: Dict[str, Dict[str, Any]] = {}
        self.embedding_features: Dict[str, Dict[str, Any]] = {}
        for name, cfg in (self.feature_files or {}).items():
            if cfg.get('enabled', False):
                self.enabled_features[name] = cfg
                if cfg.get('feature_type') == 'embedding':
                    self.embedding_features[name] = cfg

        if len(self.enabled_features) == 0:
            logger.warning("GatedFeatureFusionUnit: 未启用任何辅助特征，直接返回输入")

        # ——— 模块构建 ———
        # 1) ID特征嵌入层
        self.embedding_layers = nn.ModuleDict()
        for feat, cfg in self.embedding_features.items():
            vocab_size = int(cfg.get('vocab_size', 1024))
            emb = nn.Embedding(vocab_size, self.id_emb_dim, padding_idx=self.emb_cfg.get('padding_idx', None))
            if self.id_emb_dropout > 0:
                self.embedding_layers[feat] = nn.Sequential(emb, nn.Dropout(self.id_emb_dropout))
            else:
                self.embedding_layers[feat] = emb

        # 2) 浮点/one-hot 特征的线性头（按原始 feature_dim -> per_feature_dim）
        self.scalar_projections = nn.ModuleDict()
        for feat, cfg in self.enabled_features.items():
            if cfg.get('feature_type') == 'embedding':
                continue
            in_dim = int(cfg.get('feature_dim', 1))
            # 作用在每个残基的特征向量上
            self.scalar_projections[feat] = nn.Sequential(
                nn.Linear(in_dim, self.per_feature_dim),
                nn.GELU(),
                nn.Dropout(self.float_feature_dropout),
            )

        # 3) 融合层：拼接所有 per_feature 表示 -> df
        #    运行期根据输入拼接后的维度进行一次性初始化，避免 LazyLinear 带来的未初始化参数
        self.fuse_linear: Optional[nn.Linear] = None
        self.fuse_act = nn.GELU()

        # 4) a -> a' 到 d_e
        self.aux_to_embed = nn.Sequential(
            nn.Linear(self.df, self.embedding_dim),
            nn.GELU(),
        )

        # 5) 门控层
        if self.gate_type == 'concat':
            # 运行期按 [e; a'] 的维度 2*d_e 初始化
            self.gate_linear: Optional[nn.Linear] = None
        else:
            # simple: 仅用 a 产生门
            self.gate_linear: Optional[nn.Linear] = nn.Linear(self.df, self.embedding_dim)

        # 特征缓存与长度
        self.feature_cache = {} if self.cache_features else None
        self.protein_lengths: Dict[str, int] = {}
        self._load_protein_lengths()
        if self.cache_features:
            self._preload_features()

        logger.info(
            f"GatedFeatureFusionUnit 初始化: features={list(self.enabled_features.keys())}, "
            f"id_emb_dim={self.id_emb_dim}, per_feat_dim={self.per_feature_dim}, df={self.df}, gate={self.gate_type}"
        )

    # ——— 数据加载与解析 ———
    def _load_protein_lengths(self):
        try:
            fasta_path = Path(self.protein_fasta_path)
            if fasta_path.exists():
                for record in SeqIO.parse(fasta_path, 'fasta'):
                    self.protein_lengths[record.id] = len(record.seq)
                logger.info(f"加载蛋白长度: {len(self.protein_lengths)} 条")
            else:
                logger.warning(f"FASTA文件不存在: {fasta_path}")
        except Exception as e:
            logger.warning(f"加载蛋白长度失败: {e}")

    def _resolve_feature_file_path(self, file_path_value: Optional[Union[str, bool]]) -> Optional[str]:
        if not file_path_value or str(file_path_value).lower() == 'false':
            return None
        p = Path(str(file_path_value))
        if p.exists():
            return str(p)
        if self.all_feature_folder:
            candidate = Path(self.all_feature_folder) / p
            return str(candidate)
        return str(p)

    def _preload_features(self):
        logger.info("预加载特征文件...")
        for feat, cfg in self.enabled_features.items():
            try:
                if feat == 'protein_length':
                    # 作为ID类的内部特征时使用
                    feat_dict = {}
                    for pid, L in self.protein_lengths.items():
                        vocab_size = cfg.get('vocab_size', 5000)
                        Lc = min(L, vocab_size - 1)
                        feat_dict[pid] = np.array([Lc] * L, dtype=np.int64)
                    if self.feature_cache is not None:
                        self.feature_cache[feat] = feat_dict
                    continue

                path = self._resolve_feature_file_path(cfg.get('file_path'))
                if not path:
                    (logger.error if self.log_missing_as_error else logger.warning)(
                        f"特征 {feat} 缺少 file_path")
                    continue
                p = Path(path)
                if not p.exists():
                    (logger.error if self.log_missing_as_error else logger.warning)(
                        f"特征文件不存在: {p}")
                    continue
                with open(p, 'r') as f:
                    data = json.load(f)
                feat_dict = {}
                for item in data:
                    pid = item['protein_id']
                    arr = self._parse_feature_data(item, feat, cfg)
                    feat_dict[pid] = arr
                if self.feature_cache is not None:
                    self.feature_cache[feat] = feat_dict
                logger.info(f"预加载 {feat}: {len(feat_dict)} 条")
            except Exception as e:
                logger.error(f"预加载特征 {feat} 失败: {e}")

    def _parse_feature_data(self, item: Dict[str, Any], feat: str, cfg: Dict[str, Any]) -> np.ndarray:
        # 二级结构: 独热字符串 -> float
        if feat == 'secondary_structure_features':
            data_field = 'secondary_structure'
            data_str = item[data_field].replace(' ', '').replace('\n', '')
            segs = data_str.split('],[')
            segs[0] = segs[0].lstrip('[')
            segs[-1] = segs[-1].rstrip(']')
            rows = []
            for seg in segs:
                vals = [float(x) for x in seg.split(',')]
                if len(vals) != 3:
                    vals = [0.0, 0.0, 1.0]
                rows.append(vals)
            return np.array(rows, dtype=np.float32)

        # 其他字段推断
        if cfg.get('feature_type') == 'embedding':
            # ID序列，按逗号分割
            data_field = 'embeddings'
            values = [int(float(x)) for x in item[data_field].split(',')]
            return np.array(values, dtype=np.int64)

        # 连续特征
        if 'hydrophobicity' in feat:
            data_field = 'hydrophobicity'
        elif 'sasa' in feat:
            data_field = 'sasa'
        elif 'pqr' in feat:
            data_field = 'embeddings'
        else:
            data_field = 'embeddings'

        values = [float(x) for x in item[data_field].split(',')]
        feat_dim = int(cfg.get('feature_dim', 1))
        arr = np.array(values, dtype=np.float32)
        if feat_dim == 1:
            return arr.reshape(-1, 1)
        else:
            # 确保长度可整除
            total = (len(arr) // feat_dim) * feat_dim
            arr = arr[:total]
            return arr.reshape(-1, feat_dim)

    def _normalize_features(self, arr: np.ndarray) -> np.ndarray:
        if self.normalization_method == 'z_score':
            mean = np.mean(arr, axis=0, keepdims=True)
            std = np.std(arr, axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            return (arr - mean) / std
        elif self.normalization_method == 'min_max':
            min_v = np.min(arr, axis=0, keepdims=True)
            max_v = np.max(arr, axis=0, keepdims=True)
            denom = np.where((max_v - min_v) < 1e-8, 1.0, (max_v - min_v))
            return (arr - min_v) / denom
        else:
            return arr

    def _get_feature_matrix(self, protein_id: str, target_len: int, device: torch.device) -> torch.Tensor:
        """
        返回 [target_len, sum_dim_per_feature_before_fuse] 的拼接矩阵。
        对于ID特征：嵌入 -> [L, id_emb_dim]
        对于连续特征：线性 -> [L, per_feature_dim]
        """
        per_feat_tensors = []
        for feat, cfg in self.enabled_features.items():
            try:
                # 取数据
                if self.feature_cache is not None and feat in self.feature_cache and protein_id in self.feature_cache[feat]:
                    data = self.feature_cache[feat][protein_id]
                else:
                    data = self._load_single_feature(protein_id, feat, cfg)

                # 规范为 [L] (ID) 或 [L, d]
                if cfg.get('feature_type') == 'embedding':
                    ids = self._adjust_id_length(data, target_len, cfg)
                    ids_t = torch.from_numpy(ids).to(device)
                    emb_layer = self.embedding_layers.get(feat, None)
                    if emb_layer is None:
                        # 未配置embedding层，回退为零
                        per_feat = torch.zeros((target_len, self.id_emb_dim), device=device)
                    else:
                        per_feat = emb_layer(ids_t)  # [L, id_emb_dim]
                else:
                    mat = self._adjust_cont_length(data, target_len, int(cfg.get('feature_dim', 1)))
                    if cfg.get('normalize', False):
                        mat = self._normalize_features(mat)
                    mat_t = torch.from_numpy(mat).to(device)
                    head = self.scalar_projections.get(feat, None)
                    if head is None:
                        # 安全回退
                        if mat_t.ndim == 1:
                            mat_t = mat_t.unsqueeze(-1)
                        per_feat = F.gelu(mat_t)
                    else:
                        per_feat = head(mat_t)

                per_feat_tensors.append(per_feat)
            except Exception as e:
                logger.warning(f"特征 {feat} 加载失败: {e}")
                # 回退为零
                if cfg.get('feature_type') == 'embedding':
                    per_feat_tensors.append(torch.zeros((target_len, self.id_emb_dim), device=device))
                else:
                    per_feat_tensors.append(torch.zeros((target_len, self.per_feature_dim), device=device))

        if len(per_feat_tensors) == 0:
            return torch.zeros((target_len, 0), device=device)
        return torch.cat(per_feat_tensors, dim=-1)

    def _adjust_id_length(self, ids: np.ndarray, target_len: int, cfg: Dict[str, Any]) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            ids = ids.reshape(-1)
        if len(ids) < target_len:
            out = np.zeros(target_len, dtype=np.int64)
            out[: len(ids)] = ids
            ids = out
        elif len(ids) > target_len:
            ids = ids[:target_len]
        vocab_size = cfg.get('vocab_size', 1024)
        ids = np.clip(ids, 0, vocab_size - 1)
        return ids

    def _adjust_cont_length(self, mat: np.ndarray, target_len: int, feat_dim: int) -> np.ndarray:
        mat = np.asarray(mat)
        if feat_dim == 1:
            mat = mat.reshape(-1, 1)
        else:
            total = (len(mat) // feat_dim) * feat_dim
            mat = mat[:total].reshape(-1, feat_dim)
        L = mat.shape[0]
        if L == target_len:
            return mat.astype(np.float32)
        elif L < target_len:
            pad = np.zeros((target_len - L, mat.shape[1]), dtype=np.float32)
            return np.vstack([mat, pad])
        else:
            return mat[:target_len].astype(np.float32)

    def _load_single_feature(self, protein_id: str, feat: str, cfg: Dict[str, Any]) -> np.ndarray:
        try:
            if feat == 'protein_length':
                L = self.protein_lengths.get(protein_id, 100)
                vocab_size = cfg.get('vocab_size', 5000)
                Lc = min(L, vocab_size - 1)
                return np.array([Lc] * L, dtype=np.int64)

            path = self._resolve_feature_file_path(cfg.get('file_path'))
            if not path:
                raise FileNotFoundError(f"特征 {feat} 缺少 file_path")
            with open(Path(path), 'r') as f:
                data = json.load(f)
            for item in data:
                if item['protein_id'] == protein_id:
                    return self._parse_feature_data(item, feat, cfg)
            raise KeyError(f"蛋白 {protein_id} 不在特征 {feat} 文件中")
        except Exception as e:
            logger.error(f"加载特征 {feat} 失败: {e}")
            # 回退
            if cfg.get('feature_type') == 'embedding':
                return np.zeros((0,), dtype=np.int64)
            else:
                return np.zeros((0, int(cfg.get('feature_dim', 1))), dtype=np.float32)

    # ——— 前向 ———
    def forward(self,
                embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[List[str]] = None):
        """
        Args:
            embeddings: [B, L, d_e]
            attention_mask: [B, L] (未使用，仅用于长度对齐/兼容)
            protein_ids: [B]

        Returns:
            - 若启用正则： (enhanced, gates, gate_l1_mean)
            - 否则： enhanced
        """
        if embeddings.dim() != 3:
            raise ValueError("期望输入形状 [B, L, d_e]")
        B, L, d_e = embeddings.shape
        if d_e != self.embedding_dim:
            raise ValueError(f"嵌入维度不匹配: got {d_e}, expected {self.embedding_dim}")

        if protein_ids is None or len(protein_ids) != B:
            protein_ids = getattr(self, '_current_protein_ids', None)
            if protein_ids is None or len(protein_ids) != B:
                # 无法融合，直接返回输入
                logger.debug("GatedFeatureFusionUnit: 缺少protein_ids，直接返回输入")
                return embeddings

        device = embeddings.device
        out_list = []
        gate_list = []

        for i in range(B):
            pid = protein_ids[i]
            real_L = self.protein_lengths.get(pid, L)
            real_L = min(real_L, L)

            # 取辅助矩阵 [real_L, sum_dim]
            aux_mat = self._get_feature_matrix(pid, real_L, device)
            # pad/crop到 L
            if aux_mat.shape[0] < L:
                pad_len = L - aux_mat.shape[0]
                if aux_mat.shape[1] == 0:
                    aux_mat = torch.zeros((L, 0), device=device)
                else:
                    pad = torch.zeros((pad_len, aux_mat.shape[1]), device=device, dtype=aux_mat.dtype)
                    aux_mat = torch.cat([aux_mat, pad], dim=0)
            elif aux_mat.shape[0] > L:
                aux_mat = aux_mat[:L]

            # 融合到 df
            if aux_mat.shape[-1] == 0:
                a = torch.zeros((L, self.df), device=device)
            else:
                if self.fuse_linear is None:
                    # 首次见到输入维度，创建线性层
                    self.fuse_linear = nn.Linear(aux_mat.shape[-1], self.df).to(device)
                a = self.fuse_act(self.fuse_linear(aux_mat))  # [L, df]

            # 投影到 d_e
            a_prime = self.aux_to_embed(a)  # [L, d_e]

            # 门控
            if self.gate_type == 'concat':
                gate_in = torch.cat([embeddings[i, :L, :], a_prime[:L, :]], dim=-1)  # [L, 2*d_e]
                if self.gate_linear is None:
                    self.gate_linear = nn.Linear(2 * self.embedding_dim, self.embedding_dim).to(device)
                g = torch.sigmoid(self.gate_linear(gate_in))
                # 为了维度一致，若L<原L，补pad
            else:
                if self.gate_linear is None:
                    self.gate_linear = nn.Linear(self.df, self.embedding_dim).to(device)
                g = torch.sigmoid(self.gate_linear(a[:L, :]))  # [L, d_e]

            # 合成输出
            h_real = embeddings[i, :L, :] + g * a_prime[:L, :]
            # 如需pad到L
            if L > real_L:
                pad_rows = L - real_L
                h = torch.cat([h_real, embeddings[i, real_L:, :]], dim=0)
                g_full = torch.cat([g, torch.zeros((pad_rows, d_e), device=device)], dim=0)
            else:
                h = h_real
                g_full = g

            out_list.append(h)
            gate_list.append(g_full)

        enhanced = torch.stack(out_list, dim=0)
        gates = torch.stack(gate_list, dim=0)

        if self.enable_reconstruction and self.reconstruction_loss_weight > 0:
            gate_l1 = gates.abs().mean()
            return enhanced, gates, gate_l1
        else:
            return enhanced

    def set_current_protein_ids(self, protein_ids: List[str]):
        self._current_protein_ids = protein_ids

    @torch.no_grad()
    def compute_feature_gate_report(
        self,
        embeddings: Optional[torch.Tensor],  # [B, L, d_e] 或 None（simple 模式可为 None）
        attention_mask: Optional[torch.Tensor],  # [B, L] 或 None
        protein_ids: List[str],
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        计算“每个特征”的平均门控值（跨样本/位置/通道取均值）。

        定义：对每个特征f，仅保留该特征对应的输入通道，其他特征置零，计算门控g_f并取均值。
        - simple: g = sigmoid(Wg·a_f + b)
        - concat: g = sigmoid(Wg·[e; a'_f] + b)

        Args:
            embeddings: [B, L, d_e]，当 gate_type='concat' 时必需；simple 时可为 None
            attention_mask: [B, L]，可选，用于确定有效长度
            protein_ids: 蛋白ID列表 [B]
            max_samples: 限制参与统计的样本数，None 表示不限制

        Returns:
            Dict[str, float]: {feature_name: mean_gate_value}
        """
        self.eval()
        device = None
        if embeddings is not None:
            device = embeddings.device
        # 若为simple且无embeddings，从任意缓存的模块参数推断设备
        if device is None:
            device = next(self.parameters()).device

        # 构建特征拼接顺序与宽度
        feat_order: List[Tuple[str, int]] = []
        for feat, cfg in self.enabled_features.items():
            if cfg.get('feature_type') == 'embedding':
                feat_order.append((feat, self.id_emb_dim))
            else:
                feat_order.append((feat, self.per_feature_dim))

        sums: Dict[str, float] = {name: 0.0 for name, _ in feat_order}
        counts: Dict[str, int] = {name: 0 for name, _ in feat_order}

        B = len(protein_ids)
        limit = B if max_samples is None else min(B, max_samples)

        for i in range(limit):
            pid = protein_ids[i]
            # 确定序列长度
            if attention_mask is not None:
                L = int(attention_mask[i].sum().item())
            else:
                L = self.protein_lengths.get(pid, embeddings.shape[1] if embeddings is not None else 1024)
            L = max(1, L)

            # 构造特征矩阵
            aux_full = self._get_feature_matrix(pid, L, device)  # [L, sum_dim]
            sum_dim = aux_full.shape[-1]
            if sum_dim == 0:
                # 无特征则跳过
                continue

            # 预先构建 e（若 concat）
            if self.gate_type == 'concat':
                if embeddings is None:
                    # 无法计算concat门控
                    continue
                e_i = embeddings[i, :L, :]

            # 遍历每个特征片段，单独评估门控
            col_start = 0
            for name, width in feat_order:
                col_end = col_start + width
                # 仅保留该特征
                aux_only = torch.zeros_like(aux_full)
                aux_only[:, col_start:col_end] = aux_full[:, col_start:col_end]

                # 融合 -> a
                if self.fuse_linear is None:
                    self.fuse_linear = nn.Linear(sum_dim, self.df).to(device)
                a = self.fuse_act(self.fuse_linear(aux_only))  # [L, df]
                a_prime = self.aux_to_embed(a)  # [L, d_e]

                # 计算门控
                if self.gate_type == 'concat':
                    if self.gate_linear is None:
                        self.gate_linear = nn.Linear(2 * self.embedding_dim, self.embedding_dim).to(device)
                    gate_in = torch.cat([e_i, a_prime], dim=-1)
                    g = torch.sigmoid(self.gate_linear(gate_in))  # [L, d_e]
                else:
                    if self.gate_linear is None:
                        self.gate_linear = nn.Linear(self.df, self.embedding_dim).to(device)
                    g = torch.sigmoid(self.gate_linear(a))  # [L, d_e]

                g_mean = g.mean().item()
                sums[name] += g_mean
                counts[name] += 1

                col_start = col_end

        report = {name: (sums[name] / counts[name]) if counts[name] > 0 else 0.0 for name, _ in feat_order}
        return report
