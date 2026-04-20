import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import numpy as np
import lmdb
import yaml
from Bio import SeqIO

logger = logging.getLogger(__name__)


class ResidualConvBlock(nn.Module):
    """一维残差卷积块，用于在低维空间提取局部上下文"""

    def __init__(self, channels: int, kernel_size: int, dropout: float, activation: str):
        super().__init__()
        padding = kernel_size // 2
        act = activation.lower()
        if act == "relu":
            act_fn = nn.ReLU()
        elif act == "silu":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            act_fn,
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout)
        )
        self.activation = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        return self.activation(out + residual)


class FeatureConcatUnit(nn.Module):
    """
    多模态特征拼接预处理单元 (优化版：极致轻量化)

    数据流 (简化后):
    1. 加载残基级特征 (JSON)
    2. 逐特征编码: LayerNorm + Linear + 1D-CNN (移除Transformer, O(L²)→O(L))
    3. 提前极致降维: 每个特征立即降到 4-8 维
    4. 全局门控: 学习全局特征权重向量 (移除逐残基门控矩阵)
    5. 拼接后直接低维投影 (移除 ConcatMLP)
    6. 简化 ResNet-1D-CNN (1-2块)
    7. 低维残差连接: 元素级乘法或最小映射
    8. (训练) 轻量重构损失 + 全局稀疏正则
    """

    def __init__(self,
                 embedding_dim: int,
                 feature_files: Dict[str, Dict[str, Any]],
                 projection: Optional[Dict[str, Any]] = None,
                 data_processing: Optional[Dict[str, Any]] = None,
                 embedding: Optional[Dict[str, Any]] = None,
                 feature_gating: Optional[Dict[str, Any]] = None,
                 fusion_mlp: Optional[Dict[str, Any]] = None,
                 resnet: Optional[Dict[str, Any]] = None,
                 reconstruction: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim

        self.projection_config = projection or {}
        self.data_processing_config = data_processing or {}
        self.embedding_config = embedding or {}
        self.feature_gating_config = feature_gating or {}
        self.fusion_mlp_config = fusion_mlp or {}
        self.resnet_config = resnet or {}
        self.reconstruction_config = reconstruction or {}
        
        # 新增：提前降维配置
        self.early_compression_dim = self.projection_config.get('early_compression_dim', 8)  # 极致降维到4-8维

        self.protein_fasta_path = self.data_processing_config.get('protein_fasta_path', 'dataset/S1/protein.fasta')
        self.cache_features = self.data_processing_config.get('cache_features', True)
        self.normalization_method = self.data_processing_config.get('normalization_method', 'z_score')
        self.handle_missing = self.data_processing_config.get('handle_missing', 'zero')
        self.all_feature_folder = self.data_processing_config.get('all_feature_folder', None)
        self.log_missing_as_error = self.data_processing_config.get('log_missing_as_error', False)

        self.feature_files = feature_files or {}
        self.feature_configs: Dict[str, Dict[str, Any]] = {}
        self.enabled_features: Dict[str, Dict[str, Any]] = {}
        self.embedding_features: Dict[str, Dict[str, Any]] = {}
        self.feature_input_dims: Dict[str, int] = {}
        self.feature_encoded_dims: Dict[str, int] = {}
        self.total_feature_dim = 0

        for feature_name, config in self.feature_files.items():
            if not config.get('enabled', False):
                continue

            feature_config = dict(config)
            raw_dim = feature_config.get('raw_dim') or feature_config.get('input_dim')
            if raw_dim is None:
                if feature_config.get('feature_type') == 'embedding':
                    raw_dim = feature_config.get('embedding_dim', embedding_dim)
                elif feature_name == 'secondary_structure_features':
                    raw_dim = 3
                else:
                    raw_dim = feature_config.get('feature_dim', 1)

            # 允许通过 compression_dim 独立设置压缩维度，否则使用全局 early_compression_dim
            feature_config['_input_dim'] = int(raw_dim)
            custom_dim = feature_config.get('compression_dim')
            if custom_dim is not None:
                feature_config['_encoded_dim'] = int(custom_dim)
            else:
                feature_config['_encoded_dim'] = int(self.early_compression_dim)

            if feature_config.get('feature_type') == 'embedding':
                self.embedding_features[feature_name] = feature_config
            else:
                self.enabled_features[feature_name] = feature_config

            self.feature_configs[feature_name] = feature_config
            self.feature_input_dims[feature_name] = feature_config['_input_dim']
            self.feature_encoded_dims[feature_name] = feature_config['_encoded_dim']
            self.total_feature_dim += feature_config['_encoded_dim']
        
        # 记录特征顺序，确保后续处理一致性
        self.feature_names_ordered = list(self.feature_configs.keys())

        if self.total_feature_dim == 0:
            raise ValueError("没有启用任何特征，请检查feature_files配置")

        self.projection_activation = (self.projection_config.get('activation', 'GELU') or 'GELU').lower()

        self.embedding_layers = nn.ModuleDict()
        for feature_name, config in self.embedding_features.items():
            vocab_size = config.get('vocab_size', 1000)
            emb_dim = config.get('embedding_dim', 64)
            padding_idx = self.embedding_config.get('padding_idx', None)
            embedding_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
            emb_dropout = self.embedding_config.get('dropout', 0.1)
            if emb_dropout > 0:
                embedding_layer = nn.Sequential(embedding_layer, nn.Dropout(emb_dropout))
            self.embedding_layers[feature_name] = embedding_layer

        self.compression_layers = self._build_compression_layers()
        self.feature_encoder_config = self.data_processing_config.get('feature_encoder', {})
        self.feature_encoders = nn.ModuleDict()
        self._build_feature_encoders()
        self.active_feature_names = [name for name in self.feature_names_ordered if name in self.feature_encoders]
        self.num_active_features = len(self.active_feature_names)

        self.gating_enabled = self.feature_gating_config.get('enabled', True)
        self.gating_hidden_dim = int(self.feature_gating_config.get('hidden_dim', 256))
        self.gating_dropout = float(self.feature_gating_config.get('dropout', 0.1))
        self.gating_activation = self.feature_gating_config.get('activation', 'GELU')
        self.sparsity_loss_weight = float(self.feature_gating_config.get('sparsity_loss_weight', 0.0))
        self.gating_constant_value = float(self.feature_gating_config.get('disabled_scalar_weight', 1.0))
        
        # 构建门控对齐层：为了非对称门控，将所有特征对齐到 early_compression_dim
        self.gating_aligners = nn.ModuleDict()
        for feature_name in self.feature_names_ordered:
            encoded_dim = self.feature_encoded_dims[feature_name]
            # 如果特征维度与观察维度不一致，添加线性对齐层
            if encoded_dim != self.early_compression_dim:
                self.gating_aligners[feature_name] = nn.Linear(encoded_dim, self.early_compression_dim)
                
        self.feature_gating_layers = self._build_gating_layers()
        # 数据集级门控统计（用于导出 feature_gating.csv）
        self._gating_stats_dataset_name: Optional[str] = None
        self._gating_stats_store: Dict[str, Dict[str, float]] = {}
        self.reset_feature_gating_stats()

        self.concat_mlp = self._build_concat_mlp()

        self.low_dim = int(self.projection_config.get('output_dim', self.projection_config.get('hidden_dim', 256)))
        if self.low_dim <= 0:
            self.low_dim = 256
        self.projection_dropout = float(self.projection_config.get('dropout', 0.1))
        self.feature_projection = self._build_projection_network()

        self.resnet_blocks = self._build_resnet_blocks()

        # 优化：使用低秩分解替代直接线性映射 (low_dim -> bottleneck -> embedding_dim)
        self.use_low_rank = self.projection_config.get('use_low_rank_final', True)
        if self.use_low_rank:
            bottleneck = max(self.low_dim // 2, 64)
            self.final_projection = nn.Sequential(
                nn.Linear(self.low_dim, bottleneck),
                self._activation(self.projection_activation),
                nn.Linear(bottleneck, self.embedding_dim)
            )
        else:
            self.final_projection = nn.Linear(self.low_dim, self.embedding_dim)

        self.enable_reconstruction = bool(self.reconstruction_config.get('enabled', True))
        self.reconstruction_loss_weight = float(self.reconstruction_config.get('loss_weight', 0.1))
        self.reconstruction_head = self._build_reconstruction_head()

        self.feature_cache = {} if self.cache_features else None
        self.protein_lengths: Dict[str, int] = {}
        self._load_protein_lengths()
        
        # 门控权重日志初始化
        self.gating_log_file = None
        self._init_gating_logger()
        
        # 【新增】批量加载模式初始化
        self.use_batched_loading = False
        self.multimodal_lmdb_env = None
        self.lmdb_global_meta = None
        self._init_batched_loading()
        
        if self.cache_features and not self.use_batched_loading:
            self._preload_features()

        # 调试：打印当前使用的多模态特征根目录及批量加载状态
        logger.info(
            "FeatureConcatUnit root directory all_feature_folder=%s, batched_loading=%s",
            self.all_feature_folder,
            self.use_batched_loading,
        )

        logger.info(
            "FeatureConcatUnit initialization: features=%s, total_dim=%d, low_dim=%d",
            list(self.feature_configs.keys()),
            self.total_feature_dim,
            self.low_dim
        )

    def _init_gating_logger(self):
        """初始化门控权重日志文件"""
        # 使用 logger 的根处理器来查找输出目录
        try:
            self.output_dir = None
            
            # 第一尝试：获取 'sepal_ppi' 这个特定的 logger (由 src/utils/logger.py 定义)
            target_loggers = [logging.getLogger("sepal_ppi"), logging.getLogger()]
            
            for logger_instance in target_loggers:
                if logger_instance.handlers:
                    for handler in logger_instance.handlers:
                        if hasattr(handler, 'baseFilename'):
                            log_path = Path(handler.baseFilename)
                            self.output_dir = log_path.parent
                            break
                if self.output_dir:
                    break
            
            # 如果从 logger 找不到，尝试从 kwargs 中获取 (如果传入了)
            if not self.output_dir and hasattr(self, 'kwargs'):
                 pass # self.kwargs 没保存，暂时不处理
        except Exception:
            self.output_dir = None

    def _log_gating_weights(self, weights_dict: Dict[str, float]):
        """将权重写入独立日志文件"""
        if self.gating_log_file is None:
            # 如果找到了输出目录，写入到该目录
            try:
                log_filename = "gating_weights_stats.jsonl"
                if self.output_dir and self.output_dir.exists():
                     file_path = self.output_dir / log_filename
                else:
                    # 回退到当前目录，或者尝试从 hydra 全局配置获取（如果是hydra模式）
                    try:
                        import hydra
                        if hydra.core.hydra_config.HydraConfig.initialized():
                             file_path = Path(hydra.core.hydra_config.HydraConfig.get().run.dir) / log_filename
                        else:
                             file_path = Path(log_filename) # 当前目录
                    except ImportError:
                        file_path = Path(log_filename)

                # logger.info("Initializing gating log file at: %s", file_path) # Debug
                self.gating_log_file = open(file_path, "a")
            except Exception:
                pass
        
        if self.gating_log_file:
            log_entry = {
                "timestamp": str(np.datetime64('now')),
                "weights": weights_dict
            }
            try:
                self.gating_log_file.write(json.dumps(log_entry) + "\n")
                self.gating_log_file.flush()
            except Exception:
                pass

    def _build_gating_layers(self) -> nn.Module:
        """构建全局门控层：学习每个特征的全局权重标量"""
        if not self.gating_enabled:
            return None
        
        num_features = len(self.feature_encoded_dims)
        # 输入维度是所有特征对齐后的扁平化向量 [Num_Features * early_compression_dim]
        gating_input_dim = num_features * self.early_compression_dim
        
        if num_features == 0:
            return None
        
        # 全局门控：输出 [num_features] 的权重向量
        return nn.Sequential(
            nn.Linear(gating_input_dim, self.gating_hidden_dim),
            self._activation(self.gating_activation),
            nn.Dropout(self.gating_dropout),
            nn.Linear(self.gating_hidden_dim, num_features),
            nn.Sigmoid()
        )

    def reset_feature_gating_stats(self, dataset_name: Optional[str] = None):
        """重置门控统计缓存（按数据集评估前调用）"""
        self._gating_stats_dataset_name = dataset_name
        self._gating_stats_store = {
            name: {'count': 0.0, 'sum': 0.0, 'sum_sq': 0.0}
            for name in self.active_feature_names
        }

    def _accumulate_feature_gating_stats(self,
                                         gate_weights: Optional[torch.Tensor],
                                         feature_names: List[str]):
        """累计门控统计：count/sum/sum_sq，用于后续 mean/var 输出"""
        if gate_weights is None or not feature_names:
            return

        try:
            gw = gate_weights.detach().float().cpu()
            if gw.dim() == 1:
                gw = gw.unsqueeze(0)

            used_cols = min(gw.shape[1], len(feature_names))
            if used_cols <= 0:
                return

            gw = gw[:, :used_cols]
            sums = gw.sum(dim=0).tolist()
            sums_sq = (gw * gw).sum(dim=0).tolist()
            count = float(gw.shape[0])

            for idx in range(used_cols):
                name = feature_names[idx]
                if name not in self._gating_stats_store:
                    self._gating_stats_store[name] = {'count': 0.0, 'sum': 0.0, 'sum_sq': 0.0}
                self._gating_stats_store[name]['count'] += count
                self._gating_stats_store[name]['sum'] += float(sums[idx])
                self._gating_stats_store[name]['sum_sq'] += float(sums_sq[idx])
        except Exception:
            pass

    def get_feature_gating_stats(self) -> Dict[str, Any]:
        """返回门控统计摘要（每特征 mean/var/count）"""
        summary = {
            'dataset_name': self._gating_stats_dataset_name,
            'enabled': bool(self.gating_enabled),
            'features': {}
        }
        for name in self.active_feature_names:
            raw = self._gating_stats_store.get(name, {'count': 0.0, 'sum': 0.0, 'sum_sq': 0.0})
            count = float(raw.get('count', 0.0))
            if count > 0:
                mean = float(raw.get('sum', 0.0)) / count
                var = float(raw.get('sum_sq', 0.0)) / count - mean * mean
                var = max(var, 0.0)
            else:
                mean = 0.0
                var = 0.0
            summary['features'][name] = {
                'mean': float(mean),
                'var': float(var),
                'count': int(count)
            }
        return summary

    def _build_concat_mlp(self) -> nn.Module:
        """移除融合 MLP，直接使用 LayerNorm 归一化拼接特征"""
        in_dim = self.total_feature_dim
        if in_dim <= 0:
            return nn.Identity()
        # 只保留 LayerNorm，移除 MLP 层
        return nn.LayerNorm(in_dim)

    def _build_projection_network(self) -> nn.Module:
        """简化投影网络：直接从拼接维度降到低维，无隐藏层"""
        in_dim = self.total_feature_dim
        if in_dim <= 0:
            return nn.Identity()
        # 移除隐藏层，直接投影
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.low_dim),
            nn.Dropout(self.projection_dropout)
        )

    def _activation(self, name: str) -> nn.Module:
        name = (name or 'gelu').lower()
        if name == 'relu':
            return nn.ReLU()
        if name == 'silu':
            return nn.SiLU()
        return nn.GELU()

    def _build_resnet_blocks(self) -> nn.Module:
        """简化 ResNet：默认只用1块，减少计算量"""
        blocks = int(self.resnet_config.get('num_blocks', 1))  # 默认改为1
        if blocks <= 0:
            return nn.Identity()
        kernel_size = int(self.resnet_config.get('kernel_size', 3))
        dropout = float(self.resnet_config.get('dropout', 0.1))
        activation = self.resnet_config.get('activation', 'GELU')
        layers = nn.ModuleList([
            ResidualConvBlock(self.low_dim, kernel_size, dropout, activation)
            for _ in range(blocks)
        ])
        return layers

    def _build_reconstruction_head(self) -> Optional[nn.Module]:
        """轻量重构头：单层映射"""
        if not self.enable_reconstruction:
            return None
        # 简化：只用单层，不使用隐藏层
        return nn.Sequential(
            nn.LayerNorm(self.low_dim),
            nn.Linear(self.low_dim, self.low_dim)
        )

    def _build_feature_encoders(self):
        """构建轻量级特征编码器：Linear + 1D-CNN """
        default_dropout = self.feature_encoder_config.get('dropout', 0.1)
        default_activation = self.feature_encoder_config.get('activation', 'gelu')
        default_kernel_size = self.feature_encoder_config.get('kernel_size', 3)

        for feature_name, config in self.feature_configs.items():
            input_dim = config.get('_input_dim', config.get('feature_dim', 1))
            output_dim = config.get('_encoded_dim', input_dim)  # 现在统一为 early_compression_dim
            encoder_cfg = config.get('encoder', {})

            encoder = LightweightFeatureEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_size=encoder_cfg.get('kernel_size', default_kernel_size),
                dropout=encoder_cfg.get('dropout', default_dropout),
                activation=encoder_cfg.get('activation', default_activation)
            )
            self.feature_encoders[feature_name] = encoder

    def _build_compression_layers(self) -> nn.ModuleDict:
        compression_layers = nn.ModuleDict()
        self.embedding_compression_config = self.data_processing_config.get('embedding_compression', {})
        self.compression_method = self.embedding_compression_config.get('method', 'truncate')
        self.truncate_config = self.embedding_compression_config.get('truncate', {})
        self.linear_config = self.embedding_compression_config.get('linear', {})
        self.mlp_config = self.embedding_compression_config.get('mlp', {})

        if self.compression_method == 'truncate':
            return compression_layers

        for feature_name, config in self.embedding_features.items():
            embedding_dim = config.get('embedding_dim', 64)
            feature_dim = config.get('feature_dim', embedding_dim)
            if embedding_dim == feature_dim:
                continue
            layer = self._create_compression_layer(feature_name, embedding_dim, feature_dim)
            if layer is not None:
                compression_layers[feature_name] = layer
        return compression_layers

    def _create_compression_layer(self, feature_name: str, embedding_dim: int, feature_dim: int) -> Optional[nn.Module]:
        if self.compression_method == 'linear':
            layer = nn.Linear(embedding_dim, feature_dim)
            return nn.Sequential(layer, nn.Dropout(self.embedding_config.get('dropout', 0.1)))
        if self.compression_method == 'mlp':
            hidden = max(feature_dim * 2, 64)
            activation = (self.mlp_config.get('activation', 'gelu') or 'gelu').lower()
            if activation == 'relu':
                act_fn = nn.ReLU()
            elif activation == 'silu':
                act_fn = nn.SiLU()
            else:
                act_fn = nn.GELU()
            return nn.Sequential(
                nn.Linear(embedding_dim, hidden),
                act_fn,
                nn.Dropout(self.mlp_config.get('dropout', 0.1)),
                nn.Linear(hidden, feature_dim)
            )
        return None

    def _load_protein_lengths(self):
        try:
            fasta_path = Path(self.protein_fasta_path)
            if fasta_path.exists():
                for record in SeqIO.parse(fasta_path, 'fasta'):
                    self.protein_lengths[record.id] = len(record.seq)
                logger.info("Loading protein length: %d ", len(self.protein_lengths))
            else:
                logger.warning("FASTA file does not exist: %s", fasta_path)
        except Exception as e:
            logger.warning("Loading protein length failed: %s", e)

    def _resolve_feature_file_path(self, file_path_value: Optional[Union[str, bool]]) -> Optional[str]:
        if not file_path_value or str(file_path_value).lower() == 'false':
            return None
        p = Path(str(file_path_value))
        if p.exists():
            return str(p)
        if self.all_feature_folder:
            candidate = Path(self.all_feature_folder) / p
            if candidate.exists():
                return str(candidate)
        return str(p)

    def _init_batched_loading(self):
        """初始化批量加载模式：自动检测或生成LMDB"""
        if not self.all_feature_folder:
            return
        
        feature_folder = Path(self.all_feature_folder)
        lmdb_path = feature_folder / 'multimodal_features.lmdb'
        
        # 检查LMDB是否存在
        if lmdb_path.exists():
            try:
                # 尝试打开LMDB
                self.multimodal_lmdb_env = lmdb.open(
                    str(lmdb_path), 
                    readonly=True, 
                    lock=False,
                    readahead=False,
                    meminit=False
                )
                
                # 读取元数据
                with self.multimodal_lmdb_env.begin() as txn:
                    meta_bytes = txn.get(b'__global_meta__')
                    if meta_bytes:
                        self.lmdb_global_meta = json.loads(meta_bytes.decode())
                        self.use_batched_loading = True
                        logger.info(
                            "Enable batch loading mode: %s (feature count=%d, Total Dimension=%d)",
                            lmdb_path,
                            len(self.lmdb_global_meta['feature_order']),
                            self.lmdb_global_meta['total_dim']
                        )
                        return
            except Exception as e:
                logger.warning("Failed to open LMDB: %s, Fallback to sample by sample loading", e)
                if self.multimodal_lmdb_env:
                    self.multimodal_lmdb_env.close()
                    self.multimodal_lmdb_env = None
        
        # LMDB不存在，尝试自动生成
        logger.info("Pre computed LMDB of feaute not found: %s", lmdb_path)
        logger.info("Attempt to automatically generate LMDB (first run)...")
        
        try:
            self._generate_lmdb(lmdb_path)
            # 生成后重新初始化
            self._init_batched_loading()
        except Exception as e:
            logger.error("Automatic generation of LMDB failed: %s", e)
            logger.info("Fallback to sample by sample loading mode")
    
    def _generate_lmdb(self, output_path: Path):
        """自动生成LMDB"""
        from subprocess import run, CalledProcessError
        
        # 创建临时配置文件
        import tempfile
        import yaml
        
        temp_config = {
            'data_processing': {
                'all_feature_folder': str(self.all_feature_folder),
                'protein_fasta_path': self.protein_fasta_path
            },
            'feature_files': self.feature_files,
            'projection': {'early_compression_dim': self.early_compression_dim}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name
        
        try:
            # 调用预处理脚本
            cmd = [
                'python', 'mutifeature_tools/precompute_multimodal_lmdb.py',
                '--config', temp_config_path,
                '--output', str(output_path),
                '--map-size', '50'
            ]
            
            logger.info("run command: %s", ' '.join(cmd))
            result = run(cmd, capture_output=True, text=True, check=True)
            logger.info("LMDB generation successful: %s", output_path)
            
        except CalledProcessError as e:
            logger.error("LMDB generation failed:")
            logger.error("STDOUT: %s", e.stdout)
            logger.error("STDERR: %s", e.stderr)
            raise
        finally:
            Path(temp_config_path).unlink(missing_ok=True)
    
    def _load_batch_features_from_lmdb(self, protein_ids: List[str], 
                                       max_length: int, 
                                       device: torch.device) -> torch.Tensor:
        """从LMDB批量加载特征"""
        batch_size = len(protein_ids)
        total_dim = self.lmdb_global_meta['total_dim']
        
        # 预分配张量
        features = torch.zeros((batch_size, max_length, total_dim), 
                              dtype=torch.float32, device='cpu')
        
        with self.multimodal_lmdb_env.begin() as txn:
            for i, protein_id in enumerate(protein_ids):
                # 读取特征
                feat_bytes = txn.get(protein_id.encode())
                if feat_bytes is None:
                    continue
                
                # 读取元数据
                meta_bytes = txn.get(f'__meta_{protein_id}'.encode())
                if meta_bytes:
                    metadata = json.loads(meta_bytes.decode())
                    seq_len = metadata['seq_len']
                else:
                    # 回退：从字节长度推断
                    seq_len = len(feat_bytes) // (total_dim * 4)  # 4 bytes per float32
                
                # 解析特征数组
                feat_array = np.frombuffer(feat_bytes, dtype=np.float32)
                feat_array = feat_array.reshape(seq_len, total_dim)
                
                # Padding/Truncate
                actual_len = min(seq_len, max_length)
                # 优化：直接通过 NumPy 视图赋值，避免 torch.from_numpy 的只读警告和 .copy() 的额外开销
                features[i, :actual_len].numpy()[:] = feat_array[:actual_len]
        
        return features.to(device)

    def _preload_features(self):
        logger.info("Preloading feature files...")
        for feature_name, config in self.feature_configs.items():
            if feature_name == 'protein_length':
                length_cache = {}
                for pid, plen in self.protein_lengths.items():
                    vocab = config.get('vocab_size', 5000)
                    clipped = min(plen, vocab - 1)
                    length_cache[pid] = np.full(plen, clipped, dtype=np.int64)
                if self.feature_cache is not None:
                    self.feature_cache[feature_name] = length_cache
                continue

            file_path = self._resolve_feature_file_path(config.get('file_path'))
            if not file_path:
                continue
            path = Path(file_path)
            if not path.exists():
                (logger.error if self.log_missing_as_error else logger.warning)(
                    "The feature file does not exist: %s", path)
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                cache_dict = {}
                for item in data:
                    pid = item['protein_id']
                    cache_dict[pid] = self._parse_feature_data(item, feature_name, config)
                if self.feature_cache is not None:
                    self.feature_cache[feature_name] = cache_dict
                logger.info("preload %s: %d ", feature_name, len(cache_dict))
            except Exception as e:
                logger.error("Preloading features %s failure: %s", feature_name, e)

    def _parse_feature_data(self, item: Dict[str, Any], feature_name: str, config: Dict[str, Any]) -> np.ndarray:
        if feature_name == 'secondary_structure_features':
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

        if config.get('feature_type') == 'embedding':
            values = [int(float(x)) for x in item.get('embeddings', '').split(',') if x != '']
            return np.array(values, dtype=np.int64)

        if 'hydrophobicity' in feature_name:
            data_field = 'hydrophobicity'
        elif 'sasa' in feature_name:
            data_field = 'sasa'
        elif 'pqr' in feature_name:
            data_field = 'embeddings'
        else:
            data_field = 'embeddings'

        values = [float(x) for x in item.get(data_field, '').split(',') if x != '']
        feature_dim = config.get('_input_dim', config.get('feature_dim', 1))
        arr = np.array(values, dtype=np.float32)
        if feature_dim == 1:
            return arr.reshape(-1, 1)
        usable = (len(arr) // feature_dim) * feature_dim
        return arr[:usable].reshape(-1, feature_dim).astype(np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.normalization_method == 'z_score':
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            return (features - mean) / std
        if self.normalization_method == 'min_max':
            min_v = np.min(features, axis=0, keepdims=True)
            max_v = np.max(features, axis=0, keepdims=True)
            denom = np.where((max_v - min_v) < 1e-8, 1.0, (max_v - min_v))
            return (features - min_v) / denom
        return features

    def _adjust_feature_matrix_length(self, feature_matrix: np.ndarray, target_length: int, feature_dim: int) -> np.ndarray:
        if feature_matrix.shape[0] == target_length:
            return feature_matrix.astype(np.float32)
        if feature_matrix.shape[0] < target_length:
            pad = np.zeros((target_length - feature_matrix.shape[0], feature_dim), dtype=np.float32)
            return np.vstack([feature_matrix, pad])
        return feature_matrix[:target_length].astype(np.float32)

    def _process_embedding_feature(self,
                                   feature_data: np.ndarray,
                                   feature_name: str,
                                   config: Dict[str, Any],
                                   target_length: int,
                                   device: torch.device) -> torch.Tensor:
        if feature_name == 'protein_length':
            if len(feature_data) == 0:
                feature_data = np.zeros(target_length, dtype=np.int64)
            if len(feature_data) == 1:
                feature_data = np.full(target_length, feature_data[0], dtype=np.int64)
        else:
            if len(feature_data) < target_length:
                padded = np.zeros(target_length, dtype=np.int64)
                padded[:len(feature_data)] = feature_data
                feature_data = padded
            elif len(feature_data) > target_length:
                feature_data = feature_data[:target_length]

        vocab_size = config.get('vocab_size', 1000)
        feature_data = np.clip(feature_data, 0, vocab_size - 1)

        ids_tensor = torch.from_numpy(feature_data).to(device)
        embedding_layer = self.embedding_layers[feature_name]
        embedded = embedding_layer(ids_tensor)

        embedded = self._apply_embedding_compression(embedded, feature_name, config)

        if embedded.shape[0] < target_length:
            pad = torch.zeros(target_length - embedded.shape[0], embedded.shape[-1], device=device, dtype=embedded.dtype)
            embedded = torch.cat([embedded, pad], dim=0)
        elif embedded.shape[0] > target_length:
            embedded = embedded[:target_length]

        return embedded

    def _apply_embedding_compression(self, embedded_features: torch.Tensor,
                                     feature_name: str,
                                     config: Dict[str, Any]) -> torch.Tensor:
        embedding_dim = config.get('embedding_dim', embedded_features.shape[-1])
        feature_dim = config.get('feature_dim', embedding_dim)
        if embedding_dim == feature_dim:
            return embedded_features
        if self.compression_method == 'truncate':
            strategy = self.truncate_config.get('strategy', 'head')
            if strategy == 'tail':
                return embedded_features[:, -feature_dim:]
            if strategy == 'random':
                idx = torch.randperm(embedding_dim, device=embedded_features.device)[:feature_dim]
                idx, _ = torch.sort(idx)
                return embedded_features.index_select(dim=1, index=idx)
            return embedded_features[:, :feature_dim]
        if feature_name in self.compression_layers:
            return self.compression_layers[feature_name](embedded_features)
        return embedded_features[:, :feature_dim]
    
    def _apply_embedding_compression_batched(self, embedded_features: torch.Tensor,
                                             feature_name: str,
                                             config: Dict[str, Any]) -> torch.Tensor:
        """批量版本的 embedding 压缩"""
        embedding_dim = config.get('embedding_dim', embedded_features.shape[-1])
        feature_dim = config.get('feature_dim', embedding_dim)
        if embedding_dim == feature_dim:
            return embedded_features
        
        if self.compression_method == 'truncate':
            strategy = self.truncate_config.get('strategy', 'first')
            if strategy == 'random':
                # 对批次使用相同的随机索引
                idx = torch.randperm(embedding_dim, device=embedded_features.device)[:feature_dim]
                idx, _ = torch.sort(idx)
                return embedded_features.index_select(dim=-1, index=idx)
            return embedded_features[:, :, :feature_dim]
        
        if feature_name in self.compression_layers:
            return self.compression_layers[feature_name](embedded_features)
        
        return embedded_features[:, :, :feature_dim]

    def _get_protein_features(self,
                              protein_id: str,
                              target_length: int,
                              device: torch.device) -> Dict[str, torch.Tensor]:
        feature_tensors: Dict[str, torch.Tensor] = {}
        for feature_name, config in self.feature_configs.items():
            try:
                data = None
                if self.cache_features and self.feature_cache is not None and feature_name in self.feature_cache:
                    data = self.feature_cache[feature_name].get(protein_id)
                if data is None:
                    data = self._load_single_feature(protein_id, feature_name, config)

                if config.get('feature_type') == 'embedding':
                    tensor = self._process_embedding_feature(data, feature_name, config, target_length, device)
                else:
                    if config.get('normalize', False):
                        data = self._normalize_features(data)
                    input_dim = config.get('_input_dim', config.get('feature_dim', 1))
                    if len(data) == 0:
                        feature_matrix = np.zeros((target_length, input_dim), dtype=np.float32)
                    else:
                        if data.ndim == 1:
                            data = data.reshape(-1, input_dim)
                        feature_matrix = self._adjust_feature_matrix_length(data, target_length, input_dim)
                    tensor = torch.from_numpy(feature_matrix).to(device)
                feature_tensors[feature_name] = tensor
            except Exception as e:
                logger.warning("获取特征 %s 失败: %s", feature_name, e)
                dim = config.get('_encoded_dim', config.get('feature_dim', 1))
                feature_tensors[feature_name] = torch.zeros(target_length, dim, device=device)
        return feature_tensors

    def _load_single_feature(self, protein_id: str, feature_name: str, config: Dict[str, Any]) -> np.ndarray:
        if feature_name == 'protein_length':
            plen = self.protein_lengths.get(protein_id, 0)
            vocab = config.get('vocab_size', 5000)
            clipped = min(plen, vocab - 1)
            return np.array([clipped], dtype=np.int64)

        file_path = self._resolve_feature_file_path(config.get('file_path'))
        if not file_path:
            (logger.error if self.log_missing_as_error else logger.warning)(
                "特征 %s 缺少 file_path", feature_name)
            return np.zeros((0,), dtype=np.float32)

        path = Path(file_path)
        if not path.exists():
            (logger.error if self.log_missing_as_error else logger.warning)(
                "特征文件不存在: %s", path)
            return np.zeros((0,), dtype=np.float32)

        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item.get('protein_id') == protein_id:
                return self._parse_feature_data(item, feature_name, config)

        if self.handle_missing == 'zero':
            return np.zeros((0,), dtype=np.float32)
        raise KeyError(f"蛋白 {protein_id} 在特征 {feature_name} 中缺失")

    def _run_resnet(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.resnet_blocks, nn.Identity):
            return x
        out = x.transpose(1, 2)
        for block in self.resnet_blocks:
            out = block(out)
        return out.transpose(1, 2)

    def forward(self,
                embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[List[str]] = None):
        if embeddings.dim() != 3:
            raise ValueError("期望输入形状为 [batch_size, seq_len, embedding_dim]")
        batch_size, seq_len, embed_dim = embeddings.shape
        if embed_dim != self.embedding_dim:
            raise ValueError(f"嵌入维度不匹配: got {embed_dim}, expected {self.embedding_dim}")

        if protein_ids is None or len(protein_ids) != batch_size:
            logger.warning("FeatureConcatUnit 缺少protein_ids，跳过多模态融合")
            return embeddings

        device = embeddings.device
        
        # 【新增】批量加载分支
        if self.use_batched_loading:
            return self._forward_batched(embeddings, attention_mask, protein_ids)
        
        # 原有逐样本加载分支
        fused_list: List[torch.Tensor] = []
        low_dim_inputs: List[torch.Tensor] = []
        gate_values_for_loss: List[float] = []  # 存储标量，避免保留计算图
        batch_gate_weights: List[torch.Tensor] = [] # 存储每个样本的门控权重向量 [num_features]
        active_names_for_stats = [name for name in self.feature_names_ordered if name in self.feature_encoders]

        for i, protein_id in enumerate(protein_ids):
            real_length = self.protein_lengths.get(protein_id, seq_len)
            real_length = min(real_length, seq_len)

            feature_map = self._get_protein_features(protein_id, real_length, device)
            encoded_list = []
            gating_obs_list = []
            
            # 按严格顺序编码所有特征
            for feature_name in self.feature_names_ordered:
                if feature_name not in feature_map:
                    continue
                tensor = feature_map[feature_name]
                
                if feature_name not in self.feature_encoders:
                    continue
                
                encoder = self.feature_encoders[feature_name]
                encoded = encoder(tensor.unsqueeze(0)).squeeze(0)  # [seq_len, encoded_dim]
                encoded_list.append(encoded)
                
                # 为全局门控准备输入
                if self.gating_enabled:
                    # 如果需要对齐（大维度->小维度），先通过线性层
                    if feature_name in self.gating_aligners:
                        gating_obs = self.gating_aligners[feature_name](encoded)
                    else:
                        gating_obs = encoded
                    
                    # GAP (Global Average Pooling) -> [early_compression_dim] (e.g. 8)
                    gating_obs = gating_obs.mean(dim=0)
                    gating_obs_list.append(gating_obs)
            
            if not encoded_list:
                continue
            
            # 全局门控：学习每个特征的全局权重
            if self.gating_enabled and self.feature_gating_layers is not None:
                # 拼接所有特征的观察向量 [Num_Features * 8]
                global_input = torch.cat(gating_obs_list, dim=0) 
                
                # 通过全局门控层
                gate_weights = self.feature_gating_layers(global_input)  # [num_features]

                # 记录门控值用于稀疏损失
                gate_values_for_loss.append(gate_weights.mean().detach().item())
            else:
                # 退化门控：固定全局标量权重（默认1.0）
                gate_weights = torch.full(
                    (len(encoded_list),),
                    float(self.gating_constant_value),
                    device=device,
                    dtype=encoded_list[0].dtype,
                )

            # 应用全局权重到每个特征
            for idx, encoded in enumerate(encoded_list):
                encoded_list[idx] = encoded * gate_weights[idx]  # 广播乘法

            # 收集门控权重用于统计
            batch_gate_weights.append(gate_weights.detach())

            # 拼接所有特征
            concatenated = torch.cat(encoded_list, dim=-1)  # [seq_len, total_feature_dim]

            # 归一化（移除了 MLP 融合）
            fused = self.concat_mlp(concatenated)  # 现在只是 LayerNorm

            if real_length < seq_len:
                pad = torch.zeros(seq_len - real_length, fused.shape[-1], device=device, dtype=fused.dtype)
                fused = torch.cat([fused, pad], dim=0)
            elif real_length > seq_len:
                fused = fused[:seq_len]

            projected = self.feature_projection(fused)
            low_dim_inputs.append(projected)

            processed = self._run_resnet(projected.unsqueeze(0)).squeeze(0)
            fused_list.append(processed)

        low_dim_batch = torch.stack(low_dim_inputs, dim=0)  # [batch, seq_len, low_dim]
        processed_batch = torch.stack(fused_list, dim=0)     # [batch, seq_len, low_dim]

        # 映射回 ESM 维度
        aligned = self.final_projection(processed_batch)  # [batch, seq_len, embedding_dim]
        
        # 优化：使用元素级乘法而非加法（可选，配置控制）
        use_multiplicative = self.projection_config.get('use_multiplicative_residual', False)
        if use_multiplicative:
            # 元素级门控融合
            gate = torch.sigmoid(aligned)
            enhanced = embeddings * gate
        else:
            # 传统残差连接
            enhanced = embeddings + aligned

        # 计算并记录本批次的平均门控权重
        gate_stats = {}
        if batch_gate_weights:
            # stack -> [batch, num_features] -> mean -> [num_features]
            avg_weights = torch.stack(batch_gate_weights).mean(dim=0).cpu().numpy()
            
            # 这里的顺序对应 feature_names_ordered 中实际存在的特征
            # 由于前面的循环是按照 feature_names_ordered 遍历并添加有效特征
            # 我们需要重新确定哪些特征被添加到了 encoded_list (即 batch_gate_weights 对应的特征)
            # 在逐样本模式下，假设每个样本的有效特征列表是一致的（由 enabled 配置决定）
            
            active_names = active_names_for_stats
            
            stats_dict = {}
            for idx, name in enumerate(active_names):
                if idx < len(avg_weights):
                    stats_dict[name] = float(avg_weights[idx])
            
            gate_stats['avg_gating_weights'] = stats_dict

            # 累计统计（用于推理后导出 CSV）
            stacked_gate = torch.stack(batch_gate_weights)
            self._accumulate_feature_gating_stats(stacked_gate, active_names)
            
            # 在推理模式下记录权重
            if not self.training:
                self._log_gating_weights(stats_dict)
                # logger.info("Batch Gating Weights: %s", json.dumps(stats_dict, indent=2))

        if self.enable_reconstruction and self.training:
            recon = self.reconstruction_head(processed_batch) if self.reconstruction_head else processed_batch
            recon_loss = F.mse_loss(recon, low_dim_batch, reduction='mean')
            # 使用标量列表计算gate稀疏性损失，避免保留计算图
            if gate_values_for_loss and self.sparsity_loss_weight > 0:
                gate_penalty = sum(gate_values_for_loss) / len(gate_values_for_loss)
                recon_loss = recon_loss + self.sparsity_loss_weight * gate_penalty
            return enhanced, gate_stats, recon_loss

        return enhanced

    def _forward_batched(self,
                        embeddings: torch.Tensor,
                        attention_mask: Optional[torch.Tensor],
                        protein_ids: List[str]) -> torch.Tensor:
        """批量加载模式的前向传播（GPU优化）"""
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        # 批量加载所有特征 [batch, seq_len, total_dim]
        batch_features = self._load_batch_features_from_lmdb(protein_ids, seq_len, device)
        
        # 分割特征维度
        feature_splits = batch_features.split(self.lmdb_global_meta['feature_dims'], dim=-1)
        
        # 批量编码所有特征（完全并行），同时收集门控观察值
        encoded_list = []
        gating_obs_list = []
        active_names = []
        
        for i, feat in enumerate(feature_splits):
            feature_name = self.lmdb_global_meta['feature_order'][i]
            if feature_name not in self.feature_encoders:
                continue
            
            cfg = self.feature_configs.get(feature_name, {})
            encoded_dim = cfg.get('_encoded_dim', 8)  # 获取该特征的目标编码维度
            
            # 处理 embedding 特征
            if cfg.get('feature_type') == 'embedding':
                # feat: [batch, seq_len, 1] 包含ID
                ids = feat.squeeze(-1).long()  # [batch, seq_len]
                vocab_size = cfg.get('vocab_size', 1000)
                ids = torch.clamp(ids, 0, vocab_size - 1)
                
                # 通过 embedding 层
                embedding_layer = self.embedding_layers[feature_name]
                embedded = embedding_layer(ids)  # [batch, seq_len, embedding_dim]
                
                # 应用压缩 (注意：这里现在支持压缩到 compression_dim 而不仅是 8)
                embedded = self._apply_embedding_compression_batched(embedded, feature_name, cfg)
                
                # 通过编码器
                encoder = self.feature_encoders[feature_name]
                encoded = encoder(embedded)  # [batch, seq_len, encoded_dim]
            else:
                # 普通特征：直接编码
                encoder = self.feature_encoders[feature_name]
                encoded = encoder(feat)  # [batch, seq_len, encoded_dim]
            
            encoded_list.append(encoded)
            active_names.append(feature_name)
            
            # 为批量全局门控准备输入
            if self.gating_enabled:
                # 如果需要对齐（大维度->小维度），先通过线性层
                if feature_name in self.gating_aligners:
                    # Input: [batch, seq_len, encoded_dim] -> [batch, seq_len, 8]
                    gating_obs = self.gating_aligners[feature_name](encoded)
                else:
                    gating_obs = encoded
                
                # GAP (Global Average Pooling) -> [batch, early_compression_dim]
                gating_obs = gating_obs.mean(dim=1)
                gating_obs_list.append(gating_obs)
        
        if not encoded_list:
            return embeddings
        
        # 全局门控（批量）
        if self.gating_enabled and self.feature_gating_layers is not None:
            # 拼接所有特征的观察向量 [batch, Num_Features * 8]
            global_input = torch.cat(gating_obs_list, dim=-1)
            
            # 批量门控 [batch, num_features]
            gate_weights = self.feature_gating_layers(global_input)
        else:
            # 退化门控：固定全局标量权重（默认1.0）
            gate_weights = torch.full(
                (batch_size, len(encoded_list)),
                float(self.gating_constant_value),
                device=device,
                dtype=encoded_list[0].dtype,
            )

        # 应用门控到每个特征
        for idx in range(len(encoded_list)):
            # [batch, seq_len, dim] * [batch, 1, 1]
            encoded_list[idx] = encoded_list[idx] * gate_weights[:, idx:idx+1, None]

        gate_stats = {}
        if gate_weights is not None and active_names:
            avg_weights = gate_weights.mean(dim=0).detach().cpu().numpy() # [num_features]
            stats_dict = {}
            for idx, name in enumerate(active_names):
                if idx < len(avg_weights):
                    stats_dict[name] = float(avg_weights[idx])
            gate_stats['avg_gating_weights'] = stats_dict

            # 累计统计（用于推理后导出 CSV）
            self._accumulate_feature_gating_stats(gate_weights, active_names)

            # 推理兼容日志
            if not self.training:
                self._log_gating_weights(stats_dict)
                # logger.info("Batch Gating Weights (Batched): %s", json.dumps(stats_dict, indent=2))
        
        # 修改：使用 cat 而不是 stack，因为维度可能不同
        # 拼接 [batch, seq_len, total_feature_dim] (e.g. 576)
        concatenated = torch.cat(encoded_list, dim=-1)
        
        # LayerNorm
        fused = self.concat_mlp(concatenated)
        
        # 低维投影 [batch, seq_len, 128]
        projected = self.feature_projection(fused)
        
        # ResNet [batch, seq_len, 128]
        if not isinstance(self.resnet_blocks, nn.Identity):
            processed = projected.transpose(1, 2)  # [batch, 128, seq_len]
            for block in self.resnet_blocks:
                processed = block(processed)
            processed = processed.transpose(1, 2)  # [batch, seq_len, 128]
        else:
            processed = projected
        
        # 最终映射 [batch, seq_len, embedding_dim]
        aligned = self.final_projection(processed)
        
        # 残差连接
        use_multiplicative = self.projection_config.get('use_multiplicative_residual', False)
        if use_multiplicative:
            gate = torch.sigmoid(aligned)
            enhanced = embeddings * gate
        else:
            enhanced = embeddings + aligned
        
        # 重构损失（训练时）
        if self.enable_reconstruction and self.training:
            recon = self.reconstruction_head(processed) if self.reconstruction_head else processed
            recon_loss = F.mse_loss(recon, projected, reduction='mean')
            return enhanced, gate_stats, recon_loss
        
        return enhanced

    def set_current_protein_ids(self, protein_ids: List[str]):
        self._current_protein_ids = protein_ids


class LightweightFeatureEncoder(nn.Module):
    """
    轻量级特征编码器：使用 1D-CNN
    
    架构：Linear 降维 → LayerNorm → 1D-CNN → Activation
    """

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        activation = (activation or 'gelu').lower()
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }.get(activation, nn.GELU())

        # 1. 线性投影到目标维度
        self.project = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        # 2. 轻量级 1D-CNN 捕获局部依赖
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            output_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=padding,
            groups=max(1, output_dim // 4)  # 使用分组卷积进一步减少参数
        )
        
        self.activation = act_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feature_seq: [batch, seq_len, input_dim] or [seq_len, input_dim]
        Returns:
            [batch, seq_len, output_dim] or [seq_len, output_dim]
        """
        # 线性投影
        x = self.project(feature_seq)  # [..., seq_len, output_dim]
        x = self.norm(x)
        
        # 1D-CNN 需要 [batch, channels, length] 格式
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)  # [1, seq_len, output_dim]
        
        x = x.transpose(1, 2)  # [batch, output_dim, seq_len]
        x = self.conv(x)       # [batch, output_dim, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, output_dim]
        
        if not is_batched:
            x = x.squeeze(0)   # [seq_len, output_dim]
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


# 保留旧类名作为别名，向后兼容
FeatureSequenceEncoder = LightweightFeatureEncoder
