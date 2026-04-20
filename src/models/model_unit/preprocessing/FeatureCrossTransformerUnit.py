import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Mapping, cast, Union, Tuple, Set
import math
from pathlib import Path
import json
import numpy as np
import copy
from Bio import SeqIO

logger = logging.getLogger(__name__)


# 导入其他预处理模块
from .FlashTransformerEncoderLayer import FlashTransformerEncoderLayer
from .PositionalEncoding import PositionalEncoding
from .FlashTransformerDecoderLayer import FlashTransformerDecoderLayer
class FeatureCrossTransformerUnit(nn.Module):
    """
    特征交叉Transformer预处理单元
    
    使用完整的Transformer架构将残基级别特征嵌入到ESM嵌入中：
    - 特征作为源序列(Source)通过Transformer编码器
    - ESM嵌入作为目标序列(Target)
    - 使用Transformer解码器进行交叉注意力融合
    - 支持多种特征类型和完整的Transformer架构
    """
    
    def __init__(self,
                 embedding_dim: int,
                 feature_files: Dict[str, Dict[str, Any]],
                 transformer: Optional[Dict[str, Any]] = None,
                 projection: Optional[Dict[str, Any]] = None,
                 data_processing: Optional[Dict[str, Any]] = None,
                 embedding: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        初始化特征交叉Transformer单元
        
        Args:
            embedding_dim (int): ESM嵌入维度
            feature_files (Dict): 特征文件配置
            transformer (Dict): Transformer配置
            projection (Dict): 投影层配置  
            data_processing (Dict): 数据处理配置
            embedding (Dict): 嵌入层配置（用于ID类特征）
        """
        super(FeatureCrossTransformerUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # 默认配置
        if transformer is None:
            transformer = {}
        if projection is None:
            projection = {}
        if data_processing is None:
            data_processing = {}
        if embedding is None:
            embedding = {}
            
        self.transformer_config = transformer
        self.projection_config = projection
        self.data_processing_config = data_processing
        self.embedding_config = embedding
        
        # Transformer参数
        self.num_encoder_layers = transformer.get('num_encoder_layers', 3)
        self.num_decoder_layers = transformer.get('num_decoder_layers', 3)
        self.num_heads = transformer.get('num_heads', 8)
        self.dim_feedforward = transformer.get('dim_feedforward', 2048)
        self.transformer_dropout = transformer.get('dropout', 0.1)
        self.transformer_activation = transformer.get('activation', 'gelu')
        self.use_flash_attention = transformer.get('use_flash_attention', True)
        
        # 投影层参数
        self.projection_hidden_dim = projection.get('hidden_dim', None)
        self.projection_dropout = projection.get('dropout', 0.1)
        self.projection_activation = projection.get('activation', 'GELU')
        
        # 数据处理参数
        self.protein_fasta_path = data_processing.get('protein_fasta_path', 'dataset/S1/protein.fasta')
        self.cache_features = data_processing.get('cache_features', True)
        self.normalization_method = data_processing.get('normalization_method', 'z_score')
        self.handle_missing = data_processing.get('handle_missing', 'zero')
        # 新增：统一特征根目录与缺失报错控制
        self.all_feature_folder = data_processing.get('all_feature_folder', None)
        self.log_missing_as_error = data_processing.get('log_missing_as_error', False)
        
        # 特征配置处理
        self.feature_files = feature_files
        self.enabled_features = {}
        self.feature_configs = {}
        self.total_feature_dim = 0
        self.embedding_features = {}
        self.feature_input_dims: Dict[str, int] = {}
        self.feature_encoded_dims: Dict[str, int] = {}
        self.length_mismatch_records: Set[Tuple[str, str, str]] = set()
        
        for feature_name, config in feature_files.items():
            if not config.get('enabled', False):
                continue

            feature_config = copy.deepcopy(config)

            raw_dim = feature_config.get('raw_dim') or feature_config.get('input_dim')
            if raw_dim is None:
                if feature_config.get('feature_type') == 'embedding':
                    raw_dim = feature_config.get('embedding_dim', embedding_dim)
                elif feature_name == 'secondary_structure_features':
                    raw_dim = 3
                else:
                    raw_dim = feature_config.get('raw_feature_dim', feature_config.get('feature_dim', 1))

            encoded_dim = feature_config.get('feature_dim', raw_dim)

            feature_config['_input_dim'] = int(raw_dim)
            feature_config['_encoded_dim'] = int(encoded_dim)

            if feature_config.get('feature_type') == 'embedding':
                self.embedding_features[feature_name] = feature_config
            else:
                self.enabled_features[feature_name] = feature_config

            self.feature_configs[feature_name] = feature_config
            self.feature_input_dims[feature_name] = feature_config['_input_dim']
            self.feature_encoded_dims[feature_name] = feature_config['_encoded_dim']
            self.total_feature_dim += feature_config['_encoded_dim']
        
        logger.info(f"启用的特征: {list(self.enabled_features.keys())}")
        logger.info(f"嵌入特征: {list(self.embedding_features.keys())}")
        logger.info(f"总特征维度: {self.total_feature_dim}")
        
        if self.total_feature_dim == 0:
            raise ValueError("没有启用任何特征，请检查feature_files配置")
        
        # 选择激活函数
        if self.projection_activation == "ReLU":
            self.activation_fn = nn.ReLU()
        elif self.projection_activation == "GELU":
            self.activation_fn = nn.GELU()
        elif self.projection_activation == "SiLU":
            self.activation_fn = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {self.projection_activation}")
        
        # 创建嵌入层（用于ID类特征）
        self.embedding_layers = nn.ModuleDict()
        
        for feature_name, config in self.embedding_features.items():
            vocab_size = config.get('vocab_size', 1000)
            emb_dim = config.get('embedding_dim', 64)
            padding_idx = self.embedding_config.get('padding_idx', None)
            
            # 创建嵌入层
            embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=emb_dim,
                padding_idx=padding_idx
            )
            
            # 添加dropout
            emb_dropout = self.embedding_config.get('dropout', 0.1)
            if emb_dropout > 0:
                embedding_layer = nn.Sequential(
                    embedding_layer,
                    nn.Dropout(emb_dropout)
                )
            
            self.embedding_layers[feature_name] = embedding_layer
            logger.info(f"创建嵌入层 {feature_name}: vocab_size={vocab_size}, emb_dim={emb_dim}")
        
        # 创建嵌入压缩层（用于处理embedding_dim != feature_dim的情况）
        self.compression_layers = self._build_compression_layers()
        self.feature_encoder_config = data_processing.get('feature_encoder', {})
        self.feature_fusion_config = data_processing.get('feature_fusion', {})
        
        # 构建特征投影网络（将特征投影到embedding_dim）
        if self.projection_hidden_dim is not None:
            # 带隐藏层的投影
            self.feature_projection = nn.Sequential(
                nn.Linear(self.total_feature_dim, self.projection_hidden_dim),
                self.activation_fn,
                nn.Dropout(self.projection_dropout),
                nn.Linear(self.projection_hidden_dim, embedding_dim)
            )
        else:
            # 直接投影
            self.feature_projection = nn.Linear(self.total_feature_dim, embedding_dim)
        
        # 尝试导入flash attention
        self.flash_attn_available = False
        if self.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.flash_attn_available = True
                logger.info("Flash Attention可用，将使用Flash Attention优化Transformer")
            except ImportError:
                logger.warning("Flash Attention不可用，将使用标准Transformer")
                self.flash_attn_available = False
        
        # 初始化Flash Transformer状态
        self.flash_transformer_available = False
        
        # 构建Transformer架构
        if self.use_flash_attention:
            # 尝试使用真正的Flash Attention Transformer
            self._build_flash_transformer()
        else:
            # 使用标准的PyTorch Transformer
            self.flash_transformer_available = False
            self._build_standard_transformer()
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=5000)
        
        # 特征重构配置
        self.enable_reconstruction = transformer.get('enable_reconstruction', False)
        self.reconstruction_loss_weight = transformer.get('reconstruction_loss_weight', 0.1)
        self.reconstruction_loss_type = transformer.get('reconstruction_loss_type', 'mse')  # 'mse' or 'l1'
        
        # 构建特征重构头（如果启用）
        if self.enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embedding_dim, self.projection_hidden_dim if self.projection_hidden_dim else embedding_dim),
                self.activation_fn,
                nn.Dropout(self.projection_dropout),
                nn.Linear(self.projection_hidden_dim if self.projection_hidden_dim else embedding_dim, self.total_feature_dim)
            )
            logger.info(f"启用特征重构: 权重={self.reconstruction_loss_weight}, 损失类型={self.reconstruction_loss_type}")
        
        # 特征缓存和蛋白长度
        self.feature_cache = {} if self.cache_features else None
        self.protein_lengths = {}
        
        # 加载蛋白长度信息和预加载特征
        self._load_protein_lengths()
        if self.cache_features:
            self._preload_features()

        # 构建特征编码器与融合层
        self.feature_encoders = nn.ModuleDict()
        self._build_feature_encoders()
        self._build_feature_fusion_layer()
        
        logger.debug(f"创建特征交叉Transformer单元: embedding_dim={embedding_dim}, "
                    f"feature_dim={self.total_feature_dim}, "
                    f"encoder_layers={self.num_encoder_layers}, "
                    f"decoder_layers={self.num_decoder_layers}")

    def _build_standard_transformer(self):
        """构建标准PyTorch Transformer"""
        # Transformer编码器层（处理特征序列）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.transformer_activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_encoder_layers
        )
        
        # Transformer解码器层（将特征融合到ESM嵌入）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.transformer_activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_decoder_layers
        )
    
    def _build_flash_transformer(self):
        """构建真正的Flash Attention优化Transformer"""
        try:
            # 创建Flash Attention编码器
            flash_encoder_layers = []
            for _ in range(self.num_encoder_layers):
                layer = FlashTransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.transformer_dropout,
                    activation=self.transformer_activation
                )
                flash_encoder_layers.append(layer)
            
            self.flash_encoder_layers = nn.ModuleList(flash_encoder_layers)
            
            # 创建Flash Attention解码器
            flash_decoder_layers = []
            for _ in range(self.num_decoder_layers):
                layer = FlashTransformerDecoderLayer(
                    d_model=self.embedding_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.transformer_dropout,
                    activation=self.transformer_activation
                )
                flash_decoder_layers.append(layer)
            
            self.flash_decoder_layers = nn.ModuleList(flash_decoder_layers)
            
            # 标记Flash Transformer可用
            self.flash_transformer_available = True
            logger.info(f"Flash Attention Transformer构建成功: "
                       f"编码器{self.num_encoder_layers}层, 解码器{self.num_decoder_layers}层")
            
        except Exception as e:
            logger.warning(f"Flash Attention Transformer构建失败，回退到标准Transformer: {e}")
            self.flash_transformer_available = False
            self._build_standard_transformer()
    
    # 复用FeatureCrossAttentionUnit的特征处理方法
    def _load_protein_lengths(self):
        """从FASTA文件加载蛋白长度信息"""
        try:
            fasta_path = Path(self.protein_fasta_path)
            if fasta_path.exists():
                for record in SeqIO.parse(fasta_path, "fasta"):
                    self.protein_lengths[record.id] = len(record.seq)
                logger.info(f"加载了{len(self.protein_lengths)}个蛋白的长度信息")
            else:
                logger.warning(f"FASTA文件不存在: {fasta_path}")
        except Exception as e:
            logger.warning(f"加载蛋白长度失败: {e}")
    
    def _preload_features(self):
        """预加载所有特征文件到内存"""
        logger.info("开始预加载特征文件...")
        for feature_name, config in self.feature_configs.items():
            try:
                # 处理特殊的内部生成特征（不需要文件路径）
                if feature_name == 'protein_length':
                    # 为所有已知蛋白生成长度特征
                    feature_dict = {}
                    for protein_id, length in self.protein_lengths.items():
                        # 创建虚拟item用于解析
                        dummy_item = {'protein_id': protein_id}
                        feature_dict[protein_id] = self._parse_feature_data(
                            dummy_item, feature_name, config
                        )
                    
                    if self.feature_cache is not None:
                        self.feature_cache[feature_name] = feature_dict
                    logger.info(f"预加载内部特征 {feature_name}: {len(feature_dict)} 条记录")
                    continue  # 跳过文件处理部分
                
                # 处理需要文件路径的特征
                file_path = self._resolve_feature_file_path(config.get('file_path'))
                if not file_path:
                    (logger.error if self.log_missing_as_error else logger.warning)(
                        f"特征 {feature_name} 缺少file_path配置")
                    continue
                
                # 处理外部文件特征
                file_path = Path(file_path)
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 转换为字典格式便于查找
                    feature_dict = {}
                    for item in data:
                        protein_id = item['protein_id']
                        # 根据特征类型处理数据
                        try:
                            feature_dict[protein_id] = self._parse_feature_data(
                                item, feature_name, config
                            )
                        except Exception as e:
                            logger.warning(f"解析 {feature_name} 中蛋白 {protein_id} 失败: {e}")
                    
                    if self.feature_cache is not None:
                        self.feature_cache[feature_name] = feature_dict
                    logger.info(f"预加载特征 {feature_name}: {len(feature_dict)} 条记录")
                else:
                    (logger.error if self.log_missing_as_error else logger.warning)(
                        f"特征文件不存在: {file_path}")
            except Exception as e:
                logger.error(f"预加载特征 {feature_name} 失败: {e}")
    
    def _parse_feature_data(self, item: Dict, feature_name: str, config: Dict) -> np.ndarray:
        """解析特征数据（复用FeatureCrossAttentionUnit的实现）"""
        
        # 处理特殊的内部生成特征
        if feature_name == 'protein_length':
            # 蛋白长度特征：从FASTA长度信息生成
            protein_id = item['protein_id']
            if protein_id in self.protein_lengths:
                length = self.protein_lengths[protein_id]
                # 将长度限制在vocab_size范围内
                vocab_size = config.get('vocab_size', 5000)
                length = min(length, vocab_size - 1)
                return np.array([length], dtype=np.int64)  # 返回单个长度值
            else:
                logger.warning(f"蛋白 {protein_id} 长度信息缺失，使用默认值100")
                return np.array([100], dtype=np.int64)  # 默认长度
        
        # 确定数据字段名
        if feature_name == 'secondary_structure_features':
            data_field = 'secondary_structure'
        elif 's1ssttoken' in feature_name:
            data_field = 'embeddings'
        elif 'hydrophobicity' in feature_name:
            data_field = 'hydrophobicity'
        elif 'sasa' in feature_name:
            data_field = 'sasa'
        elif 'pqr' in feature_name:
            data_field = 'embeddings'
        else:
            data_field = 'embeddings'  # 默认字段名
        
        data_str = item[data_field]
        
        if feature_name == 'secondary_structure_features':
            # 特殊处理二级结构：解析独热编码
            data_str = data_str.replace(' ', '').replace('\n', '')
            segments = data_str.split('],[')
            
            # 处理首尾的方括号
            segments[0] = segments[0].lstrip('[')
            segments[-1] = segments[-1].rstrip(']')
            
            # 解析每个三元组
            features = []
            for segment in segments:
                values = [float(x) for x in segment.split(',')]
                if len(values) == 3:
                    features.extend(values)  # 展平3维独热编码
                else:
                    logger.warning(f"二级结构数据格式异常: {segment}")
                    features.extend([0.0, 0.0, 1.0])  # 默认为coil
            
            return np.array(features, dtype=np.float32)
        elif config.get('feature_type') == 'embedding':
            # ID类特征：解析为整数ID
            values = [int(float(x)) for x in data_str.split(',')]  # 先转float再转int，处理科学计数法
            return np.array(values, dtype=np.int64)
        else:
            # 普通特征：按逗号分割
            values = [float(x) for x in data_str.split(',')]
            return np.array(values, dtype=np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        if self.normalization_method == 'z_score':
            # Z-score标准化
            mean = np.mean(features)
            std = np.std(features)
            if std > 1e-8:  # 避免除零
                return (features - mean) / std
            else:
                return features - mean
        elif self.normalization_method == 'min_max':
            # Min-Max标准化到[0,1]
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val - min_val > 1e-8:
                return (features - min_val) / (max_val - min_val)
            else:
                return features * 0
        else:
            return features
    
    def _process_embedding_feature(self, feature_data: np.ndarray, feature_name: str,
                                   config: Dict, target_length: int, device: torch.device,
                                   protein_id: str) -> torch.Tensor:
        """处理嵌入类特征，保持二维序列格式并记录长度差异"""
        
        # 处理特殊的蛋白长度特征（全局特征，每个位置都相同）
        if feature_name == 'protein_length':
            # 蛋白长度是全局特征，每个残基位置都使用相同的长度值
            if len(feature_data) == 1:
                # 将单个长度值扩展到所有位置
                length_value = feature_data[0]
                feature_data = np.full(target_length, length_value, dtype=np.int64)
            else:
                logger.warning(f"蛋白长度特征数据格式异常，长度: {len(feature_data)}")
                # 使用第一个值或默认值
                length_value = feature_data[0] if len(feature_data) > 0 else 100
                feature_data = np.full(target_length, length_value, dtype=np.int64)
        else:
            # 调整ID序列长度（针对残基级别的特征）
            if len(feature_data) < target_length:
                # 填充0（通常表示padding或unknown）
                padded = np.zeros(target_length, dtype=np.int64)
                padded[:len(feature_data)] = feature_data
                feature_data = padded
            elif len(feature_data) > target_length:
                # 截断到目标长度
                feature_data = feature_data[:target_length]
                logger.warning(f"ID特征 {feature_name} 长度超过目标长度，已截断")
        
        # 确保ID在有效范围内
        vocab_size = config.get('vocab_size', 1000)
        feature_data = np.clip(feature_data, 0, vocab_size - 1)
        original_length = len(feature_data)
        if original_length != target_length and original_length > 0:
            action = 'truncate' if original_length > target_length else 'pad'
            self._report_length_mismatch(
                protein_id,
                feature_name,
                feature_length=original_length,
                esm_length=target_length,
                action=action,
                detail=f"embedding ids (dim={config.get('embedding_dim', self.embedding_dim)})"
            )
        
        # 转换为张量并通过嵌入层
        ids_tensor = torch.from_numpy(feature_data).to(device)  # [target_length]
        
        # 获取嵌入层
        embedding_layer = self.embedding_layers[feature_name]
        
        # 通过嵌入层：[target_length] -> [target_length, embedding_dim]
        embedded_features = embedding_layer(ids_tensor)
        
        # 应用嵌入压缩（如果需要）
        if hasattr(self, 'compression_method'):
            embedded_features = self._apply_embedding_compression(embedded_features, feature_name, config)

        current_length = embedded_features.shape[0]
        if current_length != target_length:
            if current_length < target_length:
                pad = torch.zeros(target_length - current_length, embedded_features.shape[-1],
                                   device=device, dtype=embedded_features.dtype)
                embedded_features = torch.cat([embedded_features, pad], dim=0)
            else:
                embedded_features = embedded_features[:target_length]
            # 二次保证长度一致（若嵌入层或压缩层改变长度）
            action = 'truncate' if current_length > target_length else 'pad'
            self._report_length_mismatch(
                protein_id,
                feature_name,
                feature_length=current_length,
                esm_length=target_length,
                action=action,
                detail="post-embedding"
            )
        
        return embedded_features
    
    def _adjust_feature_matrix_length(self, feature_matrix: np.ndarray, target_length: int, feature_dim: int) -> np.ndarray:
        """调整特征矩阵长度以匹配序列长度，避免扁平化"""
        current_length = feature_matrix.shape[0]
        if current_length == target_length:
            return feature_matrix

        if current_length < target_length:
            padding = np.zeros((target_length - current_length, feature_dim), dtype=feature_matrix.dtype)
            return np.vstack([feature_matrix, padding])

        logger.warning(f"特征长度 {current_length} 超过目标长度 {target_length}，已截断")
        return feature_matrix[:target_length]

    def _report_length_mismatch(self, protein_id: str, feature_name: str, feature_length: int,
                                esm_length: int, action: str, detail: Optional[str] = None) -> None:
        """记录长度不一致的详细信息，只提示一次以避免刷屏"""
        detail_key = detail or action
        key = (protein_id, feature_name, detail_key)
        if key in self.length_mismatch_records:
            return

        self.length_mismatch_records.add(key)
        action_cn = '截断' if action == 'truncate' else '填充'
        message = (f"蛋白 {protein_id} 的特征 {feature_name} 长度 {feature_length} "
                   f"与 ESM 序列长度 {esm_length} 不一致，将执行{action_cn}处理")
        if detail:
            message += f" | {detail}"
        logger.warning(message)
    
    def _get_protein_features(
        self,
        protein_id: str,
        target_length: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """获取指定蛋白的所有启用特征（保持二维形状）"""
        feature_tensors: Dict[str, torch.Tensor] = {}

        for feature_name, config in self.feature_configs.items():
            try:
                feature_data = None
                if self.cache_features and self.feature_cache is not None and feature_name in self.feature_cache:
                    feature_data = self.feature_cache[feature_name].get(protein_id)

                if feature_data is None:
                    feature_data = self._load_single_feature(protein_id, feature_name, config)

                if config.get('feature_type') == 'embedding':
                    feature_tensor = self._process_embedding_feature(
                        feature_data, feature_name, config, target_length, device, protein_id
                    )
                else:
                    if config.get('normalize', False):
                        feature_data = self._normalize_features(feature_data)

                    input_dim = config.get('_input_dim', config.get('feature_dim', 1))
                    if len(feature_data) % input_dim != 0 and len(feature_data) > 0:
                        logger.warning(
                            f"特征 {feature_name} 数据长度 {len(feature_data)} 不能被输入维度 {input_dim} 整除，自动截断")
                        usable = (len(feature_data) // input_dim) * input_dim
                        feature_data = feature_data[:usable]

                    if len(feature_data) == 0:
                        feature_matrix = np.zeros((1, input_dim), dtype=np.float32)
                    else:
                        feature_matrix = feature_data.reshape(-1, input_dim)

                    original_length = feature_matrix.shape[0]
                    if original_length != target_length:
                        action = 'truncate' if original_length > target_length else 'pad'
                        self._report_length_mismatch(
                            protein_id,
                            feature_name,
                            feature_length=original_length,
                            esm_length=target_length,
                            action=action,
                            detail=f"per-residue dim={input_dim}"
                        )

                    feature_matrix = self._adjust_feature_matrix_length(feature_matrix, target_length, input_dim)
                    feature_tensor = torch.from_numpy(feature_matrix).to(device)

                feature_tensors[feature_name] = feature_tensor

            except Exception as e:
                logger.warning(f"获取特征 {feature_name} 失败: {e}")
                input_dim = config.get('_input_dim', config.get('feature_dim', 1))
                zeros = torch.zeros(target_length, input_dim, device=device, dtype=torch.float32)
                feature_tensors[feature_name] = zeros

        return feature_tensors
    
    def _load_single_feature(self, protein_id: str, feature_name: str, config: Dict) -> np.ndarray:
        """实时加载单个特征（不推荐，仅作后备）"""
        try:
            # 处理特殊的内部生成特征
            if feature_name == 'protein_length':
                dummy_item = {'protein_id': protein_id}
                return self._parse_feature_data(dummy_item, feature_name, config)
            
            # 处理需要文件的特征
            file_path = self._resolve_feature_file_path(config.get('file_path'))
            if not file_path:
                (logger.error if self.log_missing_as_error else logger.warning)(
                    f"特征 {feature_name} 缺少file_path配置")
                return np.array([], dtype=np.float32)
            
            # 处理外部文件特征
            file_path = Path(file_path)
            if not file_path.exists():
                (logger.error if self.log_missing_as_error else logger.warning)(
                    f"特征文件不存在: {file_path}")
                return np.array([], dtype=np.float32)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                if item['protein_id'] == protein_id:
                    return self._parse_feature_data(item, feature_name, config)
            
            (logger.error if self.log_missing_as_error else logger.warning)(
                f"蛋白 {protein_id} 在 {feature_name} 中未找到")
            return np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"加载特征 {feature_name} 失败: {e}")
            return np.array([], dtype=np.float32)
    
    def forward(self, 
                embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        前向传播：使用Transformer将特征嵌入到ESM嵌入中
        
        Args:
            embeddings: ESM嵌入 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            protein_ids: 蛋白ID列表 [batch_size]，用于获取特征
        
        Returns:
            torch.Tensor: 增强后的嵌入 [batch_size, seq_len, embedding_dim]
        """
        if embeddings.dim() != 3:
            raise ValueError("期望输入形状为 [batch_size, seq_len, embedding_dim]")
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        if embed_dim != self.embedding_dim:
            raise ValueError(f"嵌入维度不匹配: got {embed_dim}, expected {self.embedding_dim}")
        
        # 如果没有提供protein_ids，返回原始嵌入
        if protein_ids is None or len(protein_ids) != batch_size:
            logger.warning(f"FeatureCrossTransformer未收到protein_ids (got {type(protein_ids)}, batch_size={batch_size})，跳过多模态特征融合")
            return embeddings
        
        logger.debug(f"FeatureCrossTransformer处理 {len(protein_ids)} 个蛋白: {protein_ids[:3]}...")
        
        device = embeddings.device
        
        batch_features = []
        for i, protein_id in enumerate(protein_ids):
            # 第一步：获取蛋白质真实长度
            if protein_id in self.protein_lengths:
                real_length = self.protein_lengths[protein_id]
            else:
                # 后备：使用ESM bucketed长度
                real_length = seq_len
                logger.warning(f"蛋白 {protein_id} 在FASTA中未找到，使用ESM长度 {seq_len}")
            
            # 第二步：用真实长度加载和验证多模态特征
            per_feature_data = self._get_protein_features(
                protein_id,
                target_length=real_length,
                device=device,
                attention_mask=None  # 暂不传mask，验证原始数据
            )
            
            # 第三步：编码特征（使用真实长度）
            encoded = self._encode_per_feature_sequences(
                per_feature_data,
                real_length,
                device,
                attention_mask=None
            )
            
            # 第四步：填充/截断到ESM bucketed长度
            if real_length < seq_len:
                # 填充
                padding = torch.zeros(seq_len - real_length, encoded.shape[-1],
                                     dtype=encoded.dtype, device=device)
                encoded = torch.cat([encoded, padding], dim=0)
            elif real_length > seq_len:
                # 截断
                logger.warning(f"蛋白 {protein_id} 真实长度 {real_length} > ESM长度 {seq_len}，已截断")
                encoded = encoded[:seq_len]
            
            batch_features.append(encoded)

        batch_features = torch.stack(batch_features, dim=0)

        if self.feature_fusion is not None:
            fusion_mask = None
            if attention_mask is not None:
                fusion_mask = ~(attention_mask.bool())
            batch_features = self.feature_fusion(batch_features, src_key_padding_mask=fusion_mask)

        projected_features = self.feature_projection(batch_features)  # [batch_size, seq_len, embedding_dim]
        
        # 添加位置编码
        projected_features = self.positional_encoding(projected_features)
        esm_with_pos = self.positional_encoding(embeddings)
        
        # 使用Transformer进行特征嵌入
        enhanced_embeddings, encoded_features = self._apply_transformer_fusion(
            projected_features, esm_with_pos, attention_mask
        )
        
        # 如果启用重构且在训练模式，计算重构特征和损失
        if self.enable_reconstruction and self.training:
            # 从编码后的特征重构原始特征
            reconstructed_features = self.reconstruction_head(encoded_features)  # [batch_size, seq_len, total_feature_dim]
            
            # 计算重构损失
            reconstruction_loss = self._compute_reconstruction_loss(
                reconstructed_features, batch_features, attention_mask
            )
            
            return enhanced_embeddings, reconstructed_features, reconstruction_loss
        
        return enhanced_embeddings
    
    def _apply_transformer_fusion(self, 
                                 feature_embeddings: torch.Tensor,
                                 esm_embeddings: torch.Tensor, 
                                 attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用Transformer将特征融合到ESM嵌入中"""
        
        # 创建注意力掩码
        src_key_padding_mask = None
        tgt_key_padding_mask = None
        if attention_mask is not None:
            # True表示需要被忽略的位置
            src_key_padding_mask = (attention_mask == 0)
            tgt_key_padding_mask = (attention_mask == 0)
        
        try:
            if self.flash_transformer_available:
                # 使用Flash Attention Transformer
                return self._flash_transformer_fusion(
                    feature_embeddings, esm_embeddings, 
                    src_key_padding_mask, tgt_key_padding_mask
                )
            else:
                # 使用标准Transformer
                return self._standard_transformer_fusion(
                    feature_embeddings, esm_embeddings,
                    src_key_padding_mask, tgt_key_padding_mask
                )
        except Exception as e:
            logger.warning(f"Flash Attention Transformer融合失败，回退到标准Transformer: {e}")
            return self._standard_transformer_fusion(
                feature_embeddings, esm_embeddings,
                src_key_padding_mask, tgt_key_padding_mask
            )
    
    def _flash_transformer_fusion(self, 
                                 feature_embeddings: torch.Tensor,
                                 esm_embeddings: torch.Tensor,
                                 src_key_padding_mask: Optional[torch.Tensor] = None,
                                 tgt_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用Flash Attention Transformer进行特征融合"""
        
        # 第一步：使用Flash编码器编码特征序列
        encoded_features = feature_embeddings
        for encoder_layer in self.flash_encoder_layers:
            encoded_features = encoder_layer(
                src=encoded_features,
                src_key_padding_mask=src_key_padding_mask
            )
        
        # 第二步：使用Flash解码器进行交叉注意力融合
        fused_embeddings = esm_embeddings
        for decoder_layer in self.flash_decoder_layers:
            fused_embeddings = decoder_layer(
                tgt=fused_embeddings,
                memory=encoded_features,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
        
        return fused_embeddings, encoded_features
    
    def _standard_transformer_fusion(self,
                                   feature_embeddings: torch.Tensor,
                                   esm_embeddings: torch.Tensor,
                                   src_key_padding_mask: Optional[torch.Tensor] = None,
                                   tgt_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用标准PyTorch Transformer进行特征融合"""
        
        # 第一步：编码特征序列
        encoded_features = self.transformer_encoder(
            feature_embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 第二步：解码融合
        fused_embeddings = self.transformer_decoder(
            tgt=esm_embeddings,
            memory=encoded_features,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return fused_embeddings, encoded_features
    
    def _compute_reconstruction_loss(self, 
                                   reconstructed_features: torch.Tensor,
                                   original_features: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算特征重构损失
        
        Args:
            reconstructed_features: 重构的特征 [batch_size, seq_len, total_feature_dim]
            original_features: 原始特征 [batch_size, seq_len, total_feature_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            torch.Tensor: 重构损失标量
        """
        # 应用注意力掩码（忽略填充位置）
        if attention_mask is not None:
            # 扩展掩码到特征维度
            mask = attention_mask.unsqueeze(-1).expand_as(reconstructed_features)  # [batch_size, seq_len, total_feature_dim]
            
            # 只计算非填充位置的损失
            reconstructed_masked = reconstructed_features * mask
            original_masked = original_features * mask
            
            # 计算损失
            if self.reconstruction_loss_type == 'mse':
                loss = F.mse_loss(reconstructed_masked, original_masked, reduction='sum')
            elif self.reconstruction_loss_type == 'l1':
                loss = F.l1_loss(reconstructed_masked, original_masked, reduction='sum')
            else:
                raise ValueError(f"不支持的重构损失类型: {self.reconstruction_loss_type}")
            
            # 标准化：除以有效位置数
            valid_positions = mask.sum()
            if valid_positions > 0:
                loss = loss / valid_positions
            
        else:
            # 没有掩码时计算所有位置的损失
            if self.reconstruction_loss_type == 'mse':
                loss = F.mse_loss(reconstructed_features, original_features)
            elif self.reconstruction_loss_type == 'l1':
                loss = F.l1_loss(reconstructed_features, original_features)
            else:
                raise ValueError(f"不支持的重构损失类型: {self.reconstruction_loss_type}")
        
        return loss

    def set_current_protein_ids(self, protein_ids: List[str]):
        """设置当前批次的蛋白质ID（用于兼容现有接口）"""
        self._current_protein_ids = protein_ids

    def _resolve_feature_file_path(self, file_path_value: Optional[Union[str, bool]]) -> Optional[str]:
        """根据all_feature_folder解析特征文件路径。
        支持以下情况：
        - file_path为绝对/相对路径：如果存在则直接使用；若不存在且提供了all_feature_folder，则在其下拼接
        - file_path为仅文件名：若提供了all_feature_folder，则拼接；否则按相对路径处理
        - file_path为False/None：返回None
        """
        if not file_path_value or str(file_path_value).lower() == 'false':
            return None
        p = Path(str(file_path_value))
        # 如果给定路径已存在，直接返回
        if p.exists():
            return str(p)
        # 若不存在且有根目录，尝试拼接
        if self.all_feature_folder:
            candidate = Path(self.all_feature_folder) / p
            return str(candidate)
        # 否则返回原始相对路径（由调用方再判断是否存在）
        return str(p)

    def _build_feature_encoders(self):
        """构建逐特征编码器，将原始序列映射到目标feature_dim"""
        if not self.feature_configs:
            return

        default_heads = self.feature_encoder_config.get('num_heads', 4)
        default_layers = self.feature_encoder_config.get('num_layers', 1)
        default_dropout = self.feature_encoder_config.get('dropout', 0.1)
        default_activation = self.feature_encoder_config.get('activation', 'gelu')

        for feature_name, config in self.feature_configs.items():
            input_dim = config.get('_input_dim', config.get('feature_dim', 1))
            output_dim = config.get('_encoded_dim', input_dim)
            encoder_cfg = config.get('encoder', {})

            encoder = FeatureSequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=encoder_cfg.get('num_layers', default_layers),
                num_heads=encoder_cfg.get('num_heads', default_heads),
                dropout=encoder_cfg.get('dropout', default_dropout),
                activation=encoder_cfg.get('activation', default_activation)
            )
            self.feature_encoders[feature_name] = encoder

    def _build_feature_fusion_layer(self):
        """构建多特征联合自注意力层，使不同模态共享上下文"""
        layers = self.feature_fusion_config.get('num_layers', 1)
        if layers <= 0 or self.total_feature_dim == 0:
            self.feature_fusion = None
            return

        desired_heads = self.feature_fusion_config.get('num_heads', 8)
        nhead = self._select_valid_heads(self.total_feature_dim, desired_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_feature_dim,
            nhead=nhead,
            dim_feedforward=self.feature_fusion_config.get('dim_feedforward', max(self.total_feature_dim * 2, 64)),
            dropout=self.feature_fusion_config.get('dropout', 0.1),
            activation=self.feature_fusion_config.get('activation', 'gelu'),
            batch_first=True
        )
        self.feature_fusion = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    @staticmethod
    def _select_valid_heads(d_model: int, desired_heads: int) -> int:
        """确保注意力头数可以被d_model整除"""
        if desired_heads <= 0:
            return 1
        if d_model % desired_heads == 0:
            return desired_heads
        for h in range(desired_heads, 0, -1):
            if d_model % h == 0:
                return h
        return 1

    def _encode_per_feature_sequences(
        self,
        feature_map: Dict[str, torch.Tensor],
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """对每个特征序列进行独立编码并拼接成统一表示"""
        encoded_list: List[torch.Tensor] = []
        for feature_name, tensor in feature_map.items():
            if feature_name not in self.feature_encoders:
                continue

            encoder = self.feature_encoders[feature_name]
            tensor = tensor.unsqueeze(0)
            encoded = encoder(tensor, attention_mask.unsqueeze(0) if attention_mask is not None else None)
            encoded_list.append(encoded.squeeze(0))

        if encoded_list:
            return torch.cat(encoded_list, dim=-1)
        return torch.zeros(seq_len, self.total_feature_dim, device=device)

    def _build_compression_layers(self) -> nn.ModuleDict:
        """构建嵌入压缩层"""
        compression_layers = nn.ModuleDict()
        
        # 如果没有配置压缩方法，则设置默认值
        if not hasattr(self, 'compression_method'):
            self.embedding_compression_config = self.data_processing_config.get('embedding_compression', {})
            self.compression_method = self.embedding_compression_config.get('method', 'truncate')
            self.truncate_config = self.embedding_compression_config.get('truncate', {})
            self.linear_config = self.embedding_compression_config.get('linear', {})
            self.mlp_config = self.embedding_compression_config.get('mlp', {})
            
        if self.compression_method == 'truncate':
            # 截断方法不需要额外的层
            return compression_layers
            
        for feature_name, config in self.embedding_features.items():
            embedding_dim = config.get('embedding_dim', 64)
            feature_dim = config.get('feature_dim', embedding_dim)
            
            # 只为embedding_dim != feature_dim的特征创建压缩层
            if embedding_dim != feature_dim:
                compression_layer = self._create_compression_layer(
                    feature_name, embedding_dim, feature_dim
                )
                if compression_layer is not None:
                    compression_layers[feature_name] = compression_layer
                    logger.info(f"创建压缩层 {feature_name}: {embedding_dim}维 -> {feature_dim}维 (方法: {self.compression_method})")
        
        return compression_layers
    
    def _create_compression_layer(self, feature_name: str, input_dim: int, output_dim: int) -> nn.Module:
        """创建单个压缩层"""
        if self.compression_method == 'truncate':
            # 截断方法不需要额外的层，在forward中处理
            return None
            
        elif self.compression_method == 'linear':
            # 单层全连接压缩
            dropout = self.linear_config.get('dropout', 0.1)
            bias = self.linear_config.get('bias', True)
            activation = self.linear_config.get('activation', None)
            
            layers = [nn.Linear(input_dim, output_dim, bias=bias)]
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            return nn.Sequential(*layers)
            
        elif self.compression_method == 'mlp':
            # 多层MLP压缩
            hidden_layers = self.mlp_config.get('hidden_layers', [512])
            dropout = self.mlp_config.get('dropout', 0.1)
            activation = self.mlp_config.get('activation', 'relu')
            final_activation = self.mlp_config.get('final_activation', None)
            
            layers = []
            dims = [input_dim] + hidden_layers + [output_dim]
            
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                
                # 除了最后一层，都添加激活函数和dropout
                if i < len(dims) - 2:
                    if activation == 'relu':
                        layers.append(nn.ReLU())
                    elif activation == 'gelu':
                        layers.append(nn.GELU())
                    elif activation == 'tanh':
                        layers.append(nn.Tanh())
                    elif activation == 'silu':
                        layers.append(nn.SiLU())
                    
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                else:
                    # 最后一层的激活函数
                    if final_activation == 'relu':
                        layers.append(nn.ReLU())
                    elif final_activation == 'gelu':
                        layers.append(nn.GELU())
                    elif final_activation == 'tanh':
                        layers.append(nn.Tanh())
            
            return nn.Sequential(*layers)
        
        else:
            logger.warning(f"未知的压缩方法: {self.compression_method}")
            return None
    
    def _apply_embedding_compression(self, embedded_features: torch.Tensor, 
                                   feature_name: str, config: Dict) -> torch.Tensor:
        """应用嵌入压缩"""
        embedding_dim = config.get('embedding_dim', 64)
        feature_dim = config.get('feature_dim', embedding_dim)
        
        # 如果维度相等，直接返回
        if embedding_dim == feature_dim:
            return embedded_features
        
        # embedded_features shape: [target_length, embedding_dim]
        if self.compression_method == 'truncate':
            # 截断方法
            strategy = self.truncate_config.get('strategy', 'head')
            
            if strategy == 'head':
                # 保留前feature_dim维
                compressed = embedded_features[:, :feature_dim]
            elif strategy == 'tail':
                # 保留后feature_dim维
                compressed = embedded_features[:, -feature_dim:]
            elif strategy == 'random':
                # 随机选择feature_dim维（注意：这会在每次调用时产生不同结果）
                import random
                indices = random.sample(range(embedding_dim), feature_dim)
                indices.sort()  # 保持顺序
                compressed = embedded_features[:, indices]
            else:
                logger.warning(f"未知的截断策略: {strategy}，使用head策略")
                compressed = embedded_features[:, :feature_dim]
                
        elif self.compression_method in ['linear', 'mlp']:
            # 使用神经网络压缩
            if feature_name in self.compression_layers:
                compression_layer = self.compression_layers[feature_name]
                compressed = compression_layer(embedded_features)
            else:
                logger.warning(f"特征 {feature_name} 没有对应的压缩层，使用截断")
                compressed = embedded_features[:, :feature_dim]
        else:
            logger.warning(f"未知的压缩方法: {self.compression_method}，使用截断")
            compressed = embedded_features[:, :feature_dim]
        
        return compressed


class FeatureSequenceEncoder(nn.Module):
    """轻量级序列编码器：特征 → 投影 → (可选)Transformer"""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1,
                 num_heads: int = 4, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        activation = (activation or 'gelu').lower()
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }.get(activation, nn.GELU())

        self.project = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            act_fn,
            nn.Dropout(dropout)
        )

        if num_layers > 0 and output_dim > 0:
            nhead = FeatureCrossTransformerUnit._select_valid_heads(output_dim, num_heads)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=nhead,
                dim_feedforward=max(output_dim * 2, 64),
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, feature_seq: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.project(feature_seq)
        if self.encoder is not None:
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~(attention_mask.bool())
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x
