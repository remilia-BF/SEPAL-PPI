import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Mapping, cast, Union, Tuple
import math
from pathlib import Path
import json
import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)

class FeatureCrossAttentionUnit(nn.Module):
    """
    特征交叉注意力预处理单元
    
    从mutifeature目录加载各种残基级别特征，通过全连接层投影到embedding_dim，
    然后使用交叉注意力机制增强ESM嵌入：
    - 查询(Q): 投影后的残基特征
    - 键(K)和值(V): ESM嵌入
    
    支持多种特征类型：
    - 浮点数特征 (需要归一化): hydrophobicity, sasa, pqr, hmm_confidence等
    - 二进制标志特征: hmm_binding_domain_flag等
    - 独热编码特征: secondary_structure (3维)
    - 高维特征: s1ssttoken (2048维)
    """
    
    def __init__(self,
                 embedding_dim: int,
                 feature_files: Dict[str, Dict[str, Any]],
                 cross_attention: Optional[Dict[str, Any]] = None,
                 projection: Optional[Dict[str, Any]] = None,
                 data_processing: Optional[Dict[str, Any]] = None,
                 embedding: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        初始化特征交叉注意力单元
        
        Args:
            embedding_dim (int): ESM嵌入维度
            feature_files (Dict): 特征文件配置
            cross_attention (Dict): 交叉注意力配置
            projection (Dict): 投影层配置  
            data_processing (Dict): 数据处理配置
            embedding (Dict): 嵌入层配置（用于ID类特征）
        """
        super(FeatureCrossAttentionUnit, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # 默认配置
        if cross_attention is None:
            cross_attention = {}
        if projection is None:
            projection = {}
        if data_processing is None:
            data_processing = {}
        if embedding is None:
            embedding = {}
            
        self.cross_attention_config = cross_attention
        self.projection_config = projection
        self.data_processing_config = data_processing
        self.embedding_config = embedding
        
        # 交叉注意力参数
        self.num_heads = cross_attention.get('num_heads', 8)
        self.num_layers = cross_attention.get('num_layers', 1)
        self.cross_dropout = cross_attention.get('dropout', 0.1)
        self.use_flash_attention = cross_attention.get('use_flash_attention', True)
        
        # 投影层参数
        self.projection_hidden_dim = projection.get('hidden_dim', None)
        self.projection_dropout = projection.get('dropout', 0.1)
        self.projection_activation = projection.get('activation', 'GELU')
        
        # 数据处理参数
        self.protein_fasta_path = data_processing.get('protein_fasta_path', 'dataset/S1/protein.fasta')
        self.cache_features = data_processing.get('cache_features', True)
        self.normalization_method = data_processing.get('normalization_method', 'z_score')
        self.handle_missing = data_processing.get('handle_missing', 'zero')
        # 统一特征根目录（训练时可设为 mutifeature/S1；预测时可覆盖）
        self.all_feature_folder = data_processing.get('all_feature_folder', None)
        # 缺失特征时是否按错误记录（仍回退为0向量继续推理）
        self.log_missing_as_error = data_processing.get('log_missing_as_error', False)
        
        # 特征配置处理
        self.feature_files = feature_files
        self.enabled_features = {}
        self.feature_configs = {}
        self.total_feature_dim = 0
        self.embedding_features = {}  # 需要嵌入层处理的特征
        
        # 计算启用的特征维度
        for feature_name, config in feature_files.items():
            if config.get('enabled', False):
                self.enabled_features[feature_name] = config
                self.feature_configs[feature_name] = config
                
                # 检查是否是嵌入类型特征
                if config.get('feature_type') == 'embedding':
                    self.embedding_features[feature_name] = config
                
                self.total_feature_dim += config.get('feature_dim', 1)
        
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
        self.vocab_mappings = {}  # 存储ID到索引的映射
        
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
        
        # 构建投影网络
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
                logger.info("Flash Attention可用，将使用Flash Attention优化交叉注意力")
            except ImportError:
                logger.warning("Flash Attention不可用，将使用标准注意力机制")
                self.flash_attn_available = False
        
        # 构建交叉注意力层
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            if not self.flash_attn_available:
                # 标准MultiheadAttention
                attn_layer = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.cross_dropout,
                    batch_first=True
                )
                self.cross_attention_layers.append(attn_layer)
        
        # 归一化和dropout层
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)  # 用于查询特征归一化
        self.dropout = nn.Dropout(self.cross_dropout)
        
        # 查询特征演化层
        self.query_evolution_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(self.num_layers)
        ])
        
        # 特征缓存
        self.feature_cache = {} if self.cache_features else None
        self.protein_lengths = {}  # 蛋白长度缓存
        
        # 加载蛋白长度信息
        self._load_protein_lengths()
        
        # 预加载特征文件（可选）
        if self.cache_features:
            self._preload_features()
        
        logger.debug(f"创建特征交叉注意力单元: embedding_dim={embedding_dim}, "
                    f"feature_dim={self.total_feature_dim}, heads={self.num_heads}, "
                    f"layers={self.num_layers}")

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
        for feature_name, config in self.enabled_features.items():
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
                        feature_dict[protein_id] = self._parse_feature_data(
                            item, feature_name, config
                        )
                    
                    if self.feature_cache is not None:
                        self.feature_cache[feature_name] = feature_dict
                    logger.info(f"预加载特征 {feature_name}: {len(feature_dict)} 条记录")
                else:
                    (logger.error if self.log_missing_as_error else logger.warning)(
                        f"特征文件不存在: {file_path}")
            except Exception as e:
                logger.error(f"预加载特征 {feature_name} 失败: {e}")
    
    def _parse_feature_data(self, item: Dict, feature_name: str, config: Dict) -> np.ndarray:
        """解析特征数据"""
        
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
            # 格式: "[0,0,1],[0,1,0],[0,0,1],..."
            # 移除空格并按],[ 分割
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
    
    def _get_protein_features(self, protein_id: str, target_length: int, device: torch.device) -> torch.Tensor:
        """获取指定蛋白的所有启用特征"""
        all_features = []
        
        for feature_name, config in self.enabled_features.items():
            try:
                if self.cache_features and self.feature_cache is not None and feature_name in self.feature_cache:
                    # 从缓存获取
                    if protein_id in self.feature_cache[feature_name]:
                        feature_data = self.feature_cache[feature_name][protein_id]
                    else:
                        logger.warning(f"蛋白 {protein_id} 在特征 {feature_name} 中不存在")
                        if config.get('feature_type') == 'embedding':
                            feature_data = np.zeros(target_length, dtype=np.int64)  # ID特征默认为0
                        else:
                            feature_data = np.zeros(target_length * config['feature_dim'], dtype=np.float32)
                else:
                    # 实时加载（不推荐，性能较差）
                    feature_data = self._load_single_feature(protein_id, feature_name, config)
                
                # 处理嵌入类特征
                if config.get('feature_type') == 'embedding':
                    # ID类特征：通过嵌入层处理
                    feature_tensor = self._process_embedding_feature(
                        feature_data, feature_name, config, target_length, device
                    )
                else:
                    # 普通特征：归一化和调整长度
                    if config.get('normalize', False):
                        feature_data = self._normalize_features(feature_data)
                    
                    # 调整长度
                    feature_data = self._adjust_feature_length(
                        feature_data, target_length, config['feature_dim']
                    )
                    
                    feature_tensor = torch.from_numpy(feature_data).to(device)
                
                all_features.append(feature_tensor)
                
            except Exception as e:
                logger.warning(f"获取特征 {feature_name} 失败: {e}")
                # 使用零填充作为后备
                if config.get('feature_type') == 'embedding':
                    fallback_data = torch.zeros(target_length * config['feature_dim'], device=device, dtype=torch.float32)
                else:
                    fallback_data = torch.zeros(target_length * config['feature_dim'], device=device, dtype=torch.float32)
                all_features.append(fallback_data)
        
        # 合并所有特征
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.zeros(target_length * self.total_feature_dim, device=device, dtype=torch.float32)
    
    def _process_embedding_feature(self, feature_data: np.ndarray, feature_name: str, 
                                 config: Dict, target_length: int, device: torch.device) -> torch.Tensor:
        """处理嵌入类特征"""
        # 调整ID序列长度
        if len(feature_data) < target_length:
            # 填充0（通常表示padding或unknown）
            padded = np.zeros(target_length, dtype=np.int64)
            padded[:len(feature_data)] = feature_data
            feature_data = padded
        elif len(feature_data) > target_length:
            # 截断到目标长度
            feature_data = feature_data[:target_length]
            logger.warning(f"ID特征 {feature_name} 长度 {len(feature_data)} 超过目标长度 {target_length}，已截断")
        
        # 确保ID在有效范围内
        vocab_size = config.get('vocab_size', 1000)
        feature_data = np.clip(feature_data, 0, vocab_size - 1)
        
        # 转换为张量并通过嵌入层
        ids_tensor = torch.from_numpy(feature_data).to(device)  # [target_length]
        
        # 获取嵌入层
        embedding_layer = self.embedding_layers[feature_name]
        
        # 通过嵌入层：[target_length] -> [target_length, embedding_dim]
        embedded_features = embedding_layer(ids_tensor)
        
        # 应用嵌入压缩（如果需要）
        if hasattr(self, 'compression_method'):
            embedded_features = self._apply_embedding_compression(embedded_features, feature_name, config)
        
        # 展平：[target_length, feature_dim] -> [target_length * feature_dim]
        return embedded_features.flatten()
    
    def _adjust_feature_length(self, feature_data: np.ndarray, target_length: int, feature_dim: int) -> np.ndarray:
        """调整特征长度以匹配目标长度"""
        # 重塑为 [length, feature_dim]
        if feature_dim == 1:
            current_length = len(feature_data)
            feature_reshaped = feature_data.reshape(-1, 1)
        else:
            # 对于多维特征（如secondary structure的3维）
            if len(feature_data) % feature_dim != 0:
                logger.warning(f"特征数据长度 {len(feature_data)} 不能被特征维度 {feature_dim} 整除")
                # 截断或填充到合适长度
                target_total = (len(feature_data) // feature_dim) * feature_dim
                feature_data = feature_data[:target_total]
            
            current_length = len(feature_data) // feature_dim
            feature_reshaped = feature_data.reshape(current_length, feature_dim)
        
        # 调整长度
        if current_length == target_length:
            result = feature_reshaped
        elif current_length < target_length:
            # 填充
            padding = np.zeros((target_length - current_length, feature_dim), dtype=np.float32)
            result = np.vstack([feature_reshaped, padding])
        else:
            # 截断
            result = feature_reshaped[:target_length]
            logger.warning(f"特征长度 {current_length} 超过目标长度 {target_length}，已截断")
        
        # 重新展平
        return result.flatten()
    
    def _load_single_feature(self, protein_id: str, feature_name: str, config: Dict) -> np.ndarray:
        """实时加载单个特征（不推荐，仅作后备）"""
        try:
            # 处理特殊的内部生成特征
            if feature_name == 'protein_length':
                dummy_item = {'protein_id': protein_id}
                return self._parse_feature_data(dummy_item, feature_name, config)
            
            # 处理需要文件的特征
            file_path = config.get('file_path')
            if not file_path:
                logger.warning(f"特征 {feature_name} 缺少file_path配置")
                return np.array([], dtype=np.float32)
            
            # 处理外部文件特征
            file_path = Path(file_path)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                if item['protein_id'] == protein_id:
                    return self._parse_feature_data(item, feature_name, config)
            
            logger.warning(f"蛋白 {protein_id} 在 {feature_name} 中未找到")
            return np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"加载特征 {feature_name} 失败: {e}")
            return np.array([], dtype=np.float32)
    
    def _convert_to_flash_attn_dtype(self, tensor: torch.Tensor):
        """将张量转换为Flash Attention支持的数据类型"""
        original_dtype = tensor.dtype
        
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            tensor = tensor.to(dtype=target_dtype)
        
        return tensor, original_dtype
    
    def _convert_back_from_flash_attn_dtype(self, tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        """从Flash Attention数据类型转换回原始数据类型"""
        if tensor.dtype != original_dtype:
            tensor = tensor.to(dtype=original_dtype)
        return tensor
    
    def forward(self, 
                embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        前向传播
        
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
        
        # 如果没有提供protein_ids，尝试从全局状态获取或使用缓存策略
        if protein_ids is None or len(protein_ids) != batch_size:
            protein_ids = getattr(self, '_current_protein_ids', None)
            if protein_ids is None or len(protein_ids) != batch_size:
                # 如果无法获取protein_ids，根据配置决定行为
                if hasattr(self, '_allow_fallback') and self._allow_fallback:
                    # 使用随机特征作为fallback
                    protein_ids = self._create_fallback_features(batch_size, seq_len, embeddings.device)
                else:
                    logger.debug("无法获取protein_ids，返回原始嵌入")
                    return embeddings
        
        device = embeddings.device
        
        # 为批次中的每个蛋白获取特征
        batch_features = []
        for i, protein_id in enumerate(protein_ids):
            # 使用蛋白质的实际序列长度，确保与HMM特征长度的一致性
            if protein_id in self.protein_lengths:
                real_length = self.protein_lengths[protein_id]
            else:
                # 后备方案：使用序列长度
                real_length = seq_len
                logger.debug(f"蛋白质 {protein_id} 长度信息缺失，使用序列长度: {real_length}")
            
            # 获取蛋白特征
            protein_features = self._get_protein_features(protein_id, real_length, device)
            
            # 调整到批次序列长度
            if real_length < seq_len:
                # 需要填充到seq_len
                padding_size = (seq_len - real_length) * self.total_feature_dim
                padding = torch.zeros(padding_size, dtype=protein_features.dtype, device=device)
                protein_features = torch.cat([protein_features, padding], dim=0)
            elif real_length > seq_len:
                # 截断到seq_len
                protein_features = protein_features[:seq_len * self.total_feature_dim]
            
            # 重塑为 [seq_len, total_feature_dim]
            protein_features = protein_features.view(seq_len, self.total_feature_dim)
            batch_features.append(protein_features)
        
        # 堆叠为批次 [batch_size, seq_len, total_feature_dim]
        batch_features = torch.stack(batch_features, dim=0)
        
        # 投影特征到embedding_dim
        projected_features = self.feature_projection(batch_features)  # [batch_size, seq_len, embedding_dim]
        
        # 交叉注意力增强
        enhanced_embeddings = self._apply_cross_attention(
            projected_features, embeddings, attention_mask
        )
        
        return enhanced_embeddings
    
    def _apply_cross_attention(self, 
                              query_features: torch.Tensor,
                              esm_embeddings: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用交叉注意力机制"""
        
        # 处理注意力掩码
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True表示需要被忽略的位置
        
        x = esm_embeddings  # 初始化输出为ESM嵌入
        q = query_features  # 初始化查询特征，在层间也会演化
        
        # 逐层应用交叉注意力
        for layer_idx in range(self.num_layers):
            # === 交叉注意力子层 ===
            residual_x = x
            residual_q = q
            
            # 层归一化
            q_norm = self.layer_norm1(q)  # 查询：演化的查询特征
            k_norm = self.layer_norm1(x)  # 键：当前嵌入
            v_norm = self.layer_norm1(x)  # 值：当前嵌入
            
            if self.flash_attn_available:
                # Flash Attention实现
                batch_size, seq_len, embed_dim = q_norm.shape
                head_dim = embed_dim // self.num_heads
                
                assert embed_dim % self.num_heads == 0, \
                    f"嵌入维度{embed_dim}不能被头数{self.num_heads}整除"
                
                # 转换数据类型
                q_norm, orig_dtype_q = self._convert_to_flash_attn_dtype(q_norm)
                k_norm, orig_dtype_k = self._convert_to_flash_attn_dtype(k_norm)
                v_norm, orig_dtype_v = self._convert_to_flash_attn_dtype(v_norm)
                
                # 重塑为Flash Attention格式
                q_reshaped = q_norm.reshape(batch_size, seq_len, self.num_heads, head_dim)
                k_reshaped = k_norm.reshape(batch_size, seq_len, self.num_heads, head_dim)
                v_reshaped = v_norm.reshape(batch_size, seq_len, self.num_heads, head_dim)
                
                # 交叉注意力计算
                attn_output = self.flash_attn_func(
                    q_reshaped, k_reshaped, v_reshaped,
                    dropout_p=self.cross_dropout if self.training else 0.0,
                    causal=False
                )
                
                if attn_output is None:
                    raise RuntimeError("Flash Attention returned None")
                
                # 重塑回原始格式
                attn_output = attn_output.view(batch_size, seq_len, embed_dim)
                
                # 转换回原始数据类型
                attn_output = self._convert_back_from_flash_attn_dtype(attn_output, orig_dtype_q)
                
            else:
                # 标准MultiheadAttention
                attn_layer = self.cross_attention_layers[layer_idx]
                attn_output, _ = attn_layer(
                    query=q_norm,
                    key=k_norm,
                    value=v_norm,
                    key_padding_mask=key_padding_mask
                )
            
            # 残差连接和归一化
            x = residual_x + self.dropout(attn_output)
            x = self.layer_norm2(x)
            
            # === 查询特征演化子层 ===
            # 让查询特征通过专门的演化层
            q_residual = q
            q_normed = self.layer_norm3(q)
            q_evolved = self.query_evolution_layers[layer_idx](q_normed)
            q_enhanced = self.activation_fn(q_evolved + attn_output)  # 结合注意力输出
            q = q_residual + self.dropout(q_enhanced)
        
        return x
    
    def _create_fallback_features(self, batch_size: int, seq_len: int, device: torch.device) -> List[str]:
        """创建fallback特征，当无法获取protein_ids时使用"""
        # 返回虚拟的protein_ids，这将导致使用零特征
        return [f"_fallback_{i}" for i in range(batch_size)]
    
    def set_current_protein_ids(self, protein_ids: List[str]):
        """设置当前批次的蛋白质ID（用于兼容现有接口）"""
        self._current_protein_ids = protein_ids

    def _resolve_feature_file_path(self, file_path_value: Optional[Union[str, bool]]) -> Optional[str]:
        """根据all_feature_folder解析特征文件路径，允许相对文件名在根目录下查找。"""
        if not file_path_value or str(file_path_value).lower() == 'false':
            return None
        p = Path(str(file_path_value))
        if p.exists():
            return str(p)
        if self.all_feature_folder:
            candidate = Path(self.all_feature_folder) / p
            return str(candidate)
        return str(p)
