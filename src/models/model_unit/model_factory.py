"""
基于YAML配置的模型工厂
从YAML配置文件创建完整的模型架构
"""

import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .pooling_units import create_pooling_unit
from .interaction_units import create_interaction_unit
from .preprocessing_units import create_preprocessing_unit
from .prepooling_units import create_prepairing_unit
from .classifier_units import create_classifier_unit
from .input_layer import create_input_layer

# 使用统一的logger命名空间
logger = logging.getLogger("sepal_ppi.model_factory")


class ConfigurableModel(nn.Module):
    """
    基于YAML配置的可配置模型
    
    包含完整的数据处理流程：
    1. 预处理：可选的transformer预处理
    2. 预池化：蛋白对交互增强（可选）
    3. 池化：将序列级嵌入转换为固定大小表示
    4. 交互：组合两个蛋白质的特征
    5. 分类：MLP分类器进行最终预测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可配置模型
        
        Args:
            config (Dict): 模型配置字典，通常从YAML文件加载
                          应包含: input_data, preprocessing, prepooling, pooling, post_pooling, learning_architecture, output
        """
        super(ConfigurableModel, self).__init__()
        
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 从配置中提取基本参数
        self.embedding_dim = self._get_embedding_dim()
        
        # 创建各个组件
        self._create_input_layer()
        self._create_preprocessing_unit()
        self._create_prepairing_unit()
        self._create_pooling_unit()
        self._create_interaction_unit()
        self._create_classifier_unit()
        
        logger.debug(f"Creating configurable model: embedding_dim={self.embedding_dim}")
        logger.debug(f"Architecture: {self._get_architecture_summary()}")
    
    # 保持原样，没有变化
    
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from config"""
        # 优先从顶级配置中获取（支持覆盖）
        if 'embedding_dim' in self.config:
            return self.config['embedding_dim']
        
        # 其次从model_config中获取（兼容旧格式）
        if 'embedding_dim' in self.model_config:
            return self.model_config['embedding_dim']
        
        # 再次从input_data中获取
        input_data = self.model_config.get('input_data', {})
        if 'embedding_dim' in input_data:
            return input_data['embedding_dim']
        
        # 默认值
        return 1280
    
    def _create_preprocessing_unit(self):
        """Create preprocessing unit"""
        preprocessing_config = self.model_config.get('preprocessing', {})
        
        # 添加嵌入维度到配置
        preprocessing_config['embedding_dim'] = self.embedding_dim
        
        self.preprocessing = create_preprocessing_unit(preprocessing_config)
        logger.debug(f"Create preprocessing unit: {preprocessing_config.get('preprocessor', 'none')}")
        logger.debug(f"Preprocessing config: {preprocessing_config}")

    def _create_input_layer(self):
        """Create input layer (optional linear projection: input_dim -> embedding_dim)"""
        input_config = self.model_config.get('input_data', {})
        # 目标维度为模型使用的 embedding_dim
        input_config['embedding_dim'] = self.embedding_dim
        # 如果未显式提供 input_dim，则默认等于 embedding_dim（即不投影）
        if 'input_dim' not in input_config or input_config['input_dim'] is None:
            input_config['input_dim'] = self.embedding_dim

        self.input_layer = create_input_layer(input_config)
        logger.debug(
            f"Create input layer: input_dim={input_config.get('input_dim')} -> embedding_dim={self.embedding_dim}"
        )
    
    def _create_prepairing_unit(self):
        """Create pre-pooling unit"""
        prepairing_config = self.model_config.get('prepooling', {})
        
        # 添加嵌入维度到配置
        prepairing_config['embedding_dim'] = self.embedding_dim
        
        logger.debug("=" * 80)
        logger.debug("ModelFactory: creating pre-pooling unit")
        logger.debug(f"Config: {prepairing_config}")
        logger.debug("=" * 80)
        
        self.prepairing = create_prepairing_unit(prepairing_config)
        
        logger.debug("=" * 80)
        logger.debug(f"ModelFactory: pre-pooling unit created - method: {prepairing_config.get('method', 'none')}")
        logger.debug("=" * 80)
    
    def _create_pooling_unit(self):
        """Create pooling unit"""
        pooling_config = self.model_config.get('pooling', {})

        # 支持顶层 attention_num 统一覆盖池化头数
        global_attention_num = self.config.get('attention_num', None)
        if global_attention_num is not None:
            try:
                pooling_config['attention_num'] = int(global_attention_num)
            except Exception:
                logger.warning(f"Invalid top-level attention_num={global_attention_num}, skip overriding pooling attention_num")

        # 优先把 preprocessing 的 all_feature_folder 透传给高级注意力池化
        preprocessing_cfg = self.model_config.get('preprocessing', {}) or {}
        data_processing_cfg = preprocessing_cfg.get('data_processing', {}) if isinstance(preprocessing_cfg, dict) else {}
        if isinstance(data_processing_cfg, dict):
            all_feature_folder = data_processing_cfg.get('all_feature_folder')
            if all_feature_folder and 'all_feature_folder' not in pooling_config:
                pooling_config['all_feature_folder'] = all_feature_folder
        
        # 添加嵌入维度到配置
        pooling_config['embedding_dim'] = self.embedding_dim
        
        self.pooling = create_pooling_unit(pooling_config)
        logger.debug(f"Create pooling unit: {pooling_config.get('method', 'average_pooling')}")
        logger.debug(f"Pooling config: {pooling_config}")
    
    def _create_interaction_unit(self):
        """Create interaction unit"""
        interaction_config = self.model_config.get('post_pooling', {})
        
        # 添加嵌入维度到配置
        interaction_config['embedding_dim'] = self.embedding_dim
        
        self.interaction = create_interaction_unit(interaction_config)
        
        # 计算交互后的输出维度
        self.interaction_output_dim = self._calculate_interaction_output_dim(interaction_config)
        
        logger.debug(f"Create interaction unit: {interaction_config.get('operation', 'hadamard_product')}, "
                f"output_dim: {self.interaction_output_dim}")
        logger.debug(f"Interaction config: {interaction_config}")
    
    def _calculate_interaction_output_dim(self, interaction_config: Dict) -> int:
        """计算交互单元的输出维度"""
        operation = interaction_config.get('operation', 'hadamard_product')
        
        if operation == 'hadamard_product':
            return self.embedding_dim
        elif operation == 'concatenation':
            return 2 * self.embedding_dim
        elif operation == 'difference':
            return self.embedding_dim
        elif operation == 'cosine':
            return 1
        elif operation == 'outer_product':
            # 检查是否有降维
            reduction_dim = interaction_config.get('reduction_dim')
            if reduction_dim is not None:
                return reduction_dim
            else:
                return self.embedding_dim * self.embedding_dim
        elif operation == 'fast_compact_bilinear':
            return interaction_config.get('output_dim', self.embedding_dim)
        else:
            raise ValueError(f"未知的交互操作: {operation}")
    
    def _create_classifier_unit(self):
        """Create classifier unit"""
        classifier_config = self.model_config.get('learning_architecture', {})
        
        # 设置输入维度
        classifier_config['input_dim'] = self.interaction_output_dim

        # muti_MLP 专家头数与池化头数自动对齐，避免出现 attention_num=4 但 gate 输出 10 头
        if classifier_config.get('type') == 'muti_MLP':
            pooling_cfg = self.model_config.get('pooling', {}) or {}
            pooling_heads = pooling_cfg.get('attention_num', None)
            global_attention_num = self.config.get('attention_num', None)
            target_heads = global_attention_num if global_attention_num is not None else pooling_heads
            if target_heads is not None:
                try:
                    target_heads = int(target_heads)
                    prev_heads = classifier_config.get('head', None)
                    classifier_config['head'] = target_heads
                    if prev_heads is not None and int(prev_heads) != target_heads:
                        logger.info(f"Auto-align muti_MLP head from {prev_heads} to {target_heads} (match pooling attention_num)")
                except Exception:
                    logger.warning(f"Invalid muti_MLP head alignment target={target_heads}, keep original head setting")
        
        self.classifier = create_classifier_unit(classifier_config)
        logger.debug(f"Create classifier unit: {classifier_config.get('type', 'standard_mlp')}")
        logger.debug(f"Classifier config: {classifier_config}")
    
    def _get_architecture_summary(self) -> str:
        """获取模型架构摘要"""
        # Input description (optional) - prefer human-friendly description when present
        input_cfg = self.model_config.get('input_data', {})
        input_desc = input_cfg.get('description')

        # Fallback to a concise projection/identity summary when description is not provided
        if not input_desc:
            proj = input_cfg.get('projection', {}) or {}
            input_dim = input_cfg.get('input_dim', None)
            if proj.get('enabled', False) or (input_dim is not None and input_dim != self.embedding_dim):
                input_desc = f"input_proj({input_dim}->{self.embedding_dim})"
            else:
                input_desc = "identity_input"

        preprocessing = self.model_config.get('preprocessing', {}).get('preprocessor', 'none')
        prepairing = self.model_config.get('prepooling', {}).get('method', 'none')
        pooling = self.model_config.get('pooling', {}).get('method', 'average_pooling')
        interaction = self.model_config.get('post_pooling', {}).get('operation', 'hadamard_product')
        classifier = self.model_config.get('learning_architecture', {}).get('type', 'standard_mlp')

        # Include input description at the start of the pipeline summary
        return f"{input_desc} -> {preprocessing} -> {prepairing} -> {pooling} -> {interaction} -> {classifier}"
    
    def forward(self, 
                protein1_seq: torch.Tensor, 
                protein2_seq: torch.Tensor,
                protein1_mask: Optional[torch.Tensor] = None,
                protein2_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[tuple] = None,
                return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            protein1_seq: 蛋白质1序列嵌入 [batch_size, seq_len, embedding_dim]
            protein2_seq: 蛋白质2序列嵌入 [batch_size, seq_len, embedding_dim]
            protein1_mask: 蛋白质1的注意力掩码 [batch_size, seq_len]
            protein2_mask: 蛋白质2的注意力掩码 [batch_size, seq_len]
            protein_ids: 蛋白质ID元组，用于某些预处理器
            return_features: 是否返回倒数第二层特征
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, 1]
            torch.Tensor: 倒数第二层特征 [batch_size, feature_dim] (如果return_features=True)
        """
        # 重置注意力权重跟踪（如果支持）
        if hasattr(self.pooling, 'reset_attention_tracking'):
            self.pooling.reset_attention_tracking()
        
        # 提取蛋白质ID
        protein1_ids = None
        protein2_ids = None
        if protein_ids is not None:
            protein1_ids, protein2_ids = protein_ids

        
        # 步骤1: 预处理（可选的transformer变换）
        # 在预处理前，首先执行输入线性投影（当输入维度与模型维度不一致时）
        protein1_seq = self.input_layer(protein1_seq)
        protein2_seq = self.input_layer(protein2_seq)

        # 检查预处理器是否支持protein_ids参数
        reconstruction_loss = 0.0
        if hasattr(self.preprocessing, 'forward') and 'protein_ids' in self.preprocessing.forward.__code__.co_varnames:
            protein1_result = self.preprocessing(protein1_seq, protein1_mask, protein1_ids)
            protein2_result = self.preprocessing(protein2_seq, protein2_mask, protein2_ids)
            
            # 处理可能的重构损失返回
            if isinstance(protein1_result, tuple) and len(protein1_result) == 3:
                protein1_preprocessed, _, protein1_recon_loss = protein1_result
                reconstruction_loss += protein1_recon_loss
            else:
                protein1_preprocessed = protein1_result
                
            if isinstance(protein2_result, tuple) and len(protein2_result) == 3:
                protein2_preprocessed, _, protein2_recon_loss = protein2_result
                reconstruction_loss += protein2_recon_loss
            else:
                protein2_preprocessed = protein2_result
        else:
            protein1_preprocessed = self.preprocessing(protein1_seq, protein1_mask)
            protein2_preprocessed = self.preprocessing(protein2_seq, protein2_mask)
        
        # 存储重构损失供训练循环使用
        if self.training and hasattr(self.preprocessing, 'enable_reconstruction') and self.preprocessing.enable_reconstruction:
            self._reconstruction_loss = reconstruction_loss * self.preprocessing.reconstruction_loss_weight
        else:
            self._reconstruction_loss = 0.0
        
        # 步骤2: 预池化（蛋白对交互增强）
        protein1_enhanced, protein2_enhanced = self.prepairing(
            protein1_preprocessed, protein2_preprocessed, 
            protein1_mask, protein2_mask
        )
        
        # 步骤3: 池化（序列 -> 固定大小表示）
        protein1_pooled = self.pooling(
            protein1_enhanced,
            protein1_mask,
            protein_ids=protein1_ids,
            protein_tag='protein1',
        )  # [batch_size, embedding_dim]
        protein2_pooled = self.pooling(
            protein2_enhanced,
            protein2_mask,
            protein_ids=protein2_ids,
            protein_tag='protein2',
        )  # [batch_size, embedding_dim]

        # 步骤4: 交互（组合两个蛋白质特征）
        # muti_MLP 场景：优先使用 head 级池化表示，构造 [B, H, interaction_dim]
        interaction_features = None
        if hasattr(self.classifier, 'get_last_gate_weights') and hasattr(self.pooling, 'get_head_pooled_outputs'):
            try:
                head_outputs = self.pooling.get_head_pooled_outputs(live=True)
                if isinstance(head_outputs, dict) and 'protein1' in head_outputs and 'protein2' in head_outputs:
                    protein1_heads = head_outputs['protein1']  # [B, H, D]
                    protein2_heads = head_outputs['protein2']  # [B, H, D]
                    if protein1_heads.dim() == 3 and protein2_heads.dim() == 3:
                        num_heads = min(protein1_heads.size(1), protein2_heads.size(1))
                        per_head_interactions = []
                        for head_idx in range(num_heads):
                            head_inter = self.interaction(protein1_heads[:, head_idx, :], protein2_heads[:, head_idx, :])
                            per_head_interactions.append(head_inter)
                        if per_head_interactions:
                            interaction_features = torch.stack(per_head_interactions, dim=1)  # [B, H, F]
            except Exception:
                interaction_features = None

        if interaction_features is None:
            interaction_features = self.interaction(protein1_pooled, protein2_pooled)  # [batch_size, feature_dim]
        
        # 步骤5: 分类
        if return_features:
            predictions, features = self.classifier(interaction_features, return_features=True)
            return predictions, features
        else:
            predictions = self.classifier(interaction_features)  # [batch_size, 1]
            return predictions
    
    def forward_with_pooled(self, 
                           protein1_pooled: torch.Tensor, 
                           protein2_pooled: torch.Tensor,
                           return_features: bool = False) -> torch.Tensor:
        """
        使用预池化嵌入的前向传播（用于效率优化）
        
        Args:
            protein1_pooled (torch.Tensor): 第一个蛋白质池化嵌入 [batch_size, embedding_dim]
            protein2_pooled (torch.Tensor): 第二个蛋白质池化嵌入 [batch_size, embedding_dim]
            return_features: 是否返回倒数第二层特征
            
        Returns:
            torch.Tensor: 交互预测 [batch_size, 1]
            torch.Tensor: 倒数第二层特征 [batch_size, feature_dim] (如果return_features=True)
        """
        # 跳过预处理和池化，直接进行交互和分类
        interaction_features = self.interaction(protein1_pooled, protein2_pooled)
        if return_features:
            prediction, features = self.classifier(interaction_features, return_features=True)
            return prediction, features
        else:
            prediction = self.classifier(interaction_features)
            return prediction
    
    def get_pooled_embeddings(self, 
                             protein_seq: torch.Tensor,
                             protein_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取单个蛋白质的池化嵌入（用于缓存）
        
        Args:
            protein_seq (torch.Tensor): 蛋白质序列嵌入 [batch_size, seq_len, embedding_dim]
            protein_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
            
        Returns:
            torch.Tensor: 池化嵌入 [batch_size, embedding_dim]
        """
        projected = self.input_layer(protein_seq)
        preprocessed = self.preprocessing(projected, protein_mask)
        # 注意：这里跳过了预池化步骤，因为预池化需要两个蛋白序列
        pooled = self.pooling(preprocessed, protein_mask)
        return pooled
    
    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取注意力权重（如果池化层支持）
        
        Returns:
            Dict[str, torch.Tensor]: 包含protein1和protein2注意力权重的字典，或None
        """
        if hasattr(self.pooling, 'get_attention_weights'):
            return self.pooling.get_attention_weights()
        return None

    def get_head_gating_weights(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的 head-gating 权重（若分类器支持）。"""
        if hasattr(self.classifier, 'get_last_gate_weights'):
            return self.classifier.get_last_gate_weights()
        return None

    def get_head_entropy_regularization_loss(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的 head-gating 熵正则损失（若分类器支持）。"""
        if hasattr(self.classifier, 'get_gate_entropy_regularization_loss'):
            return self.classifier.get_gate_entropy_regularization_loss()
        return None


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path (str): YAML配置文件路径
        
    Returns:
        Dict: 解析后的配置字典
        
    Raises:
        FileNotFoundError: 当配置文件不存在时
        yaml.YAMLError: 当YAML格式错误时
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"Successfully loaded config file: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析YAML配置文件失败: {e}")


def create_model_from_yaml(config_path: str, 
                          device: Optional[torch.device] = None,
                          **override_params) -> ConfigurableModel:
    """
    从YAML配置文件创建模型
    
    Args:
        config_path (str): YAML配置文件路径
        device (Optional[torch.device]): 目标设备
        **override_params: 覆盖配置文件中的参数
        
    Returns:
        ConfigurableModel: 配置好的模型实例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    config = load_yaml_config(config_path)
    
    # 应用覆盖参数
    if override_params:
        config.update(override_params)
        logger.debug(f"Applied override parameters: {override_params}")
    
    model = ConfigurableModel(config)
    model.to(device)
    
    # 记录模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created and moved to device: {device}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_model_from_config(config: Dict[str, Any], 
                            device: Optional[torch.device] = None) -> ConfigurableModel:
    """
    从配置字典创建模型
    
    Args:
        config (Dict): 模型配置字典
        device (Optional[torch.device]): 目标设备
        
    Returns:
        ConfigurableModel: 配置好的模型实例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConfigurableModel(config)
    model.to(device)
    
    # 记录模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.debug(f"Model created and moved to device: {device}")
    logger.debug(f"Total parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")
    
    return model


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证配置文件的完整性和正确性
    
    Args:
        config (Dict): 模型配置字典
        
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    try:
        # 检查必要的顶级键
        if 'model_config' not in config:
            return False, "缺少 'model_config' 键"
        
        model_config = config['model_config']
        
        # 检查必要的配置部分
        required_sections = ['pooling', 'post_pooling', 'learning_architecture']
        for section in required_sections:
            if section not in model_config:
                return False, f"缺少 '{section}' 配置部分"
        
        # 验证预池化配置
        prepooling_config = model_config.get('prepooling', {})
        if 'method' not in prepooling_config and prepooling_config:
            return False, "prepooling配置中缺少 'method' 键"
        
        # 验证池化配置
        pooling_config = model_config.get('pooling', {})
        if 'method' not in pooling_config:
            return False, "pooling配置中缺少 'method' 键"
        
        # 验证交互配置
        interaction_config = model_config.get('post_pooling', {})
        if 'operation' not in interaction_config:
            return False, "post_pooling配置中缺少 'operation' 键"
        
        # 验证分类器配置
        classifier_config = model_config.get('learning_architecture', {})
        if 'type' not in classifier_config:
            return False, "learning_architecture配置中缺少 'type' 键"
        
        return True, "配置验证通过"
        
    except Exception as e:
        return False, f"配置验证失败: {str(e)}"