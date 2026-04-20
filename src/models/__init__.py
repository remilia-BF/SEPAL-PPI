"""
Model modules for SEPAL-PPI
"""

from .model import create_mlp_model, AvgPoolMLP, create_avg_pool_mlp
from .train_model import train_model, get_default_training_config
from .predict_model import predict_interactions
from .pooling import create_pooling_layer, AveragePooling, MaxPooling, AttentionPooling
from .interaction import create_interaction_layer, HadamardInteraction, ConcatenationInteraction
from .classifier import create_mlp_classifier, MLPClassifier

# 新的模块化模型系统
from .model import create_yaml_model, create_config_model, print_model_architecture, get_available_model_configs
from .model_unit import ConfigurableModel, create_model_from_yaml, create_model_from_config
from .model_unit import (
    # 池化模块
    AveragePoolingUnit,
    AttentionPoolingUnit, 
    AdvancedAttentionPoolingUnit,
    MaxPoolingUnit,
    NonePoolingUnit,
    
    # 交互模块
    HadamardInteractionUnit,
    OuterProductInteractionUnit,
    ConcatenationInteractionUnit,
    
    # 预处理模块
    BaseTransformerUnit,
    IdentityPreprocessingUnit,
    
    # 预池化模块
    NonePrepoolingUnit,
    CrossAttentionPoolingUnit,
    
    # 分类器模块
    StandardMLPUnit,
    ResidualMLPUnit,
    MutiMLPUnit,
)

__all__ = [
    # 原有的模型创建函数 (Legacy)
    'create_mlp_model', 'train_model', 'get_default_training_config', 'predict_interactions',
    'AvgPoolMLP', 'create_avg_pool_mlp',
    'create_pooling_layer', 'AveragePooling', 'MaxPooling', 'AttentionPooling',
    'create_interaction_layer', 'HadamardInteraction', 'ConcatenationInteraction',
    'create_mlp_classifier', 'MLPClassifier',
    
    # 新的模块化模型系统 (推荐使用)
    'create_yaml_model',           # 从YAML创建模型
    'create_config_model',         # 从配置字典创建模型
    'ConfigurableModel',           # 可配置模型类
    'create_model_from_yaml',      # 底层YAML模型创建函数
    'create_model_from_config',    # 底层配置模型创建函数
    'print_model_architecture',    # 模型架构分析
    'get_available_model_configs', # 获取可用配置
    
    # 模块化组件
    'AveragePoolingUnit',
    'AttentionPoolingUnit', 
    'AdvancedAttentionPoolingUnit',
    'MaxPoolingUnit',
    'NonePoolingUnit',
    'HadamardInteractionUnit',
    'OuterProductInteractionUnit',
    'ConcatenationInteractionUnit',
    'BaseTransformerUnit',
    'IdentityPreprocessingUnit',
    'NonePrepoolingUnit',
    'CrossAttentionPoolingUnit',
    'StandardMLPUnit',
    'ResidualMLPUnit',
    'MutiMLPUnit',
]