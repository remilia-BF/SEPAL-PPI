"""
模块化的模型组件
包含可配置的池化、交互、预处理和分类器模块
"""

from .pooling_units import *
from .interaction_units import *
from .preprocessing_units import *
from .classifier_units import *
from .prepooling_units import *
from .model_factory import create_model_from_yaml, create_model_from_config, ConfigurableModel

__all__ = [
    # 池化模块
    'AveragePoolingUnit',
    'AttentionPoolingUnit', 
    'AdvancedAttentionPoolingUnit',
    'MaxPoolingUnit',
    'NonePoolingUnit',
    
    # 交互模块
    'HadamardInteractionUnit',
    'OuterProductInteractionUnit',
    'ConcatenationInteractionUnit',
    
    # 预处理模块
    'BaseTransformerUnit',
    'IdentityPreprocessingUnit',
    
    # 预池化模块
    'NonePrepoolingUnit',
    'CrossAttentionPoolingUnit',
    
    # 分类器模块
    'StandardMLPUnit',
    'ResidualMLPUnit',
    'MutiMLPUnit',
    
    # 模型工厂
    'create_model_from_yaml',
    'create_model_from_config',
    'ConfigurableModel'
]