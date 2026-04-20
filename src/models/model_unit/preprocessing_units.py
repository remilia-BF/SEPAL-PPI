"""
预处理单元模块
包含各种预处理方法的可配置实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Mapping, cast, Union, Tuple
import math
from pathlib import Path
import json
import numpy as np
import math

logger = logging.getLogger(__name__)


from .preprocessing.IdentityPreprocessingUnit import IdentityPreprocessingUnit


from .preprocessing.BaseTransformerUnit import BaseTransformerUnit


from .preprocessing.FlashAttentionTransformerUnit import FlashAttentionTransformerUnit


from .preprocessing.HMMPreprocessingUnit import HMMPreprocessingUnit


from .preprocessing.CrossAttentionEnhancedTransformerUnit import CrossAttentionEnhancedTransformerUnit


from .preprocessing.FeatureCrossAttentionUnit import FeatureCrossAttentionUnit


from .preprocessing.FlashTransformerEncoderLayer import FlashTransformerEncoderLayer


from .preprocessing.FlashTransformerDecoderLayer import FlashTransformerDecoderLayer


from .preprocessing.FeatureCrossTransformerUnit import FeatureCrossTransformerUnit
from .preprocessing.FeatureConcatUnit import FeatureConcatUnit


from .preprocessing.PositionalEncoding import PositionalEncoding
from .preprocessing.GatedFeatureFusionUnit import GatedFeatureFusionUnit


def create_preprocessing_unit(preprocessing_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建预处理单元
    
    Args:
        preprocessing_config (Dict): 预处理配置，包含preprocessor和其他参数
                                   必须包含: preprocessor (str) - 预处理器名称
                                   可选包含: embedding_dim, transformer_layers, transformer_heads等
    
    Returns:
        nn.Module: 预处理单元实例
    
    Raises:
        ValueError: 当预处理器不支持时
    
    Examples:
        >>> config = {
        ...     'preprocessor': 'none',
        ...     'embedding_dim': 1280
        ... }
        >>> preprocessor = create_preprocessing_unit(config)
        
        >>> config = {
        ...     'preprocessor': 'base_transformer',
        ...     'embedding_dim': 1280,
        ...     'transformer_layers': [1024, 512],
        ...     'transformer_heads': 8,
        ...     'transformer_layer_num': 2
        ... }
        >>> preprocessor = create_preprocessing_unit(config)
    """
    preprocessor = preprocessing_config.get('preprocessor', 'none')
    embedding_dim = preprocessing_config.get('embedding_dim', None)
    
    if embedding_dim is None:
        raise ValueError("预处理单元需要指定embedding_dim参数")
    
    # 移除preprocessor和embedding_dim，将其余参数传递给具体的预处理单元
    kwargs = {k: v for k, v in preprocessing_config.items() if k not in ['preprocessor', 'embedding_dim']}
    
    if preprocessor == 'none':
        return IdentityPreprocessingUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'base_transformer':
        return BaseTransformerUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'flash_transformer':
        return FlashAttentionTransformerUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'hmm':
        return HMMPreprocessingUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'feature_cross_att':
        return FeatureCrossAttentionUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'feature_cross_transformer':
        return FeatureCrossTransformerUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'feature_concat':
        return FeatureConcatUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'gated_feature_fusion':
        return GatedFeatureFusionUnit(embedding_dim=embedding_dim, **kwargs)
    elif preprocessor == 'cross_attention_transformer':
        # 不再支持cross_attention_transformer预处理器
        raise ValueError(f"不再支持的预处理器: {preprocessor}. "
                        f"支持的预处理器: none, base_transformer, flash_transformer, hmm, feature_cross_att, feature_cross_transformer, feature_concat, gated_feature_fusion")
    else:
        raise ValueError(f"不支持的预处理器: {preprocessor}. "
                        f"支持的预处理器: none, base_transformer, flash_transformer, hmm, feature_cross_att, feature_cross_transformer, feature_concat, gated_feature_fusion")