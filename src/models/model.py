"""
Neural network models for protein-protein interaction prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, List, Dict
from .pooling import create_pooling_layer
from .interaction import create_interaction_layer
from .classifier import create_mlp_classifier

# Import new modular components
from .model_unit import create_model_from_yaml, create_model_from_config, ConfigurableModel

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-Layer Perceptron for protein-protein interaction prediction"""
    
    def __init__(self, input_size=1280, hidden_sizes=None, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 128, 16]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, input_pairs, return_features: bool = False):
        """
        Forward pass for protein pair embeddings
        
        Args:
            input_pairs: Tensor of shape (batch_size, 2, embedding_dim)
                        where input_pairs[:, 0, :] is first protein embedding
                        and input_pairs[:, 1, :] is second protein embedding
            return_features: Whether to return last layer features
        
        Returns:
            Tensor of shape (batch_size, 1) with interaction probabilities
            Tensor of shape (batch_size, feature_dim) with last layer features (if return_features=True)
        """
        # Extract protein embeddings
        protein1 = input_pairs[:, 0, :].reshape(-1, self.input_size)
        protein2 = input_pairs[:, 1, :].reshape(-1, self.input_size)
        
        # Element-wise multiplication (Hadamard product)
        combined_features = torch.mul(protein1, protein2)
        
        # Pass through network
        if return_features:
            # Get features from the second-to-last layer
            features = None
            x = combined_features
            for i, layer in enumerate(self.network[:-2]):  # Exclude last two layers (linear + sigmoid)
                x = layer(x)
                if i == len(self.network) - 4:  # Second-to-last layer
                    features = x
            # Get final output
            output = self.network(x)
            return output, features
        else:
            output = self.network(combined_features)
            return output


def create_mlp_model(input_size=1280, hidden_sizes=None, dropout_rate=0.0, device=None):
    """
    Factory function to create and initialize an MLP model
    
    Args:
        input_size (int): Size of input embeddings (default: 1280 for ESM-1b)
        hidden_sizes (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        device (torch.device): Device to place the model on
    
    Returns:
        MLP: Initialized model
    """
    if hidden_sizes is None:
        hidden_sizes = [1024, 512, 128, 16]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    )
    
    model.to(device)
    
    return model


class AvgPoolMLP(nn.Module):
    """
    Complete model: Average Pooling + Interaction + MLP Classifier
    
    This model processes sequence-level embeddings through:
    1. Pooling: Convert [L, embedding_dim] to [embedding_dim] 
    2. Interaction: Combine two protein embeddings
    3. Classification: MLP classifier for interaction prediction
    """
    
    def __init__(self,
                 embedding_dim: int = 1280,
                 pooling_type: str = 'avg',
                 interaction_type: str = 'hadamard',
                 classifier_type: str = 'standard',
                 pooling_config: Dict = None,
                 interaction_config: Dict = None,
                 classifier_config: Dict = None):
        super(AvgPoolMLP, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        self.interaction_type = interaction_type
        self.classifier_type = classifier_type
        
        # Initialize configs
        pooling_config = pooling_config or {}
        interaction_config = interaction_config or {}
        classifier_config = classifier_config or {}
        
        # Create pooling layer
        self.pooling = create_pooling_layer(
            pooling_type=pooling_type,
            embedding_dim=embedding_dim,
            **pooling_config
        )
        
        # Create interaction layer
        self.interaction = create_interaction_layer(
            interaction_type=interaction_type,
            embedding_dim=embedding_dim,
            **interaction_config
        )
        
        # Determine classifier input dimension based on interaction type
        classifier_input_dim = self._get_classifier_input_dim()
        
        # Create classifier
        self.classifier = create_mlp_classifier(
            input_dim=classifier_input_dim,
            classifier_type=classifier_type,
            **classifier_config
        )
    
    def _get_classifier_input_dim(self) -> int:
        """Calculate classifier input dimension based on interaction type"""
        if self.interaction_type == 'hadamard':
            return self.embedding_dim
        elif self.interaction_type == 'concatenation':
            return 2 * self.embedding_dim
        elif self.interaction_type == 'difference':
            return self.embedding_dim
        elif self.interaction_type == 'cosine':
            return 1
        elif self.interaction_type == 'bilinear':
            return self.embedding_dim  # Default output dim
        elif self.interaction_type == 'attention':
            return self.embedding_dim
        elif self.interaction_type == 'gated':
            return self.embedding_dim
        elif self.interaction_type == 'multi':
            # This would need to be calculated based on specific interaction types
            # For now, assume hadamard + concatenation
            return 3 * self.embedding_dim
        else:
            return self.embedding_dim
    
    def forward(self, 
                protein1_seq: torch.Tensor, 
                protein2_seq: torch.Tensor,
                protein1_mask: Optional[torch.Tensor] = None,
                protein2_mask: Optional[torch.Tensor] = None,
                protein_ids: Optional[tuple] = None,
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the complete model
        
        Args:
            protein1_seq (torch.Tensor): First protein sequence embeddings [batch_size, seq_len1, embedding_dim]
            protein2_seq (torch.Tensor): Second protein sequence embeddings [batch_size, seq_len2, embedding_dim]
            protein1_mask (torch.Tensor): Attention mask for first protein [batch_size, seq_len1]
            protein2_mask (torch.Tensor): Attention mask for second protein [batch_size, seq_len2]
            protein_ids (Optional[tuple]): Tuple of (protein1_ids, protein2_ids) for multimodal features
            return_features (bool): Whether to return last layer features
            
        Returns:
            torch.Tensor: Interaction predictions [batch_size, 1]
            torch.Tensor: Last layer features [batch_size, feature_dim] (if return_features=True)
        """
        # Note: AvgPoolMLP doesn't have preprocessing module, so protein_ids are not used here
        # This parameter is kept for API compatibility with ConfigurableModel
        
        # Step 1: Pool sequence embeddings to fixed-size representations
        protein1_pooled = self.pooling(protein1_seq, protein1_mask)  # [batch_size, embedding_dim]
        protein2_pooled = self.pooling(protein2_seq, protein2_mask)  # [batch_size, embedding_dim]
        
        # Step 2: Compute interaction features
        interaction_features = self.interaction(protein1_pooled, protein2_pooled)  # [batch_size, interaction_dim]
        
        # Step 3: Classify interaction
        if return_features:
            prediction, features = self.classifier(interaction_features, return_features=True)
            return prediction, features
        else:
            prediction = self.classifier(interaction_features)
            return prediction
    
    def forward_with_pooled(self, 
                           protein1_pooled: torch.Tensor, 
                           protein2_pooled: torch.Tensor,
                           protein_ids: Optional[tuple] = None,
                           return_features: bool = False) -> torch.Tensor:
        """
        Forward pass with pre-pooled embeddings (for efficiency when embeddings are cached)
        
        Args:
            protein1_pooled (torch.Tensor): First protein pooled embedding [batch_size, embedding_dim]
            protein2_pooled (torch.Tensor): Second protein pooled embedding [batch_size, embedding_dim]
            protein_ids (Optional[tuple]): Tuple of (protein1_ids, protein2_ids) - not used in pooled mode
            return_features (bool): Whether to return last layer features
            
        Returns:
            torch.Tensor: Interaction predictions [batch_size, 1]
            torch.Tensor: Last layer features [batch_size, feature_dim] (if return_features=True)
        """
        # Compute interaction features
        interaction_features = self.interaction(protein1_pooled, protein2_pooled)
        
        # Classify interaction
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
        Get pooled embedding for a single protein (useful for caching)
        
        Args:
            protein_seq (torch.Tensor): Protein sequence embeddings [batch_size, seq_len, embedding_dim]
            protein_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Pooled embedding [batch_size, embedding_dim]
        """
        return self.pooling(protein_seq, protein_mask)


def create_avg_pool_mlp(embedding_dim: int = 1280,
                       pooling_type: str = 'avg',
                       interaction_type: str = 'hadamard',
                       classifier_type: str = 'standard',
                       device: Optional[torch.device] = None,
                       **kwargs) -> AvgPoolMLP:
    """
    Factory function to create AvgPoolMLP model
    
    Args:
        embedding_dim (int): Embedding dimension
        pooling_type (str): Type of pooling ('avg', 'max', 'attention', etc.)
        interaction_type (str): Type of interaction ('hadamard', 'concatenation', etc.)
        classifier_type (str): Type of classifier ('standard', 'residual', etc.)
        device (Optional[torch.device]): Device to place model on
        **kwargs: Additional configurations for each component
        
    Returns:
        AvgPoolMLP: Initialized model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract component-specific configs
    pooling_config = kwargs.get('pooling_config', {})
    interaction_config = kwargs.get('interaction_config', {})
    classifier_config = kwargs.get('classifier_config', {})
    
    model = AvgPoolMLP(
        embedding_dim=embedding_dim,
        pooling_type=pooling_type,
        interaction_type=interaction_type,
        classifier_type=classifier_type,
        pooling_config=pooling_config,
        interaction_config=interaction_config,
        classifier_config=classifier_config
    )
    
    model.to(device)
    
    logger.info(f"Create model: {pooling_type}_pool + {interaction_type}_interaction + {classifier_type}_classifier")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_custom_mlp(layer_config):
    """
    Create MLP with custom layer configuration (legacy function)
    
    Args:
        layer_config (dict): Configuration dictionary with keys:
            - input_size: Input embedding size
            - hidden_sizes: List of hidden layer sizes
            - dropout_rate: Dropout rate
            - activation: Activation function name
            - device: Target device
    
    Returns:
        MLP: Configured model
    """
    return create_mlp_model(
        input_size=layer_config.get('input_size', 1280),
        hidden_sizes=layer_config.get('hidden_sizes', [1024, 512, 128, 16]),
        dropout_rate=layer_config.get('dropout_rate', 0.0),
        device=layer_config.get('device', None)
    )


# =============================================================================
# 新的模块化模型创建函数
# =============================================================================

def create_yaml_model(config_path: str, device: Optional[torch.device] = None, **kwargs) -> ConfigurableModel:
    """
    从YAML配置文件创建模型 (推荐使用)
    
    Args:
        config_path (str): YAML配置文件路径
        device (Optional[torch.device]): 目标设备
        **kwargs: 覆盖配置参数
        
    Returns:
        ConfigurableModel: 基于YAML配置的模型
        
    Examples:
        >>> # 使用平均池化配置
        >>> model = create_yaml_model('model_config/avg_pooling_v2.yaml')
        >>> 
        >>> # 使用注意力池化配置  
        >>> model = create_yaml_model('model_config/attention_pooling_v2.yaml')
        >>>
        >>> # 覆盖部分配置
        >>> model = create_yaml_model(
        ...     'model_config/avg_pooling_v2.yaml',
        ...     embedding_dim=768
        ... )
    """
    return create_model_from_yaml(config_path, device=device, **kwargs)


def create_config_model(config: Dict, device: Optional[torch.device] = None) -> ConfigurableModel:
    """
    从配置字典创建模型
    
    Args:
        config (Dict): 模型配置字典
        device (Optional[torch.device]): 目标设备
        
    Returns:
        ConfigurableModel: 基于配置的模型
    """
    return create_model_from_config(config, device=device)


def get_available_model_configs() -> List[str]:
    """
    获取可用的模型配置文件列表
    
    Returns:
        List[str]: 可用的配置文件路径列表
    """
    from pathlib import Path
    
    config_dir = Path(__file__).parent.parent.parent / "model_config"
    available_configs = []
    
    if config_dir.exists():
        for config_file in config_dir.glob("*_v2.yaml"):
            available_configs.append(str(config_file))
    
    return available_configs


def print_model_architecture(model: ConfigurableModel, logger: Optional[logging.Logger] = None) -> None:
    """
    Print model architecture information
    
    Args:
        model (ConfigurableModel): Model to analyze
    """
    log = logger if logger is not None else logging.getLogger("sepal_ppi")
    log.debug("=" * 60)
    log.debug("Model architecture")
    log.debug("=" * 60)
    
    # Basic info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    log.info(f"Total parameters: {total_params:,}")
    log.debug(f"Trainable parameters: {trainable_params:,}")
    log.debug(f"Embedding dimension: {model.embedding_dim}")
    log.debug(f"Interaction output dimension: {model.interaction_output_dim}")
    
    # Architecture pipeline
    # Only the architecture pipeline uses info level
    log.info(f"Architecture pipeline: {model._get_architecture_summary()}")
    
    # Module details
    log.debug("Module details:")
    log.debug(f"  Preprocessing: {model.preprocessing.__class__.__name__}")
    log.debug(f"  Pooling: {model.pooling.__class__.__name__}")
    log.debug(f"  Interaction: {model.interaction.__class__.__name__}")
    log.debug(f"  Classifier: {model.classifier.__class__.__name__}")
    
    log.debug("=" * 60)