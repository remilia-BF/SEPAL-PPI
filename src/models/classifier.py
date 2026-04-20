"""
MLP classifiers for protein-protein interaction prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 activation: str = 'relu',
                 use_batch_norm: bool = False):
        super(MLPClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 128, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Get activation function
        self.activation_fn = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation_fn)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (sigmoid for binary classification)
        if output_dim == 1:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through MLP classifier
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            return_features (bool): Whether to return last layer features
            
        Returns:
            torch.Tensor: Output predictions [batch_size, output_dim]
            torch.Tensor: Last layer features [batch_size, last_hidden_dim] (if return_features=True)
        """
        if return_features:
            # 获取倒数第二层的特征
            features = x
            for i, layer in enumerate(self.network[:-2]):  # 除了最后两个层（Linear + Sigmoid）
                features = layer(features)
            prediction = self.network(features)
            return prediction, features
        else:
            return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper MLPs"""
    
    def __init__(self, dim: int, dropout_rate: float = 0.0, use_batch_norm: bool = False):
        super(ResidualBlock, self).__init__()
        
        layers = [nn.Linear(dim, dim)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        
        layers.append(nn.ReLU())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(dim, dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        
        self.block = nn.Sequential(*layers)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        out = self.block(x)
        out = out + residual
        return self.activation(out)


class ResidualMLPClassifier(nn.Module):
    """MLP classifier with residual blocks"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_residual_blocks: int = 2,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        super(ResidualMLPClassifier, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate, use_batch_norm)
            for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual MLP"""
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_proj(x)


class AttentionClassifier(nn.Module):
    """Classifier with self-attention mechanism"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0):
        super(AttentionClassifier, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention classifier
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output predictions [batch_size, output_dim]
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Self-attention
        attended, _ = self.attention(x, x, x)  # [batch_size, 1, hidden_dim]
        
        # Layer norm and residual connection
        x = self.layer_norm(attended + x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Classify
        return self.classifier(x)


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers"""
    
    def __init__(self, classifiers: List[nn.Module], ensemble_method: str = 'average'):
        super(EnsembleClassifier, self).__init__()
        
        self.classifiers = nn.ModuleList(classifiers)
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            # Learnable weights for ensemble
            self.ensemble_weights = nn.Parameter(torch.ones(len(classifiers)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        predictions = []
        
        for classifier in self.classifiers:
            pred = classifier(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_classifiers, batch_size, output_dim]
        
        if self.ensemble_method == 'average':
            return predictions.mean(dim=0)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_preds = predictions * weights.view(-1, 1, 1)
            return weighted_preds.sum(dim=0)
        elif self.ensemble_method == 'max':
            return predictions.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def create_mlp_classifier(input_dim: int,
                         classifier_type: str = 'standard',
                         **kwargs) -> nn.Module:
    """
    Factory function to create MLP classifier
    
    Args:
        input_dim (int): Input feature dimension
        classifier_type (str): Type of classifier
        **kwargs: Additional arguments for specific classifier types
        
    Returns:
        nn.Module: MLP classifier
    """
    if classifier_type == 'standard':
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [512, 128, 32]),
            output_dim=kwargs.get('output_dim', 1),
            dropout_rate=kwargs.get('dropout_rate', 0.0),
            activation=kwargs.get('activation', 'relu'),
            use_batch_norm=kwargs.get('use_batch_norm', False)
        )
    
    elif classifier_type == 'residual':
        return ResidualMLPClassifier(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_residual_blocks=kwargs.get('num_residual_blocks', 2),
            output_dim=kwargs.get('output_dim', 1),
            dropout_rate=kwargs.get('dropout_rate', 0.0),
            use_batch_norm=kwargs.get('use_batch_norm', False)
        )
    
    elif classifier_type == 'attention':
        return AttentionClassifier(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_heads=kwargs.get('num_heads', 8),
            output_dim=kwargs.get('output_dim', 1),
            dropout_rate=kwargs.get('dropout_rate', 0.0)
        )
    
    elif classifier_type == 'ensemble':
        # Create base classifiers
        num_classifiers = kwargs.get('num_classifiers', 3)
        base_classifiers = []
        
        for i in range(num_classifiers):
            classifier = MLPClassifier(
                input_dim=input_dim,
                hidden_dims=kwargs.get('hidden_dims', [512, 128, 32]),
                output_dim=kwargs.get('output_dim', 1),
                dropout_rate=kwargs.get('dropout_rate', 0.1),
                activation=kwargs.get('activation', 'relu')
            )
            base_classifiers.append(classifier)
        
        return EnsembleClassifier(
            classifiers=base_classifiers,
            ensemble_method=kwargs.get('ensemble_method', 'average')
        )
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def test_classifier(classifier: nn.Module, batch_size: int = 4, input_dim: int = 1280):
    """
    Test classifier with dummy data
    
    Args:
        classifier (nn.Module): Classifier to test
        batch_size (int): Batch size for testing
        input_dim (int): Input dimension for testing
    """
    logger.debug(f"Testing {classifier.__class__.__name__}...")
    
    # Create dummy input
    x = torch.randn(batch_size, input_dim)
    
    # Test classifier
    output = classifier(x)
    
    logger.debug(f"Input shape: {x.shape}")
    logger.debug(f"Output shape: {output.shape}")
    logger.debug(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    logger.debug(f"Test passed for {classifier.__class__.__name__}")
    
    return True