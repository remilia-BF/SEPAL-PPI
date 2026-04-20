"""
Pooling modules for sequence-level embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AveragePooling(nn.Module):
    """Average pooling with attention mask support"""
    
    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply average pooling to sequence embeddings
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] (1 for real tokens, 0 for padding)
            
        Returns:
            torch.Tensor: Pooled embeddings [batch_size, embedding_dim]
        """
        if attention_mask is None:
            # Simple average if no mask
            return embeddings.mean(dim=1)
        
        # Masked average pooling
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        masked_embeddings = embeddings * mask
        
        # Sum and divide by actual lengths
        summed = masked_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        lengths = mask.sum(dim=1)  # [batch_size, 1]
        
        # Avoid division by zero
        lengths = torch.clamp(lengths, min=1.0)
        
        return summed / lengths


class MaxPooling(nn.Module):
    """Max pooling with attention mask support"""
    
    def __init__(self):
        super(MaxPooling, self).__init__()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply max pooling to sequence embeddings
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Pooled embeddings [batch_size, embedding_dim]
        """
        if attention_mask is None:
            # Simple max pooling
            return embeddings.max(dim=1)[0]
        
        # Masked max pooling
        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Set padding positions to large negative value
        masked_embeddings = embeddings.masked_fill(~mask, float('-inf'))
        
        return masked_embeddings.max(dim=1)[0]


class AttentionPooling(nn.Module):
    """Attention-based pooling"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to sequence embeddings
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Pooled embeddings [batch_size, embedding_dim]
        """
        # Compute attention scores
        attention_scores = self.attention(embeddings).squeeze(-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Mask attention scores
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum
        pooled = torch.bmm(attention_weights.unsqueeze(1), embeddings).squeeze(1)  # [batch_size, embedding_dim]
        
        return pooled


class CLSTokenPooling(nn.Module):
    """Extract CLS token (first token) as pooled representation"""
    
    def __init__(self):
        super(CLSTokenPooling, self).__init__()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract CLS token (first token) as pooled representation
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask (not used for CLS pooling)
            
        Returns:
            torch.Tensor: CLS token embeddings [batch_size, embedding_dim]
        """
        return embeddings[:, 0, :]  # First token


class NonePooling(nn.Module):
    """Identity transformation - return embeddings without any modification"""
    
    def __init__(self):
        super(NonePooling, self).__init__()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return embeddings without any modification (identity transformation)
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask (not used for None pooling)
            
        Returns:
            torch.Tensor: Unmodified embeddings [batch_size, seq_len, embedding_dim]
        """
        return embeddings


class MultiPooling(nn.Module):
    """Combine multiple pooling strategies"""
    
    def __init__(self, embedding_dim: int, pooling_types: Optional[list] = None):
        super(MultiPooling, self).__init__()
        
        if pooling_types is None:
            pooling_types = ['avg', 'max']
        
        self.pooling_types = pooling_types
        self.poolers = nn.ModuleDict()
        
        for pool_type in pooling_types:
            if pool_type == 'avg':
                self.poolers[pool_type] = AveragePooling()
            elif pool_type == 'max':
                self.poolers[pool_type] = MaxPooling()
            elif pool_type == 'attention':
                self.poolers[pool_type] = AttentionPooling(embedding_dim)
            elif pool_type == 'cls':
                self.poolers[pool_type] = CLSTokenPooling()
            elif pool_type == 'none':
                self.poolers[pool_type] = NonePooling()
            else:
                raise ValueError(f"Unknown pooling type: {pool_type}")
        
        # Output dimension is sum of all pooling dimensions
        self.output_dim = len(pooling_types) * embedding_dim
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multiple pooling strategies and concatenate results
        
        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Concatenated pooled embeddings [batch_size, output_dim]
        """
        pooled_outputs = []
        
        for pool_type, pooler in self.poolers.items():
            pooled = pooler(embeddings, attention_mask)
            pooled_outputs.append(pooled)
        
        return torch.cat(pooled_outputs, dim=-1)


def create_pooling_layer(pooling_type: str = 'avg', embedding_dim: int = 1280, **kwargs) -> nn.Module:
    """
    Factory function to create pooling layer
    
    Args:
        pooling_type (str): Type of pooling ('avg', 'max', 'attention', 'cls', 'multi')
        embedding_dim (int): Embedding dimension
        **kwargs: Additional arguments for specific pooling types
        
    Returns:
        nn.Module: Pooling layer
    """
    if pooling_type == 'avg':
        return AveragePooling()
    elif pooling_type == 'max':
        return MaxPooling()
    elif pooling_type == 'attention':
        hidden_dim = kwargs.get('hidden_dim', 128)
        return AttentionPooling(embedding_dim, hidden_dim)
    elif pooling_type == 'cls':
        return CLSTokenPooling()
    elif pooling_type == 'none':
        return NonePooling()
    elif pooling_type == 'multi':
        pooling_types = kwargs.get('pooling_types', ['avg', 'max'])
        return MultiPooling(embedding_dim, pooling_types)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


def test_pooling_layer(pooling_layer: nn.Module, batch_size: int = 4, seq_len: int = 100, embedding_dim: int = 1280):
    """
    Test pooling layer with dummy data
    
    Args:
        pooling_layer (nn.Module): Pooling layer to test
        batch_size (int): Batch size for testing
        seq_len (int): Sequence length for testing
        embedding_dim (int): Embedding dimension for testing
    """
    logger.debug(f"Testing {pooling_layer.__class__.__name__}...")
    
    # Create dummy data
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Create attention mask (some sequences are shorter)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        # Random sequence length
        length = torch.randint(seq_len // 2, seq_len, (1,)).item()
        attention_mask[i, length:] = False
    
    # Test pooling
    pooled = pooling_layer(embeddings, attention_mask)
    
    logger.debug(f"Input shape: {embeddings.shape}")
    logger.debug(f"Attention mask shape: {attention_mask.shape}")
    logger.debug(f"Output shape: {pooled.shape}")
    logger.debug(f"Test passed for {pooling_layer.__class__.__name__}")
    
    return True