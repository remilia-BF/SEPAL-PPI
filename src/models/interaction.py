"""
Protein interaction modules for combining protein embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class HadamardInteraction(nn.Module):
    """Element-wise multiplication (Hadamard product) for protein interaction"""
    
    def __init__(self):
        super(HadamardInteraction, self).__init__()
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using Hadamard product
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, embedding_dim]
        """
        return torch.mul(protein1_emb, protein2_emb)


class ConcatenationInteraction(nn.Module):
    """Concatenation for protein interaction"""
    
    def __init__(self):
        super(ConcatenationInteraction, self).__init__()
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using concatenation
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, 2 * embedding_dim]
        """
        return torch.cat([protein1_emb, protein2_emb], dim=-1)


class DifferenceInteraction(nn.Module):
    """Absolute difference for protein interaction"""
    
    def __init__(self):
        super(DifferenceInteraction, self).__init__()
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using absolute difference
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, embedding_dim]
        """
        return torch.abs(protein1_emb - protein2_emb)


class CosineSimilarityInteraction(nn.Module):
    """Cosine similarity for protein interaction"""
    
    def __init__(self):
        super(CosineSimilarityInteraction, self).__init__()
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between protein embeddings
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Cosine similarity [batch_size, 1]
        """
        cosine_sim = F.cosine_similarity(protein1_emb, protein2_emb, dim=-1, eps=1e-8)
        return cosine_sim.unsqueeze(-1)  # Add dimension for consistency


class BilinearInteraction(nn.Module):
    """Bilinear interaction for protein embeddings"""
    
    def __init__(self, embedding_dim: int, output_dim: int = None):
        super(BilinearInteraction, self).__init__()
        
        if output_dim is None:
            output_dim = embedding_dim
        
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, output_dim)
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using bilinear transformation
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, output_dim]
        """
        return self.bilinear(protein1_emb, protein2_emb)


class AttentionInteraction(nn.Module):
    """Attention-based interaction between protein embeddings"""
    
    def __init__(self, embedding_dim: int, attention_dim: int = 128):
        super(AttentionInteraction, self).__init__()
        
        self.attention_dim = attention_dim
        
        # Attention mechanism
        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        self.key_proj = nn.Linear(embedding_dim, attention_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.scale = attention_dim ** -0.5
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using cross-attention
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, embedding_dim]
        """
        # Project embeddings
        q1 = self.query_proj(protein1_emb)  # [batch_size, attention_dim]
        k2 = self.key_proj(protein2_emb)    # [batch_size, attention_dim]
        v2 = self.value_proj(protein2_emb)  # [batch_size, embedding_dim]
        
        # Compute attention scores
        scores = torch.sum(q1 * k2, dim=-1, keepdim=True) * self.scale  # [batch_size, 1]
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to get interaction representation
        interaction = attention_weights * v2  # [batch_size, embedding_dim]
        
        return interaction


class MultiInteraction(nn.Module):
    """Combine multiple interaction methods"""
    
    def __init__(self, embedding_dim: int, interaction_types: list = None):
        super(MultiInteraction, self).__init__()
        
        if interaction_types is None:
            interaction_types = ['hadamard', 'concatenation']
        
        self.interaction_types = interaction_types
        self.interactions = nn.ModuleDict()
        
        # Calculate output dimension
        self.output_dim = 0
        
        for interaction_type in interaction_types:
            if interaction_type == 'hadamard':
                self.interactions[interaction_type] = HadamardInteraction()
                self.output_dim += embedding_dim
            elif interaction_type == 'concatenation':
                self.interactions[interaction_type] = ConcatenationInteraction()
                self.output_dim += 2 * embedding_dim
            elif interaction_type == 'difference':
                self.interactions[interaction_type] = DifferenceInteraction()
                self.output_dim += embedding_dim
            elif interaction_type == 'cosine':
                self.interactions[interaction_type] = CosineSimilarityInteraction()
                self.output_dim += 1
            elif interaction_type == 'bilinear':
                self.interactions[interaction_type] = BilinearInteraction(embedding_dim)
                self.output_dim += embedding_dim
            elif interaction_type == 'attention':
                self.interactions[interaction_type] = AttentionInteraction(embedding_dim)
                self.output_dim += embedding_dim
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply multiple interaction methods and concatenate results
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined interaction features [batch_size, output_dim]
        """
        interaction_outputs = []
        
        for interaction_type, interaction_module in self.interactions.items():
            output = interaction_module(protein1_emb, protein2_emb)
            interaction_outputs.append(output)
        
        return torch.cat(interaction_outputs, dim=-1)


class GatedInteraction(nn.Module):
    """Gated interaction mechanism"""
    
    def __init__(self, embedding_dim: int):
        super(GatedInteraction, self).__init__()
        
        # Gate networks
        self.gate1 = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        self.gate2 = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Interaction transformation
        self.interaction_transform = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, protein1_emb: torch.Tensor, protein2_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine protein embeddings using gated interaction
        
        Args:
            protein1_emb (torch.Tensor): First protein embedding [batch_size, embedding_dim]
            protein2_emb (torch.Tensor): Second protein embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embedding [batch_size, embedding_dim]
        """
        # Concatenate for gate computation
        concat_emb = torch.cat([protein1_emb, protein2_emb], dim=-1)
        
        # Compute gates
        gate1 = self.gate1(concat_emb)  # [batch_size, embedding_dim]
        gate2 = self.gate2(concat_emb)  # [batch_size, embedding_dim]
        
        # Apply gates and combine
        gated1 = gate1 * protein1_emb
        gated2 = gate2 * protein2_emb
        
        # Interaction
        interaction = torch.mul(gated1, gated2)
        
        return self.interaction_transform(interaction)


def create_interaction_layer(interaction_type: str = 'hadamard', embedding_dim: int = 1280, **kwargs) -> nn.Module:
    """
    Factory function to create interaction layer
    
    Args:
        interaction_type (str): Type of interaction
        embedding_dim (int): Embedding dimension
        **kwargs: Additional arguments for specific interaction types
        
    Returns:
        nn.Module: Interaction layer
    """
    if interaction_type == 'hadamard':
        return HadamardInteraction()
    elif interaction_type == 'concatenation':
        return ConcatenationInteraction()
    elif interaction_type == 'difference':
        return DifferenceInteraction()
    elif interaction_type == 'cosine':
        return CosineSimilarityInteraction()
    elif interaction_type == 'bilinear':
        output_dim = kwargs.get('output_dim', embedding_dim)
        return BilinearInteraction(embedding_dim, output_dim)
    elif interaction_type == 'attention':
        attention_dim = kwargs.get('attention_dim', 128)
        return AttentionInteraction(embedding_dim, attention_dim)
    elif interaction_type == 'multi':
        interaction_types = kwargs.get('interaction_types', ['hadamard', 'concatenation'])
        return MultiInteraction(embedding_dim, interaction_types)
    elif interaction_type == 'gated':
        return GatedInteraction(embedding_dim)
    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")


def test_interaction_layer(interaction_layer: nn.Module, batch_size: int = 4, embedding_dim: int = 1280):
    """
    Test interaction layer with dummy data
    
    Args:
        interaction_layer (nn.Module): Interaction layer to test
        batch_size (int): Batch size for testing
        embedding_dim (int): Embedding dimension for testing
    """
    logger.debug(f"Testing {interaction_layer.__class__.__name__}...")
    
    # Create dummy protein embeddings
    protein1_emb = torch.randn(batch_size, embedding_dim)
    protein2_emb = torch.randn(batch_size, embedding_dim)
    
    # Test interaction
    interaction_output = interaction_layer(protein1_emb, protein2_emb)
    
    logger.debug(f"Protein 1 shape: {protein1_emb.shape}")
    logger.debug(f"Protein 2 shape: {protein2_emb.shape}")
    logger.debug(f"Interaction output shape: {interaction_output.shape}")
    logger.debug(f"Test passed for {interaction_layer.__class__.__name__}")
    
    return True