"""
Embedding Generator Module for One-Step Prediction

Handles ESM embedding generation with two pathways:
1. CIS embeddings: First token (BF16 -> FP32) through input-layer linear projection (5120 -> 1280)
2. Pooled embeddings: Full residue embeddings through MLP adapter + multimodal fusion + attention pooling

Supports both online (per-pair) and prebuild (batch) modes.
"""

import os
import sys
import json
import warnings
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
    BitsAndBytesConfig = None

# Try importing esm-efficient
try:
    from esme import ESM2, ESM1b
    from esme.alphabet import tokenize, tokenize_unpad
    ESME_AVAILABLE = True
except ImportError:
    ESME_AVAILABLE = False
    ESM2 = None
    ESM1b = None
    tokenize = None
    tokenize_unpad = None


# Model configurations
HF_MODEL_NAME_MAP: Dict[str, str] = {
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b": "facebook/esm2_t36_3B_UR50D",
    "esm2_15b": "facebook/esm2_t48_15B_UR50D",
    "esm1b_650m": "facebook/esm1b_t33_650M_UR50S",
}

MODEL_DIM_MAP: Dict[str, int] = {
    "esm2_15b": 5120,
    "esm2_3b": 2560,
    "esm2_650m": 1280,
    "esm2_150m": 640,
    "esm1b_650m": 1280,
}


class NetSurfStyleHead(nn.Module):
    """NetSurfP-style residue head for RSA + Q3 prediction."""

    def __init__(self, in_dim: int, model_dim: int = 1280, lstm_hidden: int = 1024, lstm_dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Identity() if in_dim == model_dim else nn.Linear(in_dim, model_dim)
        self.conv1 = nn.Conv1d(model_dim, 32, kernel_size=129, padding=64)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=257, padding=128)
        self.bn = nn.BatchNorm1d(model_dim + 32)
        self.lstm = nn.LSTM(
            input_size=model_dim + 32,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout,
        )
        out_dim = lstm_hidden * 2
        self.rsa_head = nn.Linear(out_dim, 1)
        self.q3_head = nn.Linear(out_dim, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x_t = x.transpose(1, 2)
        c = F.relu(self.conv1(x_t))
        c = F.relu(self.conv2(c))
        x_cat = torch.cat([x_t, c], dim=1)
        x_cat = self.bn(x_cat)
        x_seq = x_cat.transpose(1, 2)
        h, _ = self.lstm(x_seq)
        rsa = torch.sigmoid(self.rsa_head(h)).squeeze(-1)
        q3_logits = self.q3_head(h)
        return rsa, q3_logits

SUPPORTED_ESM_PRECISIONS = {"bf16", "int8", "int6", "int4"}
ESME_QUANT_MAP = {
    "int4": "4bit",
    "int8": "8bit",
}

# ESM2-15B BF16 requires approximately 30GB VRAM
ESM_VRAM_REQUIREMENTS: Dict[str, int] = {
    "esm2_15b": 30 * 1024 ** 3,  # 30 GB
    "esm2_3b": 8 * 1024 ** 3,    # 8 GB
    "esm2_650m": 3 * 1024 ** 3,  # 3 GB
    "esm2_150m": 1 * 1024 ** 3,  # 1 GB
}


def _safe_torch_load(path: str, map_location: torch.device):
    """Safe checkpoint loading with proper warnings handling."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)
        return torch.load(path, map_location=map_location)


def _compute_file_md5(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _verify_checkpoint_md5(checkpoint_path: str, expected_md5: Optional[str] = None, 
                           logger: Optional[logging.Logger] = None) -> bool:
    """Verify checkpoint file integrity using MD5.
    
    Args:
        checkpoint_path: Path to checkpoint file
        expected_md5: Expected MD5 hash (if None, only compute and return)
        logger: Logger instance for output
        
    Returns:
        True if MD5 matches or expected_md5 is None, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return False
    
    computed_md5 = _compute_file_md5(checkpoint_path)
    
    if expected_md5 is None:
        if logger:
            logger.debug(f"Checkpoint MD5: {computed_md5}")
        return True
    
    if computed_md5 == expected_md5:
        if logger:
            logger.debug(f"Checkpoint MD5 verified: {computed_md5}")
        return True
    else:
        if logger:
            logger.error(f"MD5 mismatch! Expected: {expected_md5}, Got: {computed_md5}")
        return False


@dataclass
class EmbeddingResult:
    """Container for embedding generation results."""
    protein_id: str
    cis_embedding: Optional[np.ndarray] = None  # [1280] FP32
    pooled_embedding: Optional[np.ndarray] = None  # [1280] FP32
    attention_weights: Optional[Dict[str, Any]] = None  # For JSONL output
    residue_embeddings: Optional[np.ndarray] = None  # [seq_len, 1280] for IG analysis


class InputLayerProjector(nn.Module):
    """
    Input layer projector for dimension reduction.
    
    Supports two modes:
    1. Single linear projection (hidden_dim=None): input_dim -> output_dim
    2. MLP projection (hidden_dim!=None): input_dim -> hidden_dim -> output_dim
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None,
                 activation: str = 'gelu', residual: bool = True, layernorm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.residual = residual and (input_dim == output_dim)
        self._use_mlp = hidden_dim is not None
        
        if hidden_dim is None:
            # Single linear projection
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            # MLP projection
            self.proj1 = nn.Linear(input_dim, hidden_dim)
            self.proj2 = nn.Linear(hidden_dim, output_dim)
            
            if activation.lower() == 'gelu':
                self.act = nn.GELU()
            elif activation.lower() == 'relu':
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()
        
        # LayerNorm
        if layernorm:
            if hidden_dim is not None:
                self.ln = nn.LayerNorm(output_dim)
            else:
                self.layernorm = nn.LayerNorm(output_dim)
        else:
            self.ln = None
            self.layernorm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mlp:
            out = self.proj1(x)
            out = self.act(out)
            out = self.proj2(out)
            if self.residual:
                out = out + x
            if self.ln is not None:
                out = self.ln(out)
        else:
            out = self.proj(x)
            if self.residual:
                out = out + x
            if self.layernorm is not None:
                out = self.layernorm(out)
        return out
    
    @staticmethod
    def from_checkpoint(
        ckpt_path: str,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        strict_mode: bool = False,
        component_name: str = "input_layer"
    ) -> 'InputLayerProjector':
        """Load from checkpoint file."""
        ckpt = _safe_torch_load(ckpt_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        meta = ckpt.get('meta', {})
        model_config = ckpt.get('model_config', {})
        
        # Extract configuration
        input_dim = meta.get('input_dim')
        embedding_dim = meta.get('embedding_dim')
        activation = meta.get('activation', 'gelu')
        
        # Get projection config from model_config
        projection_config = {}
        if 'model_config' in model_config:
            input_data = model_config['model_config'].get('input_data', {})
            projection_config = input_data.get('projection', {})
        
        hidden_dim = projection_config.get('hidden_dim')
        residual = projection_config.get('residual', True)
        # Important: for old/exported linear input_layer checkpoints (proj.weight/proj.bias
        # only), LMDB generation used pure linear projection without LayerNorm. If we default
        # to layernorm=True here, outputs will diverge significantly from LMDB.
        if 'layernorm' in projection_config:
            layernorm = projection_config.get('layernorm', True)
        else:
            has_ln_weights = (
                'layernorm.weight' in state or 'layernorm.bias' in state or
                'ln.weight' in state or 'ln.bias' in state
            )
            layernorm = has_ln_weights
        
        # Infer dimensions from weights
        weight_key = 'proj.weight' if 'proj.weight' in state else 'weight'
        if weight_key in state:
            weight = state[weight_key]
            if input_dim is None:
                input_dim = weight.shape[1]
            if embedding_dim is None:
                embedding_dim = weight.shape[0]
        
        if input_dim is None or embedding_dim is None:
            raise RuntimeError(f"Cannot infer dimensions from checkpoint: input_dim={input_dim}, embedding_dim={embedding_dim}")
        
        projector = InputLayerProjector(
            input_dim=input_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=residual,
            layernorm=layernorm
        )
        
        missing_keys, unexpected_keys = projector.load_state_dict(state, strict=False)
        if logger:
            logger.info(
                f"{component_name} load_state_dict: "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )
            if missing_keys:
                logger.warning(f"{component_name} missing_keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"{component_name} unexpected_keys: {unexpected_keys}")
        if strict_mode and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Strict mode: {component_name} load_state_dict mismatch "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
            )
        projector.to(device)
        projector.eval()
        
        return projector


class AttentionPoolingUnit(nn.Module):
    """
    Attention pooling unit with multi-head support.
    
    Stores attention weights for interpretability analysis.
    """
    
    def __init__(self, embedding_dim: int, attention_num: int = 1, dropout: float = 0.1,
                 save_attention: bool = True, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_num = attention_num
        self.save_attention = save_attention
        self.temperature = temperature
        
        self.attention_weights = nn.Linear(embedding_dim, attention_num)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for IG analysis
        self.last_attention_weights = None
        self.last_attention_scores = None  # Pre-softmax scores for IG
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention weight storage.
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]
            attention_mask: [batch_size, seq_len] or [seq_len]
            
        Returns:
            pooled: [batch_size, embedding_dim] or [embedding_dim]
        """
        squeeze_batch = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            squeeze_batch = True
        
        batch_size, seq_len, _ = embeddings.shape
        
        # Compute attention scores [batch_size, seq_len, attention_num]
        attention_scores = self.attention_weights(embeddings)
        attention_scores = self.dropout(attention_scores)
        
        # Store pre-softmax scores for IG
        if self.save_attention:
            self.last_attention_scores = attention_scores.detach().clone()
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            attention_scores = attention_scores + (1.0 - mask) * (-1e9)
        
        # Temperature scaling + softmax
        scaled_scores = attention_scores / max(self.temperature, 1e-4)
        attention_weights = F.softmax(scaled_scores, dim=1)
        
        # Store attention weights
        if self.save_attention:
            self.last_attention_weights = attention_weights.detach().cpu()
        
        # Weighted average
        if self.attention_num == 1:
            weighted_embeddings = embeddings * attention_weights
            pooled = weighted_embeddings.sum(dim=1)
        else:
            pooled_heads = []
            for head_idx in range(self.attention_num):
                head_weights = attention_weights[:, :, head_idx:head_idx+1]
                weighted_embeddings = embeddings * head_weights
                head_pooled = weighted_embeddings.sum(dim=1)
                pooled_heads.append(head_pooled)
            pooled = torch.stack(pooled_heads, dim=0).mean(dim=0)
        
        if squeeze_batch:
            pooled = pooled.squeeze(0)
        
        return pooled
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last attention weights."""
        return self.last_attention_weights
    
    def get_attention_scores(self) -> Optional[torch.Tensor]:
        """Get last pre-softmax attention scores (for IG)."""
        return self.last_attention_scores


def _normalize_attention_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention weights to [0.001, 1.0] range.
    
    Consistent with inference_engine.py normalization.
    """
    if weights.numel() == 0:
        return weights
    
    # Softmax normalization
    weights_softmax = torch.softmax(weights, dim=0)
    
    # Map to [0.001, 1.0] range
    min_val, max_val = 0.001, 1.0
    w_min, w_max = weights_softmax.min(), weights_softmax.max()
    
    if w_max > w_min:
        normalized = (weights_softmax - w_min) / (w_max - w_min)
        normalized = normalized * (max_val - min_val) + min_val
    else:
        normalized = torch.full_like(weights_softmax, (max_val + min_val) / 2)
    
    return normalized


def format_attention_data(protein_id: str, sequence: str, 
                          attention_weights: torch.Tensor) -> Dict[str, Any]:
    """
    Format attention weights for JSONL output.
    
    Args:
        protein_id: Protein identifier
        sequence: Protein sequence
        attention_weights: [seq_len, attention_num]
        
    Returns:
        Dict with attention data in JSONL format
    """
    seq_len, attention_num = attention_weights.shape
    effective_len = min(seq_len, len(sequence))
    
    attention_data = {
        "protein_id": protein_id,
        "length": len(sequence),
        "attention_heads": attention_num,
        "attention": {}
    }
    
    for head_idx in range(attention_num):
        head_weights = attention_weights[:effective_len, head_idx]
        head_weights_normalized = _normalize_attention_weights(head_weights)
        weights_list = [round(float(w), 4) for w in head_weights_normalized]
        attention_data["attention"][f"head_{head_idx + 1}"] = weights_list
    
    return attention_data


class EmbeddingGenerator:
    """
    Generate ESM embeddings for protein sequences.
    
    Provides two embedding pathways:
    1. CIS embeddings: First token through linear projection
    2. Pooled embeddings: Full sequence through attention pooling
    
    Supports both online (per-pair) and prebuild (batch) modes.
    """
    
    def __init__(
        self,
        esm_model_name: str = "esm2_15b",
        esm_precision: str = "bf16",
        force_hf_esm: bool = False,
        input_layer_ckpt: str = None,
        model_pretrain_ckpt: str = None,
        cis_input_layer_ckpt: str = None,  # CIS projector (5120 -> 2560 -> 5120)
        pooled_classifier_ckpt: Optional[str] = None,
        multifeature_dir: Optional[str] = None,
        enable_online_multifeature: bool = False,
        netsurfp_head_ckpt: Optional[str] = None,
        strict_mode: bool = False,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            esm_model_name: ESM model name (e.g., esm2_15b)
            esm_precision: ESM model precision (bf16, int8, int6, int4)
            force_hf_esm: Force HuggingFace ESM backend instead of esm-efficient
            input_layer_ckpt: Path to input_layer.pth (5120 -> 1280 projection) for pooled embeddings
            model_pretrain_ckpt: Path to complete_model.pth (MLP adapter + pooling) for pooled embeddings
            cis_input_layer_ckpt: Path to CIS input_layer.pth (5120 -> 2560 -> 5120 MLP) for CIS embeddings
            pooled_classifier_ckpt: Path to pooled best_model.pth for compatibility checks
            multifeature_dir: Optional path to multimodal features directory
            enable_online_multifeature: If True, lazily predict SASA/Q3 from the same ESM residue embeddings
            netsurfp_head_ckpt: Optional checkpoint path for online NetSurfP head
            strict_mode: If True, fail immediately when checkpoint keys mismatch
            device: Computation device
            logger: Optional logger instance
        """
        self.esm_model_name = esm_model_name
        self.esm_precision = esm_precision
        self.force_hf_esm = force_hf_esm
        self.input_layer_ckpt = input_layer_ckpt
        self.model_pretrain_ckpt = model_pretrain_ckpt
        self.cis_input_layer_ckpt = cis_input_layer_ckpt
        self.pooled_classifier_ckpt = pooled_classifier_ckpt
        self.multifeature_dir = multifeature_dir
        self.enable_online_multifeature = enable_online_multifeature
        self.netsurfp_head_ckpt = netsurfp_head_ckpt
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Check model name
        if esm_model_name not in HF_MODEL_NAME_MAP:
            raise ValueError(
                f"Unsupported ESM model: {esm_model_name}. "
                f"Available models: {list(HF_MODEL_NAME_MAP.keys())}"
            )

        if esm_precision not in SUPPORTED_ESM_PRECISIONS:
            raise ValueError(
                f"Unsupported ESM precision: {esm_precision}. "
                f"Available precisions: {sorted(SUPPORTED_ESM_PRECISIONS)}"
            )
        
        # Warn if not using default model
        if esm_model_name != "esm2_15b":
            self.logger.warning(
                f"Using {esm_model_name} instead of esm2_15b. "
                f"Make sure you have retrained models for this embedding dimension."
            )
        
        # Set device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Resolve quantization precision (int6 falls back to int4 with warning)
        self.esm_quant_precision = self._resolve_quant_precision()

        # Autocast dtype for ESM forward
        self.esm_autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None

        # Prefer esm-efficient for all precisions unless user explicitly forces HuggingFace.
        self._use_esme = not self.force_hf_esm
        backend = "HuggingFace" if not self._use_esme else "esm-efficient"
        self.logger.info(f"ESM backend selected: {backend} (precision={self.esm_precision})")
        
        # Check VRAM availability
        self._check_vram_availability()
        
        # Initialize components (lazy loading)
        self.esm_model = None
        self.esm_tokenizer = None
        self.input_layer = None
        self.internal_projector = None
        self.pooling_layer = None
        self.preprocessing_layer = None
        self.cis_projector = None  # CIS-specific MLP projector (5120 -> 2560 -> 5120)
        self.netsurfp_head = None
        
        # Model configuration
        self.esm_dim = MODEL_DIM_MAP[esm_model_name]
        self.embedding_dim = 1280  # Output dimension after projection
        
        # Embedding cache for prebuild mode
        self.cis_cache: Dict[str, np.ndarray] = {}
        self.pooled_cache: Dict[str, np.ndarray] = {}
        self.cis_tensor_cache: Dict[str, torch.Tensor] = {}
        self.pooled_tensor_cache: Dict[str, torch.Tensor] = {}
        self.raw_cis_tensor_cache: Dict[str, torch.Tensor] = {}
        self.raw_residue_avg_tensor_cache: Dict[str, torch.Tensor] = {}
        self.attention_cache: Dict[str, Dict[str, Any]] = {}
        self.residue_cache: Dict[str, np.ndarray] = {}  # For IG analysis
        self.sequence_input_cache: Dict[str, np.ndarray] = {}  # LMDB-equivalent [L, 1280]
        
        # Flags
        self._models_loaded = False
        self._use_multifeature = multifeature_dir is not None
        self._online_multifeature_enabled = False

    def _load_state_dict_with_checks(
        self,
        module: nn.Module,
        state_dict: Dict[str, Any],
        component_name: str
    ) -> Tuple[List[str], List[str]]:
        """Load state dict with key mismatch logging and strict-mode enforcement."""
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

        self.logger.info(
            f"{component_name} load_state_dict: "
            f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
        )
        if missing_keys:
            self.logger.warning(f"{component_name} missing_keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"{component_name} unexpected_keys: {unexpected_keys}")

        if self.strict_mode and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Strict mode: {component_name} load_state_dict mismatch "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
            )

        return missing_keys, unexpected_keys
    
    def _check_vram_availability(self) -> None:
        """Check if sufficient VRAM is available."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU (may be very slow)")
            return
        
        try:
            device_props = torch.cuda.get_device_properties(0)
            total_vram = device_props.total_memory
            required_vram = ESM_VRAM_REQUIREMENTS.get(self.esm_model_name, 0)
            
            # Reserve some margin for embeddings
            margin = 4 * 1024 ** 3  # 4 GB margin
            
            if total_vram < required_vram + margin:
                self.logger.warning(
                    f"Limited VRAM detected: {total_vram / 1024**3:.1f} GB. "
                    f"{self.esm_model_name} requires ~{required_vram / 1024**3:.1f} GB. "
                    f"Using online mode to avoid OOM."
                )
        except Exception as e:
            self.logger.warning(f"Could not check VRAM: {e}")
    
    def load_models(self) -> None:
        """Load all required models."""
        if self._models_loaded:
            return
        
        self.logger.info("Loading ESM model and adapters...")
        
        # Check dependencies for the selected backend
        if self._use_esme and not ESME_AVAILABLE:
            raise ImportError(
                "esm-efficient is required for the selected backend. "
                "Install with: pip install esm-efficient"
            )
        if (not self._use_esme) and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required when using HuggingFace backend. "
                "Install with: pip install transformers"
            )
        
        # Load ESM model
        self._load_esm_model()
        
        # Load CIS projector (5120 -> 2560 -> 5120) for CIS model
        # The CIS model uses: projector(hidden_2560) -> average_pooling -> hadamard -> MLP
        if self.cis_input_layer_ckpt:
            self._load_cis_projector()
        else:
            # Try to auto-detect CIS projector from precision-aware cache first.
            cis_candidates = [
                Path("cache") / "sepal-st50" / self.esm_precision / "cis" / "input_layer.pth",
                Path("cache/sepal-st50/cis/input_layer.pth"),
            ]
            resolved = next((p for p in cis_candidates if p.exists()), None)
            if resolved is not None:
                self.cis_input_layer_ckpt = str(resolved)
                self._load_cis_projector()
            else:
                self.logger.warning("CIS projector not found - CIS embeddings will be incorrect!")
        
        # Load input layer (5120 -> 1280) for pooled embeddings
        if self.input_layer_ckpt:
            self._load_input_layer()
        
        # Load model components (MLP adapter, pooling, preprocessing)
        if self.model_pretrain_ckpt:
            self._load_model_components()
        
        self._models_loaded = True
        self.logger.debug("All models loaded successfully")
    
    
    
    
    def _load_esm_model(self) -> None:
        """
        Load ESM model from selected backend.
        
        Default backend is esm-efficient (including bf16).
        Use `force_hf_esm=True` to switch to HuggingFace backend.
        
        """
        if self._use_esme:
            self._load_esme_model()
            return

        hf_name = HF_MODEL_NAME_MAP[self.esm_model_name]
        
        # Load tokenizer from HuggingFace
        self.logger.debug(f"Loading tokenizer from HuggingFace: {hf_name}")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(hf_name)
        
        # Load model from HuggingFace
        self.logger.debug(f"Loading ESM model from HuggingFace: {hf_name}")
        self.esm_model = AutoModel.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.esm_model.eval()
        
        hidden_size = getattr(self.esm_model.config, "hidden_size", self.esm_dim)
        self.logger.debug(f"ESM model loaded, hidden_size={hidden_size}")

    def _get_esme_model_path(self) -> Path:
        return Path("cache/esm") / f"{self.esm_model_name}.safetensors"

    def _get_esme_quantization(self) -> Optional[str]:
        if self.esm_quant_precision in ESME_QUANT_MAP:
            return ESME_QUANT_MAP[self.esm_quant_precision]
        return None

    def _load_esme_model(self) -> None:
        if not ESME_AVAILABLE:
            raise ImportError("esm-efficient is required for selected backend")

        model_path = self._get_esme_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"ESM-Efficient model not found: {model_path}. "
                "Download weights and place the safetensors file in cache/esm."
            )

        if self.esm_model_name.startswith("esm2"):
            model_cls = ESM2
        elif self.esm_model_name.startswith("esm1b"):
            model_cls = ESM1b
        else:
            raise ValueError(f"ESM-Efficient does not support model: {self.esm_model_name}")

        quantization = self._get_esme_quantization()
        if quantization is not None:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for esm-efficient quantization. "
                    "Install with: pip install bitsandbytes"
                ) from e
        self.logger.info(
            f"Loading ESM-Efficient model from {model_path} "
            f"(quantization={quantization})"
        )

        self.esm_model = model_cls.from_pretrained(
            str(model_path),
            device=0 if self.device.type == "cuda" else "cpu",
            quantization=quantization,
        )
        self.esm_model.eval()

    def _esme_forward_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Run esm-efficient forward to get embeddings.
        """
        activations: Dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                activations["last"] = output[0]
            else:
                activations["last"] = output

        hook_module = None
        if hasattr(self.esm_model, "layers") and self.esm_model.layers:
            hook_module = self.esm_model.layers[-1]
        elif hasattr(self.esm_model, "transformer") and hasattr(self.esm_model.transformer, "layers"):
            hook_module = self.esm_model.transformer.layers[-1]

        handle = None
        if hook_module is not None:
            handle = hook_module.register_forward_hook(_hook)

        outputs = self.esm_model(tokens)

        if handle is not None:
            handle.remove()

        if "last" in activations and isinstance(activations["last"], torch.Tensor):
            return activations["last"]

        if isinstance(outputs, (tuple, list)) and outputs:
            outputs = outputs[0]

        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            return outputs.logits

        if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
            return outputs.last_hidden_state

        raise RuntimeError(
            f"ESM-Efficient output does not contain embeddings. Output type: {type(outputs)}"
        )

    def _esme_forward_embeddings_batch(self, sequences: List[str]) -> List[torch.Tensor]:
        if tokenize_unpad is not None:
            tokens, _indices, cu_lens, max_len = tokenize_unpad(sequences)
            tokens = tokens.to(self.device)
            cu_lens = cu_lens.to(self.device)
            max_len = max_len
            activations: Dict[str, torch.Tensor] = {}

            def _hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    activations["last"] = output[0]
                else:
                    activations["last"] = output

            hook_module = None
            if hasattr(self.esm_model, "layers") and self.esm_model.layers:
                hook_module = self.esm_model.layers[-1]
            elif hasattr(self.esm_model, "transformer") and hasattr(self.esm_model.transformer, "layers"):
                hook_module = self.esm_model.transformer.layers[-1]

            handle = None
            if hook_module is not None:
                handle = hook_module.register_forward_hook(_hook)

            with torch.no_grad():
                try:
                    _ = self.esm_model(tokens, (cu_lens, max_len))
                except TypeError:
                    _ = self.esm_model(tokens)

            if handle is not None:
                handle.remove()

            if "last" not in activations:
                raise RuntimeError("ESM-Efficient hook did not capture embeddings")

            emb_flat = activations["last"]
            if emb_flat.dim() != 2:
                raise RuntimeError(f"Unexpected esm-efficient batch embedding shape: {tuple(emb_flat.shape)}")

            cu_lens_cpu = cu_lens.detach().cpu().tolist()
            embeddings = []
            for idx in range(len(sequences)):
                start = int(cu_lens_cpu[idx])
                end = int(cu_lens_cpu[idx + 1])
                embeddings.append(emb_flat[start:end])
            return embeddings

        tokens = tokenize(sequences).to(self.device)
        with torch.no_grad():
            emb = self._esme_forward_embeddings(tokens)
        if emb.dim() == 3:
            return [emb[i] for i in range(emb.shape[0])]
        if emb.dim() == 2:
            lengths = (tokens != 0).sum(dim=1).tolist()
            embeddings = []
            offset = 0
            for length in lengths:
                embeddings.append(emb[offset:offset + int(length)])
                offset += int(length)
            return embeddings
        raise RuntimeError(f"Unexpected esm-efficient batch embedding shape: {tuple(emb.shape)}")

    def _resolve_quant_precision(self) -> str:
        if self.esm_precision == "int6":
            self.logger.warning("int6 is not supported by bitsandbytes; falling back to int4.")
            return "int4"
        return self.esm_precision

    def _get_quant_cache_dir(self) -> Path:
        return Path("cache/esm") / self.esm_model_name / self.esm_quant_precision

    def _get_quantization_config(self) -> "BitsAndBytesConfig":
        if BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig not available. Install transformers with bitsandbytes support.")

        if self.esm_quant_precision == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.esm_quant_precision == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        raise ValueError(f"Unsupported quantized precision: {self.esm_quant_precision}")

    def _save_quantized_model(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.esm_model.save_pretrained(str(cache_dir), safe_serialization=True)
            if self.esm_tokenizer is not None:
                self.esm_tokenizer.save_pretrained(str(cache_dir))
            meta_path = cache_dir / "quantization_meta.json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(
                    {
                        "model": self.esm_model_name,
                        "requested_precision": self.esm_precision,
                        "cached_precision": self.esm_quant_precision,
                    },
                    meta_file,
                    indent=2,
                )
        except Exception as e:
            self.logger.warning(f"Failed to save quantized model cache: {e}")

    def _load_esm_model_quantized(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes quantization requires CUDA-enabled GPU")

        if not TRANSFORMERS_AVAILABLE or BitsAndBytesConfig is None:
            raise ImportError(
                "Transformers with bitsandbytes support is required for int8/int6/int4. "
                "Install with: pip install transformers bitsandbytes"
            )

        hf_name = HF_MODEL_NAME_MAP[self.esm_model_name]
        cache_dir = self._get_quant_cache_dir()

        # Tokenizer cache (reuses local cache if present)
        self._load_tokenizer(hf_name, cache_dir)

        model_safetensors = cache_dir / "model.safetensors"
        model_bin = cache_dir / "pytorch_model.bin"

        if model_safetensors.exists() or model_bin.exists():
            self.logger.info(
                f"Loading cached {self.esm_quant_precision} ESM model from: {cache_dir}"
            )
            self.esm_model = AutoModel.from_pretrained(
                str(cache_dir),
                local_files_only=True,
                device_map="auto",
            )
            self.esm_model.eval()
        else:
            quant_config = self._get_quantization_config()
            self.logger.info(
                f"Quantizing {self.esm_model_name} to {self.esm_quant_precision} with bitsandbytes..."
            )
            self.esm_model = AutoModel.from_pretrained(
                hf_name,
                device_map="auto",
                quantization_config=quant_config,
            )
            self.esm_model.eval()
            self._save_quantized_model(cache_dir)

        hidden_size = getattr(self.esm_model.config, "hidden_size", self.esm_dim)
        self.logger.debug(f"ESM model loaded (quantized), hidden_size={hidden_size}")
    
    def _load_tokenizer(self, hf_name: str, cache_dir: Path) -> None:
        """
        Load tokenizer with local cache support.
        
        If local cache exists, load from there (fully offline).
        Otherwise, load from HuggingFace and save to cache.
        """
        if cache_dir.exists() and (cache_dir / "tokenizer_config.json").exists():
            # Load from local cache (fully offline)
            self.logger.debug(f"Loading tokenizer from local cache: {cache_dir}")
            self.esm_tokenizer = AutoTokenizer.from_pretrained(str(cache_dir), local_files_only=True)
        else:
            # Load from HuggingFace and cache locally
            self.logger.debug(f"Loading tokenizer from HuggingFace: {hf_name}")
            self.esm_tokenizer = AutoTokenizer.from_pretrained(hf_name)
            
            # Save to local cache
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.esm_tokenizer.save_pretrained(str(cache_dir))
            self.logger.debug(f"Tokenizer cached to: {cache_dir}")
    
    def _save_config_to_cache(self, config_path: Path) -> None:
        """Save model config to local cache."""
        try:
            config_dict = self.esm_model.config.to_dict()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.debug(f"Config cached to: {config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache config: {e}")
    
    def _load_esm_from_safetensors_offline(self, cache_path: Path, config_path: Path) -> None:
        """
        Load ESM model entirely offline from local cache.
        
        Uses cached config and weights - no network access required.
        Uses AutoModel to match HuggingFace direct loading behavior.
        """
        try:
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoConfig
            
            # Load config from local cache
            self.logger.debug("Loading model config from local cache...")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Fix torch_dtype in config (should be actual dtype, not string)
            if 'torch_dtype' in config_dict and isinstance(config_dict['torch_dtype'], str):
                if config_dict['torch_dtype'] == 'bfloat16':
                    config_dict['torch_dtype'] = 'bfloat16'  # Will be parsed by AutoConfig
            
            config = AutoConfig.for_model(**config_dict)
            
            # Load weights directly to GPU using safetensors
            self.logger.debug("Loading weights from safetensors directly to GPU...")
            state_dict = load_file(str(cache_path), device=str(self.device))
            
            # Filter out keys that don't belong to base model
            # Remove pooler, contact_head, lm_head if present
            filtered_state_dict = {}
            excluded_prefixes = ['pooler.', 'contact_head.', 'lm_head.', 'esm.pooler.', 'esm.contact_head.']
            for k, v in state_dict.items():
                # Skip excluded keys
                if any(k.startswith(prefix) for prefix in excluded_prefixes):
                    self.logger.debug(f"Skipping key: {k}")
                    continue
                filtered_state_dict[k] = v
            
            # Create model directly (AutoModel matches HuggingFace behavior)
            self.logger.debug("Creating model with AutoModel (matches HuggingFace behavior)...")
            self.esm_model = AutoModel.from_config(config)
            
            # Load weights (already on GPU, already bf16)
            self._load_state_dict_with_checks(
                self.esm_model,
                filtered_state_dict,
                component_name="esm_model(local_cache)"
            )
            
            # Move to GPU with bf16 dtype
            self.esm_model = self.esm_model.to(device=self.device, dtype=torch.bfloat16)
            self.esm_model.eval()
            
            self.logger.debug(f"Successfully loaded from local cache (fully offline, using AutoModel)")
            
        except ImportError:
            self.logger.warning("safetensors not installed. Install with: pip install safetensors")
            raise RuntimeError("Cannot load cached model without safetensors library")
        except Exception as e:
            self.logger.error(f"Failed to load from local cache: {e}")
            raise
    
    def _load_esm_from_safetensors(self, cache_path: Path, hf_name: str, config_cache_path: Path = None) -> None:
        """
        Load ESM model from safetensors cache directly to GPU.
        
        Uses memory-mapped loading to avoid doubling memory usage.
        Also caches config locally for future offline loads.
        Uses AutoModel to match HuggingFace direct loading behavior.
        """
        try:
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel
            
            # Load config from HuggingFace (small, fast)
            self.logger.debug("Loading model config from HuggingFace...")
            config = AutoConfig.from_pretrained(hf_name)
            
            # Load weights directly to GPU using safetensors
            self.logger.debug("Loading weights from safetensors directly to GPU...")
            state_dict = load_file(str(cache_path), device=str(self.device))
            
            # Filter out keys that don't belong to base model
            filtered_state_dict = {}
            excluded_prefixes = ['pooler.', 'contact_head.', 'lm_head.', 'esm.pooler.', 'esm.contact_head.']
            for k, v in state_dict.items():
                if any(k.startswith(prefix) for prefix in excluded_prefixes):
                    self.logger.debug(f"Skipping key: {k}")
                    continue
                filtered_state_dict[k] = v
            
            # Create model directly (AutoModel matches HuggingFace behavior)
            self.logger.debug("Creating model with AutoModel (matches HuggingFace behavior)...")
            self.esm_model = AutoModel.from_config(config)
            
            # Load weights (already on GPU, already bf16)
            self._load_state_dict_with_checks(
                self.esm_model,
                filtered_state_dict,
                component_name="esm_model(safetensors)"
            )
            
            # Move to GPU with bf16 dtype
            self.esm_model = self.esm_model.to(device=self.device, dtype=torch.bfloat16)
            self.esm_model.eval()
            
            # Cache config for future offline loads
            if config_cache_path is not None:
                self._save_config_to_cache(config_cache_path)
            
            self.logger.debug(f"Successfully loaded from safetensors cache (direct GPU load, using AutoModel)")
            
        except ImportError:
            self.logger.warning("safetensors not installed. Install with: pip install safetensors")
            self.logger.warning("Falling back to standard HuggingFace loading...")
            self._load_esm_fallback(hf_name)
        except Exception as e:
            self.logger.warning(f"Failed to load from safetensors with direct GPU: {e}")
            # Fallback to CPU-first loading
            self._load_esm_from_safetensors_fallback(cache_path, hf_name)
    
    def _load_esm_from_safetensors_fallback(self, cache_path: Path, hf_name: str) -> None:
        """Fallback safetensors loading via CPU (for older PyTorch versions). Uses AutoModel."""
        try:
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel
            
            self.logger.debug("Using CPU-first loading (fallback mode)...")
            
            config = AutoConfig.from_pretrained(hf_name)
            
            # Load weights to CPU first
            state_dict = load_file(str(cache_path))
            
            # Filter out excluded keys
            filtered_state_dict = {}
            excluded_prefixes = ['pooler.', 'contact_head.', 'lm_head.', 'esm.pooler.', 'esm.contact_head.']
            for k, v in state_dict.items():
                if any(k.startswith(prefix) for prefix in excluded_prefixes):
                    continue
                filtered_state_dict[k] = v
            
            # Create model and load weights
            self.esm_model = AutoModel.from_config(config)
            self._load_state_dict_with_checks(
                self.esm_model,
                filtered_state_dict,
                component_name="esm_model(safetensors_fallback)"
            )
            
            # Move to GPU with bf16
            self.esm_model = self.esm_model.to(dtype=torch.bfloat16, device=self.device)
            self.esm_model.eval()
            
            self.logger.debug("Loaded via CPU fallback (using AutoModel)")
            
        except Exception as e:
            self.logger.warning(f"Safetensors fallback failed: {e}")
            self._load_esm_fallback(hf_name)
    
    def _save_esm_to_safetensors(self, cache_path: Path) -> None:
        """Save ESM model to safetensors cache."""
        try:
            from safetensors.torch import save_file
            
            self.logger.debug(f"Saving bf16 weights to safetensors cache: {cache_path}")
            
            # Get state dict
            state_dict = self.esm_model.state_dict()
            
            # Convert to bf16 (should already be, but ensure)
            state_dict_bf16 = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v 
                              for k, v in state_dict.items()}
            
            # Save to safetensors
            save_file(state_dict_bf16, str(cache_path))
            
            self.logger.debug(f"Saved bf16 ESM weights to: {cache_path}")
            self.logger.debug("Future loads will be much faster!")
            
        except ImportError:
            self.logger.warning("safetensors not installed. Skipping cache save.")
            self.logger.warning("Install with: pip install safetensors")
        except Exception as e:
            self.logger.warning(f"Failed to save safetensors cache: {e}")
    
    def _load_esm_fallback(self, hf_name: str) -> None:
        """Fallback ESM loading without safetensors."""
        self.esm_model = AutoModel.from_pretrained(
            hf_name, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.esm_model.eval()
        
        hidden_size = getattr(self.esm_model.config, "hidden_size", self.esm_dim)
        self.logger.debug(f"ESM model loaded, hidden_size={hidden_size}")
    
    def _load_input_layer(self) -> None:
        """Load input layer for dimension projection."""
        self.logger.debug(f"Loading input layer: {self.input_layer_ckpt}")
        self.input_layer = InputLayerProjector.from_checkpoint(
            self.input_layer_ckpt, 
            self.device,
            logger=self.logger,
            strict_mode=self.strict_mode,
            component_name="input_layer(pooled)"
        )
        self.logger.debug(
            f"Input layer loaded: {self.input_layer.input_dim} -> {self.input_layer.output_dim}"
        )
    
    def _load_cis_projector(self) -> None:
        """
        Load CIS-specific MLP projector (5120 -> 2560 -> 5120).
        
        CIS model uses a different projection than pooled model:
        - Pooled: 5120 -> 1280 (dimension reduction)
        - CIS: 5120 -> 2560 -> 5120 (MLP with residual, keeps dimension)
        """
        self.logger.debug(f"Loading CIS projector: {self.cis_input_layer_ckpt}")
        
        ckpt = _safe_torch_load(self.cis_input_layer_ckpt, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        meta = ckpt.get('meta', {})
        
        # CIS projector: 5120 -> 2560 -> 5120 (MLP with residual + LayerNorm)
        input_dim = meta.get('input_dim', 5120)
        output_dim = meta.get('embedding_dim', 5120)
        hidden_dim = 2560  # From config: projection.hidden_dim = 2560
        
        # Check if this is an MLP projector (has proj1/proj2) or linear (has proj)
        has_mlp = 'proj1.weight' in state or 'ln.weight' in state
        
        if has_mlp:
            self.cis_projector = InputLayerProjector(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                activation='gelu',
                residual=True,
                layernorm=True
            )
            self._load_state_dict_with_checks(
                self.cis_projector,
                state,
                component_name="cis_projector"
            )
        else:
            # Linear projector (unlikely for CIS but handle it)
            self.cis_projector = InputLayerProjector(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=None,
                residual=False,
                layernorm=False
            )
            self._load_state_dict_with_checks(
                self.cis_projector,
                state,
                component_name="cis_projector"
            )
        
        self.cis_projector.to(self.device)
        self.cis_projector.eval()
        
        self.logger.debug(
            f"CIS projector loaded: {input_dim} -> {hidden_dim} -> {output_dim} (MLP with residual)"
        )

    def _load_model_components(self) -> None:
        """Load MLP adapter, pooling layer, and optional preprocessing."""
        import yaml
        
        self.logger.debug(f"Loading model components: {self.model_pretrain_ckpt}")
        
        # Find config file
        model_dir = Path(self.model_pretrain_ckpt).parent
        config_path = None
        for name in ["resolved_config.yaml", "best_config.yaml"]:
            if (model_dir / name).exists():
                config_path = model_dir / name
                break
        
        if config_path is None:
            error_msg = (
                f"No config file found in {model_dir}!\n"
                f"Required: resolved_config.yaml or best_config.yaml\n"
                f"This file is needed to load model architecture correctly.\n"
                f"Please ensure the model checkpoint is from a complete training run."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.debug(f"Loading model config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            resolved_config = yaml.safe_load(f)
        model_config = resolved_config.get('model', {}).get('model_config', {})
        
        # Load checkpoint
        ckpt = _safe_torch_load(self.model_pretrain_ckpt, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        
        # Load pooling layer
        pooling_config = model_config.get('pooling', {
            'embedding_dim': 1280,
            'attention_num': 2,
            'dropout': 0.05,
            'save_attention': True,
            'attention_config': {'temperature': 0.3}
        })
        
        self.pooling_layer = AttentionPoolingUnit(
            embedding_dim=pooling_config.get('embedding_dim', 1280),
            attention_num=pooling_config.get('attention_num', 2),
            dropout=pooling_config.get('dropout', 0.05),
            save_attention=True,
            temperature=pooling_config.get('attention_config', {}).get('temperature', 0.3)
        )
        
        # Load pooling weights
        pooling_state = {
            k[len('pooling.'):]: v for k, v in state.items() 
            if k.startswith('pooling.')
        }
        if pooling_state:
            self._load_state_dict_with_checks(
                self.pooling_layer,
                pooling_state,
                component_name="pooling_layer"
            )
            self.logger.debug(f"Loaded {len(pooling_state)} pooling layer weights")
        
        self.pooling_layer.to(self.device)
        self.pooling_layer.eval()
        
        # Load internal projector (MLP adapter)
        internal_state = {
            k[len('input_layer.'):]: v for k, v in state.items()
            if k.startswith('input_layer.')
        }

        # Determine whether the weights in internal_state describe a refinement layer
        # (input_dim == self.embedding_dim → 1280→2560→1280, apply it) or a
        # dimension-reduction layer (input_dim != embedding_dim, e.g. 5120→1280, skip it).
        # NOTE: best_model.pth has NO input_layer.* keys because separate_save_models=True
        # strips them into input_layer.pth.  Inspecting best_model.pth therefore always
        # returned False and incorrectly disabled the projector.  We now check the actual
        # weight shapes from complete_model.pth instead.
        enable_internal_projector = False  # default: disabled until positively confirmed
        if internal_state:
            # Try to infer input-dim from proj1.weight [hidden, input] or weight [out, input]
            proj1_w = internal_state.get('proj1.weight')
            if proj1_w is None:
                proj1_w = internal_state.get('weight')
            if proj1_w is not None and proj1_w.dim() >= 2:
                detected_input_dim = proj1_w.shape[1]
                if detected_input_dim == self.embedding_dim:
                    enable_internal_projector = True
                    self.logger.debug(
                        f"internal_projector input_dim={detected_input_dim} matches "
                        f"embedding_dim={self.embedding_dim}; enabling projector"
                    )
                else:
                    self.logger.info(
                        f"internal_projector input_dim={detected_input_dim} != "
                        f"embedding_dim={self.embedding_dim} (likely a dimension-reduction "
                        f"layer not intended for this path); disabling projector"
                    )
            else:
                # Cannot determine shape — enable and let load_state_dict fail loudly if wrong
                enable_internal_projector = True
                self.logger.debug("Could not infer projector input_dim; enabling internal_projector speculatively")
        
        if internal_state and enable_internal_projector:
            input_data_config = model_config.get('input_data', {})
            projection_config = input_data_config.get('projection', {})
            hidden_dim = projection_config.get('hidden_dim')
            
            if hidden_dim is not None:
                self.internal_projector = InputLayerProjector(
                    input_dim=self.embedding_dim,
                    output_dim=self.embedding_dim,
                    hidden_dim=hidden_dim,
                    activation=projection_config.get('activation', 'gelu'),
                    residual=projection_config.get('residual', True),
                    layernorm=projection_config.get('layernorm', True)
                )
                self._load_state_dict_with_checks(
                    self.internal_projector,
                    internal_state,
                    component_name="internal_projector"
                )
                self.internal_projector.to(self.device)
                self.internal_projector.eval()
                self.logger.debug(f"Loaded internal projector: {self.embedding_dim} -> {hidden_dim} -> {self.embedding_dim}")
        elif internal_state and not enable_internal_projector:
            self.internal_projector = None
            self.logger.info("internal_projector weights found in complete_model but disabled for legacy consistency")
        
        # Load preprocessing layer (multimodal features)
        # Always attempt to load if model config requires it
        self._load_preprocessing_layer(model_config, state)
    
    def _load_preprocessing_layer(self, model_config: Dict, state: Dict) -> None:
        """Load preprocessing layer for multimodal features."""
        preprocessing_config = model_config.get('preprocessing', {})
        
        if preprocessing_config.get('preprocessor') == 'feature_concat':
            # Model requires multifeature - check if directory is provided
            if self.multifeature_dir is None and not self.enable_online_multifeature:
                error_msg = (
                    "Model architecture requires multimodal features but --multifeature-dir not specified!\n"
                    "This model was trained with feature_concat preprocessing and requires multifeature data.\n\n"
                    "Please provide multifeature directory using:\n"
                    "  --multifeature-dir /path/to/multifeature\n\n"
                    "To generate multifeature data, run:\n"
                    "  python mutifeature_tools/one_step_mutifeature.py -p <pdb_dir> -f <fasta> -o <output_dir>"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            if self.multifeature_dir is None and self.enable_online_multifeature:
                self.logger.info(
                    "feature_concat detected without --multifeature-dir; enabling online RSA/Q3 generation from ESM residue embeddings"
                )
            
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from src.models.model_unit.preprocessing.FeatureConcatUnit import FeatureConcatUnit
                
                # Update feature path
                if self.multifeature_dir is not None:
                    preprocessing_config['data_processing']['all_feature_folder'] = self.multifeature_dir
                
                self.preprocessing_layer = FeatureConcatUnit(
                    embedding_dim=self.embedding_dim,
                    **{k: v for k, v in preprocessing_config.items() 
                       if k not in ['preprocessor', 'description', 'embedding_dim']}
                )
                
                # Load weights
                preprocessing_state = {
                    k[len('preprocessing.'):]: v for k, v in state.items()
                    if k.startswith('preprocessing.')
                }
                if preprocessing_state:
                    self._load_state_dict_with_checks(
                        self.preprocessing_layer,
                        preprocessing_state,
                        component_name="preprocessing_layer"
                    )
                    self.logger.debug(f"Loaded {len(preprocessing_state)} preprocessing weights")
                
                self.preprocessing_layer.to(self.device)
                self.preprocessing_layer.eval()
                self._online_multifeature_enabled = bool(self.enable_online_multifeature and self.multifeature_dir is None)
                
            except ImportError as e:
                error_msg = f"Failed to import FeatureConcatUnit: {e}"
                self.logger.error(error_msg)
                raise ImportError(error_msg) from e
    
    def get_esm_embeddings(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ESM embeddings for a sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Tuple of (cls_embedding, residue_embeddings)
            - cls_embedding: [esm_dim] FP32 (first token)
            - residue_embeddings: [seq_len, esm_dim] FP32 (without CLS/EOS)
        """
        cls_embedding_t, residue_embeddings_t = self._get_esm_embeddings_torch(sequence)
        return cls_embedding_t.cpu().numpy(), residue_embeddings_t.cpu().numpy()

    def _get_esm_embeddings_torch(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ESM embeddings as torch tensors on the current device."""
        with torch.no_grad():
            if self._use_esme:
                tokens = tokenize([sequence]).to(self.device)
                emb = self._esme_forward_embeddings(tokens)
            else:
                inputs = self.esm_tokenizer(
                    sequence,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(self.device)

                if self.esm_autocast_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.esm_autocast_dtype):
                        outputs = self.esm_model(**inputs)
                else:
                    outputs = self.esm_model(**inputs)

                emb = outputs.last_hidden_state

            if emb.dim() == 3:
                emb_seq = emb[0]
            elif emb.dim() == 2:
                emb_seq = emb
            else:
                raise RuntimeError(f"Unexpected ESM embedding shape: {tuple(emb.shape)}")

            if emb_seq.shape[-1] != self.esm_dim:
                raise RuntimeError(
                    f"ESM embedding dim mismatch: got {emb_seq.shape[-1]}, expected {self.esm_dim}"
                )

            cls_embedding = emb_seq[0].float()
            residue_embeddings = emb_seq[1:-1].float()
            return cls_embedding, residue_embeddings

    def _resolve_netsurfp_head_ckpt(self) -> Path:
        if self.netsurfp_head_ckpt:
            return Path(self.netsurfp_head_ckpt)
        return Path("cache") / "netsurfp_head" / self.esm_precision / "netsurfp_head_best.pt"

    def _load_netsurfp_head(self) -> None:
        if self.netsurfp_head is not None:
            return
        ckpt_path = self._resolve_netsurfp_head_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Online multifeature enabled but NetSurfP head checkpoint not found: {ckpt_path}"
            )
        payload = _safe_torch_load(str(ckpt_path), map_location=self.device)
        model_cfg = payload.get("model_config", {})
        head = NetSurfStyleHead(
            in_dim=int(model_cfg.get("in_dim", self.esm_dim)),
            model_dim=int(model_cfg.get("model_dim", 1280)),
            lstm_hidden=int(model_cfg.get("lstm_hidden", 1024)),
            lstm_dropout=float(model_cfg.get("lstm_dropout", 0.5)),
        ).to(self.device)
        head.load_state_dict(payload["model_state_dict"])
        head.eval()
        self.netsurfp_head = head
        self.logger.info("Loaded online NetSurfP head: %s", ckpt_path)

    def _ensure_online_multifeature(self, protein_id: str, residue_emb_t: torch.Tensor) -> None:
        if not self._online_multifeature_enabled or self.preprocessing_layer is None:
            return
        if not hasattr(self.preprocessing_layer, "feature_cache"):
            return

        feature_cache = getattr(self.preprocessing_layer, "feature_cache", None)
        if feature_cache is None:
            self.preprocessing_layer.feature_cache = {}
            feature_cache = self.preprocessing_layer.feature_cache

        if "sasa_features" not in feature_cache:
            feature_cache["sasa_features"] = {}
        if "secondary_structure_features" not in feature_cache:
            feature_cache["secondary_structure_features"] = {}

        has_sasa = protein_id in feature_cache["sasa_features"]
        has_ss = protein_id in feature_cache["secondary_structure_features"]
        if has_sasa and has_ss:
            return

        self._load_netsurfp_head()

        with torch.no_grad():
            x = residue_emb_t.unsqueeze(0)
            rsa_pred, q3_logits = self.netsurfp_head(x)
            rsa_np = rsa_pred.squeeze(0).detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
            q3_idx = torch.argmax(q3_logits.squeeze(0), dim=-1).detach().cpu().numpy()
            q3_onehot = np.eye(3, dtype=np.float32)[q3_idx]

        feature_cache["sasa_features"][protein_id] = rsa_np
        feature_cache["secondary_structure_features"][protein_id] = q3_onehot
        if hasattr(self.preprocessing_layer, "protein_lengths"):
            self.preprocessing_layer.protein_lengths[protein_id] = int(residue_emb_t.shape[0])
    
    def generate_cis_embedding(self, protein_id: str, sequence: str) -> np.ndarray:
        """
        Generate CIS embedding (CLS token through MLP projection).
        
        Pipeline: ESM CLS (5120) -> cis_projector (5120 -> 2560 -> 5120) -> output (5120)
        
        The CIS model uses a projector MLP (5120 -> 2560 -> 5120 with LayerNorm and residual)
        before average pooling and hadamard interaction.
        
        Args:
            protein_id: Protein identifier
            sequence: Protein sequence
            
        Returns:
            CIS embedding [5120] FP32
        """
        # Check cache
        if protein_id in self.cis_cache:
            return self.cis_cache[protein_id]

        # Reuse tensor cache if available
        cached_tensor = self.cis_tensor_cache.get(protein_id)
        if cached_tensor is not None:
            cis_embedding = cached_tensor.detach().cpu().numpy().astype(np.float32)
            self.cis_cache[protein_id] = cis_embedding
            return cis_embedding
        
        # Get ESM embedding
        cls_emb, _ = self.get_esm_embeddings(sequence)
        
        # Apply CIS projector (5120 -> 2560 -> 5120 MLP with LayerNorm)
        # This matches the CIS model architecture: projector(hidden_2560) -> average_pooling -> hadamard -> MLP
        if self.cis_projector is not None:
            with torch.no_grad():
                cls_tensor = torch.from_numpy(cls_emb).to(self.device).float()
                projected = self.cis_projector(cls_tensor)
                cis_embedding = projected.cpu().numpy().astype(np.float32)
        else:
            # Fallback: raw ESM CLS without projection (will produce incorrect results!)
            self.logger.warning(f"CIS projector not loaded, using raw ESM CLS for {protein_id} - results may be incorrect!")
            cis_embedding = cls_emb.astype(np.float32)
        
        # Cache result
        self.cis_cache[protein_id] = cis_embedding
        self.cis_tensor_cache[protein_id] = torch.from_numpy(cis_embedding).to(self.device)
        
        return cis_embedding

    def generate_cis_and_pooled_tensors(
        self,
        protein_id: str,
        sequence: str,
        return_attention: bool = True,
        return_residue_emb: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate CIS and pooled embeddings in one ESM forward pass on GPU."""
        features = self.generate_single_feature_tensors(
            protein_id,
            sequence,
            return_attention=return_attention,
            return_residue_emb=return_residue_emb,
        )
        return features['processed_cis'], features['processed_residue_attn']

    def generate_single_feature_tensors(
        self,
        protein_id: str,
        sequence: str,
        return_attention: bool = True,
        return_residue_emb: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate raw/processed single-protein vectors in one ESM forward pass."""
        cached_raw_cis = self.raw_cis_tensor_cache.get(protein_id)
        cached_raw_avg = self.raw_residue_avg_tensor_cache.get(protein_id)
        cached_cis = self.cis_tensor_cache.get(protein_id)
        cached_pooled = self.pooled_tensor_cache.get(protein_id)

        if (
            cached_raw_cis is not None
            and cached_raw_avg is not None
            and cached_cis is not None
            and cached_pooled is not None
            and not return_residue_emb
        ):
            return {
                'raw_cis': cached_raw_cis,
                'raw_residue_avg': cached_raw_avg,
                'processed_cis': cached_cis,
                'processed_residue_attn': cached_pooled,
            }

        cls_emb_t, residue_emb_t = self._get_esm_embeddings_torch(sequence)

        with torch.no_grad():
            raw_cis_t = cls_emb_t.detach().float()
            raw_residue_avg_t = residue_emb_t.mean(dim=0).detach().float()

            if self.cis_projector is not None:
                cis_t = self.cis_projector(cls_emb_t)
            else:
                self.logger.warning(
                    f"CIS projector not loaded, using raw ESM CLS for {protein_id} - results may be incorrect!"
                )
                cis_t = cls_emb_t

            projected = residue_emb_t
            if self.input_layer is not None:
                projected = self.input_layer(projected)
            if self.internal_projector is not None:
                projected = self.internal_projector(projected)

            if return_residue_emb:
                self.residue_cache[protein_id] = projected.detach().cpu().numpy().astype(np.float32)

            self._ensure_online_multifeature(protein_id, residue_emb_t)

            if self.preprocessing_layer is not None:
                projected = projected.unsqueeze(0)
                projected = self.preprocessing_layer(projected, protein_ids=[protein_id])
                projected = projected.squeeze(0)

            pooled_t = self.pooling_layer(projected)

            if return_attention:
                attn_weights = self.pooling_layer.get_attention_weights()
                if attn_weights is not None:
                    weights = attn_weights.squeeze(0) if attn_weights.dim() == 3 else attn_weights
                    self.attention_cache[protein_id] = format_attention_data(protein_id, sequence, weights)

        cis_t = cis_t.detach().float()
        pooled_t = pooled_t.detach().float()

        self.raw_cis_tensor_cache[protein_id] = raw_cis_t
        self.raw_residue_avg_tensor_cache[protein_id] = raw_residue_avg_t
        self.cis_tensor_cache[protein_id] = cis_t
        self.pooled_tensor_cache[protein_id] = pooled_t

        return {
            'raw_cis': raw_cis_t,
            'raw_residue_avg': raw_residue_avg_t,
            'processed_cis': cis_t,
            'processed_residue_attn': pooled_t,
        }
    
    def generate_pooled_embedding(
        self, 
        protein_id: str, 
        sequence: str,
        return_attention: bool = True,
        return_residue_emb: bool = False
    ) -> EmbeddingResult:
        """
        Generate pooled embedding through full pipeline.
        
        Pipeline: ESM -> input_layer -> internal_projector -> preprocessing -> pooling
        
        Args:
            protein_id: Protein identifier
            sequence: Protein sequence
            return_attention: Whether to return attention weights
            return_residue_emb: Whether to return residue embeddings (for IG)
            
        Returns:
            EmbeddingResult with pooled embedding and optional attention weights
        """
        # Check cache
        if protein_id in self.pooled_cache and not return_residue_emb:
            return EmbeddingResult(
                protein_id=protein_id,
                pooled_embedding=self.pooled_cache[protein_id],
                attention_weights=self.attention_cache.get(protein_id)
            )
        
        # Get ESM embeddings
        _, residue_emb_t = self._get_esm_embeddings_torch(sequence)
        embeddings = residue_emb_t  # [seq_len, esm_dim]
        
        # Debug logging for component status
        self.logger.debug(f"[{protein_id}] Pipeline components:")
        self.logger.debug(f"  - input_layer: {self.input_layer is not None}")
        self.logger.debug(f"  - internal_projector: {self.internal_projector is not None}")
        self.logger.debug(f"  - preprocessing_layer: {self.preprocessing_layer is not None}")
        self.logger.debug(f"  - pooling_layer: {self.pooling_layer is not None}")
        self.logger.debug(f"  - multifeature_dir: {self.multifeature_dir}")
        
        with torch.no_grad():
            # 1. Input layer projection (ESM dim -> embedding_dim)
            if self.input_layer is not None:
                projected = self.input_layer(embeddings)  # [seq_len, 1280]
                self.logger.debug(
                    f"[{protein_id}] After input_layer: shape={projected.shape}, "
                    f"mean={projected.mean().item():.6f}, std={projected.std().item():.6f}"
                )
            else:
                projected = embeddings
                self.logger.debug(f"[{protein_id}] No input_layer, using raw ESM embeddings")
            
            # 2. Internal projector (MLP adapter)
            if self.internal_projector is not None:
                projected = self.internal_projector(projected)  # [seq_len, 1280]
                self.logger.debug(
                    f"[{protein_id}] After internal_projector: shape={projected.shape}, "
                    f"mean={projected.mean().item():.6f}, std={projected.std().item():.6f}"
                )
            
            # Store residue embeddings for IG analysis
            residue_embeddings_output = None
            if return_residue_emb:
                residue_embeddings_output = projected.cpu().numpy().astype(np.float32)
                self.residue_cache[protein_id] = residue_embeddings_output

            self._ensure_online_multifeature(protein_id, residue_emb_t)
            
            # 3. Preprocessing (multimodal features)
            if self.preprocessing_layer is not None:
                projected = projected.unsqueeze(0)  # [1, seq_len, 1280]
                projected = self.preprocessing_layer(projected, protein_ids=[protein_id])
                projected = projected.squeeze(0)  # [seq_len, 1280]
                self.logger.debug(
                    f"[{protein_id}] After preprocessing: shape={projected.shape}, "
                    f"mean={projected.mean().item():.6f}, std={projected.std().item():.6f}"
                )
            else:
                self.logger.debug(f"[{protein_id}] No preprocessing_layer, skipping multifeature fusion")
            
            # 4. Attention pooling
            pooled = self.pooling_layer(projected)  # [1280]
            self.logger.debug(
                f"[{protein_id}] After pooling: shape={pooled.shape}, "
                f"mean={pooled.mean().item():.6f}, std={pooled.std().item():.6f}"
            )
            
            # Get attention weights
            attention_data = None
            if return_attention:
                attn_weights = self.pooling_layer.get_attention_weights()
                if attn_weights is not None:
                    weights = attn_weights.squeeze(0) if attn_weights.dim() == 3 else attn_weights
                    attention_data = format_attention_data(protein_id, sequence, weights)
        
        # Convert to numpy
        pooled_embedding = pooled.cpu().numpy().astype(np.float32)
        
        # Cache results
        self.pooled_cache[protein_id] = pooled_embedding
        if attention_data is not None:
            self.attention_cache[protein_id] = attention_data
        
        return EmbeddingResult(
            protein_id=protein_id,
            pooled_embedding=pooled_embedding,
            attention_weights=attention_data,
            residue_embeddings=residue_embeddings_output
        )

    def _generate_cis_from_cls_torch(
        self,
        protein_id: str,
        cls_emb_t: torch.Tensor,
        cache_numpy: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            cls_tensor = cls_emb_t.to(self.device).float()
            if self.cis_projector is not None:
                cis_t = self.cis_projector(cls_tensor)
            else:
                self.logger.warning(
                    f"CIS projector not loaded, using raw ESM CLS for {protein_id} - results may be incorrect!"
                )
                cis_t = cls_tensor

        cis_t = cis_t.detach().float()
        self.cis_tensor_cache[protein_id] = cis_t
        if cache_numpy:
            self.cis_cache[protein_id] = cis_t.cpu().numpy().astype(np.float32)
        return cis_t

    def _generate_cis_from_cls(self, protein_id: str, cls_emb: np.ndarray) -> np.ndarray:
        cls_tensor = torch.from_numpy(cls_emb).to(self.device).float()
        cis_t = self._generate_cis_from_cls_torch(protein_id, cls_tensor, cache_numpy=True)
        return cis_t.cpu().numpy().astype(np.float32)

    def generate_residue_embedding(self, protein_id: str, sequence: str) -> np.ndarray:
        """
        Generate LMDB-equivalent residue embeddings for sequence models.

        Pipeline: ESM residue embeddings -> input_layer only

        This returns the same semantic stage as legacy LMDB inputs: [L, 1280]
        after the initial 5120 -> 1280 projection, before the model's own
        internal projector / preprocessing / pooling stack.
        """
        cached = self.sequence_input_cache.get(protein_id)
        if cached is not None:
            return cached

        _, residue_emb = self.get_esm_embeddings(sequence)
        embeddings = torch.from_numpy(residue_emb).to(self.device)

        with torch.no_grad():
            if self.input_layer is not None:
                projected = self.input_layer(embeddings)
            else:
                projected = embeddings

        sequence_input = projected.cpu().float().numpy().astype(np.float32)
        self.sequence_input_cache[protein_id] = sequence_input
        return sequence_input

    def _generate_pooled_from_residue_torch(
        self,
        protein_id: str,
        sequence: str,
        residue_emb_t: torch.Tensor,
        return_attention: bool,
        return_residue_emb: bool,
        cache_numpy: bool = True,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        embeddings = residue_emb_t.to(self.device).float()

        with torch.no_grad():
            if self.input_layer is not None:
                projected = self.input_layer(embeddings)
            else:
                projected = embeddings

            if self.internal_projector is not None:
                projected = self.internal_projector(projected)

            residue_embeddings_output = None
            if return_residue_emb:
                residue_embeddings_output = projected.cpu().numpy().astype(np.float32)
                self.residue_cache[protein_id] = residue_embeddings_output

            self._ensure_online_multifeature(protein_id, embeddings)

            if self.preprocessing_layer is not None:
                projected = projected.unsqueeze(0)
                projected = self.preprocessing_layer(projected, protein_ids=[protein_id])
                projected = projected.squeeze(0)

            pooled_t = self.pooling_layer(projected)

            attention_data = None
            if return_attention:
                attn_weights = self.pooling_layer.get_attention_weights()
                if attn_weights is not None:
                    weights = attn_weights.squeeze(0) if attn_weights.dim() == 3 else attn_weights
                    attention_data = format_attention_data(protein_id, sequence, weights)

        pooled_t = pooled_t.detach().float()
        self.pooled_tensor_cache[protein_id] = pooled_t
        if cache_numpy:
            self.pooled_cache[protein_id] = pooled_t.cpu().numpy().astype(np.float32)
        if attention_data is not None:
            self.attention_cache[protein_id] = attention_data

        return pooled_t, residue_embeddings_output, attention_data

    def _generate_pooled_from_residue(
        self,
        protein_id: str,
        sequence: str,
        residue_emb: np.ndarray,
        return_attention: bool,
        return_residue_emb: bool
    ) -> EmbeddingResult:
        embeddings = torch.from_numpy(residue_emb).to(self.device).float()
        pooled_t, residue_embeddings_output, attention_data = self._generate_pooled_from_residue_torch(
            protein_id,
            sequence,
            embeddings,
            return_attention=return_attention,
            return_residue_emb=return_residue_emb,
            cache_numpy=True,
        )

        pooled_embedding = pooled_t.cpu().numpy().astype(np.float32)

        return EmbeddingResult(
            protein_id=protein_id,
            pooled_embedding=pooled_embedding,
            attention_weights=attention_data,
            residue_embeddings=residue_embeddings_output
        )
    
    def generate_embeddings_batch(
        self,
        protein_sequences: Dict[str, str],
        mode: str = "both",
        show_progress: bool = True,
        batch_size: int = 16,
        return_residue_emb: bool = True,
        collect_results: bool = True,
    ) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings for multiple proteins.
        
        Args:
            protein_sequences: Dict mapping protein_id to sequence
            mode: "cis", "pooled", or "both"
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping protein_id to EmbeddingResult
        """
        from tqdm import tqdm
        
        results = {}
        
        if self._use_esme:
            items = list(protein_sequences.items())
            items.sort(key=lambda kv: len(kv[1]))
            progress = None
            if show_progress:
                progress = tqdm(total=len(items), desc="Generating embeddings")
            effective_batch_size = max(1, int(batch_size))
            start = 0
            while start < len(items):
                batch = items[start:start + effective_batch_size]
                batch_ids = [item[0] for item in batch]
                batch_seqs = [item[1] for item in batch]
                try:
                    batch_embs = self._esme_forward_embeddings_batch(batch_seqs)
                    for protein_id, sequence, emb_seq in zip(batch_ids, batch_seqs, batch_embs):
                        if emb_seq.dim() == 2:
                            emb_tensor = emb_seq
                        elif emb_seq.dim() == 3:
                            emb_tensor = emb_seq[0]
                        else:
                            raise RuntimeError(
                                f"Unexpected esm-efficient embedding shape: {tuple(emb_seq.shape)}"
                            )

                        if emb_tensor.requires_grad:
                            emb_tensor = emb_tensor.detach()

                        if emb_tensor.shape[-1] != self.esm_dim:
                            raise RuntimeError(
                                f"ESM embedding dim mismatch: got {emb_tensor.shape[-1]}, expected {self.esm_dim}"
                            )

                        cls_emb_t = emb_tensor[0].float()
                        residue_emb_t = emb_tensor[1:-1].float()

                        result = EmbeddingResult(protein_id=protein_id) if collect_results else None
                        if mode in ["cis", "both"]:
                            cis_t = self._generate_cis_from_cls_torch(
                                protein_id,
                                cls_emb_t,
                                cache_numpy=False,
                            )
                            if result is not None:
                                result.cis_embedding = cis_t.detach().cpu().numpy().astype(np.float32)
                        if mode in ["pooled", "both"]:
                            pooled_t, residue_embeddings_output, attention_data = self._generate_pooled_from_residue_torch(
                                protein_id,
                                sequence,
                                residue_emb_t,
                                return_attention=True,
                                return_residue_emb=return_residue_emb,
                                cache_numpy=False,
                            )
                            if result is not None:
                                result.pooled_embedding = pooled_t.detach().cpu().numpy().astype(np.float32)
                                result.attention_weights = attention_data
                                result.residue_embeddings = residue_embeddings_output

                        if result is not None:
                            results[protein_id] = result
                    start += len(batch_ids)
                    if progress is not None:
                        progress.update(len(batch_ids))
                except torch.cuda.OutOfMemoryError as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if effective_batch_size <= 1:
                        for protein_id in batch_ids:
                            self.logger.error(f"Failed to generate embedding for {protein_id}: {e}")
                        start += len(batch_ids)
                        if progress is not None:
                            progress.update(len(batch_ids))
                    else:
                        next_batch_size = max(1, effective_batch_size // 2)
                        self.logger.warning(
                            "CUDA OOM in embedding batch generation "
                            f"(current_batch={effective_batch_size}). Retrying with batch_size={next_batch_size}."
                        )
                        effective_batch_size = next_batch_size
                        continue
                except RuntimeError as e:
                    err_msg = str(e).lower()
                    if "out of memory" in err_msg and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if effective_batch_size <= 1:
                            for protein_id in batch_ids:
                                self.logger.error(f"Failed to generate embedding for {protein_id}: {e}")
                            start += len(batch_ids)
                            if progress is not None:
                                progress.update(len(batch_ids))
                        else:
                            next_batch_size = max(1, effective_batch_size // 2)
                            self.logger.warning(
                                "RuntimeError OOM in embedding batch generation "
                                f"(current_batch={effective_batch_size}). Retrying with batch_size={next_batch_size}."
                            )
                            effective_batch_size = next_batch_size
                            continue
                    else:
                        for protein_id in batch_ids:
                            self.logger.error(f"Failed to generate embedding for {protein_id}: {e}")
                        start += len(batch_ids)
                        if progress is not None:
                            progress.update(len(batch_ids))
                except Exception as e:
                    for protein_id in batch_ids:
                        self.logger.error(f"Failed to generate embedding for {protein_id}: {e}")
                    start += len(batch_ids)
                    if progress is not None:
                        progress.update(len(batch_ids))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if progress is not None:
                progress.close()
            return results

        iterator = protein_sequences.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", total=len(protein_sequences))
        
        for protein_id, sequence in iterator:
            try:
                result = EmbeddingResult(protein_id=protein_id) if collect_results else None
                
                if mode in ["cis", "both"]:
                    cis_value = self.generate_cis_embedding(protein_id, sequence)
                    if result is not None:
                        result.cis_embedding = cis_value
                
                if mode in ["pooled", "both"]:
                    pooled_result = self.generate_pooled_embedding(
                        protein_id, 
                        sequence,
                        return_attention=True,
                        return_residue_emb=return_residue_emb
                    )
                    if result is not None:
                        result.pooled_embedding = pooled_result.pooled_embedding
                        result.attention_weights = pooled_result.attention_weights
                        result.residue_embeddings = pooled_result.residue_embeddings
                
                if result is not None:
                    results[protein_id] = result
                
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for {protein_id}: {e}")
                continue
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and len(results) % 50 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def clear_cache(self) -> None:
        """Clear embedding caches."""
        self.cis_cache.clear()
        self.pooled_cache.clear()
        self.cis_tensor_cache.clear()
        self.pooled_tensor_cache.clear()
        self.raw_cis_tensor_cache.clear()
        self.raw_residue_avg_tensor_cache.clear()
        self.attention_cache.clear()
        self.residue_cache.clear()
        self.sequence_input_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cached_cis_embedding(self, protein_id: str) -> Optional[np.ndarray]:
        """Get CIS embedding from cache."""
        return self.cis_cache.get(protein_id)

    def get_cached_cis_tensor(self, protein_id: str) -> Optional[torch.Tensor]:
        """Get CIS embedding tensor from cache."""
        return self.cis_tensor_cache.get(protein_id)
    
    def get_cached_pooled_embedding(self, protein_id: str) -> Optional[np.ndarray]:
        """Get pooled embedding from cache."""
        return self.pooled_cache.get(protein_id)

    def get_cached_pooled_tensor(self, protein_id: str) -> Optional[torch.Tensor]:
        """Get pooled embedding tensor from cache."""
        return self.pooled_tensor_cache.get(protein_id)
    
    def get_cached_attention(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """Get attention weights from cache."""
        return self.attention_cache.get(protein_id)
    
    def get_cached_residue_embedding(self, protein_id: str) -> Optional[np.ndarray]:
        """Get residue embeddings from cache (for IG analysis)."""
        return self.residue_cache.get(protein_id)

    def get_cached_sequence_input_embedding(self, protein_id: str) -> Optional[np.ndarray]:
        """Get LMDB-equivalent residue embeddings from cache."""
        return self.sequence_input_cache.get(protein_id)
    
    # ==================== IG Analysis Interface ====================
    
    def get_input_layer_for_ig(self) -> Optional[nn.Module]:
        """Get input layer module for IG analysis."""
        return self.input_layer
    
    def get_internal_projector_for_ig(self) -> Optional[nn.Module]:
        """Get internal projector module for IG analysis."""
        return self.internal_projector
    
    def get_preprocessing_layer_for_ig(self) -> Optional[nn.Module]:
        """Get preprocessing layer module for IG analysis."""
        return self.preprocessing_layer
    
    def get_pooling_layer_for_ig(self) -> Optional[nn.Module]:
        """Get pooling layer module for IG analysis."""
        return self.pooling_layer
