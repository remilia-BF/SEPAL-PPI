"""Generate SASA/secondary-structure features from sequence via NetSurfP-style head."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.onestep_predict.embedding_generator import EmbeddingGenerator

from .feature_io import save_sasa_features_json, save_secondary_structure_features_json


LOGGER = logging.getLogger(__name__)


class NetSurfStyleHead(nn.Module):
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


def resolve_netsurfp_checkpoint(precision: str) -> Path:
    precision = (precision or "bf16").strip().lower()
    ckpt = Path("cache") / "netsurfp_head" / precision / "netsurfp_head_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"NetSurfP head checkpoint not found: {ckpt}")
    return ckpt


def load_head_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    payload = torch.load(str(ckpt_path), map_location=device)
    model_cfg = payload["model_config"]
    model = NetSurfStyleHead(
        in_dim=int(model_cfg["in_dim"]),
        model_dim=int(model_cfg["model_dim"]),
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_dropout=float(model_cfg.get("lstm_dropout", 0.5)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def generate_netsurfp_multifeature(
    protein_sequences: Dict[str, str],
    output_dir: Path,
    esm_model: str,
    esm_precision: str,
    force_hf_esm: bool = False,
    checkpoint_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Generate RSA and Q3 one-hot features from sequence-only inputs."""
    log = logger or LOGGER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(checkpoint_path) if checkpoint_path else resolve_netsurfp_checkpoint(esm_precision)
    head_model = load_head_checkpoint(ckpt, device)

    generator = EmbeddingGenerator(
        esm_model_name=esm_model,
        esm_precision=esm_precision,
        force_hf_esm=force_hf_esm,
        logger=log,
    )
    generator.load_models()

    sasa_by_id: Dict[str, np.ndarray] = {}
    ss_by_id: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for protein_id, seq in protein_sequences.items():
            if not seq:
                continue
            _, residue = generator.get_esm_embeddings(seq)
            emb = torch.from_numpy(residue.astype(np.float32)).unsqueeze(0).to(device)
            rsa_pred, q3_logits = head_model(emb)
            rsa_np = rsa_pred.squeeze(0).cpu().numpy().astype(np.float32)
            q3_np = np.argmax(q3_logits.squeeze(0).cpu().numpy(), axis=-1)
            onehot = np.eye(3, dtype=np.int32)[q3_np]
            sasa_by_id[protein_id] = rsa_np
            ss_by_id[protein_id] = onehot

    output_dir.mkdir(parents=True, exist_ok=True)
    save_sasa_features_json(sasa_by_id, output_dir / "sasa_features.json")
    save_secondary_structure_features_json(ss_by_id, output_dir / "secondary_structure_features.json")
    return output_dir
