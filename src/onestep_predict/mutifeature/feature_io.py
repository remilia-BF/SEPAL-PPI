"""Serialization and parsing helpers for residue-level multimodal features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def format_secondary_structure_onehot(values: np.ndarray) -> str:
    """Format [L,3] one-hot matrix as '[1,0,0],[0,1,0],...' string."""
    rows: List[str] = []
    for row in values:
        rows.append(f"[{int(row[0])},{int(row[1])},{int(row[2])}]")
    return ",".join(rows)


def parse_secondary_structure_onehot(ss_str: str) -> np.ndarray:
    """Parse '[1,0,0],[0,1,0],...' to np.ndarray [L,3]."""
    ss_str = ss_str.replace(" ", "").replace("\n", "")
    if not ss_str:
        return np.zeros((0, 3), dtype=np.float32)
    segs = ss_str.split("],[")
    segs[0] = segs[0].lstrip("[")
    segs[-1] = segs[-1].rstrip("]")
    rows: List[List[float]] = []
    for seg in segs:
        vals = [float(x) for x in seg.split(",") if x != ""]
        if len(vals) != 3:
            vals = [0.0, 0.0, 1.0]
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def save_sasa_features_json(sasa_by_id: Dict[str, Iterable[float]], output_path: Path) -> None:
    items = []
    for protein_id, values in sasa_by_id.items():
        items.append(
            {
                "protein_id": protein_id,
                "sasa": ",".join(f"{float(v):.6f}" for v in values),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def save_secondary_structure_features_json(ss_by_id: Dict[str, np.ndarray], output_path: Path) -> None:
    items = []
    for protein_id, onehot in ss_by_id.items():
        items.append(
            {
                "protein_id": protein_id,
                "secondary_structure": format_secondary_structure_onehot(onehot),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
