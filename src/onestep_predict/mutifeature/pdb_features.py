"""Generate SASA and secondary structure features directly from PDB files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_io import save_sasa_features_json, save_secondary_structure_features_json

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    PDBParser = None
    DSSP = None


LOGGER = logging.getLogger(__name__)


SS_MAP = {
    "H": [1, 0, 0],
    "G": [1, 0, 0],
    "I": [1, 0, 0],
    "E": [0, 1, 0],
    "B": [0, 1, 0],
}


def _extract_residue_ids(structure) -> List[Tuple[str, int]]:
    """Extract standard residue ids from first model in sequence order."""
    out: List[Tuple[str, int]] = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue
            out.append((chain.id, residue.id[1]))
    return out


def _compute_dssp_maps(structure, pdb_path: Path, logger: logging.Logger) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], List[int]]]:
    """Return RSA and Q3 one-hot maps keyed by (chain, resseq)."""
    rsa_map: Dict[Tuple[str, int], float] = {}
    ss_map: Dict[Tuple[str, int], List[int]] = {}

    try:
        dssp = DSSP(structure[0], str(pdb_path))
    except Exception as e:
        logger.warning("DSSP failed for %s: %s", pdb_path.name, e)
        return rsa_map, ss_map

    for key in dssp.keys():
        chain_id = key[0]
        resseq = key[1][1]
        dssp_item = dssp[key]
        ss_char = dssp_item[2]
        rsa = float(dssp_item[3])
        rsa_map[(chain_id, resseq)] = max(0.0, min(1.0, rsa))
        ss_map[(chain_id, resseq)] = SS_MAP.get(ss_char, [0, 0, 1])

    return rsa_map, ss_map


def generate_pdb_multifeature(
    pdb_dir: Path,
    protein_ids: List[str],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Generate `sasa_features.json` and `secondary_structure_features.json` from PDB.

    Returns output_dir path.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for PDB feature generation")

    log = logger or LOGGER
    parser = PDBParser(QUIET=True)

    sasa_by_id: Dict[str, List[float]] = {}
    ss_by_id: Dict[str, np.ndarray] = {}

    for pid in protein_ids:
        pdb_file = None
        for ext in (".pdb", ".PDB"):
            candidate = pdb_dir / f"{pid}{ext}"
            if candidate.exists():
                pdb_file = candidate
                break
        if pdb_file is None:
            log.warning("PDB not found for %s in %s", pid, pdb_dir)
            continue

        try:
            structure = parser.get_structure(pid, str(pdb_file))
            residue_ids = _extract_residue_ids(structure)
            rsa_map, ss_map = _compute_dssp_maps(structure, pdb_file, log)

            rsa_vals: List[float] = []
            ss_vals: List[List[int]] = []
            for rid in residue_ids:
                rsa_vals.append(float(rsa_map.get(rid, 0.0)))
                ss_vals.append(ss_map.get(rid, [0, 0, 1]))

            sasa_by_id[pid] = rsa_vals
            ss_by_id[pid] = np.array(ss_vals, dtype=np.int32)
        except Exception as e:
            log.warning("Failed to generate PDB multifeature for %s: %s", pid, e)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_sasa_features_json(sasa_by_id, output_dir / "sasa_features.json")
    save_secondary_structure_features_json(ss_by_id, output_dir / "secondary_structure_features.json")
    return output_dir
