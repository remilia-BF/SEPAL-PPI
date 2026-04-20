import json
import logging
from pathlib import Path
from typing import Dict

import lmdb
import numpy as np


class SingleProteinLMDBExporter:
    """Export four single-protein representations into independent LMDB files."""

    LMDB_NAMES = {
        "raw_cis": "single_raw_cis.lmdb",
        "raw_residue_avg": "single_raw_residue_avg.lmdb",
        "processed_cis": "single_processed_cis.lmdb",
        "processed_residue_attn": "single_processed_residue_attn.lmdb",
    }

    def __init__(self, embedding_generator, logger: logging.Logger):
        self.embedding_generator = embedding_generator
        self.logger = logger

    def _estimate_map_size(self, num_proteins: int, dims: Dict[str, int]) -> int:
        total_bytes = 0
        for key in self.LMDB_NAMES:
            dim = int(dims.get(key, 0))
            total_bytes += num_proteins * max(1, dim) * 4

        # Reserve margin for metadata and LMDB overhead.
        map_size = int(max(512 * 1024 * 1024, total_bytes * 3))
        return map_size

    def export(self, protein_sequences: Dict[str, str], output_dir: Path) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not protein_sequences:
            raise ValueError("No protein sequences provided for single LMDB export")

        first_protein = next(iter(protein_sequences))
        first_seq = protein_sequences[first_protein]
        first_features = self.embedding_generator.generate_single_feature_tensors(
            first_protein,
            first_seq,
            return_attention=True,
            return_residue_emb=False,
        )

        dims = {k: int(v.shape[-1]) for k, v in first_features.items() if k in self.LMDB_NAMES}
        map_size = self._estimate_map_size(len(protein_sequences), dims)

        envs = {}
        txns = {}
        output_paths: Dict[str, str] = {}

        try:
            for feature_key, lmdb_name in self.LMDB_NAMES.items():
                lmdb_path = output_dir / lmdb_name
                if lmdb_path.exists():
                    # LMDB can be a directory; overwrite by reusing path.
                    pass
                env = lmdb.open(str(lmdb_path), map_size=map_size)
                txn = env.begin(write=True)
                envs[feature_key] = env
                txns[feature_key] = txn
                output_paths[feature_key] = str(lmdb_path)

                meta = {
                    "feature": feature_key,
                    "dtype": "float32",
                    "vector_dim": int(dims.get(feature_key, 0)),
                    "count": int(len(protein_sequences)),
                    "source": "sepal-ppi single ig-mode export",
                }
                txn.put(b"_meta", json.dumps(meta, ensure_ascii=False).encode("utf-8"))

            for idx, (protein_id, sequence) in enumerate(protein_sequences.items()):
                if idx == 0:
                    features = first_features
                else:
                    features = self.embedding_generator.generate_single_feature_tensors(
                        protein_id,
                        sequence,
                        return_attention=True,
                        return_residue_emb=False,
                    )

                key = protein_id.encode("utf-8")
                for feature_key in self.LMDB_NAMES:
                    vector = features[feature_key].detach().float().cpu().numpy().astype(np.float32, copy=False)
                    txns[feature_key].put(key, vector.tobytes())

            for feature_key, txn in txns.items():
                txn.commit()
                envs[feature_key].sync()

        finally:
            for env in envs.values():
                env.close()

        self.logger.info(
            "Exported single-mode LMDB representations for %d proteins", len(protein_sequences)
        )
        for feature_key, path in output_paths.items():
            self.logger.info("  %s -> %s", feature_key, path)

        return output_paths
