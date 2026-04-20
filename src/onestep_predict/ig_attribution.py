"""
Integrated Gradients Attribution Module for One-Step Prediction

Computes Integrated Gradients (IG) attribution scores for:
1. Internal projector (MLP adapter)
2. Pooling layer (attention pooling)
3. Preprocessing layer (multimodal fusion)

Supports saving attribution scores to PDB B-factor columns.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_ig_baselines(
    baseline_path: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Load IG baseline vectors from file.

    Supported formats:
    - .npz: keys `residue`, `cis`, `pooled`
    - .npy: a single vector used for all branches
    - .json: dict with optional keys `residue`, `cis`, `pooled`
    """
    log = logger or logging.getLogger(__name__)
    out: Dict[str, Optional[np.ndarray]] = {
        "residue": None,
        "cis": None,
        "pooled": None,
    }

    if not baseline_path:
        return out

    path = Path(baseline_path)
    if not path.exists():
        log.warning("IG baseline file not found: %s. Falling back to zero baseline.", path)
        return out

    try:
        suffix = path.suffix.lower()

        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for k in out.keys():
                    if k in data:
                        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
        elif suffix == ".npy":
            vec = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
            out = {"residue": vec, "cis": vec, "pooled": vec}
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k in out.keys():
                    if k in payload:
                        out[k] = np.asarray(payload[k], dtype=np.float32).reshape(-1)
            elif isinstance(payload, list):
                vec = np.asarray(payload, dtype=np.float32).reshape(-1)
                out = {"residue": vec, "cis": vec, "pooled": vec}
        else:
            log.warning(
                "Unsupported IG baseline format: %s. Use .npz/.npy/.json. Falling back to zero baseline.",
                path,
            )
            return out

        shape_msg = {
            k: (None if v is None else int(v.shape[0]))
            for k, v in out.items()
        }
        log.info("Loaded IG baseline vectors from %s with dims: %s", path, shape_msg)
        return out
    except Exception as e:
        log.warning("Failed to load IG baseline from %s: %s. Falling back to zero baseline.", path, e)
        return {"residue": None, "cis": None, "pooled": None}


@dataclass
class IGResult:
    """Container for IG attribution results."""
    protein_id: str
    sequence: str
    
    # Per-residue IG attributions [seq_len]
    internal_projector_ig: Optional[np.ndarray] = None
    pooling_ig: Optional[np.ndarray] = None  
    preprocessing_ig: Optional[np.ndarray] = None
    
    # Aggregated IG (sum of all components)
    aggregated_ig: Optional[np.ndarray] = None
    
    # Normalized IG for PDB B-factor [0, 100]
    normalized_ig: Optional[np.ndarray] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairIGResult:
    """Container for pair-level IG attribution results."""
    protein1: str
    protein2: str
    cis_ig_protein1: Optional[np.ndarray] = None
    cis_ig_protein2: Optional[np.ndarray] = None
    residue_ig_protein1: Optional[np.ndarray] = None
    residue_ig_protein2: Optional[np.ndarray] = None
    total_ig_protein1: Optional[np.ndarray] = None
    total_ig_protein2: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegratedGradientsCalculator:
    """
    Calculate Integrated Gradients attributions for protein interaction prediction.
    
    IG measures the contribution of each input feature (residue embedding) to the
    model's prediction by integrating gradients along the path from a baseline
    to the input.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
    """
    
    def __init__(
        self,
        internal_projector: Optional[nn.Module] = None,
        pooling_layer: Optional[nn.Module] = None,
        preprocessing_layer: Optional[nn.Module] = None,
        residue_baseline_vector: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        n_steps: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize IG calculator.
        
        Args:
            internal_projector: MLP adapter module
            pooling_layer: Attention pooling module  
            preprocessing_layer: Multimodal fusion module
            device: Computation device
            n_steps: Number of interpolation steps for IG
            logger: Logger instance
        """
        self.internal_projector = internal_projector
        self.pooling_layer = pooling_layer
        self.preprocessing_layer = preprocessing_layer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps
        self.logger = logger or logging.getLogger(__name__)
        self.residue_baseline_vector = residue_baseline_vector
        self._residue_baseline_warned = False
        
        # Move modules to device
        if self.internal_projector is not None:
            self.internal_projector.to(self.device)
        if self.pooling_layer is not None:
            self.pooling_layer.to(self.device)
        if self.preprocessing_layer is not None:
            self.preprocessing_layer.to(self.device)
    
    def _get_baseline(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get baseline input for IG calculation.
        
        Uses zero baseline (default for IG).
        
        Args:
            embeddings: Input embeddings [seq_len, embedding_dim]
            
        Returns:
            Baseline embeddings (zeros with same shape)
        """
        if self.residue_baseline_vector is None:
            return torch.zeros_like(embeddings)

        if embeddings.dim() != 2:
            return torch.zeros_like(embeddings)

        emb_dim = int(embeddings.shape[-1])
        baseline_vec = np.asarray(self.residue_baseline_vector, dtype=np.float32).reshape(-1)
        if baseline_vec.shape[0] != emb_dim:
            if not self._residue_baseline_warned:
                self.logger.warning(
                    "Residue baseline dim mismatch: baseline=%d, embedding=%d. Using zero baseline.",
                    baseline_vec.shape[0],
                    emb_dim,
                )
                self._residue_baseline_warned = True
            return torch.zeros_like(embeddings)

        baseline_t = torch.from_numpy(baseline_vec).to(embeddings.device, dtype=embeddings.dtype)
        baseline_t = baseline_t.unsqueeze(0).expand(embeddings.shape[0], -1)
        return baseline_t
    
    def _interpolate_embeddings(
        self, 
        baseline: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Create interpolated embeddings for IG path integral.
        
        Args:
            baseline: Baseline embeddings [seq_len, embedding_dim]
            embeddings: Target embeddings [seq_len, embedding_dim]
            
        Returns:
            Interpolated embeddings [n_steps, seq_len, embedding_dim]
        """
        alphas = torch.linspace(0, 1, self.n_steps, device=self.device)
        
        # Shape: [n_steps, seq_len, embedding_dim]
        delta = embeddings - baseline
        interpolated = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * delta.unsqueeze(0)
        
        return interpolated
    
    def _compute_gradients(
        self,
        module: nn.Module,
        embeddings: torch.Tensor,
        target_fn: callable = None,
        protein_id: str = None
    ) -> torch.Tensor:
        """
        Compute gradients of output w.r.t. input embeddings.
        
        Args:
            module: Neural network module
            embeddings: Input embeddings [seq_len, embedding_dim] or [batch, seq_len, embedding_dim]
            target_fn: Function to extract scalar target from output
            protein_id: Protein ID for preprocessing layer
            
        Returns:
            Gradients [seq_len, embedding_dim] or [batch, seq_len, embedding_dim]
        """
        embeddings_clone = embeddings.detach().clone()
        embeddings_clone.requires_grad_(True)
        
        # Forward pass
        if module == self.preprocessing_layer and protein_id is not None:
            if embeddings_clone.dim() == 2:
                embeddings_clone_batch = embeddings_clone.unsqueeze(0)
                output = module(embeddings_clone_batch, protein_ids=[protein_id])
                output = output.squeeze(0)
            else:
                # Batch processing
                output = module(embeddings_clone, protein_ids=[protein_id] * embeddings_clone.shape[0])
        else:
            output = module(embeddings_clone)
        
        # Get target scalar
        if target_fn is not None:
            target = target_fn(output)
        else:
            # Default: sum of output (works for pooling)
            target = output.sum()
        
        # Backward pass
        target.backward()
        
        return embeddings_clone.grad
    
    def compute_internal_projector_ig(
        self,
        residue_embeddings: np.ndarray,
        protein_id: str = None
    ) -> Optional[np.ndarray]:
        """
        Compute IG attributions for internal projector (MLP adapter).
        
        Args:
            residue_embeddings: [seq_len, embedding_dim]
            protein_id: Protein identifier
            
        Returns:
            Per-residue IG scores [seq_len]
        """
        if self.internal_projector is None:
            return None
        
        embeddings = torch.from_numpy(residue_embeddings).float().to(self.device)
        baseline = self._get_baseline(embeddings)
        
        # Interpolate
        interpolated = self._interpolate_embeddings(baseline, embeddings)  # [n_steps, seq_len, emb_dim]
        
        # Compute gradients for each step
        all_gradients = []
        
        for step_idx in range(self.n_steps):
            step_emb = interpolated[step_idx]
            grad = self._compute_gradients(self.internal_projector, step_emb)
            all_gradients.append(grad)
        
        # Stack gradients [n_steps, seq_len, embedding_dim]
        gradients = torch.stack(all_gradients, dim=0)
        
        # Integral approximation (trapezoidal rule)
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = avg_gradients.mean(dim=0)  # [seq_len, embedding_dim]
        
        # Scale by (input - baseline)
        ig = integrated_gradients * (embeddings - baseline)
        
        # Aggregate per-residue (L2 norm across embedding dimension)
        per_residue_ig = torch.norm(ig, p=2, dim=1)  # [seq_len]
        
        return per_residue_ig.cpu().numpy()
    
    def compute_pooling_ig(
        self,
        residue_embeddings: np.ndarray,
        protein_id: str = None
    ) -> Optional[np.ndarray]:
        """
        Compute IG attributions for attention pooling layer.
        
        Args:
            residue_embeddings: [seq_len, embedding_dim]
            protein_id: Protein identifier
            
        Returns:
            Per-residue IG scores [seq_len]
        """
        if self.pooling_layer is None:
            return None
        
        embeddings = torch.from_numpy(residue_embeddings).float().to(self.device)
        baseline = self._get_baseline(embeddings)
        
        # Interpolate
        interpolated = self._interpolate_embeddings(baseline, embeddings)
        
        # Compute gradients for each step
        all_gradients = []
        
        # Define target function: L2 norm of pooled output
        def target_fn(output):
            return torch.norm(output, p=2)
        
        for step_idx in range(self.n_steps):
            step_emb = interpolated[step_idx]
            grad = self._compute_gradients(self.pooling_layer, step_emb, target_fn)
            all_gradients.append(grad)
        
        # Stack and integrate
        gradients = torch.stack(all_gradients, dim=0)
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = avg_gradients.mean(dim=0)
        
        # Scale and aggregate
        ig = integrated_gradients * (embeddings - baseline)
        per_residue_ig = torch.norm(ig, p=2, dim=1)
        
        return per_residue_ig.cpu().numpy()
    
    def compute_preprocessing_ig(
        self,
        residue_embeddings: np.ndarray,
        protein_id: str
    ) -> Optional[np.ndarray]:
        """
        Compute IG attributions for preprocessing (multimodal fusion) layer.
        
        Args:
            residue_embeddings: [seq_len, embedding_dim]
            protein_id: Protein identifier
            
        Returns:
            Per-residue IG scores [seq_len]
        """
        if self.preprocessing_layer is None:
            return None
        
        embeddings = torch.from_numpy(residue_embeddings).float().to(self.device)
        baseline = self._get_baseline(embeddings)
        
        # Interpolate
        interpolated = self._interpolate_embeddings(baseline, embeddings)
        
        # Compute gradients for each step
        all_gradients = []
        
        def target_fn(output):
            return output.sum()
        
        for step_idx in range(self.n_steps):
            step_emb = interpolated[step_idx]
            grad = self._compute_gradients(
                self.preprocessing_layer, 
                step_emb, 
                target_fn,
                protein_id=protein_id
            )
            all_gradients.append(grad)
        
        # Stack and integrate
        gradients = torch.stack(all_gradients, dim=0)
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = avg_gradients.mean(dim=0)
        
        # Scale and aggregate
        ig = integrated_gradients * (embeddings - baseline)
        per_residue_ig = torch.norm(ig, p=2, dim=1)
        
        return per_residue_ig.cpu().numpy()
    
    def compute_full_ig(
        self,
        residue_embeddings: np.ndarray,
        protein_id: str,
        sequence: str
    ) -> IGResult:
        """
        Compute full IG attributions for all available layers.
        
        Args:
            residue_embeddings: [seq_len, embedding_dim]
            protein_id: Protein identifier
            sequence: Protein sequence
            
        Returns:
            IGResult with all attribution scores
        """
        result = IGResult(protein_id=protein_id, sequence=sequence)
        
        # Internal projector IG
        try:
            result.internal_projector_ig = self.compute_internal_projector_ig(
                residue_embeddings, protein_id
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute internal projector IG for {protein_id}: {e}")
        
        # Pooling IG
        try:
            result.pooling_ig = self.compute_pooling_ig(
                residue_embeddings, protein_id
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute pooling IG for {protein_id}: {e}")
        
        # Preprocessing IG
        try:
            result.preprocessing_ig = self.compute_preprocessing_ig(
                residue_embeddings, protein_id
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute preprocessing IG for {protein_id}: {e}")
        
        # Aggregate IG scores
        ig_components = [
            result.internal_projector_ig,
            result.pooling_ig,
            result.preprocessing_ig
        ]
        valid_components = [ig for ig in ig_components if ig is not None]
        
        if valid_components:
            # Sum all valid IG components
            result.aggregated_ig = np.sum(valid_components, axis=0)

            res_component = None
            if result.pooling_ig is not None or result.preprocessing_ig is not None:
                pooling = result.pooling_ig if result.pooling_ig is not None else np.zeros_like(result.aggregated_ig)
                preprocessing = (
                    result.preprocessing_ig
                    if result.preprocessing_ig is not None
                    else np.zeros_like(result.aggregated_ig)
                )
                res_component = pooling + preprocessing

            additivity = self._compute_additivity_check(
                total_ig=result.aggregated_ig,
                res_ig=res_component,
                cis_ig=result.internal_projector_ig,
                tolerance=1e-5,
            )
            result.metadata["additivity_check"] = additivity

            # Non-blocking validation: always continue and save outputs.
            if additivity["passed"]:
                self.logger.info(
                    "Additivity check for %s passed (rel_l2=%.3e, max_abs=%.3e)",
                    protein_id,
                    additivity["error_rel_l2"],
                    additivity["error_max_abs"],
                )
            else:
                self.logger.warning(
                    "Additivity check for %s failed but is non-blocking "
                    "(rel_l2=%.3e > tol=%.3e, max_abs=%.3e)",
                    protein_id,
                    additivity["error_rel_l2"],
                    additivity["tolerance"],
                    additivity["error_max_abs"],
                )
            
            # Normalize to [0, 100] for PDB B-factor
            result.normalized_ig = self._normalize_for_bfactor(result.aggregated_ig)
        
        return result
    
    def _normalize_for_bfactor(self, ig_scores: np.ndarray) -> np.ndarray:
        """
        Normalize IG scores to [0, 100] range for PDB B-factor column.
        
        Args:
            ig_scores: Raw IG scores [seq_len]
            
        Returns:
            Normalized scores [0, 100]
        """
        if ig_scores.size == 0:
            return ig_scores
        
        min_val = ig_scores.min()
        max_val = ig_scores.max()
        
        if max_val > min_val:
            normalized = (ig_scores - min_val) / (max_val - min_val) * 100.0
        else:
            normalized = np.full_like(ig_scores, 50.0)
        
        return normalized

    def _compute_additivity_check(
        self,
        total_ig: np.ndarray,
        res_ig: Optional[np.ndarray],
        cis_ig: Optional[np.ndarray],
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """Compute non-blocking additivity diagnostics for G_total ~= G_res + G_cis."""
        total_term = total_ig.astype(np.float64, copy=False)
        res_term = np.zeros_like(total_term)
        cis_term = np.zeros_like(total_term)

        if res_ig is not None:
            res_term = res_ig.astype(np.float64, copy=False)
        if cis_ig is not None:
            cis_term = cis_ig.astype(np.float64, copy=False)

        reconstructed = res_term + cis_term
        error_vec = total_term - reconstructed

        error_l2 = float(np.linalg.norm(error_vec, ord=2))
        total_l2 = float(np.linalg.norm(total_term, ord=2))
        error_rel_l2 = float(error_l2 / (total_l2 + 1e-12))
        error_max_abs = float(np.max(np.abs(error_vec))) if error_vec.size > 0 else 0.0
        passed = bool(error_rel_l2 <= tolerance)

        return {
            "equation": "G_total ~= G_res + G_cis",
            "definition": {
                "G_total": "aggregated_ig",
                "G_res": "pooling_ig + preprocessing_ig",
                "G_cis": "internal_projector_ig"
            },
            "tolerance": float(tolerance),
            "passed": passed,
            "error_l2": error_l2,
            "error_rel_l2": error_rel_l2,
            "error_max_abs": error_max_abs,
            "components_present": {
                "pooling_ig": res_ig is not None,
                "internal_projector_ig": cis_ig is not None,
            },
        }


class PDBBFactorWriter:
    """
    Write IG attribution scores to PDB B-factor columns.
    
    Enables visualization of important residues in molecular viewers like PyMOL.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def write_ig_to_pdb(
        self,
        input_pdb: str,
        output_pdb: str,
        ig_scores: np.ndarray,
        chain_id: str = None
    ) -> bool:
        """
        Write IG scores to PDB B-factor column.
        
        Args:
            input_pdb: Input PDB file path
            output_pdb: Output PDB file path
            ig_scores: Normalized IG scores [seq_len] in [0, 100] range
            chain_id: Optional chain ID to modify (if None, modifies all chains)
            
        Returns:
            True if successful
        """
        try:
            from Bio.PDB import PDBParser, PDBIO, Select
        except ImportError:
            self.logger.error("BioPython is required for PDB writing. Install with: pip install biopython")
            return False
        
        class BFactorSelect(Select):
            """Select class to modify B-factors during writing."""
            def __init__(self, ig_scores, target_chain=None):
                self.ig_scores = ig_scores
                self.target_chain = target_chain
                self.residue_idx = 0
            
            def accept_residue(self, residue):
                return 1
            
            def accept_atom(self, atom):
                return 1
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", input_pdb)
            
            # Iterate through structure and set B-factors
            score_idx = 0
            for model in structure:
                for chain in model:
                    if chain_id is not None and chain.id != chain_id:
                        continue
                    
                    for residue in chain:
                        # Skip hetero residues (water, ligands, etc.)
                        if residue.id[0] != ' ':
                            continue
                        
                        if score_idx < len(ig_scores):
                            bfactor = float(ig_scores[score_idx])
                            for atom in residue:
                                atom.bfactor = bfactor
                            score_idx += 1
            
            # Write output
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_pdb)
            
            self.logger.debug(f"Wrote IG scores to {output_pdb} (modified {score_idx} residues)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write IG to PDB: {e}")
            return False
    
    def batch_write_ig(
        self,
        ig_results: Dict[str, IGResult],
        pdb_dir: str,
        output_dir: str,
        suffix: str = "_ig"
    ) -> Dict[str, str]:
        """
        Write IG scores for multiple proteins.
        
        Args:
            ig_results: Dict mapping protein_id to IGResult
            pdb_dir: Directory containing input PDB files
            output_dir: Directory for output PDB files
            suffix: Suffix to add to output filenames
            
        Returns:
            Dict mapping protein_id to output PDB path
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs): return x

        os.makedirs(output_dir, exist_ok=True)
        output_paths = {}
        
        pdb_path = Path(pdb_dir)
        
        iterator = tqdm(ig_results.items(), desc="Writing IG PDBs", unit="pdb")

        for protein_id, ig_result in iterator:
            if ig_result.normalized_ig is None:
                self.logger.warning(f"No IG scores for {protein_id}, skipping")
                continue
            
            # Find input PDB file
            input_pdb = None
            for ext in [".pdb", ".PDB"]:
                candidate = pdb_path / f"{protein_id}{ext}"
                if candidate.exists():
                    input_pdb = str(candidate)
                    break
            
            if input_pdb is None:
                self.logger.warning(f"PDB file not found for {protein_id}")
                continue
            
            # Output path
            output_pdb = os.path.join(output_dir, f"{protein_id}{suffix}.pdb")
            
            if self.write_ig_to_pdb(input_pdb, output_pdb, ig_result.normalized_ig):
                output_paths[protein_id] = output_pdb
        
        return output_paths


class IGAttributionAnalyzer:
    """
    High-level interface for IG attribution analysis.
    
    Integrates with EmbeddingGenerator for seamless analysis.
    """
    
    def __init__(
        self,
        embedding_generator: 'EmbeddingGenerator' = None,
        residue_baseline_vector: Optional[np.ndarray] = None,
        n_steps: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize IG analyzer.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            n_steps: Number of IG interpolation steps
            logger: Logger instance
        """
        self.embedding_generator = embedding_generator
        self.residue_baseline_vector = residue_baseline_vector
        self.n_steps = n_steps
        self.logger = logger or logging.getLogger(__name__)
        
        self.ig_calculator = None
        self.pdb_writer = PDBBFactorWriter(logger=self.logger)
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize IG calculator from embedding generator."""
        if self.embedding_generator is None:
            raise ValueError("EmbeddingGenerator is required for IG analysis")
        
        # Ensure models are loaded
        if not self.embedding_generator._models_loaded:
            self.embedding_generator.load_models()
        
        # Create IG calculator
        self.ig_calculator = IntegratedGradientsCalculator(
            internal_projector=self.embedding_generator.get_internal_projector_for_ig(),
            pooling_layer=self.embedding_generator.get_pooling_layer_for_ig(),
            preprocessing_layer=self.embedding_generator.get_preprocessing_layer_for_ig(),
            residue_baseline_vector=self.residue_baseline_vector,
            device=self.embedding_generator.device,
            n_steps=self.n_steps,
            logger=self.logger
        )
        
        self._initialized = True
        self.logger.info("IG analyzer initialized")
    
    def analyze_protein(
        self,
        protein_id: str,
        sequence: str,
        residue_embeddings: np.ndarray = None
    ) -> IGResult:
        """
        Compute IG attributions for a single protein.
        
        Args:
            protein_id: Protein identifier
            sequence: Protein sequence
            residue_embeddings: Optional pre-computed residue embeddings
            
        Returns:
            IGResult with attribution scores
        """
        if not self._initialized:
            self.initialize()
        
        # Get residue embeddings if not provided
        if residue_embeddings is None:
            cached = self.embedding_generator.get_cached_residue_embedding(protein_id)
            if cached is not None:
                residue_embeddings = cached
            else:
                # Generate embeddings
                result = self.embedding_generator.generate_pooled_embedding(
                    protein_id, sequence, 
                    return_attention=False, 
                    return_residue_emb=True
                )
                residue_embeddings = result.residue_embeddings
        
        if residue_embeddings is None:
            self.logger.error(f"Could not get residue embeddings for {protein_id}")
            return IGResult(protein_id=protein_id, sequence=sequence)
        
        # Compute full IG
        return self.ig_calculator.compute_full_ig(
            residue_embeddings, 
            protein_id, 
            sequence
        )
    
    def analyze_batch(
        self,
        protein_sequences: Dict[str, str],
        show_progress: bool = True
    ) -> Dict[str, IGResult]:
        """
        Compute IG attributions for multiple proteins.
        
        Args:
            protein_sequences: Dict mapping protein_id to sequence
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping protein_id to IGResult
        """
        from tqdm import tqdm
        
        results = {}
        
        iterator = protein_sequences.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Computing IG attributions", total=len(protein_sequences))
        
        for protein_id, sequence in iterator:
            try:
                results[protein_id] = self.analyze_protein(protein_id, sequence)
            except Exception as e:
                self.logger.error(f"Failed to compute IG for {protein_id}: {e}")
        
        return results
    
    def save_to_pdb(
        self,
        ig_results: Dict[str, IGResult],
        pdb_dir: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save IG scores to PDB B-factor columns.
        
        Args:
            ig_results: Dict mapping protein_id to IGResult
            pdb_dir: Input PDB directory
            output_dir: Output PDB directory
            
        Returns:
            Dict mapping protein_id to output PDB path
        """
        return self.pdb_writer.batch_write_ig(
            ig_results, 
            pdb_dir, 
            output_dir
        )
    
    def save_ig_json(
        self,
        ig_results: Dict[str, IGResult],
        output_path: str
    ) -> None:
        """
        Save IG results to JSONL file.
        
        Args:
            ig_results: Dict mapping protein_id to IGResult
            output_path: Output JSONL file path
        """
        output_file = Path(output_path)
        failure_log_path = output_file.parent / "ig_additivity_failures.jsonl"
        failure_count = 0

        # Ensure this run's sidecar log starts fresh.
        if failure_log_path.exists():
            failure_log_path.unlink()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for protein_id, result in ig_results.items():
                entry = {
                    "protein_id": protein_id,
                    # "sequence": result.sequence,
                    "length": len(result.sequence) if result.sequence else 0,
                }
                
                if result.internal_projector_ig is not None:
                    entry["internal_projector_ig"] = [round(float(x), 6) for x in result.internal_projector_ig]
                
                if result.pooling_ig is not None:
                    entry["pooling_ig"] = [round(float(x), 6) for x in result.pooling_ig]
                
                if result.preprocessing_ig is not None:
                    entry["preprocessing_ig"] = [round(float(x), 6) for x in result.preprocessing_ig]
                
                if result.aggregated_ig is not None:
                    entry["aggregated_ig"] = [round(float(x), 6) for x in result.aggregated_ig]
                
                if result.normalized_ig is not None:
                    entry["normalized_ig"] = [round(float(x), 2) for x in result.normalized_ig]

                additivity = result.metadata.get("additivity_check") if result.metadata else None
                if additivity is not None:
                    entry["additivity_check"] = {
                        "equation": additivity.get("equation"),
                        "tolerance": float(additivity.get("tolerance", 1e-5)),
                        "passed": bool(additivity.get("passed", False)),
                        "error_l2": float(additivity.get("error_l2", 0.0)),
                        "error_rel_l2": float(additivity.get("error_rel_l2", 0.0)),
                        "error_max_abs": float(additivity.get("error_max_abs", 0.0)),
                        "definition": additivity.get("definition", {}),
                    }

                    if not entry["additivity_check"]["passed"]:
                        failure_count += 1
                        with open(failure_log_path, 'a', encoding='utf-8') as ff:
                            ff.write(
                                json.dumps(
                                    {
                                        "protein_id": protein_id,
                                        "additivity_check": entry["additivity_check"],
                                    },
                                    ensure_ascii=False,
                                ) + '\n'
                            )
                
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved IG results to {output_path}")
        if failure_count > 0:
            self.logger.warning(
                "Additivity diagnostics: %d samples exceeded tolerance. "
                "This is non-blocking; details logged at %s",
                failure_count,
                failure_log_path,
            )


class PairIGAttributionAnalyzer:
    """Pair-level IG analyzer for final pair scoring pathways."""

    def __init__(
        self,
        embedding_generator: 'EmbeddingGenerator',
        model_names: List[str],
        model_configs: Dict[str, Dict[str, Any]],
        model_weights: Dict[str, float],
        classifiers: Dict[str, Optional[nn.Module]],
        device: torch.device,
        cis_baseline_vector: Optional[np.ndarray] = None,
        pooled_baseline_vector: Optional[np.ndarray] = None,
        n_steps: int = 50,
        logger: Optional[logging.Logger] = None,
    ):
        self.embedding_generator = embedding_generator
        self.model_names = model_names
        self.model_configs = model_configs
        self.model_weights = model_weights
        self.classifiers = classifiers
        self.device = device
        self.cis_baseline_vector = cis_baseline_vector
        self.pooled_baseline_vector = pooled_baseline_vector
        self.n_steps = n_steps
        self.logger = logger or logging.getLogger(__name__)
        self._cis_baseline_warned = False
        self._pooled_baseline_warned = False

    def _residue_baseline_like(
        self,
        residue_value: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-residue baseline [L, D] using pooled baseline vector when available."""
        if residue_value.dim() != 2:
            return torch.zeros_like(residue_value)

        if self.pooled_baseline_vector is None:
            return torch.zeros_like(residue_value)

        vec = np.asarray(self.pooled_baseline_vector, dtype=np.float32).reshape(-1)
        target_dim = int(residue_value.shape[-1])
        if vec.shape[0] != target_dim:
            if not self._pooled_baseline_warned:
                self.logger.warning(
                    "Pair IG pooled baseline dim mismatch for residue path: baseline=%d, tensor=%d. Using zero baseline.",
                    vec.shape[0],
                    target_dim,
                )
                self._pooled_baseline_warned = True
            return torch.zeros_like(residue_value)

        baseline_vec = torch.from_numpy(vec).to(residue_value.device, dtype=residue_value.dtype)
        return baseline_vec.unsqueeze(0).expand(int(residue_value.shape[0]), -1)

    def _residue_to_pooled(self, residue_embeddings: torch.Tensor, protein_id: str) -> torch.Tensor:
        """Forward residue embeddings through preprocessing + pooling to pooled embedding."""
        projected = residue_embeddings

        if self.embedding_generator.preprocessing_layer is not None:
            projected = projected.unsqueeze(0)
            projected = self.embedding_generator.preprocessing_layer(projected, protein_ids=[protein_id])
            projected = projected.squeeze(0)

        pooled = self.embedding_generator.pooling_layer(projected)
        return pooled

    def _integrated_gradients_pair_residue_per_token(
        self,
        protein1: str,
        protein2: str,
        cis1: torch.Tensor,
        cis2: torch.Tensor,
        residue1: torch.Tensor,
        residue2: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-residue IG for pair residue branch logit L_res."""
        baseline_res1 = self._residue_baseline_like(residue1)
        baseline_res2 = self._residue_baseline_like(residue2)

        alphas = torch.linspace(0.0, 1.0, self.n_steps, device=self.device)
        grad_res1: List[torch.Tensor] = []
        grad_res2: List[torch.Tensor] = []

        cis1_const = cis1.detach()
        cis2_const = cis2.detach()

        for alpha in alphas:
            x_res1 = (baseline_res1 + alpha * (residue1 - baseline_res1)).detach().clone().requires_grad_(True)
            x_res2 = (baseline_res2 + alpha * (residue2 - baseline_res2)).detach().clone().requires_grad_(True)

            pooled1 = self._residue_to_pooled(x_res1, protein1)
            pooled2 = self._residue_to_pooled(x_res2, protein2)

            _, l_res, _ = self._pair_logits(cis1_const, cis2_const, pooled1, pooled2)

            g1, g2 = torch.autograd.grad(
                outputs=l_res,
                inputs=[x_res1, x_res2],
                retain_graph=False,
                allow_unused=True,
            )

            grad_res1.append(torch.zeros_like(x_res1) if g1 is None else g1)
            grad_res2.append(torch.zeros_like(x_res2) if g2 is None else g2)

        def _integrate_per_residue(grad_list: List[torch.Tensor], delta: torch.Tensor) -> np.ndarray:
            grad_tensor = torch.stack(grad_list, dim=0)  # [steps, L, D]
            avg_grad = (grad_tensor[:-1] + grad_tensor[1:]) / 2.0
            ig = avg_grad.mean(dim=0) * delta  # [L, D]
            per_residue = torch.norm(ig, p=2, dim=1)  # [L]
            return per_residue.detach().cpu().numpy()

        residue_ig1 = _integrate_per_residue(grad_res1, residue1 - baseline_res1)
        residue_ig2 = _integrate_per_residue(grad_res2, residue2 - baseline_res2)
        return residue_ig1, residue_ig2

    def _vector_baseline_like(
        self,
        value: torch.Tensor,
        baseline_vector: Optional[np.ndarray],
        branch_name: str,
    ) -> torch.Tensor:
        if baseline_vector is None:
            return torch.zeros_like(value)

        vec = np.asarray(baseline_vector, dtype=np.float32).reshape(-1)
        target_dim = int(value.shape[0])
        if vec.shape[0] != target_dim:
            warned_flag = "_cis_baseline_warned" if branch_name == "cis" else "_pooled_baseline_warned"
            if not getattr(self, warned_flag):
                self.logger.warning(
                    "Pair IG %s baseline dim mismatch: baseline=%d, tensor=%d. Using zero baseline.",
                    branch_name,
                    vec.shape[0],
                    target_dim,
                )
                setattr(self, warned_flag, True)
            return torch.zeros_like(value)

        return torch.from_numpy(vec).to(value.device, dtype=value.dtype)

    def _pair_logits(
        self,
        cis1: torch.Tensor,
        cis2: torch.Tensor,
        pooled1: torch.Tensor,
        pooled2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (L_cis, L_res, L_total) from the same forward graph."""
        l_cis = torch.zeros((), device=self.device, dtype=torch.float32)
        l_res = torch.zeros((), device=self.device, dtype=torch.float32)

        for model_name in self.model_names:
            cfg = self.model_configs.get(model_name, {})
            emb_dim = int(cfg.get('embedding_dim', 5120))
            interaction_type = cfg.get('interaction_type', 'hadamard')
            weight = float(self.model_weights.get(model_name, 0.0))
            classifier = self.classifiers.get(model_name)

            if emb_dim == 5120:
                emb_a, emb_b = cis1, cis2
                target_bucket = 'cis'
            else:
                emb_a, emb_b = pooled1, pooled2
                target_bucket = 'res'

            if interaction_type == 'hadamard':
                interaction = emb_a * emb_b
            elif interaction_type == 'concatenation':
                interaction = torch.cat([emb_a, emb_b], dim=0)
            elif interaction_type == 'difference':
                interaction = torch.abs(emb_a - emb_b)
            else:
                interaction = emb_a * emb_b

            if classifier is None:
                # Fallback consistent with prediction engine fallback branch.
                cos_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=1).squeeze(0)
                logit = cos_sim * 2.0
            else:
                logits_or_prob = classifier(interaction.unsqueeze(0).float()).squeeze()
                # Classifier outputs probability in this codebase.
                prob = torch.clamp(logits_or_prob, 1e-7, 1.0 - 1e-7)
                logit = torch.log(prob / (1.0 - prob))

            weighted = torch.as_tensor(weight, device=self.device, dtype=torch.float32) * logit
            if target_bucket == 'cis':
                l_cis = l_cis + weighted
            else:
                l_res = l_res + weighted

        l_total = l_cis + l_res
        return l_cis, l_res, l_total

    def _integrated_gradients_pair(
        self,
        cis1: torch.Tensor,
        cis2: torch.Tensor,
        pooled1: torch.Tensor,
        pooled2: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Compute G_cis, G_res, G_total for pair-level inputs in one integration routine."""
        baseline_cis1 = self._vector_baseline_like(cis1, self.cis_baseline_vector, "cis")
        baseline_cis2 = self._vector_baseline_like(cis2, self.cis_baseline_vector, "cis")
        baseline_pooled1 = self._vector_baseline_like(pooled1, self.pooled_baseline_vector, "pooled")
        baseline_pooled2 = self._vector_baseline_like(pooled2, self.pooled_baseline_vector, "pooled")

        alphas = torch.linspace(0.0, 1.0, self.n_steps, device=self.device)

        grads = {
            'cis': {'cis1': [], 'cis2': [], 'pool1': [], 'pool2': []},
            'res': {'cis1': [], 'cis2': [], 'pool1': [], 'pool2': []},
            'total': {'cis1': [], 'cis2': [], 'pool1': [], 'pool2': []},
        }

        for alpha in alphas:
            x_cis1 = (baseline_cis1 + alpha * (cis1 - baseline_cis1)).detach().clone().requires_grad_(True)
            x_cis2 = (baseline_cis2 + alpha * (cis2 - baseline_cis2)).detach().clone().requires_grad_(True)
            x_pool1 = (baseline_pooled1 + alpha * (pooled1 - baseline_pooled1)).detach().clone().requires_grad_(True)
            x_pool2 = (baseline_pooled2 + alpha * (pooled2 - baseline_pooled2)).detach().clone().requires_grad_(True)

            l_cis, l_res, l_total = self._pair_logits(x_cis1, x_cis2, x_pool1, x_pool2)

            for target_name, target_scalar in [('cis', l_cis), ('res', l_res), ('total', l_total)]:
                grad_vals = torch.autograd.grad(
                    outputs=target_scalar,
                    inputs=[x_cis1, x_cis2, x_pool1, x_pool2],
                    retain_graph=True,
                    allow_unused=True,
                )
                grads[target_name]['cis1'].append(torch.zeros_like(x_cis1) if grad_vals[0] is None else grad_vals[0])
                grads[target_name]['cis2'].append(torch.zeros_like(x_cis2) if grad_vals[1] is None else grad_vals[1])
                grads[target_name]['pool1'].append(torch.zeros_like(x_pool1) if grad_vals[2] is None else grad_vals[2])
                grads[target_name]['pool2'].append(torch.zeros_like(x_pool2) if grad_vals[3] is None else grad_vals[3])

        def _integrate(grad_list: List[torch.Tensor], delta: torch.Tensor) -> np.ndarray:
            grad_tensor = torch.stack(grad_list, dim=0)
            avg_grad = (grad_tensor[:-1] + grad_tensor[1:]) / 2.0
            ig = avg_grad.mean(dim=0) * delta
            return ig.detach().cpu().numpy()

        outputs = {}
        for target_name in ('cis', 'res', 'total'):
            outputs[f'{target_name}_cis1'] = _integrate(grads[target_name]['cis1'], cis1 - baseline_cis1)
            outputs[f'{target_name}_cis2'] = _integrate(grads[target_name]['cis2'], cis2 - baseline_cis2)
            outputs[f'{target_name}_pool1'] = _integrate(grads[target_name]['pool1'], pooled1 - baseline_pooled1)
            outputs[f'{target_name}_pool2'] = _integrate(grads[target_name]['pool2'], pooled2 - baseline_pooled2)

        # Diagnostic only: never blocks output.
        g_total = np.concatenate([outputs['total_cis1'], outputs['total_cis2'], outputs['total_pool1'], outputs['total_pool2']])
        g_res = np.concatenate([outputs['res_cis1'], outputs['res_cis2'], outputs['res_pool1'], outputs['res_pool2']])
        g_cis = np.concatenate([outputs['cis_cis1'], outputs['cis_cis2'], outputs['cis_pool1'], outputs['cis_pool2']])
        err = g_total - (g_res + g_cis)
        rel = float(np.linalg.norm(err, ord=2) / (np.linalg.norm(g_total, ord=2) + 1e-12))
        outputs['additivity'] = {
            'equation': 'G_total ~= G_res + G_cis',
            'error_rel_l2': rel,
            'error_max_abs': float(np.max(np.abs(err))) if err.size > 0 else 0.0,
            'tolerance': 1e-5,
            'passed': bool(rel <= 1e-5),
        }

        return outputs

    def analyze_pairs(
        self,
        protein_pairs: List[Tuple[str, str]],
        protein_sequences: Dict[str, str],
        show_progress: bool = True,
    ) -> List[PairIGResult]:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs):
                return x

        results: List[PairIGResult] = []
        iterator = tqdm(protein_pairs, desc='Computing pair-level IG', total=len(protein_pairs)) if show_progress else protein_pairs

        for protein1, protein2 in iterator:
            seq1 = protein_sequences.get(protein1)
            seq2 = protein_sequences.get(protein2)
            if seq1 is None or seq2 is None:
                self.logger.warning(f"Missing sequence for pair ({protein1}, {protein2}), skipping")
                continue

            try:
                cis1, pooled1 = self.embedding_generator.generate_cis_and_pooled_tensors(
                    protein1, seq1, return_attention=False, return_residue_emb=True
                )
                cis2, pooled2 = self.embedding_generator.generate_cis_and_pooled_tensors(
                    protein2, seq2, return_attention=False, return_residue_emb=True
                )

                residue1_np = self.embedding_generator.residue_cache.get(protein1)
                residue2_np = self.embedding_generator.residue_cache.get(protein2)
                if residue1_np is None or residue2_np is None:
                    raise RuntimeError("Residue embeddings unavailable for pair-level residue IG")

                residue1_t = torch.from_numpy(residue1_np).to(self.device).float()
                residue2_t = torch.from_numpy(residue2_np).to(self.device).float()

                out = self._integrated_gradients_pair(cis1, cis2, pooled1, pooled2)
                residue_ig1, residue_ig2 = self._integrated_gradients_pair_residue_per_token(
                    protein1=protein1,
                    protein2=protein2,
                    cis1=cis1,
                    cis2=cis2,
                    residue1=residue1_t,
                    residue2=residue2_t,
                )

                result = PairIGResult(
                    protein1=protein1,
                    protein2=protein2,
                    cis_ig_protein1=np.abs(out['cis_cis1']),
                    cis_ig_protein2=np.abs(out['cis_cis2']),
                    residue_ig_protein1=np.abs(residue_ig1),
                    residue_ig_protein2=np.abs(residue_ig2),
                    total_ig_protein1=np.abs(out['total_pool1']),
                    total_ig_protein2=np.abs(out['total_pool2']),
                    metadata={
                        'additivity_check': out['additivity'],
                        'n_steps': self.n_steps,
                    },
                )
                results.append(result)

                if not out['additivity']['passed']:
                    self.logger.warning(
                        "Pair additivity check failed but non-blocking for (%s, %s): rel_l2=%.3e",
                        protein1,
                        protein2,
                        out['additivity']['error_rel_l2'],
                    )
            except Exception as e:
                self.logger.error(f"Failed pair-level IG for ({protein1}, {protein2}): {e}")

        return results

    def save_pair_ig_json(self, pair_results: List[PairIGResult], output_path: str) -> None:
        failure_log_path = Path(output_path).parent / 'pair_ig_additivity_failures.jsonl'
        failure_count = 0
        if failure_log_path.exists():
            failure_log_path.unlink()

        with open(output_path, 'w', encoding='utf-8') as f:
            for r in pair_results:
                entry = {
                    'protein1': r.protein1,
                    'protein2': r.protein2,
                    'cis_ig_protein1': [round(float(x), 6) for x in (r.cis_ig_protein1 if r.cis_ig_protein1 is not None else np.array([]))],
                    'cis_ig_protein2': [round(float(x), 6) for x in (r.cis_ig_protein2 if r.cis_ig_protein2 is not None else np.array([]))],
                    'residue_ig_protein1': [round(float(x), 6) for x in (r.residue_ig_protein1 if r.residue_ig_protein1 is not None else np.array([]))],
                    'residue_ig_protein2': [round(float(x), 6) for x in (r.residue_ig_protein2 if r.residue_ig_protein2 is not None else np.array([]))],
                    'total_ig_protein1': [round(float(x), 6) for x in (r.total_ig_protein1 if r.total_ig_protein1 is not None else np.array([]))],
                    'total_ig_protein2': [round(float(x), 6) for x in (r.total_ig_protein2 if r.total_ig_protein2 is not None else np.array([]))],
                    'metadata': r.metadata,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                additivity = (r.metadata or {}).get('additivity_check', {})
                if additivity and (not bool(additivity.get('passed', False))):
                    failure_count += 1
                    with open(failure_log_path, 'a', encoding='utf-8') as ff:
                        ff.write(
                            json.dumps(
                                {
                                    'protein1': r.protein1,
                                    'protein2': r.protein2,
                                    'additivity_check': additivity,
                                },
                                ensure_ascii=False,
                            ) + '\n'
                        )

        self.logger.info("Saved pair-level IG results to %s", output_path)
        if failure_count > 0:
            self.logger.warning(
                "Pair additivity diagnostics: %d samples exceeded tolerance. Non-blocking; details at %s",
                failure_count,
                failure_log_path,
            )
