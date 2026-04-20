#!/usr/bin/env python3
"""
SEPAL-PPI One-Step Prediction Script

One-step ensemble prediction for protein-protein interactions from PDB files.

Usage:
    python sepal-ppi.predict.py \
        --pdb-dir /path/to/pdb_files \
        --interaction-list /path/to/pairs.csv \
        --ensemble-config cache/sepal-st50/ensemble_inference_config.yaml

Features:
    - Automatic sequence extraction from PDB files
    - On-the-fly ESM embedding generation (CIS + Pooled)
    - Ensemble prediction with meta-learner
    - Attention weight analysis
    - Optional Integrated Gradients attribution analysis (pair/both modes)
    - Single IG mode exports 4 single-protein LMDB representations
    - Genome mode: on-the-fly upper-triangular pairing (keeps self-pairs, skips reverse duplicates)

Input format:
    - PDB directory: Directory containing .pdb files (named as {protein_id}.pdb)
    - Interaction list: CSV file with protein1,protein2[,label] format

Output:
    - ensemble_predictions.csv: Prediction results
    - attention_weights.jsonl: Per-protein attention weights
    - prediction_summary.json: Summary with metrics and metadata
    - single_raw_cis.lmdb (single ig_mode)
    - single_raw_residue_avg.lmdb (single ig_mode)
    - single_processed_cis.lmdb (single ig_mode)
    - single_processed_residue_attn.lmdb (single ig_mode)
    - ig_attributions.jsonl (optional, pair/both modes)
    - ig_pdbs/ (optional, pair/both modes)


"""

import os
import sys
import argparse
import logging
import csv
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> tuple[logging.Logger, Optional[str]]:
    """Set up logging with concise console + detailed file output."""
    logger = logging.getLogger('sepal-ppi.predict')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    else:
        # Keep console concise in default mode. Detailed logs go to file.
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    resolved_log_file = None
    if log_file:
        resolved_log_file = log_file
    elif output_dir is not None:
        resolved_log_file = str(output_dir / 'run.log')

    if resolved_log_file:
        log_path = Path(resolved_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger, resolved_log_file


def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def _resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"sepal-ppi-outputdata/{args.dataset_name}_{timestamp}")


def _resolve_ensemble_config(args) -> tuple[str, bool]:
    """Resolve ensemble config path and whether user explicitly provided it."""
    explicit = ('--ensemble-config' in sys.argv) or ('-e' in sys.argv)
    if explicit:
        return args.ensemble_config, True

    precision = (args.esm_precision or 'bf16').strip().lower()
    precision_path = Path(f"cache/sepal-st50/{precision}/ensemble_inference_config.yaml")
    if precision_path.exists():
        return str(precision_path), False

    legacy_default = Path("cache/sepal-st50/ensemble_inference_config.yaml")
    return str(legacy_default), False


def _print_line(char: str = '-', width: int = 70) -> None:
    print(char * width)


def _summarize_label_counts(labels: Optional[list]) -> tuple[Optional[int], Optional[int]]:
    if not labels:
        return None, None
    positives = sum(1 for v in labels if int(v) == 1)
    negatives = sum(1 for v in labels if int(v) == 0)
    return positives, negatives


def _quick_scan_input_pairs(interaction_list: Optional[str]) -> Optional[int]:
    if not interaction_list or (not os.path.isfile(interaction_list)):
        return None
    try:
        with open(interaction_list, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() in {'protein1', '#protein1'}:
                    continue
                count += 1
            return count
    except Exception:
        return None


def check_vram_availability(model_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if sufficient VRAM is available for the specified ESM model.
    
    Args:
        model_name: ESM model name (e.g., 'esm2_15b')
        logger: Logger instance
        
    Returns:
        Dict with VRAM check status and details
    """
    import torch
    
    vram_requirements = {
        "esm2_15b": 30,  # GB
        "esm2_3b": 8,
        "esm2_650m": 3,
        "esm2_150m": 1,
    }
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (may be very slow)")
        return {
            'available': False,
            'passed': False,
            'total_vram_gb': None,
            'required_vram_gb': vram_requirements.get(model_name, 30),
            'message': 'CUDA not available'
        }
    
    try:
        device_props = torch.cuda.get_device_properties(0)
        total_vram_gb = device_props.total_memory / (1024 ** 3)
        required_vram_gb = vram_requirements.get(model_name, 30)
        
        # Reserve 4GB margin
        margin_gb = 4
        
        if total_vram_gb < required_vram_gb + margin_gb:
            logger.warning(
                f"Limited VRAM: {total_vram_gb:.1f} GB available, "
                f"{required_vram_gb} GB required for {model_name}. "
                f"Consider using a smaller model."
            )
            return {
                'available': True,
                'passed': False,
                'total_vram_gb': total_vram_gb,
                'required_vram_gb': required_vram_gb,
                'message': 'VRAM below recommended threshold'
            }
        
        logger.info(
            f"VRAM check passed: {total_vram_gb:.1f} GB available, "
            f"{required_vram_gb} GB required for {model_name}"
        )
        return {
            'available': True,
            'passed': True,
            'total_vram_gb': total_vram_gb,
            'required_vram_gb': required_vram_gb,
            'message': 'VRAM check passed'
        }
        
    except Exception as e:
        logger.warning(f"Could not check VRAM: {e}")
        return {
            'available': True,
            'passed': True,
            'total_vram_gb': None,
            'required_vram_gb': vram_requirements.get(model_name, 30),
            'message': f'VRAM check skipped: {e}'
        }


def validate_inputs(args, logger: logging.Logger) -> bool:
    """
    Validate input arguments.
    
    Args:
        args: Parsed arguments
        logger: Logger instance
        
    Returns:
        True if all inputs are valid
    """
    valid = True
    
    legacy_mode = args.legacy_predict_config is not None

    # Check input mode (not required in legacy LMDB reuse mode)
    if not legacy_mode:
        has_pdb = bool(args.pdb_dir)
        has_fasta = bool(args.fasta)

        if has_pdb and has_fasta:
            logger.error("--pdb-dir and --fasta are mutually exclusive")
            valid = False
        elif (not has_pdb) and (not has_fasta):
            logger.error("Either --pdb-dir or --fasta is required unless --legacy-predict-config is used")
            valid = False

        if has_pdb:
            if not os.path.isdir(args.pdb_dir):
                logger.error(f"PDB directory does not exist: {args.pdb_dir}")
                valid = False
            else:
                pdb_files = list(Path(args.pdb_dir).glob("*.pdb")) + list(Path(args.pdb_dir).glob("*.PDB"))
                if not pdb_files:
                    logger.error(f"No PDB files found in: {args.pdb_dir}")
                    valid = False
                else:
                    logger.info(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")

        if has_fasta:
            if not os.path.isfile(args.fasta):
                logger.error(f"FASTA file does not exist: {args.fasta}")
                valid = False
            if getattr(args, 'ensemble_config_explicit', False):
                logger.error("FASTA mode does not allow explicit --ensemble-config; remove it to use auto-selected precision config")
                valid = False
    
    # Check interaction list vs genome mode (legacy mode uses predict config for data entry)
    if not legacy_mode:
        if args.genome_mode and args.interaction_list:
            logger.error("--interaction-list and --genome-mode are mutually exclusive")
            valid = False
        elif (not args.genome_mode) and (not args.interaction_list):
            logger.error("--interaction-list is required unless --genome-mode is enabled")
            valid = False

        if args.interaction_list:
            if not os.path.isfile(args.interaction_list):
                logger.error(f"Interaction list file does not exist: {args.interaction_list}")
                valid = False
        elif args.genome_mode:
            logger.info("Genome mode enabled: on-the-fly upper-triangular pairing (no pairs.csv)")

    if legacy_mode:
        if not os.path.isfile(args.legacy_predict_config):
            logger.error(f"Legacy predict config file does not exist: {args.legacy_predict_config}")
            valid = False
        if args.genome_mode:
            logger.warning("--genome-mode is ignored in --legacy-predict-config mode")
        if args.interaction_list:
            logger.warning("--interaction-list is ignored in --legacy-predict-config mode")
    
    # Check ensemble config (not required in legacy LMDB reuse mode)
    if not legacy_mode:
        if not os.path.isfile(args.ensemble_config):
            logger.error(f"Ensemble config file does not exist: {args.ensemble_config}")
            valid = False
    
    # Validate ESM model name
    valid_models = ["esm2_15b", "esm2_3b", "esm2_650m", "esm2_150m", "esm1b_650m"]
    if args.esm_model not in valid_models:
        logger.error(f"Invalid ESM model: {args.esm_model}. Valid options: {valid_models}")
        valid = False
    
    # Warn if not using default model
    if args.esm_model != "esm2_15b":
        logger.warning(
            f"Using {args.esm_model} instead of default esm2_15b. "
            f"Make sure your model checkpoints match this embedding dimension."
        )
    
    # Check optional multifeature directory
    if args.multifeature_dir:
        if not os.path.isdir(args.multifeature_dir):
            logger.warning(f"Multifeature directory does not exist: {args.multifeature_dir}")
    
    # Check optional checkpoint overrides
    if args.input_layer_ckpt:
        if not os.path.isfile(args.input_layer_ckpt):
            logger.error(f"Input layer checkpoint does not exist: {args.input_layer_ckpt}")
            valid = False
    
    if args.model_pretrain_ckpt:
        if not os.path.isfile(args.model_pretrain_ckpt):
            logger.error(f"Model pretrain checkpoint does not exist: {args.model_pretrain_ckpt}")
            valid = False

    if args.lenth <= 0:
        logger.error(f"--length/--lenth must be a positive integer, got: {args.lenth}")
        valid = False
    
    return valid


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SEPAL-PPI One-Step Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses default ensemble config)
  python sepal-ppi.predict.py \\
      --pdb-dir /path/to/pdb_files \\
      --interaction-list /path/to/pairs.csv

  # Genome mode (auto-generate all-vs-all pairs)
  python sepal-ppi.predict.py \\
      --pdb-dir /path/to/pdb_files \\
      --ensemble-config /path/to/custom_config.yaml \\
      --genome-mode

  # With IG attribution analysis
  python sepal-ppi.predict.py \\
      --pdb-dir /path/to/pdb_files \\
      --interaction-list /path/to/pairs.csv \\
      --enable-ig

  # With custom ensemble config and output directory
  python sepal-ppi.predict.py \\
      --pdb-dir /path/to/pdb_files \\
      --interaction-list /path/to/pairs.csv \\
      --ensemble-config /path/to/custom_config.yaml \\
      --output-dir /path/to/output

  # With multifeature for better accuracy
  python sepal-ppi.predict.py \\
      --pdb-dir /path/to/pdb_files \\
      --interaction-list /path/to/pairs.csv \\
      --multifeature-dir /path/to/multifeature

Input format:
  Interaction list should be a CSV file with format:
    protein1,protein2[,label]
  
  Where label (0 or 1) is optional and used for evaluation metrics.
"""
    )
    
    # Required arguments
    parser.add_argument(
        '--pdb-dir', '-p',
        type=str,
        required=False,
        help='Directory containing PDB files (named as {protein_id}.pdb); optional in --legacy-predict-config mode'
    )

    parser.add_argument(
        '--fasta', '-f',
        type=str,
        default=None,
        help='Input FASTA file for sequence-only mode (mutually exclusive with --pdb-dir)'
    )

    parser.add_argument(
        '--legacy-predict-config',
        type=str,
        default=None,
        help='Reuse legacy ensemble_predict data entry directly from a predict YAML (FASTA + LMDB path mapping)'
    )
    
    parser.add_argument(
        '--interaction-list', '-i',
        type=str,
        default=None,
        help='CSV file with protein pairs: protein1,protein2[,label] (required unless --genome-mode)'
    )

    parser.add_argument(
        '--genome-mode',
        action='store_true',
        help='Generate all-vs-all pairs from PDB-derived FASTA (no interaction-list needed)'
    )
    
    parser.add_argument(
        '--ensemble-config', '-e',
        type=str,
        default='cache/sepal-st50/ensemble_inference_config.yaml',
        help='Path to ensemble inference configuration YAML file; if omitted, auto-selects cache/sepal-st50/{esm_precision}/ensemble_inference_config.yaml'
    )
    
    # ESM model configuration
    parser.add_argument(
        '--esm-model',
        type=str,
        default='esm2_15b',
        choices=['esm2_15b', 'esm2_3b', 'esm2_650m', 'esm2_150m', 'esm1b_650m'],
        help='ESM model to use for embeddings (default: esm2_15b)'
    )

    parser.add_argument(
        '--esm-precision',
        type=str,
        default='bf16',
        choices=['bf16', 'int8', 'int6', 'int4'],
        help='ESM model precision: bf16 (default), int8/int6/int4 via esm-efficient quantization'
    )
    
    # Optional checkpoint overrides
    parser.add_argument(
        '--input-layer-ckpt',
        type=str,
        default=None,
        help='Override path to input_layer.pth (5120 -> 1280 projection)'
    )
    
    parser.add_argument(
        '--model-pretrain-ckpt',
        type=str,
        default=None,
        help='Override path to complete_model.pth (MLP adapter + pooling)'
    )
    
    # Multifeature
    parser.add_argument(
        '--multifeature-dir',
        type=str,
        default=None,
        help='Directory containing multimodal features (optional)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: sepal-ppi-outputdata/{dataset}_{timestamp})'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='onestep',
        help='Dataset name for output directory naming (default: onestep)'
    )
    
    # Processing options
    parser.add_argument(
        '--prebuild-embeddings',
        action='store_true',
        help='Pre-build all embeddings before inference (uses more memory but faster)'
    )

    parser.add_argument(
        '--emb-batch',
        type=int,
        default=16,
        help='Batch size for single-protein embedding preparation (prebuild + runtime prep, default: 16)'
    )

    parser.add_argument(
        '--predict-batch',
        type=int,
        default=5120,
        help='Batch size for pair prediction (default: 5120)'
    )

    parser.add_argument(
        '--length', '--lenth',
        dest='lenth',
        type=int,
        default=1022,
        help='Maximum sequence length allowed in --genome-mode (default: 1022). Sequences longer than this are excluded from FASTA and prediction'
    )
    
    parser.add_argument(
        '--enable-ig',
        action='store_true',
        help='Enable Integrated Gradients attribution analysis'
    )

    parser.add_argument(
        '--ig-mode',
        type=str,
        default='single',
        choices=['single', 'pair', 'both'],
        help='IG mode: single=export 4 single-protein LMDBs, pair=pair-level IG attribution, both=single IG + pair-level IG (default: single)'
    )

    parser.add_argument(
        '--single-lmdb-export-dir',
        type=str,
        default=None,
        help='Optional output directory for single-mode LMDB export (default: --output-dir)'
    )

    parser.add_argument(
        '--ig-baseline-path',
        type=str,
        default=None,
        help='Path to global-mean baseline file for IG (.npz/.npy/.json). Supported keys for .npz/.json: residue, cis, pooled'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (optional)'
    )
    
    # Skip checks (for advanced users)
    parser.add_argument(
        '--skip-vram-check',
        action='store_true',
        help='Skip VRAM availability check'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict mode: verify all components load successfully and perform MD5 checks'
    )
    
    parser.add_argument(
        '--verify-md5',
        type=str,
        default=None,
        help='Path to JSON file with expected MD5 hashes for checkpoint files'
    )
    
    parser.add_argument(
        '--force-hf-esm',
        action='store_true',
        help='Force load ESM from HuggingFace (same as Pooling_lmdb_creat.py, bypasses local cache)'
    )
    
    return parser.parse_args()


def print_multifeature_command(pdb_dir: str, fasta_path: str, output_dir: str, concise: bool = False):
    """Print command for generating multifeature files."""
    cmd = (
        f"python mutifeature_tools/one_step_mutifeature.py "
        f"-p {pdb_dir} -f {fasta_path} -o {output_dir}"
    )
    if concise:
        print(f"Multifeature hint: {cmd}")
    else:
        print("\n" + "=" * 70)
        print("MULTIFEATURE GENERATION COMMAND")
        print("=" * 70)
        print("If you need multimodal features, run the following command:")
        print()
        print(f"  {cmd}")
        print()
        print("Then re-run prediction with --multifeature-dir option:")
        print(f"  --multifeature-dir {output_dir}")
        print("=" * 70 + "\n")
    return cmd


def print_compact_summary(
    args,
    start_time: datetime,
    end_time: datetime,
    results: Dict[str, Any],
    vram_info: Optional[Dict[str, Any]],
    input_pairs_hint: Optional[int],
    log_file_path: Optional[str]
) -> None:
    stats = results.get('summary_stats', {})
    unique_proteins = stats.get('unique_proteins')
    interaction_pairs = stats.get('interaction_pairs', input_pairs_hint)
    labels_pos = stats.get('labels_positive')
    labels_neg = stats.get('labels_negative')
    embeddings_generated = stats.get('embeddings_generated', unique_proteins)
    pairs_inferred = stats.get('pairs_inferred', interaction_pairs)
    multifeature_dir = stats.get('multifeature_dir')
    model_components_loaded = stats.get('model_components_loaded')
    model_classifier_sources = stats.get('model_classifier_sources') or {}

    print("=" * 70)
    print(f"SEPAL-PPI One-Step Prediction ({args.esm_model}, {args.esm_precision})")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_line('-')
    print("Step 1: Preparing data...")
    print("Data loaded:")

    pdb_count = stats.get('pdb_files_count')
    if pdb_count is not None:
        print(f"  - PDB files: {pdb_count}")
    if unique_proteins is not None:
        print(f"  - Unique proteins: {unique_proteins}")
    if interaction_pairs is not None:
        if labels_pos is not None and labels_neg is not None:
            print(
                f"  - Interaction pairs: {interaction_pairs} "
                f"(labels: {labels_pos} positive, {labels_neg} negative)"
            )
        else:
            print(f"  - Interaction pairs: {interaction_pairs}")

    _print_line('-')
    if vram_info is not None:
        if vram_info.get('total_vram_gb') is not None:
            check_mark = "✓" if vram_info.get('passed') else "!"
            print(
                f"Resource check: VRAM {vram_info['total_vram_gb']:.1f}GB "
                f"(≥{vram_info['required_vram_gb']}GB required) {check_mark}"
            )
        else:
            print(f"Resource check: {vram_info.get('message', 'N/A')}")

    if model_components_loaded is not None:
        print(f"Model loading: ensemble components loaded ({model_components_loaded} classifiers) ✓")
        if isinstance(model_classifier_sources, dict):
            for model_name, source_info in model_classifier_sources.items():
                if not isinstance(source_info, dict):
                    continue
                if source_info.get('status') != 'loaded':
                    continue
                model_kind = 'cis' if source_info.get('is_cis') else 'residue'
                print(
                    f"  - {model_name} ({model_kind}): {source_info.get('path')} "
                    f"[source={source_info.get('source')}]"
                )
    else:
        print("Model loading: completed ✓")

    if multifeature_dir:
        print(f"Multimodal features loaded from {multifeature_dir}")
    else:
        print("Multimodal features: not enabled")

    _print_line('-')
    print("Step 2: Loading ensemble components...")
    _print_line('-')
    if embeddings_generated is not None:
        print(f"Processing: {embeddings_generated} protein embeddings generated")
    _print_line('-')
    print("Step 3: Generating embeddings...")
    _print_line('-')
    if pairs_inferred is not None:
        print(f"{pairs_inferred} pairs inferred ✓")

    metrics = results.get('evaluation_metrics')
    _print_line('-')
    if metrics and ('auroc' in metrics) and ('aupr' in metrics):
        print(f"Evaluation: AUROC = {metrics['auroc']:.4f}, AUPR = {metrics['aupr']:.4f}")
    else:
        print("Evaluation: N/A (no labels)")

    _print_line('-')
    print(f"Output saved to {results['output_dir']}:")
    for _, path in results.get('output_paths', {}).items():
        print(f"  - {Path(path).name}")

    if log_file_path:
        print(f"  - {Path(log_file_path).name} (detailed log)")

    _print_line('-')
    duration = _format_duration((end_time - start_time).total_seconds())
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (duration {duration})")
    print("=" * 70)


def print_compact_header(args, start_time: datetime) -> None:
    print("=" * 70)
    print(f"SEPAL-PPI One-Step Prediction ({args.esm_model}, {args.esm_precision})")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_line('-')


def print_compact_resource_line(vram_info: Optional[Dict[str, Any]]) -> None:
    if vram_info is None:
        return
    if vram_info.get('total_vram_gb') is not None:
        check_mark = "✓" if vram_info.get('passed') else "!"
        print(
            f"Resource check: VRAM {vram_info['total_vram_gb']:.1f}GB "
            f"(≥{vram_info['required_vram_gb']}GB required) {check_mark}"
        )
    else:
        print(f"Resource check: {vram_info.get('message', 'N/A')}")
    _print_line('-')


def print_compact_footer(
    start_time: datetime,
    end_time: datetime,
    results: Dict[str, Any],
    log_file_path: Optional[str]
) -> None:
    print(f"Output saved to {results['output_dir']}:")
    for _, path in results.get('output_paths', {}).items():
        print(f"  - {Path(path).name}")
    if log_file_path:
        print(f"  - {Path(log_file_path).name} (detailed log)")
    _print_line('-')
    duration = _format_duration((end_time - start_time).total_seconds())
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (duration {duration})")
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()
    start_time = datetime.now()

    resolved_output_dir = _resolve_output_dir(args)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger, log_file_path = setup_logging(args.verbose, args.log_file, resolved_output_dir)

    if args.verbose:
        logger.info("=" * 70)
        logger.info("SEPAL-PPI One-Step Prediction")
        logger.info("=" * 70)
    else:
        print_compact_header(args, start_time)
    
    # Load MD5 verification data if provided
    expected_md5_hashes = None

    # Resolve effective ensemble config path and explicitness.
    args.ensemble_config, args.ensemble_config_explicit = _resolve_ensemble_config(args)
    logger.info(
        f"Using ensemble config: {args.ensemble_config} "
        f"(explicit={args.ensemble_config_explicit})"
    )
    if args.verify_md5:
        import json
        if not os.path.isfile(args.verify_md5):
            logger.error(f"MD5 verification file not found: {args.verify_md5}")
            sys.exit(1)
        
        with open(args.verify_md5, 'r', encoding='utf-8') as f:
            expected_md5_hashes = json.load(f)
        logger.info(f"Loaded MD5 verification file: {args.verify_md5}")
    
    # Validate inputs
    if not validate_inputs(args, logger):
        logger.error("Input validation failed, exiting")
        sys.exit(1)
    
    # Check VRAM availability
    vram_info = None
    if not args.skip_vram_check:
        vram_info = check_vram_availability(args.esm_model, logger)
    if not args.verbose:
        print_compact_resource_line(vram_info)
    
    # Import prediction module
    try:
        from src.onestep_predict import (
            OnestepPredictEngine,
            OnestepPredictConfig,
            format_evaluation_metrics_report
        )
        from src.onestep_predict.embedding_generator import _verify_checkpoint_md5
    except ImportError as e:
        logger.error(f"Failed to import prediction module: {e}")
        logger.error("Make sure you're running from the SEPAL-PPI root directory")
        sys.exit(1)
    
    # Perform MD5 verification if requested
    if expected_md5_hashes and (args.strict or args.verify_md5):
        logger.info("Performing MD5 verification...")
        verification_failed = False
        
        # Check ensemble config checkpoints
        if os.path.isfile(args.ensemble_config):
            import yaml
            with open(args.ensemble_config, 'r', encoding='utf-8') as f:
                ensemble_config = yaml.safe_load(f)
            
            for model_name, model_config in ensemble_config.get('models', {}).items():
                for ckpt_key in ['input_layer_ckpt', 'model_pretrain_ckpt']:
                    ckpt_path = model_config.get(ckpt_key)
                    if ckpt_path and ckpt_key in expected_md5_hashes:
                        expected_md5 = expected_md5_hashes[ckpt_key]
                        if not _verify_checkpoint_md5(ckpt_path, expected_md5, logger):
                            verification_failed = True
        
        # Check override checkpoints
        if args.input_layer_ckpt and 'input_layer_ckpt' in expected_md5_hashes:
            if not _verify_checkpoint_md5(args.input_layer_ckpt, 
                                          expected_md5_hashes['input_layer_ckpt'], logger):
                verification_failed = True
        
        if args.model_pretrain_ckpt and 'model_pretrain_ckpt' in expected_md5_hashes:
            if not _verify_checkpoint_md5(args.model_pretrain_ckpt,
                                          expected_md5_hashes['model_pretrain_ckpt'], logger):
                verification_failed = True
        
        if verification_failed:
            logger.error("MD5 verification failed! Checkpoint files may be corrupted or incorrect.")
            if args.strict:
                logger.error("Strict mode enabled, exiting.")
                sys.exit(1)
            else:
                logger.warning("Continuing despite MD5 mismatch...")
    
    # Create configuration
    config = OnestepPredictConfig(
        pdb_dir=args.pdb_dir,
        fasta_path=args.fasta,
        interaction_list=args.interaction_list,
        ensemble_config=args.ensemble_config,
        ensemble_config_explicit=args.ensemble_config_explicit,
        legacy_predict_config=args.legacy_predict_config,
        input_layer_ckpt=args.input_layer_ckpt,
        model_pretrain_ckpt=args.model_pretrain_ckpt,
        esm_model=args.esm_model,
        esm_precision=args.esm_precision,
        force_hf_esm=args.force_hf_esm,
        multifeature_dir=args.multifeature_dir,
        dataset_name=args.dataset_name,
        prebuild_embeddings=args.prebuild_embeddings,
        emb_batch=args.emb_batch,
        predict_batch=args.predict_batch,
        enable_ig=args.enable_ig,
        ig_mode=args.ig_mode,
        single_lmdb_export_dir=args.single_lmdb_export_dir,
        ig_baseline_path=args.ig_baseline_path,
        strict_mode=args.strict,  # Pass strict mode flag
        genome_mode=args.genome_mode,
        genome_max_length=args.lenth,
        concise_logging=not args.verbose,
        output_log_file=log_file_path,
        output_dir=str(resolved_output_dir),
    )
    
    # Create and run prediction engine
    try:
        engine = OnestepPredictEngine(config, logger=logger)
        results = engine.run()

        if args.verbose:
            # Verbose mode keeps detailed report output on console.
            logger.info("=" * 70)
            logger.info("PREDICTION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Output directory: {results['output_dir']}")

            for name, path in results.get('output_paths', {}).items():
                logger.info(f"  {name}: {path}")

            if results.get('evaluation_metrics'):
                metrics_report = format_evaluation_metrics_report(results['evaluation_metrics'])
                print(metrics_report)
        else:
            end_time = datetime.now()
            print_compact_footer(
                start_time=start_time,
                end_time=end_time,
                results=results,
                log_file_path=log_file_path
            )
        
        # Print multifeature command only if multifeature was NOT used at all
        # (both args and auto-detection)
        if (not args.legacy_predict_config) and args.pdb_dir and args.multifeature_dir is None and not results.get('multifeature_used', False):
            fasta_path = Path(results['output_dir']) / "proteins.fasta"
            multifeature_out = Path(results['output_dir']) / "multifeature"
            print_multifeature_command(
                args.pdb_dir,
                str(fasta_path),
                str(multifeature_out),
                concise=not args.verbose
            )

        if args.verbose:
            logger.info("Done!")
        else:
            print("Done!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
