import os
import sys
import time
import shutil
import argparse
import subprocess
import warnings
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Dict

import lmdb
import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
def _safe_torch_load(path: str, map_location: torch.device):
    """Load checkpoints while suppressing the FutureWarning about weights_only.

    Tries torch.load(..., weights_only=True) first (preferred for safety),
    and falls back to the legacy behavior if the running PyTorch does not
    support the argument (older versions).
    """
    try:
        # Preferred secure path (and silences the FutureWarning in newer PyTorch)
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older PyTorch without weights_only support
        warnings.filterwarnings(
            "ignore",
            message=r".*weights_only=False.*",
            category=FutureWarning,
        )
        return torch.load(path, map_location=map_location)


# Lazy, on-demand imports (esme or ESMC)
ESME_AVAILABLE = False
try:
    from esme import ESM2, ESM1b
    from esme.alphabet import tokenize, tokenize_unpad
    ESME_AVAILABLE = True
except Exception:
    ESM2 = None
    ESM1b = None
    tokenize = None
    tokenize_unpad = None

# ESMC (kept for compatibility if requested)
ESMC_AVAILABLE = False
try:
    from esm.models.esmc import ESMC, ESMProteinTensor  # type: ignore
    from esm.sdk.api import LogitsConfig  # type: ignore
    from esm.tokenization import EsmSequenceTokenizer  # type: ignore
    ESMC_AVAILABLE = True
except Exception:
    # If esmc dependencies are not installed, later requests for esmc will error out
    pass



# =============================================
# Model maps & quantization
# =============================================
ESME_REPO_BASE = "https://huggingface.co/mhcelik/esm-efficient/resolve/main"
ESME_CACHE_DIR = Path("cache/esm")

# Short name -> safetensors filename (as provided by user list)
ESME_WEIGHT_FILES: Dict[str, str] = {
    "esm2_15b": "esm2_15b.safetensors",
    "esm2_3b": "esm2_3b.safetensors",
    "esm2_650m": "esm2_650m.safetensors",
    "esm2_150m": "esm2_150m.safetensors",
    "esm2_35m": "esm2_35m.safetensors",
    "esm2_8m": "esm2_8m.safetensors",
    "esm1b": "esm1b.safetensors",
    "esm1v_1": "esm1v_1.safetensors",
    "esm1v_2": "esm1v_2.safetensors",
    "esm1v_3": "esm1v_3.safetensors",
    "esm1v_4": "esm1v_4.safetensors",
    "esm1v_5": "esm1v_5.safetensors",
    "esmc_300m": "esmc_300m.safetensors",
    "esmc_600m": "esmc_600m.safetensors",
}

ESME_QUANT_MAP = {"int4": "4bit", "int8": "8bit"}
SUPPORTED_PRECISIONS = {"bf16", "int8", "int4"}

# ESMC local weights kept for legacy fallback path (not used when esme weights exist)
ESMC_MODEL_NAME_MAP: Dict[str, str] = {
    "esmc_300m": "esmc_300m",
    "esmc_600m": "esmc_600m",
}

ESMC_LOCAL_WEIGHTS: Dict[str, str] = {
    "esmc_300m": "data/weights/esmc_300m_2024_12_v0.pth",
    "esmc_600m": "data/weights/esmc_600m_2024_12_v0.pth",
}


def _resolve_quantization(precision: str, device: torch.device) -> Optional[str]:
    if precision == "bf16":
        return None
    if precision not in ESME_QUANT_MAP:
        raise ValueError(f"Unsupported precision: {precision}")
    if device.type != "cuda":
        raise RuntimeError("Quantized esme (int8/int4) requires CUDA device")
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        raise RuntimeError("bitsandbytes is required for int8/int4 quantization") from e
    return ESME_QUANT_MAP[precision]


def _ensure_esme_weights(file_name: str) -> Path:
    """Ensure the safetensors weights exist locally; download if missing."""
    ESME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = ESME_CACHE_DIR / file_name
    if target.exists():
        return target

    url = f"{ESME_REPO_BASE}/{file_name}?download=1"
    print(f"[esme] Downloading weights from {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(target, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
    except Exception as e:
        if target.exists():
            target.unlink()
        raise RuntimeError(f"Failed to download {file_name}: {e}") from e
    print(f"[esme] Saved to {target}")
    return target


def _select_esme_class(model_name: str):
    if model_name.startswith("esm2"):
        return ESM2
    # esm1v/esm1b/esmc share esm1b architecture in esm-efficient release
    return ESM1b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LMDB embeddings (CLS & middle tokens) for protein sequences using esme models (quantizable)"
    )
    parser.add_argument("-m", "--model", required=True, help="Model short name (e.g. esm2_150m, esm2_650m, esmc_600m)")
    parser.add_argument("-f", "--fasta", required=True, help="Input FASTA file path")
    parser.add_argument("-o", "--output-dir", required=True, help="Base output directory (e.g. emb/S5)")
    parser.add_argument("--suffix", required=True, help="Suffix to compose final directory name")
    parser.add_argument("--commit-interval", type=int, default=100, help="Number of sequences per LMDB commit batch")
    parser.add_argument("--save-interval", type=int, default=10, help="GPU cache cleanup interval (in processed sequences)")
    parser.add_argument("--force", action="store_true", help="If target directory exists, delete without prompt")
    parser.add_argument("--append", action="store_true", help="Append to existing LMDB without deleting (overrides --force)")
    parser.add_argument("--no-bucket", action="store_true", help="Do not run bucketization after LMDB generation")
    parser.add_argument("--precision", choices=sorted(SUPPORTED_PRECISIONS), default="bf16",
                        help="esme quantization precision (bf16/int8/int4); outputs remain fp32")
    # Optional: input-layer checkpoint for on-the-fly linear projection (dimension reduction) during LMDB generation
    parser.add_argument("--input-layer-ckpt", type=str, default=None, help="Path to input_layer.pth exported from training")
    parser.add_argument("--source-lmdb", type=str, default=None, help="Path to source LMDB directory to generate from (skips model inference)")
    return parser.parse_args()


def resolve_model(short_name: str) -> Tuple[str, str]:
    """Resolve backend and weight identifier.

    Returns (backend, weight_id)
    backend in {"esme", "esmc"}
    weight_id for esme is safetensors filename; for esmc is the pretrained name.
    """
    sn = short_name.lower()
    if sn in ESME_WEIGHT_FILES:
        return "esme", ESME_WEIGHT_FILES[sn]
    if sn in ESMC_MODEL_NAME_MAP:
        return "esmc", ESMC_MODEL_NAME_MAP[sn]
    raise ValueError(f"未支持的模型短名称: {short_name}")


def load_model(backend: str, weight_id: str, short_name: str, device: torch.device, precision: str):
    """Load model according to backend.

    Returns (model_or_client, tokenizer_obj, meta_dict)
    tokenizer_obj is None for esme (we use tokenize/ tokenize_unpad).
    """
    if backend == "esme":
        if not ESME_AVAILABLE:
            raise RuntimeError("esme is not installed; pip install esme")

        quantization = _resolve_quantization(precision, device)
        weight_path = _ensure_esme_weights(weight_id)
        model_cls = _select_esme_class(short_name)
        print(f"[esme] Loading {short_name} from {weight_path} (quant={quantization or 'bf16'})")
        model = model_cls.from_pretrained(
            str(weight_path),
            device=0 if device.type == "cuda" else "cpu",
            quantization=quantization,
        )
        model.eval()
        return model, None, {}

    if backend == "esmc":
        if not ESMC_AVAILABLE:
            raise RuntimeError("ESMC-related libraries not detected; please install first.")
        client = ESMC.from_pretrained(weight_id, device=device)
        tokenizer = EsmSequenceTokenizer()
        weight_path = ESMC_LOCAL_WEIGHTS.get(short_name)
        if weight_path and os.path.isfile(weight_path):
            try:
                state = _safe_torch_load(weight_path, map_location=device)
                missing, unexpected = client.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[esmc] Local weight load mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    print(f"[esmc] Loaded local weights {weight_path}")
            except Exception as e:
                print(f"[esmc] Failed to load local weights {weight_path}: {e}, continuing with pretrained weights")
        client.eval()
        return client, tokenizer, {}

    raise ValueError(f"Unknown backend: {backend}")


# Map from model short names to embedding dimensions (override-embedding-dim)
MODEL_DIM_MAP: Dict[str, int] = {
    # HF esm2 series
    "esm2_15b": 5120,
    "esm2_3b": 2560,
    "esm2_650m": 1280,
    "esm2_150m": 640,
    # Common extensions (if supported in the future)
    "esm2_35m": 480,
    "esm2_8m": 320,
    # esm1v/esmb
    "esm1b": 1280,
    "esm1b_650m": 1280,
    "esm1v_1": 1280,
    "esm1v_2": 1280,
    "esm1v_3": 1280,
    "esm1v_4": 1280,
    "esm1v_5": 1280,
    # ESMC series
    "esmc_300m": 960,
    "esmc_600m": 1152,
}


def run_bucketization(
    no_cls_lmdb_dir: str,
    fasta_path: str,
    model_short_name: str,
    override_dim: Optional[int] = None,
):
    """Run bucketization preprocessing on noCLSeos.lmdb.
    Priority: override_dim (if provided) > MODEL_DIM_MAP[model_short_name].

    - no_cls_lmdb_dir: path like <...>/noCLSeos.lmdb
    - fasta_path: original fasta path
    - model_short_name: e.g., esm2_150m / esmc_600m
    - override_dim: projected dimension when input-layer-ckpt is provided
    """
    dim = override_dim if override_dim is not None else MODEL_DIM_MAP.get(model_short_name, None)
    if dim is None:
        print(f"[bucket] Embedding dimension for model {model_short_name} not found; skipping automatic bucketization.")
        return

    script_path = os.path.join("emb_tools", "preprocess_bucketed_lmdb.py")
    if not os.path.isfile(script_path):
        print(f"[bucket] Bucketization script not found: {script_path}; skipping.")
        return

    cmd = [
        sys.executable,
        script_path,
        "--source-lmdb", no_cls_lmdb_dir,
        "--output-dir", no_cls_lmdb_dir,
        "--precision", "bf16",
        "--verbose",
        "--override-embedding-dim", str(dim),
        "--max-length-samples", "0",
        "--fasta-file", fasta_path,
    ]
    print("[bucket] Running bucketization command:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("[bucket] Bucketization completed.")
    except subprocess.CalledProcessError as e:
        print(f"[bucket] Bucketization failed with return code {e.returncode}")

def read_fasta_sequences(fasta_file_path):
    """Read FASTA file and return a dict {seq_id: sequence}"""
    sequences = {}
    try:
        with open(fasta_file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq)
    except FileNotFoundError:
        print(f"Error: file {fasta_file_path} does not exist.")
    except Exception as e:
        print(f"Error while reading FASTA file: {e}")
    return sequences


def get_embeddings(backend: str, model_or_client, tokenizer, device, sequence: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get (CLS, middle_tokens) as numpy.float32 arrays in a unified way.

    esme: hook last layer, slice CLS/middle, return fp32
    esmc: same as before
    """
    try:
        if backend == "esme":
            if not ESME_AVAILABLE:
                raise RuntimeError("esme not available")
            if tokenize is None:
                raise RuntimeError("esme tokenizers not available")

            activations: Dict[str, torch.Tensor] = {}

            def _hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    activations["last"] = output[0]
                else:
                    activations["last"] = output

            hook_module = None
            if hasattr(model_or_client, "layers") and model_or_client.layers:
                hook_module = model_or_client.layers[-1]
            elif hasattr(model_or_client, "transformer") and hasattr(model_or_client.transformer, "layers"):
                hook_module = model_or_client.transformer.layers[-1]

            handle = None
            if hook_module is not None:
                handle = hook_module.register_forward_hook(_hook)

            with torch.no_grad():
                if tokenize_unpad is not None:
                    tokens, _indices, cu_lens, max_len = tokenize_unpad([sequence])
                    tokens = tokens.to(device)
                    cu_lens = cu_lens.to(device)
                    _ = model_or_client(tokens, (cu_lens, max_len))
                else:
                    tokens = tokenize([sequence]).to(device)
                    _ = model_or_client(tokens)

            if handle is not None:
                handle.remove()

            if "last" not in activations:
                raise RuntimeError("esme forward did not return embeddings")

            emb = activations["last"]
            if emb.dim() == 3:
                emb_seq = emb[0]
            elif emb.dim() == 2:
                emb_seq = emb
            else:
                raise RuntimeError(f"Unexpected esme embedding shape: {tuple(emb.shape)}")

            cls_embedding = emb_seq[0].float().cpu().numpy()
            middle_embeddings = emb_seq[1:-1].float().cpu().numpy()
            return cls_embedding, middle_embeddings

        if backend == "esmc":
            with torch.no_grad():
                tokens = tokenizer.encode(sequence)
                protein_tensor = ESMProteinTensor(sequence=torch.tensor(tokens).to(device))
                logits_output = model_or_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                emb = logits_output.embeddings  # [layers, tokens, hidden]
                last = emb[-1]  # [tokens, hidden]
                cls_embedding = last[0].float().cpu().numpy()
                middle_embeddings = last[1:-1].float().cpu().numpy()
                return cls_embedding, middle_embeddings
    except Exception as e:
        print(f"Embedding computation error: {e}")
    return None, None


class NumpyLinearProjector:
    """
    Project residue-level embeddings from input_dim to embedding_dim using weights/bias loaded from input_layer.pth.
    Uses numpy on CPU within creatlmdb to avoid introducing internal project dependencies.
    """

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        # weight: [embedding_dim, input_dim] 与 nn.Linear 权重一致
        self.W = weight.astype(np.float32)
        self.b = None if bias is None else bias.astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self.W.shape[0]

    @staticmethod
    def from_checkpoint(path: str, device: torch.device) -> "NumpyLinearProjector":
        ckpt = _safe_torch_load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # Support two key names: proj.weight or weight (if user saved raw layer)
        weight = None
        bias = None
        # Common prefixes
        candidate_w = [
            "proj.weight",
            "weight",
        ]
        candidate_b = [
            "proj.bias",
            "bias",
        ]
        for k in candidate_w:
            if k in state:
                weight = state[k].detach().cpu().numpy()
                break
        for k in candidate_b:
            if k in state:
                bias = state[k].detach().cpu().numpy()
                break
        if weight is None:
            raise RuntimeError(f"Linear layer weights not found in {path} (proj.weight/weight)")
        return NumpyLinearProjector(weight=weight, bias=bias)

    def project_cls(self, cls_vec: np.ndarray) -> np.ndarray:
        # cls_vec: [input_dim] -> [embedding_dim]
        y = self.W @ cls_vec.astype(np.float32)
        if self.b is not None:
            y = y + self.b
        return y

    def project_tokens(self, tokens: np.ndarray) -> np.ndarray:
        # tokens: [L, input_dim] -> [L, embedding_dim]
        # Project in one matmul: (L, C_in) @ (C_in, C_out) = (L, C_out)
        y = tokens.astype(np.float32) @ self.W.T
        if self.b is not None:
            y = y + self.b
        return y


def calculate_embedding_and_save(backend, model_or_client, tokenizer, device, cls_txn, middle_txn, avg_txn, seq_id, seq):
    """Compute embeddings and save them into three separate LMDBs."""
    cls_embedding, middle_embeddings = get_embeddings(backend, model_or_client, tokenizer, device, seq)
    if cls_embedding is None or middle_embeddings is None:
        return False
    cls_txn.put(seq_id.encode(), cls_embedding.tobytes())
    middle_txn.put(seq_id.encode(), middle_embeddings.tobytes())

    # Calculate average pooling of middle embeddings
    avg_embedding = np.mean(middle_embeddings, axis=0).astype(np.float32)
    avg_txn.put(seq_id.encode(), avg_embedding.tobytes())
    return True


def estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq):
    """Estimate remaining time."""
    avg_time = sum(time_per_seq) / len(time_per_seq) if time_per_seq else 0
    remaining_time = avg_time * (total_sequences - processed_count)
    elapsed_time = time.time() - start_time
    return elapsed_time, remaining_time


def process_and_save_sequences(backend, model_or_client, tokenizer, device, fasta_file_path, output_dir_path,
                               save_interval=10, commit_interval=100, force=False, append=False,
                               input_layer_ckpt: Optional[str] = None):
    """Iteratively process sequences, compute embeddings, and stream-save to three LMDB files"""
    sequences = read_fasta_sequences(fasta_file_path)
    total_sequences = len(sequences)
    processed_count = 0
    start_time = time.time()
    time_per_seq = []

    print(f"Batch commit interval: commit every {commit_interval} sequences")
    os.makedirs(output_dir_path, exist_ok=True)

    cls_lmdb_path = os.path.join(output_dir_path, 'onlyCLS.lmdb')
    middle_lmdb_path = os.path.join(output_dir_path, 'noCLSeos.lmdb')
    avg_lmdb_path = os.path.join(output_dir_path, 'avg.lmdb')

    # Handle existing outputs
    exists = os.path.exists(cls_lmdb_path) or os.path.exists(middle_lmdb_path) or os.path.exists(avg_lmdb_path)
    if exists and not append:
        if force:
            for p in (cls_lmdb_path, middle_lmdb_path, avg_lmdb_path):
                if os.path.exists(p):
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
            print("Old databases forcibly deleted.")
        else:
            # Interactive choice
            print(f"Database files already exist in {output_dir_path}.")
            while True:
                choice = input("Do you want to (1) delete and rebuild, or (2) append to the existing database? Enter 1 or 2: ")
                if choice == '1':
                    for p in (cls_lmdb_path, middle_lmdb_path, avg_lmdb_path):
                        if os.path.exists(p):
                            if os.path.isdir(p):
                                shutil.rmtree(p)
                            else:
                                os.remove(p)
                    print("Deleted existing databases; will recreate.")
                    break
                elif choice == '2':
                    print("Will append to existing databases.")
                    append = True
                    break
                else:
                    print("Invalid input, please enter 1 or 2.")

    # Check existing keys if in append mode
    existing_keys = set()
    if append:
        print("Checking existing keys in databases...")
        existing_cls_keys = set()
        existing_mid_keys = set()
        existing_avg_keys = set()

        if os.path.exists(cls_lmdb_path):
            try:
                with lmdb.open(cls_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_cls_keys.add(key.decode())
                print(f"  Found {len(existing_cls_keys)} keys in onlyCLS.lmdb")
            except Exception as e:
                print(f"  Warning: Failed to read onlyCLS.lmdb: {e}")

        if os.path.exists(middle_lmdb_path):
            try:
                with lmdb.open(middle_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_mid_keys.add(key.decode())
                print(f"  Found {len(existing_mid_keys)} keys in noCLSeos.lmdb")
            except Exception as e:
                print(f"  Warning: Failed to read noCLSeos.lmdb: {e}")

        if os.path.exists(avg_lmdb_path):
            try:
                with lmdb.open(avg_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_avg_keys.add(key.decode())
                print(f"  Found {len(existing_avg_keys)} keys in avg.lmdb")
            except Exception as e:
                print(f"  Warning: Failed to read avg.lmdb: {e}")

        # Use intersection: only keys that exist in ALL databases are considered complete
        existing_keys = existing_cls_keys & existing_mid_keys & existing_avg_keys
        print(f"  {len(existing_keys)} sequences are complete in all databases (intersection)")

        # Report inconsistencies
        only_cls = existing_cls_keys - existing_mid_keys - existing_avg_keys
        only_mid = existing_mid_keys - existing_cls_keys - existing_avg_keys
        only_avg = existing_avg_keys - existing_cls_keys - existing_mid_keys
        if only_cls:
            print(f"  Warning: {len(only_cls)} sequences only in CLS database (will be regenerated)")
        if only_mid:
            print(f"  Warning: {len(only_mid)} sequences only in middle database (will be regenerated)")
        if only_avg:
            print(f"  Warning: {len(only_avg)} sequences only in avg database (will be regenerated)")

        # Filter sequences to process
        sequences_to_process = {seq_id: seq for seq_id, seq in sequences.items() if seq_id not in existing_keys}
        skipped_count = len(sequences) - len(sequences_to_process)
        print(f"  Skipping {skipped_count} already-complete sequences")
        print(f"  Will process {len(sequences_to_process)} missing/incomplete sequences")
        sequences = sequences_to_process
        total_sequences = len(sequences)

        if total_sequences == 0:
            print("All sequences already exist in databases. Nothing to do.")
            return

    print(f"Total {total_sequences} sequences to process.")
    print(f"CLS token embeddings will be saved to: {cls_lmdb_path}")
    print(f"Middle tokens embeddings will be saved to: {middle_lmdb_path}")
    print(f"Average pooled embeddings will be saved to: {avg_lmdb_path}")

    projector: Optional[NumpyLinearProjector] = None
    if input_layer_ckpt:
        try:
            projector = NumpyLinearProjector.from_checkpoint(input_layer_ckpt, device=torch.device("cpu"))
            print(f"[input-layer] Loaded projection layer: output_dim={projector.output_dim}")
        except Exception as e:
            print(f"[input-layer] Failed to load; ignoring input-layer projection: {e}")

    try:
        with lmdb.open(cls_lmdb_path, map_size=int(1.5 * 10**12)) as cls_env, \
             lmdb.open(middle_lmdb_path, map_size=int(1.5 * 10**12)) as middle_env, \
             lmdb.open(avg_lmdb_path, map_size=int(1.5 * 10**12)) as avg_env:

            progress_bar = tqdm(sequences.items(), desc="Processing sequences", total=total_sequences)
            cls_batch_buffer = []
            middle_batch_buffer = []
            avg_batch_buffer = []

            for seq_id, seq in progress_bar:
                seq_start = time.time()
                cls_emb, middle_emb = get_embeddings(backend, model_or_client, tokenizer, device, seq)
                if cls_emb is None or middle_emb is None:
                    continue

                # Calculate average pooling of middle embeddings (before projection)
                avg_emb = np.mean(middle_emb, axis=0).astype(np.float32)

                # If input-layer checkpoint is provided, only project residue-level embeddings; keep CLS in original dimension
                if projector is not None:
                    try:
                        # Keep cls_emb original values (e.g., 5120 for esm2_15b)
                        middle_emb = projector.project_tokens(middle_emb)
                    except Exception as e:
                        print(f"[input-layer] Projection failed; skipping sequence {seq_id}: {e}")
                        continue

                cls_batch_buffer.append((seq_id.encode(), cls_emb.tobytes()))
                middle_batch_buffer.append((seq_id.encode(), middle_emb.tobytes()))
                avg_batch_buffer.append((seq_id.encode(), avg_emb.tobytes()))

                seq_time = time.time() - seq_start
                time_per_seq.append(seq_time)
                if len(time_per_seq) > 10:
                    time_per_seq.pop(0)
                processed_count += 1

                elapsed_time, remaining_time = estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq)
                progress_bar.set_postfix_str(
                    f"Elapsed: {elapsed_time:.2f}s, ETA: {remaining_time / 60:.2f} minutes, Buffer: {len(cls_batch_buffer)}"
                )

                if len(cls_batch_buffer) >= commit_interval or processed_count == total_sequences:
                    with cls_env.begin(write=True) as cls_txn:
                        for key, value in cls_batch_buffer:
                            cls_txn.put(key, value)
                    with middle_env.begin(write=True) as middle_txn:
                        for key, value in middle_batch_buffer:
                            middle_txn.put(key, value)
                    with avg_env.begin(write=True) as avg_txn:
                        for key, value in avg_batch_buffer:
                            avg_txn.put(key, value)
                    cls_batch_buffer.clear()
                    middle_batch_buffer.clear()
                    avg_batch_buffer.clear()
                    if processed_count % save_interval == 0:
                        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nManual interruption detected; saving processed data...")
        # 尝试写入剩余缓冲
        try:
            if cls_batch_buffer or middle_batch_buffer or avg_batch_buffer:
                with lmdb.open(cls_lmdb_path, map_size=int(1.5 * 10**12)) as cls_env, \
                     lmdb.open(middle_lmdb_path, map_size=int(1.5 * 10**12)) as middle_env, \
                     lmdb.open(avg_lmdb_path, map_size=int(1.5 * 10**12)) as avg_env:
                    with cls_env.begin(write=True) as cls_txn:
                        for key, value in cls_batch_buffer:
                            cls_txn.put(key, value)
                    with middle_env.begin(write=True) as middle_txn:
                        for key, value in middle_batch_buffer:
                            middle_txn.put(key, value)
                    with avg_env.begin(write=True) as avg_txn:
                        for key, value in avg_batch_buffer:
                            avg_txn.put(key, value)
                print(f"Saved remaining buffer: CLS {len(cls_batch_buffer)}, MID {len(middle_batch_buffer)}, AVG {len(avg_batch_buffer)}")
        except Exception as e:
            print(f"Error while saving remaining data: {e}")
        print("Program stopped.")
        exit(0)

    print("\nAll sequences processed; embeddings saved to:")
    print(f"  CLS token : {cls_lmdb_path}")
    print(f"  Middle tok: {middle_lmdb_path}")
    print(f"  Average   : {avg_lmdb_path}")


def process_and_save_from_source(source_lmdb_path, fasta_file_path, output_dir_path, input_dim,
                               save_interval=10, commit_interval=100, force=False, append=False,
                               input_layer_ckpt: Optional[str] = None):
    """
    Generate embeddings by projecting from an existing LMDB source.
    Avoids loading the heavy ESM model.
    """
    sequences = read_fasta_sequences(fasta_file_path)
    total_sequences = len(sequences)
    processed_count = 0
    start_time = time.time()

    print(f"Source LMDB Mode. Input dim: {input_dim}")
    print(f"Batch commit interval: commit every {commit_interval} sequences")
    os.makedirs(output_dir_path, exist_ok=True)

    cls_lmdb_path = os.path.join(output_dir_path, 'onlyCLS.lmdb')
    middle_lmdb_path = os.path.join(output_dir_path, 'noCLSeos.lmdb')
    avg_lmdb_path = os.path.join(output_dir_path, 'avg.lmdb')

    # Source paths
    src_cls_path = os.path.join(source_lmdb_path, 'onlyCLS.lmdb')
    src_mid_path = os.path.join(source_lmdb_path, 'noCLSeos.lmdb')
    
    if not os.path.exists(src_mid_path):
        raise FileNotFoundError(f"Source middle LMDB not found: {src_mid_path}")

    # Handle existing outputs
    exists = os.path.exists(cls_lmdb_path) or os.path.exists(middle_lmdb_path) or os.path.exists(avg_lmdb_path)
    if exists and not append:
        if force:
            for p in (cls_lmdb_path, middle_lmdb_path, avg_lmdb_path):
                if os.path.exists(p):
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
            print("Old databases forcibly deleted.")
        else:
            print(f"Database files already exist in {output_dir_path}.")
            while True:
                choice = input("Do you want to (1) delete and rebuild, or (2) append to the existing database? Enter 1 or 2: ")
                if choice == '1':
                    for p in (cls_lmdb_path, middle_lmdb_path, avg_lmdb_path):
                        if os.path.exists(p):
                            if os.path.isdir(p):
                                shutil.rmtree(p)
                            else:
                                os.remove(p)
                    print("Deleted existing databases; will recreate.")
                    break
                elif choice == '2':
                    print("Will append to existing databases.")
                    append = True
                    break
                else:
                    print("Invalid input, please enter 1 or 2.")

    # Check existing keys if in append mode
    existing_keys = set()
    if append:
        print("Checking existing keys in databases...")
        existing_cls_keys = set()
        existing_mid_keys = set()
        existing_avg_keys = set()

        if os.path.exists(cls_lmdb_path):
            try:
                with lmdb.open(cls_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_cls_keys.add(key.decode())
            except Exception: pass

        if os.path.exists(middle_lmdb_path):
            try:
                with lmdb.open(middle_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_mid_keys.add(key.decode())
            except Exception: pass
            
        if os.path.exists(avg_lmdb_path):
            try:
                with lmdb.open(avg_lmdb_path, readonly=True, lock=False) as env:
                    with env.begin() as txn:
                        cursor = txn.cursor()
                        for key, _ in cursor:
                            existing_avg_keys.add(key.decode())
            except Exception: pass

        existing_keys = existing_cls_keys & existing_mid_keys & existing_avg_keys
        print(f"  {len(existing_keys)} sequences are complete in all databases.")
        
        sequences_to_process = {seq_id: seq for seq_id, seq in sequences.items() if seq_id not in existing_keys}
        sequences = sequences_to_process
        total_sequences = len(sequences)
        if total_sequences == 0:
            print("All sequences already exist. Nothing to do.")
            return

    projector: Optional[NumpyLinearProjector] = None
    if input_layer_ckpt:
        try:
            projector = NumpyLinearProjector.from_checkpoint(input_layer_ckpt, device=torch.device("cpu"))
            print(f"[input-layer] Loaded projection layer: output_dim={projector.output_dim}")
        except Exception as e:
            print(f"[input-layer] Failed to load; ignoring input-layer projection: {e}")

    print("Opening source LMDBs...")
    src_mid_env = None
    src_cls_env = None
    try:
        src_mid_env = lmdb.open(src_mid_path, readonly=True, lock=False)
        if os.path.exists(src_cls_path):
            src_cls_env = lmdb.open(src_cls_path, readonly=True, lock=False)
        
        with src_mid_env.begin() as mid_txn:
            cls_txn = src_cls_env.begin() if src_cls_env else None
            try: 
                with lmdb.open(cls_lmdb_path, map_size=int(1.5 * 10**12)) as cls_env, \
                     lmdb.open(middle_lmdb_path, map_size=int(1.5 * 10**12)) as middle_env, \
                     lmdb.open(avg_lmdb_path, map_size=int(1.5 * 10**12)) as avg_env:

                    progress_bar = tqdm(sequences.items(), desc="Projecting sequences", total=total_sequences)
                    cls_batch = []
                    mid_batch = []
                    avg_batch = []

                    for seq_id, seq in progress_bar:
                        # Read Middle
                        mid_val = mid_txn.get(seq_id.encode())
                        if mid_val is None:
                            # print(f"Warning: {seq_id} not found in source LMDB")
                            continue
                            
                        # If source is float32, standard numpy works. If bf16 bytes, we need special handling.
                        # Assuming float32 source as per user description.
                        mid_arr = np.frombuffer(mid_val, dtype=np.float32)
                        
                        if mid_arr.size % input_dim != 0:
                            print(f"Dimension mismatch for {seq_id}: size {mid_arr.size} not divisible by {input_dim}")
                            continue
                        mid_arr = mid_arr.reshape(-1, input_dim)
                        
                        # Read CLS if available
                        cls_arr = None
                        if cls_txn:
                            cls_val = cls_txn.get(seq_id.encode())
                            if cls_val:
                                cls_arr = np.frombuffer(cls_val, dtype=np.float32)

                        # Compute Avg (before projection)
                        avg_arr = np.mean(mid_arr, axis=0).astype(np.float32)

                        # Project
                        if projector:
                            mid_arr = projector.project_tokens(mid_arr)
                            # Logic matches process_and_save_sequences: Keep CLS original
                        
                        mid_batch.append((seq_id.encode(), mid_arr.tobytes()))
                        avg_batch.append((seq_id.encode(), avg_arr.tobytes()))
                        if cls_arr is not None:
                            cls_batch.append((seq_id.encode(), cls_arr.tobytes()))
                            
                        processed_count += 1
                        
                        if len(mid_batch) >= commit_interval:
                            with middle_env.begin(write=True) as txn:
                                for k, v in mid_batch: txn.put(k, v)
                            with avg_env.begin(write=True) as txn:
                                for k, v in avg_batch: txn.put(k, v)
                            if cls_batch:
                                with cls_env.begin(write=True) as txn:
                                    for k, v in cls_batch: txn.put(k, v)
                            mid_batch = []; avg_batch = []; cls_batch = []
                            
                    # Flush final
                    if mid_batch:
                         with middle_env.begin(write=True) as txn:
                            for k, v in mid_batch: txn.put(k, v)
                         with avg_env.begin(write=True) as txn:
                            for k, v in avg_batch: txn.put(k, v)
                         if cls_batch:
                            with cls_env.begin(write=True) as txn:
                                for k, v in cls_batch: txn.put(k, v)
            finally:
                if cls_txn: cls_txn.abort() # Readonly, typically fine.
    
    except Exception as e:
        print(f"Error checking source or processing: {e}")
        raise
    finally:
        if src_mid_env: src_mid_env.close()
        if src_cls_env: src_cls_env.close()

    print("\nProcessing complete from source LMDB.")


def main():
    args = parse_args()
    
    # Validate input-layer-ckpt file existence if provided
    if args.input_layer_ckpt:
        if not os.path.isfile(args.input_layer_ckpt):
            print(f"Error: input-layer checkpoint file not found: {args.input_layer_ckpt}")
            print("Please check the file path and try again.")
            sys.exit(1)
        print(f"Using input-layer checkpoint: {args.input_layer_ckpt}")
    
    os.environ["INFRA_PROVIDER"] = "True"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 组合输出目录名: <base>/<model_short>.<suffix>.lmdb
    final_dir_name = f"{args.model}.{args.suffix}.lmdb"
    output_dir_path = os.path.join(args.output_dir, final_dir_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Final output directory: {output_dir_path}")

    if args.source_lmdb:
        # Check source exists
        if not os.path.isdir(args.source_lmdb):
             print(f"Error: Source LMDB directory not found: {args.source_lmdb}")
             sys.exit(1)
             
        input_dim = MODEL_DIM_MAP.get(args.model)
        if input_dim is None:
             raise ValueError(f"Unknown input dimension for model {args.model}, cannot reshape from source bytes.")
        
        process_and_save_from_source(
            source_lmdb_path=args.source_lmdb,
            fasta_file_path=args.fasta,
            output_dir_path=output_dir_path,
            input_dim=input_dim,
            commit_interval=args.commit_interval,
            force=args.force,
            append=args.append,
            input_layer_ckpt=args.input_layer_ckpt,
        )
    else:
        backend, internal_name = resolve_model(args.model)
        model_or_client, tokenizer, meta = load_model(backend, internal_name, args.model, device, args.precision)

        process_and_save_sequences(
            backend=backend,
            model_or_client=model_or_client,
            tokenizer=tokenizer,
            device=device,
            fasta_file_path=args.fasta,
            output_dir_path=output_dir_path,
            save_interval=args.save_interval,
            commit_interval=args.commit_interval,
            force=args.force,
            append=args.append,
            input_layer_ckpt=args.input_layer_ckpt,
        )

    # After generation, automatically bucketize noCLSeos.lmdb
    if not args.no_bucket:
        no_cls_lmdb = os.path.join(output_dir_path, "noCLSeos.lmdb")
        # If input-layer projection exists, override dim with projected dim
        override_dim = None
        if args.input_layer_ckpt:
            try:
                proj = NumpyLinearProjector.from_checkpoint(args.input_layer_ckpt, device=torch.device("cpu"))
                override_dim = int(proj.output_dim)
                print(f"[bucket] Using input-layer projection dimension override: {override_dim}")
            except Exception as e:
                print(f"[bucket] Failed to parse input-layer dimension; using model default mapping: {e}")
        run_bucketization(
            no_cls_lmdb_dir=no_cls_lmdb,
            fasta_path=args.fasta,
            model_short_name=args.model,
            override_dim=override_dim,
        )
    else:
        print("[bucket] Automatic bucketization disabled (--no-bucket)")


if __name__ == "__main__":
    main()
