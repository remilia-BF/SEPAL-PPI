#!/usr/bin/env python3
"""
Generate RSA-based sasa_features.json and secondary_structure_features.json from NetSurfP-3.0 output.

Key behavior:
- Input is FASTA (not PDB).
- Calls NetSurfP-3.0 standalone via conda environment (default env: nsp3).
- Keeps NetSurfP intermediate files under <output_dir>/tmp.
- Only writes final feature files to <output_dir> root:
  - sasa_features.json
        (field name stays "sasa", but values are RSA)
  - secondary_structure_features.json
"""

from __future__ import annotations

import argparse
import configparser
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


Q3_TO_ONEHOT: Dict[str, Tuple[int, int, int]] = {
    "H": (1, 0, 0),
    "E": (0, 1, 0),
    "C": (0, 0, 1),
}


def parse_fasta_ids(fasta_path: Path) -> List[str]:
    ids: List[str] = []
    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                token = line[1:].split()[0]
                if token:
                    ids.append(token)
    return ids


def parse_fasta_records(fasta_path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id = None
    current_seq: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                token = line[1:].split()[0]
                current_id = token if token else None
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    return records


def write_fasta_records(fasta_path: Path, records: List[Tuple[str, str]]) -> None:
    with fasta_path.open("w", encoding="utf-8") as f:
        for protein_id, seq in records:
            f.write(f">{protein_id}\n{seq}\n")


def sanitize_for_filename(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "protein"


def format_sasa_values(values: List[float]) -> str:
    return ",".join(f"{float(v):.6f}" for v in values)


def format_ss_onehot(q3_seq: str) -> str:
    rows: List[str] = []
    for ch in q3_seq:
        onehot = Q3_TO_ONEHOT.get(ch, (0, 0, 1))
        rows.append(f"[{onehot[0]},{onehot[1]},{onehot[2]}]")
    return ",".join(rows)


def load_env_config(config_path: Path) -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg.read(config_path, encoding="utf-8")
    if "nsp3" not in cfg:
        raise ValueError(f"Missing [nsp3] section in config: {config_path}")

    section = cfg["nsp3"]
    return {
        "conda_executable": section.get("conda_executable", "conda"),
        "conda_env": section.get("conda_env", "nsp3"),
        "python_executable": section.get("python_executable", "python"),
        "nsp3_script": section.get("nsp3_script", ""),
        "model_path": section.get("model_path", ""),
        "use_gpu": section.get("use_gpu", "true"),
        "worker_id": section.get("worker_id", "01"),
        "run_mode": section.get("run_mode", "batch"),
        "sequences_per_run": section.get("sequences_per_run", "1"),
    }


def as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def as_positive_int(raw: str, field_name: str) -> int:
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{field_name} must be an integer, got: {raw}") from e
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got: {value}")
    return value


def collect_per_sequence_jsons(run_dir: Path) -> List[Path]:
    # nsp3 stores each protein under: <run_dir>/<0000_xxx>/<0000_xxx>.json
    json_files = sorted(run_dir.glob("*/*.json"), key=lambda p: p.parent.name)
    return [p for p in json_files if p.name != f"{run_dir.name}.json"]


def _invoke_nsp3_once(
    fasta_path: Path,
    output_tmp_root: Path,
    env_cfg: Dict[str, str],
    worker_id: str,
    use_gpu: bool,
) -> Tuple[bool, Path, Path, Path, str]:
    nsp3_script = Path(env_cfg["nsp3_script"]).expanduser()
    model_path = Path(env_cfg["model_path"]).expanduser()

    cmd = [
        env_cfg["conda_executable"],
        "run",
        "-n",
        env_cfg["conda_env"],
        env_cfg["python_executable"],
        str(nsp3_script),
        "-m",
        str(model_path),
        "-i",
        str(fasta_path),
        "-o",
        str(output_tmp_root),
        "-w",
        worker_id,
    ]

    if use_gpu:
        cmd.extend(["-gpu", "True"])

    result = subprocess.run(cmd, capture_output=True, text=True)

    mode = "gpu" if use_gpu else "cpu"
    stdout_path = output_tmp_root / f"nsp3_stdout_{worker_id}_{mode}.log"
    stderr_path = output_tmp_root / f"nsp3_stderr_{worker_id}_{mode}.log"
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")

    run_dir = output_tmp_root / worker_id
    per_seq_jsons = collect_per_sequence_jsons(run_dir) if run_dir.exists() else []
    success = (result.returncode == 0) and bool(per_seq_jsons)

    details = (
        f"returncode={result.returncode}, run_dir={run_dir}, "
        f"json_count={len(per_seq_jsons)}, stdout={stdout_path}, stderr={stderr_path}"
    )
    return success, run_dir, stdout_path, stderr_path, details


def run_nsp3(
    fasta_path: Path,
    output_tmp_root: Path,
    env_cfg: Dict[str, str],
    worker_id_override: str | None,
) -> Path:
    nsp3_script = Path(env_cfg["nsp3_script"]).expanduser()
    model_path = Path(env_cfg["model_path"]).expanduser()

    if not nsp3_script.exists():
        raise FileNotFoundError(f"nsp3_script not found: {nsp3_script}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    output_tmp_root.mkdir(parents=True, exist_ok=True)

    worker_id = worker_id_override or env_cfg["worker_id"]
    if not worker_id:
        worker_id = time.strftime("%m%d_%H%M%S")

    # First attempt follows config (usually GPU), then falls back to CPU once.
    prefer_gpu = as_bool(env_cfg["use_gpu"])
    ok, run_dir, stdout_path, stderr_path, details = _invoke_nsp3_once(
        fasta_path=fasta_path,
        output_tmp_root=output_tmp_root,
        env_cfg=env_cfg,
        worker_id=worker_id,
        use_gpu=prefer_gpu,
    )
    if ok:
        return run_dir

    if prefer_gpu:
        fallback_worker = f"{worker_id}_cpu"
        print(
            "WARN: NetSurfP GPU run failed or produced empty outputs; retrying on CPU...",
            file=sys.stderr,
        )
        ok2, run_dir2, stdout_path2, stderr_path2, details2 = _invoke_nsp3_once(
            fasta_path=fasta_path,
            output_tmp_root=output_tmp_root,
            env_cfg=env_cfg,
            worker_id=fallback_worker,
            use_gpu=False,
        )
        if ok2:
            return run_dir2

        raise RuntimeError(
            "NetSurfP failed on both GPU and CPU. "
            f"GPU: {details}; CPU: {details2}; "
            f"GPU logs: {stdout_path}, {stderr_path}; "
            f"CPU logs: {stdout_path2}, {stderr_path2}"
        )

    raise RuntimeError(
        "NetSurfP run failed. "
        f"{details}; logs: {stdout_path}, {stderr_path}"
    )


def convert_nsp3_to_feature_json(
    fasta_ids: List[str],
    per_seq_json_files: List[Path],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sasa_items: List[Dict[str, str]] = []
    ss_items: List[Dict[str, str]] = []

    for idx, pred_json in enumerate(per_seq_json_files):
        with pred_json.open("r", encoding="utf-8") as f:
            pred = json.load(f)

        rsa_values_raw = pred.get("rsa", [])
        q3_seq = pred.get("q3", "")

        rsa_values = [float(v) for v in rsa_values_raw]
        usable_len = min(len(rsa_values), len(q3_seq))
        if usable_len == 0:
            continue

        rsa_values = rsa_values[:usable_len]
        q3_seq = q3_seq[:usable_len]

        if idx < len(fasta_ids):
            protein_id = fasta_ids[idx]
        else:
            protein_id = pred.get("desc", f"seq_{idx + 1}")

        sasa_items.append(
            {
                "protein_id": protein_id,
                "sasa": format_sasa_values(rsa_values),
            }
        )
        ss_items.append(
            {
                "protein_id": protein_id,
                "secondary_structure": format_ss_onehot(q3_seq),
            }
        )

    return sasa_items, ss_items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RSA + SS onehot JSON via NetSurfP-3.0 standalone"
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--env-config",
        default="mutifeature_tools/nsp3_env_config.ini",
        help="Path to NetSurfP environment config file",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Override worker id in nsp3 outputs (default: from config)",
    )
    parser.add_argument(
        "--run-mode",
        choices=["batch", "per_sequence"],
        default=None,
        help="batch: one nsp3 process for all sequences; per_sequence: one nsp3 process per sequence (better VRAM stability)",
    )
    parser.add_argument(
        "--sequences-per-run",
        type=int,
        default=None,
        help="Only for per_sequence mode: number of FASTA records per nsp3 subprocess (default from env config, usually 1)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_config(Path(args.env_config))
    fasta_ids = parse_fasta_ids(fasta_path)
    if not fasta_ids:
        print(f"ERROR: no sequence id found in FASTA: {fasta_path}", file=sys.stderr)
        return 1

    run_mode = args.run_mode if args.run_mode is not None else env_cfg.get("run_mode", "batch")
    if run_mode not in {"batch", "per_sequence"}:
        print(f"ERROR: invalid run mode: {run_mode}", file=sys.stderr)
        return 1

    if args.sequences_per_run is not None:
        sequences_per_run = args.sequences_per_run
    else:
        sequences_per_run = as_positive_int(env_cfg.get("sequences_per_run", "1"), "sequences_per_run")

    if sequences_per_run <= 0:
        print(f"ERROR: sequences_per_run must be > 0, got {sequences_per_run}", file=sys.stderr)
        return 1

    tmp_dir = output_dir / "tmp"
    sasa_items: List[Dict[str, str]] = []
    ss_items: List[Dict[str, str]] = []

    if run_mode == "batch":
        run_dir = run_nsp3(
            fasta_path=fasta_path,
            output_tmp_root=tmp_dir,
            env_cfg=env_cfg,
            worker_id_override=args.worker_id,
        )

        per_seq_json_files = collect_per_sequence_jsons(run_dir)
        if not per_seq_json_files:
            print(f"ERROR: no per-sequence json found under: {run_dir}", file=sys.stderr)
            return 1

        sasa_items, ss_items = convert_nsp3_to_feature_json(fasta_ids, per_seq_json_files)
    else:
        records = parse_fasta_records(fasta_path)
        if not records:
            print(f"ERROR: no valid FASTA records in: {fasta_path}", file=sys.stderr)
            return 1

        split_fasta_dir = tmp_dir / "split_fastas"
        split_fasta_dir.mkdir(parents=True, exist_ok=True)

        base_worker = args.worker_id or env_cfg.get("worker_id", "01")

        total = len(records)
        total_chunks = (total + sequences_per_run - 1) // sequences_per_run

        for chunk_idx in range(total_chunks):
            start = chunk_idx * sequences_per_run
            end = min(start + sequences_per_run, total)
            chunk_records = records[start:end]
            chunk_ids = [pid for pid, _ in chunk_records]

            first_id_safe = sanitize_for_filename(chunk_ids[0])
            last_id_safe = sanitize_for_filename(chunk_ids[-1])
            single_fasta = split_fasta_dir / (
                f"{start + 1:06d}_{end:06d}_{first_id_safe}_{last_id_safe}.fasta"
            )
            write_fasta_records(single_fasta, chunk_records)

            worker_id = f"{base_worker}_{chunk_idx + 1:06d}"
            print(
                f"[chunk {chunk_idx + 1}/{total_chunks}] running nsp3 for "
                f"records {start + 1}-{end} (size={len(chunk_records)}) ..."
            )

            run_dir = run_nsp3(
                fasta_path=single_fasta,
                output_tmp_root=tmp_dir,
                env_cfg=env_cfg,
                worker_id_override=worker_id,
            )

            per_seq_json_files = collect_per_sequence_jsons(run_dir)
            if not per_seq_json_files:
                print(
                    f"WARN: no output json for chunk {chunk_idx + 1} under {run_dir}",
                    file=sys.stderr,
                )
                continue

            chunk_sasa, chunk_ss = convert_nsp3_to_feature_json(chunk_ids, per_seq_json_files)
            sasa_items.extend(chunk_sasa)
            ss_items.extend(chunk_ss)

            if len(chunk_sasa) != len(chunk_ids):
                print(
                    f"WARN: chunk {chunk_idx + 1} produced {len(chunk_sasa)}/{len(chunk_ids)} entries",
                    file=sys.stderr,
                )

            # Ensure local temporary Python objects are collected between chunk runs.
            del chunk_records, chunk_ids, chunk_sasa, chunk_ss, per_seq_json_files
            gc.collect()

    sasa_path = output_dir / "sasa_features.json"
    ss_path = output_dir / "secondary_structure_features.json"

    with sasa_path.open("w", encoding="utf-8") as f:
        json.dump(sasa_items, f, ensure_ascii=False, indent=2)

    with ss_path.open("w", encoding="utf-8") as f:
        json.dump(ss_items, f, ensure_ascii=False, indent=2)

    print(f"Saved: {sasa_path}")
    print(f"Saved: {ss_path}")
    print(f"NetSurfP intermediates: {tmp_dir}")
    print(f"Run mode: {run_mode}")
    if run_mode == "per_sequence":
        print(f"Sequences per run: {sequences_per_run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
