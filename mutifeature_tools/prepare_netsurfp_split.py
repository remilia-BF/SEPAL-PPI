#!/usr/bin/env python3
"""
Prepare NetSurfP head split with DIAMOND non-redundancy filtering.

Rules:
1) Merge proteins from multiple mutifeature datasets.
2) Remove redundancy by DIAMOND all-vs-all, keeping one representative per
   connected component where sequence identity >= threshold (default 40%).
3) Test split:
   - All SOYBN representatives.
   - Plus 200 proteins from each other source (length-diverse sampling).
4) Validation split:
   - From remaining train candidates, sample 100 proteins per source
     (length-diverse sampling).
5) Save train/val/test ids under mutifeature/infer_netsurfp_head.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


LOGGER = logging.getLogger("prepare_netsurfp_split")


@dataclass
class ProteinRecord:
    source: str
    protein_id: str
    sequence: str

    @property
    def key(self) -> str:
        return f"{self.source}|{self.protein_id}"

    @property
    def length(self) -> int:
        return len(self.sequence)


def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    current_id = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                seqs[current_id] = []
            elif current_id is not None:
                seqs[current_id].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def parse_label_ids(path: Path, key: str = "protein_id") -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Set[str] = set()
    for item in data:
        pid = item.get(key)
        if pid:
            out.add(str(pid))
    return out


def collect_records(dataset_dirs: List[Path]) -> Dict[str, ProteinRecord]:
    records: Dict[str, ProteinRecord] = {}
    for ds in dataset_dirs:
        source = ds.name
        fasta_candidates = [
            ds / "pdb_generated.fasta",
            Path("dataset") / source / "protein.fasta",
        ]
        fasta_path = None
        for cand in fasta_candidates:
            if cand.exists():
                fasta_path = cand
                break

        if fasta_path is None:
            LOGGER.warning("Skip %s: missing fasta in candidates %s", ds, fasta_candidates)
            continue

        sasa_path = ds / "sasa_features.json"
        ss_path = ds / "secondary_structure_features.json"
        if not (sasa_path.exists() and ss_path.exists()):
            LOGGER.warning("Skip %s: missing sasa/ss json labels", ds)
            continue

        seqs = parse_fasta(fasta_path)
        label_ids = parse_label_ids(sasa_path) & parse_label_ids(ss_path)
        if len(label_ids) == 0:
            LOGGER.warning("Skip %s: no overlapped label ids", ds)
            continue

        for pid, seq in seqs.items():
            if pid not in label_ids:
                continue
            rec = ProteinRecord(source=source, protein_id=pid, sequence=seq)
            records[rec.key] = rec
    LOGGER.warning("Collected proteins: %d", len(records))
    return records


def write_merged_fasta(records: Dict[str, ProteinRecord], fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            rec = records[key]
            f.write(f">{rec.key}\n{rec.sequence}\n")


def run_cmd(cmd: List[str]) -> None:
    LOGGER.info("Run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_diamond_all_vs_all(merged_fasta: Path, work_dir: Path, threshold: float, threads: int) -> Path:
    db_path = work_dir / "merged_db"
    out_path = work_dir / "diamond_all_vs_all.tsv"

    run_cmd([
        "diamond",
        "makedb",
        "--in",
        str(merged_fasta),
        "-d",
        str(db_path),
    ])

    run_cmd([
        "diamond",
        "blastp",
        "--query",
        str(merged_fasta),
        "--db",
        str(db_path),
        "--out",
        str(out_path),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "bitscore",
        "--id",
        str(threshold),
        "--max-target-seqs",
        "0",
        "--threads",
        str(threads),
        "--sensitive",
    ])

    return out_path


def build_similarity_graph(keys: List[str], diamond_tsv: Path, threshold: float) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {k: set() for k in keys}
    with diamond_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            q, s, pident = parts[0], parts[1], float(parts[2])
            if q == s:
                continue
            if pident < threshold:
                continue
            if q not in graph or s not in graph:
                continue
            graph[q].add(s)
            graph[s].add(q)
    return graph


def connected_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    comps: List[List[str]] = []
    for node in graph:
        if node in seen:
            continue
        stack = [node]
        comp: List[str] = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def select_nonredundant(records: Dict[str, ProteinRecord], comps: List[List[str]]) -> Dict[str, ProteinRecord]:
    chosen: Dict[str, ProteinRecord] = {}
    for comp in comps:
        rep = max(comp, key=lambda k: records[k].length)
        chosen[rep] = records[rep]
    LOGGER.warning("Non-redundant proteins kept: %d", len(chosen))
    return chosen


def sample_diverse_by_length(keys: List[str], records: Dict[str, ProteinRecord], n: int, seed: int) -> List[str]:
    if len(keys) <= n:
        return sorted(keys, key=lambda k: records[k].length)

    rng = random.Random(seed)
    sorted_keys = sorted(keys, key=lambda k: records[k].length)

    picked: List[str] = []
    used: Set[str] = set()

    for i in range(n):
        pos = int((i + 0.5) * len(sorted_keys) / n)
        pos = min(max(pos, 0), len(sorted_keys) - 1)
        cand = sorted_keys[pos]
        if cand not in used:
            picked.append(cand)
            used.add(cand)

    if len(picked) < n:
        rest = [k for k in sorted_keys if k not in used]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    return picked[:n]


def build_split(
    nr_records: Dict[str, ProteinRecord],
    seed: int,
    per_source_test: int,
    per_source_val: int,
) -> Tuple[List[str], List[str], List[str]]:
    by_source: Dict[str, List[str]] = {}
    for key, rec in nr_records.items():
        by_source.setdefault(rec.source, []).append(key)

    test_ids: List[str] = []
    val_ids: List[str] = []

    soybn_ids = by_source.get("SOYBN", [])
    test_ids.extend(sorted(soybn_ids))

    other_sources = [s for s in sorted(by_source.keys()) if s != "SOYBN"]
    for idx, source in enumerate(other_sources):
        source_ids = by_source[source]
        sampled = sample_diverse_by_length(source_ids, nr_records, per_source_test, seed + 11 * (idx + 1))
        test_ids.extend(sampled)

    test_set = set(test_ids)

    train_candidates_by_source: Dict[str, List[str]] = {}
    for source in other_sources:
        source_ids = by_source[source]
        remaining = [k for k in source_ids if k not in test_set]
        train_candidates_by_source[source] = remaining

    for idx, source in enumerate(other_sources):
        remaining = train_candidates_by_source[source]
        sampled = sample_diverse_by_length(remaining, nr_records, per_source_val, seed + 101 * (idx + 1))
        val_ids.extend(sampled)

    val_set = set(val_ids)
    train_ids: List[str] = []
    for source in other_sources:
        remaining = train_candidates_by_source[source]
        train_ids.extend([k for k in remaining if k not in val_set])

    train_ids = sorted(set(train_ids))
    val_ids = sorted(set(val_ids))
    test_ids = sorted(set(test_ids))

    return train_ids, val_ids, test_ids


def save_split(
    out_dir: Path,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    nr_records: Dict[str, ProteinRecord],
    threshold: float,
    per_source_test: int,
    per_source_val: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (out_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    stats = {
        "nr_threshold_percent": threshold,
        "per_source_test": per_source_test,
        "per_source_val": per_source_val,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "by_source": {},
    }

    for split_name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        source_counter: Dict[str, int] = {}
        for key in ids:
            source = nr_records[key].source
            source_counter[source] = source_counter.get(source, 0) + 1
        stats["by_source"][split_name] = source_counter

    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NetSurfP split with DIAMOND de-redundancy")
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=[
            "mutifeature/ARATH",
            "mutifeature/MAIZE",
            "mutifeature/ORYSJ",
            "mutifeature/SOYBN",
            "mutifeature/Strings_plant50",
        ],
    )
    parser.add_argument("--out-dir", type=str, default="mutifeature/infer_netsurfp_head")
    parser.add_argument("--work-dir", type=str, default="mutifeature/infer_netsurfp_head/work")
    parser.add_argument("--identity-threshold", type=float, default=40.0)
    parser.add_argument("--test-per-source", type=int, default=200)
    parser.add_argument("--val-per-source", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    dataset_dirs = [Path(p) for p in args.dataset_dirs]
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(dataset_dirs)
    if len(records) == 0:
        raise RuntimeError("No proteins collected from dataset dirs.")

    merged_fasta = work_dir / "merged_all.fasta"
    write_merged_fasta(records, merged_fasta)

    try:
        diamond_tsv = run_diamond_all_vs_all(merged_fasta, work_dir, args.identity_threshold, args.threads)
    except FileNotFoundError as e:
        raise RuntimeError("DIAMOND not found in PATH. Please install DIAMOND first.") from e

    graph = build_similarity_graph(list(records.keys()), diamond_tsv, args.identity_threshold)
    comps = connected_components(graph)
    nr_records = select_nonredundant(records, comps)

    train_ids, val_ids, test_ids = build_split(
        nr_records,
        seed=args.seed,
        per_source_test=args.test_per_source,
        per_source_val=args.val_per_source,
    )

    save_split(
        out_dir,
        train_ids,
        val_ids,
        test_ids,
        nr_records,
        threshold=args.identity_threshold,
        per_source_test=args.test_per_source,
        per_source_val=args.val_per_source,
    )

    LOGGER.warning("Split saved to %s", out_dir)
    LOGGER.warning("train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids))


if __name__ == "__main__":
    main()
