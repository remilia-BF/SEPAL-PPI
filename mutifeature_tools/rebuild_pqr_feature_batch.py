#!/usr/bin/env python3
"""
批量重建 pqr_feature.json

用途:
  - 读取各数据集目录下已存在的 PQR/*.pqr
  - 重新解析每个残基电荷并生成 pqr_feature.json
  - 修复由于 PQR 坐标字段粘连导致的长度不匹配问题

默认处理范围:
  1) ./mutifeature/*
  2) ./predict/Human/mutifeature
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("rebuild_pqr_feature_batch")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger


@dataclass
class DatasetResult:
    dataset_dir: Path
    processed: bool
    skipped_reason: Optional[str]
    pqr_files: int
    proteins_targeted: int
    proteins_written: int
    proteins_failed: int
    output_file: Optional[Path]


MERGED_COORD_PATTERN = re.compile(r"(\d)(-)")
NUM_PREFIX_PATTERN = re.compile(r"^-?\d+")


def parse_pqr_file_robust(
    pqr_file: Path, logger: logging.Logger
) -> Dict[str, float]:
    """
    解析 PQR，返回 残基ID -> 总电荷。

    兼容两类行格式:
      - 无 chain: ATOM serial name resname resseq x y z charge radius
      - 有 chain: ATOM serial name resname chain resseq x y z charge radius

    并修复坐标粘连，例如 "-22.464-102.231"。
    """
    residue_charges: Dict[str, float] = defaultdict(float)

    try:
        with open(pqr_file, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                if not (raw_line.startswith("ATOM") or raw_line.startswith("HETATM")):
                    continue

                line = MERGED_COORD_PATTERN.sub(r"\1 \2", raw_line)
                parts = line.split()

                try:
                    if len(parts) >= 11:
                        residue_name = parts[3]
                        chain_id = parts[4]
                        residue_number = parts[5]
                        charge = float(parts[9])
                        residue_id = f"{residue_name}_{chain_id}_{residue_number}"
                    elif len(parts) >= 10:
                        residue_name = parts[3]
                        residue_number = parts[4]
                        charge = float(parts[8])
                        residue_id = f"{residue_name}_{residue_number}"
                    else:
                        continue

                    residue_charges[residue_id] += charge
                except (ValueError, IndexError):
                    logger.debug("跳过无法解析行: %s", raw_line.rstrip("\n"))
                    continue
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("解析失败 %s: %s", pqr_file, exc)
        return {}

    return {residue_id: round(charge, 4) for residue_id, charge in residue_charges.items()}


def residue_sort_key(residue_id: str) -> Tuple[str, int, str]:
    """为残基ID提供稳定排序键，兼容插入码/链ID。"""
    parts = residue_id.split("_")

    if len(parts) >= 3:
        chain = parts[-2]
        resnum = parts[-1]
    else:
        chain = ""
        resnum = parts[-1]

    prefix_match = NUM_PREFIX_PATTERN.match(resnum)
    if prefix_match:
        numeric = int(prefix_match.group(0))
        suffix = resnum[prefix_match.end() :]
    else:
        numeric = 10**9
        suffix = resnum

    return chain, numeric, suffix


def charge_dict_to_ordered_list(residue_charges: Dict[str, float]) -> List[float]:
    sorted_items = sorted(residue_charges.items(), key=lambda kv: residue_sort_key(kv[0]))
    return [charge for _, charge in sorted_items]


def save_pqr_feature_json(
    protein_charges: Dict[str, List[float]], output_path: Path
) -> None:
    data = []
    for protein_id in sorted(protein_charges.keys()):
        charge_str = ",".join(f"{charge:.4f}" for charge in protein_charges[protein_id])
        data.append({"protein_id": protein_id, "embeddings": charge_str})

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_existing_pqr_feature_json(json_path: Path, logger: logging.Logger) -> Dict[str, List[float]]:
    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("读取现有 pqr_feature.json 失败，将重建为空基底: %s (%s)", json_path, exc)
        return {}

    protein_charges: Dict[str, List[float]] = {}
    if not isinstance(data, list):
        logger.warning("现有 pqr_feature.json 不是列表格式，将重建为空基底: %s", json_path)
        return {}

    for item in data:
        if not isinstance(item, dict):
            continue
        protein_id = item.get("protein_id")
        embeddings = item.get("embeddings")
        if not protein_id or not isinstance(embeddings, str):
            continue
        try:
            values = [float(x) for x in embeddings.split(",") if x.strip()]
            protein_charges[protein_id] = values
        except ValueError:
            continue
    return protein_charges


def load_failed_pqr_proteins(dataset_dir: Path, logger: logging.Logger) -> List[str]:
    """从 failed_proteins.txt 提取 pqr_feature.json 报错的 protein_id 列表。"""
    failed_file = dataset_dir / "failed_proteins.txt"
    if not failed_file.exists():
        return []

    proteins: List[str] = []
    current_file = None
    current_protein = None

    try:
        with open(failed_file, "r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("file:"):
                    current_file = line.split(":", 1)[1].strip()
                elif line.startswith("protein_id:"):
                    current_protein = line.split(":", 1)[1].strip()
                elif line.startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
                    if current_file == "pqr_feature.json" and current_protein:
                        if reason:
                            proteins.append(current_protein)
                    current_file = None
                    current_protein = None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("读取 failed_proteins.txt 失败: %s (%s)", failed_file, exc)
        return []

    seen = set()
    deduped = []
    for p in proteins:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def process_dataset(
    dataset_dir: Path,
    logger: logging.Logger,
    workers: int,
    failed_only: bool,
) -> DatasetResult:
    pqr_dir = dataset_dir / "PQR"
    if not dataset_dir.exists():
        return DatasetResult(dataset_dir, False, "dataset_dir_not_found", 0, 0, 0, 0, None)
    if not pqr_dir.exists() or not pqr_dir.is_dir():
        return DatasetResult(dataset_dir, False, "pqr_dir_missing", 0, 0, 0, 0, None)

    all_pqr_files = sorted(pqr_dir.glob("*.pqr"))
    if not all_pqr_files:
        return DatasetResult(dataset_dir, False, "no_pqr_files", 0, 0, 0, 0, None)

    pqr_map = {p.stem: p for p in all_pqr_files}

    target_proteins: List[str]
    if failed_only:
        target_proteins = load_failed_pqr_proteins(dataset_dir, logger)
        if not target_proteins:
            return DatasetResult(dataset_dir, False, "no_failed_pqr_proteins", 0, 0, 0, 0, None)
    else:
        target_proteins = sorted(pqr_map.keys())

    selected_pqr_files: List[Path] = []
    missing_targets = 0
    for protein_id in target_proteins:
        pqr_file = pqr_map.get(protein_id)
        if pqr_file is None:
            missing_targets += 1
            logger.warning("缺少对应PQR文件，跳过蛋白: %s (%s)", protein_id, dataset_dir)
            continue
        selected_pqr_files.append(pqr_file)

    if not selected_pqr_files:
        reason = "failed_targets_missing_pqr" if failed_only else "no_selected_pqr_files"
        return DatasetResult(dataset_dir, False, reason, 0, len(target_proteins), 0, missing_targets, None)

    output_file = dataset_dir / "pqr_feature.json"
    if failed_only:
        protein_charges = load_existing_pqr_feature_json(output_file, logger)
    else:
        protein_charges = {}

    failures = missing_targets

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_pqr_file_robust, p, logger): p for p in selected_pqr_files}
        for future in as_completed(futures):
            pqr_file = futures[future]
            protein_id = pqr_file.stem
            try:
                residue_charges = future.result()
                if not residue_charges:
                    failures += 1
                    continue
                protein_charges[protein_id] = charge_dict_to_ordered_list(residue_charges)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("处理失败 %s: %s", pqr_file, exc)
                failures += 1

    save_pqr_feature_json(protein_charges, output_file)

    return DatasetResult(
        dataset_dir=dataset_dir,
        processed=True,
        skipped_reason=None,
            pqr_files=len(selected_pqr_files),
            proteins_targeted=len(target_proteins),
        proteins_written=len(protein_charges),
        proteins_failed=failures,
        output_file=output_file,
    )


def discover_default_targets(repo_root: Path) -> List[Path]:
    targets: List[Path] = []

    mutifeature_root = repo_root / "mutifeature"
    if mutifeature_root.exists() and mutifeature_root.is_dir():
        for child in sorted(mutifeature_root.iterdir()):
            if child.is_dir():
                targets.append(child)

    targets.append(repo_root / "predict" / "Human" / "mutifeature")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量重建 pqr_feature.json（基于已生成 PQR 文件）"
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="仓库根目录（默认自动推断）",
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="*",
        default=None,
        help="可选：显式指定要处理的数据集目录（绝对或相对 repo_root）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行解析 PQR 的线程数（默认 8）",
    )
    parser.add_argument(
        "--failed_only",
        action="store_true",
        help="仅处理 failed_proteins.txt 中 file=pqr_feature.json 的报错蛋白",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)

    repo_root = Path(args.repo_root).resolve()
    if args.dataset_dirs:
        targets = []
        for raw in args.dataset_dirs:
            p = Path(raw)
            targets.append(p if p.is_absolute() else (repo_root / p))
    else:
        targets = discover_default_targets(repo_root)

    logger.info("待处理目录数: %d", len(targets))
    logger.info("处理模式: %s", "failed_only" if args.failed_only else "all_pqr")

    results: List[DatasetResult] = []
    for target in targets:
        logger.info("处理目录: %s", target)
        result = process_dataset(target, logger, args.workers, args.failed_only)
        results.append(result)

        if result.processed:
            logger.info(
                "完成: %s | 目标=%d 解析=%d 写入总数=%d 失败=%d 输出=%s",
                target.name,
                result.proteins_targeted,
                result.pqr_files,
                result.proteins_written,
                result.proteins_failed,
                result.output_file,
            )
        else:
            logger.warning("跳过: %s | 原因=%s", target, result.skipped_reason)

    processed = [r for r in results if r.processed]
    skipped = [r for r in results if not r.processed]

    logger.info("=" * 70)
    logger.info("批处理结束")
    logger.info("成功处理目录: %d", len(processed))
    logger.info("跳过目录: %d", len(skipped))
    logger.info("累计目标蛋白数: %d", sum(r.proteins_targeted for r in processed))
    logger.info("累计写入蛋白数: %d", sum(r.proteins_written for r in processed))
    logger.info("累计失败蛋白数: %d", sum(r.proteins_failed for r in processed))

    if skipped:
        logger.info("跳过详情:")
        for item in skipped:
            logger.info("  - %s (%s)", item.dataset_dir, item.skipped_reason)


if __name__ == "__main__":
    main()
