#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总消融实验配置与各数据集 AUPR 到单个 CSV。

默认扫描：results/Strings_plant50_bf16/ab-nosst-

输出列：
- experiment_name: 顶层实验目录名
- run_dir: 相对 root 的实际运行目录（兼容多一层嵌套的情况）
- config_short: 缩写配置名，如 ss+sasa+hmm
- enabled_features: 逗号分隔的具体启用特征
- residue_c2_aupr / residue_c3_aupr
- ensemble_c2_aupr / ensemble_c3_aupr
- hh70_aupr / hh50_aupr / hl_aupr / ll_aupr

示例：
python -m src.tools.collect_ablation_aupr \
  --root results/Strings_plant50_bf16/ab-nosst- \
  --out results/Strings_plant50_bf16/ab-nosst-/ablation_aupr_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

LOGGER = logging.getLogger("collect_ablation_aupr")

DEFAULT_ROOT = Path("results/Strings_plant50_bf16/ab-nosst-")
DEFAULT_OUT = DEFAULT_ROOT / "ablation_aupr_summary.csv"
PREDICTION_DATASETS: Sequence[str] = ("hh70", "hh50", "hl", "ll")

SHORT_TOKEN_ORDER: Sequence[str] = (
    "ss",
    "sst",
    "sasa",
    "hmm",
    "pqr",
    "hydro",
    "pfam",
    "len",
)


@dataclass
class ExperimentRun:
    experiment_name: str
    run_dir: Path
    projector_dir: Path
    config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 ab-nosst 消融实验 AUPR")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="消融实验根目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 CSV 路径")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def find_experiment_runs(root: Path) -> List[ExperimentRun]:
    runs: List[ExperimentRun] = []
    for config_path in sorted(root.rglob("resolved_config.yaml")):
        if config_path.parent.name != "sepal-ppi-projector-feature-contant":
            continue
        projector_dir = config_path.parent
        run_dir = projector_dir.parent
        try:
            relative_parts = run_dir.relative_to(root).parts
        except ValueError:
            continue
        if not relative_parts:
            continue
        runs.append(
            ExperimentRun(
                experiment_name=relative_parts[0],
                run_dir=run_dir,
                projector_dir=projector_dir,
                config_path=config_path,
            )
        )
    return runs


def _dig(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def normalize_feature_name(feature_key: str, feature_cfg: Dict[str, Any]) -> str:
    raw_key = (feature_key or "").strip().lower()
    file_path = str(feature_cfg.get("file_path") or "").strip().lower()
    file_stem = Path(file_path).stem.lower() if file_path else ""
    token_source = f"{raw_key} {file_stem}"

    if "secondary_structure_features" in token_source:
        return "ss"
    if "s1ssttoken" in token_source or "prosst_features" in token_source:
        return "sst"
    if "sasa_features" in token_source:
        return "sasa"
    if "hydrophobicity_features" in token_source:
        return "hydrophobicity"
    if "pqr_feature" in token_source or "pqr_features" in token_source:
        return "pqr"
    if "hmm_binding_domain_flag" in token_source:
        return "hmm_binding_domain"
    if "hmm_boundary_site_flag" in token_source:
        return "hmm_boundary_site"
    if "hmm_confidence_score" in token_source:
        return "hmm_confidence"
    if "hmm_core_site_flag" in token_source:
        return "hmm_core_site"
    if "hmm_domain_flag" in token_source:
        return "hmm_domain"
    if "hmm_emission_prob" in token_source:
        return "hmm_emission_prob"
    if "hmm_pfam_id" in token_source:
        return "hmm_pfam"
    if raw_key == "protein_length" or file_stem == "protein_length":
        return "protein_length"
    return raw_key or file_stem or feature_key


def collect_enabled_features(config_data: Dict[str, Any], config_path: Path) -> List[str]:
    primary = _dig(config_data, "model", "model_config", "preprocessing", "feature_files")
    secondary = _dig(config_data, "model", "model_unit", "preprocessing", "feature_files")

    primary = primary if isinstance(primary, dict) else {}
    secondary = secondary if isinstance(secondary, dict) else {}

    def _enabled(feature_dict: Dict[str, Any]) -> List[str]:
        normalized: List[str] = []
        for feature_key, feature_cfg in feature_dict.items():
            if not isinstance(feature_cfg, dict):
                continue
            if feature_cfg.get("enabled") is True:
                normalized.append(normalize_feature_name(feature_key, feature_cfg))
        return sorted(set(normalized))

    primary_enabled = _enabled(primary)
    secondary_enabled = _enabled(secondary)

    if primary_enabled and secondary_enabled and primary_enabled != secondary_enabled:
        LOGGER.warning(
            "feature_files 两处配置不一致，已取并集: %s (%s vs %s)",
            config_path,
            primary_enabled,
            secondary_enabled,
        )
        return sorted(set(primary_enabled) | set(secondary_enabled))
    if primary_enabled:
        return primary_enabled
    return secondary_enabled


def build_short_config(enabled_features: Iterable[str]) -> str:
    feature_set = set(enabled_features)
    tokens: List[str] = []

    if "ss" in feature_set:
        tokens.append("ss")
    if "sst" in feature_set:
        tokens.append("sst")
    if "sasa" in feature_set:
        tokens.append("sasa")
    if any(feature.startswith("hmm_") for feature in feature_set):
        tokens.append("hmm")
    if "pqr" in feature_set:
        tokens.append("pqr")
    if "hydrophobicity" in feature_set:
        tokens.append("hydro")
    if "hmm_pfam" in feature_set:
        tokens.append("pfam")
    if "protein_length" in feature_set:
        tokens.append("len")

    ordered_tokens = [token for token in SHORT_TOKEN_ORDER if token in tokens]
    return "+".join(ordered_tokens) if ordered_tokens else "none"


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def extract_pr_auc_by_scan(summary_path: Path, split: str) -> Optional[float]:
    split_pattern = re.compile(rf'^\s*"{re.escape(split)}"\s*:\s*\{{')
    pr_auc_pattern = re.compile(r'"pr_auc"\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')

    depth = 0
    active = False

    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not active and depth == 1 and split_pattern.search(line):
                active = True

            if active:
                match = pr_auc_pattern.search(line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None

            depth += line.count("{") - line.count("}")

            if active and depth < 2:
                active = False

    return None


def extract_split_pr_auc(summary_path: Path, split: str) -> Optional[float]:
    if not summary_path.exists():
        LOGGER.warning("缺少结果文件: %s", summary_path)
        return None

    try:
        value = extract_pr_auc_by_scan(summary_path, split)
        if value is not None:
            return value
    except Exception as exc:
        LOGGER.debug("快速扫描 pr_auc 失败，准备回退 JSON 解析: %s (%s)", summary_path, exc)

    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        LOGGER.warning("读取 JSON 失败: %s (%s)", summary_path, exc)
        return None

    candidate = None
    if isinstance(_dig(data, split), dict):
        candidate = data[split]
    elif isinstance(_dig(data, "results", split), dict):
        candidate = data["results"][split]
    elif isinstance(_dig(data, "datasets", split), dict):
        candidate = data["datasets"][split]

    if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict):
        candidate = candidate["metrics"]

    if isinstance(candidate, dict):
        try:
            value = candidate.get("pr_auc")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def extract_prediction_aupr(summary_path: Path) -> Optional[float]:
    if not summary_path.exists():
        LOGGER.warning("缺少预测汇总文件: %s", summary_path)
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        LOGGER.warning("读取 prediction_summary 失败: %s (%s)", summary_path, exc)
        return None

    value = _dig(data, "evaluation_metrics", "aupr")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def find_prediction_summary(run_dir: Path, dataset_key: str) -> Optional[Path]:
    candidates: List[Path] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "sepal-ppi-projector-feature-contant":
            continue
        if child.name == "cis_residue_ensemble":
            continue
        if child.name.lower().startswith(dataset_key.lower()):
            summary_path = child / "prediction_summary.json"
            if summary_path.exists():
                candidates.append(summary_path)

    if not candidates:
        return None
    if len(candidates) > 1:
        LOGGER.warning("%s 下发现多个 %s 结果，默认取第一个: %s", run_dir, dataset_key, candidates)
    return candidates[0]


def collect_row(root: Path, experiment: ExperimentRun) -> Dict[str, Any]:
    config_data = read_yaml(experiment.config_path)
    enabled_features = collect_enabled_features(config_data, experiment.config_path)
    residue_summary = experiment.projector_dir / "final_evaluation_summary.json"
    ensemble_summary = experiment.run_dir / "cis_residue_ensemble" / "final_evaluation_summary.json"

    row: Dict[str, Any] = {
        "experiment_name": experiment.experiment_name,
        "run_dir": experiment.run_dir.relative_to(root).as_posix(),
        "config_short": build_short_config(enabled_features),
        "enabled_features": ",".join(enabled_features),
        "residue_c2_aupr": extract_split_pr_auc(residue_summary, "c2"),
        "residue_c3_aupr": extract_split_pr_auc(residue_summary, "c3"),
        "ensemble_c2_aupr": extract_split_pr_auc(ensemble_summary, "c2"),
        "ensemble_c3_aupr": extract_split_pr_auc(ensemble_summary, "c3"),
    }

    for dataset_key in PREDICTION_DATASETS:
        summary_path = find_prediction_summary(experiment.run_dir, dataset_key)
        if summary_path is None:
            LOGGER.warning("%s 缺少 %s 的 prediction_summary.json", experiment.run_dir, dataset_key)
            row[f"{dataset_key}_aupr"] = None
            continue
        row[f"{dataset_key}_aupr"] = extract_prediction_aupr(summary_path)
    return row


def write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "experiment_name",
        "run_dir",
        "config_short",
        "enabled_features",
        "residue_c2_aupr",
        "residue_c3_aupr",
        "ensemble_c2_aupr",
        "ensemble_c3_aupr",
        "hh70_aupr",
        "hh50_aupr",
        "hl_aupr",
        "ll_aupr",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    root = args.root.resolve()
    output_path = args.out.resolve()

    if not root.exists():
        raise FileNotFoundError(f"根目录不存在: {root}")

    experiments = find_experiment_runs(root)
    if not experiments:
        raise FileNotFoundError(f"未在 {root} 下找到任何 resolved_config.yaml")

    rows = [collect_row(root, experiment) for experiment in experiments]
    rows.sort(key=lambda item: (item["config_short"], item["enabled_features"], item["run_dir"]))
    write_csv(rows, output_path)

    LOGGER.info("已汇总 %d 个实验 -> %s", len(rows), output_path)


if __name__ == "__main__":
    main()