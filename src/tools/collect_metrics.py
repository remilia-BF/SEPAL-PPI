#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

JSON 约定:
- 顶层包含 "c2" 和 "c3" 两个键，每个键的对象内含若干指标字段。
- 若存在其他结构，会尽量做容错处理（如 data["results"]["c2"] 等）。

用法:
python -m src.tools.collect_metrics \
  --root results/final \
  --out  results/summary_metrics.csv
"""
from __future__ import annotations
import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

LOGGER = logging.getLogger("collect_metrics")

# 需要提取的指标字段（按用户要求顺序）
METRIC_KEYS: List[str] = [
    "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "threshold",
    "true_positives", "false_positives", "true_negatives", "false_negatives",
    "n_samples", "n_positive", "n_negative", "positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 final_evaluation_summary.json 指标为 CSV")
    parser.add_argument("--root", type=str, default="results", help="扫描的根目录")
    parser.add_argument("--out", type=str, default="results/summary_metrics.csv", help="输出 CSV 路径")
    parser.add_argument("--strict", action="store_true", help="严格模式：缺失字段时报错")
    parser.add_argument("--verbose", action="store_true", help="打印更多日志")
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def find_json_files(root: Path) -> List[Path]:
    # 查找所有 final_evaluation_summary.json 和 prediction_summary.json
    files = []
    # 使用 set 去重，防止多次添加（虽然 globs 不同应该不会，但安全起见）
    # 排除路径中包含 interpretability_chunks 的文件
    for pattern in ["final_evaluation_summary.json", "prediction_summary.json"]:
        for p in root.rglob(pattern):
            if "interpretability_chunks" in p.parts:
                continue
            files.append(p)
    return sorted(list(set(files)))


def infer_triplet_from_path(path: Path) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """从路径推断 (dataset, embedding, method, subset)。
    支持以下更宽松的目录结构：
      .../results/final/<dataset>/<embedding>/<method>/final_evaluation_summary.json
      .../results/final/<dataset>/<embedding>/<subset>/<method>/final_evaluation_summary.json
    解析策略：定位 `final`（优先），然后把 `final` 之后到文件名前的路径片段拆成队列：
      - dataset = segments[0]
      - embedding = segments[1]
      - method = segments[-1]
      - subset = 中间的片段（segments[2:-1]）如果存在则用 '/' 连接，否则为 None
    返回 None 表示无法解析。
    """
    parts = list(path.parts)
    try:
        if "final" in parts:
            idx = parts.index("final")
        elif "results" in parts:
            idx = parts.index("results")
        else:
            return None

        # 取 final 之后到文件名前的片段
        segments = parts[idx + 1:-1]
        if len(segments) < 2:
            return None
        dataset = segments[0]
        if len(segments) == 2:
            embedding = "default"
            method = segments[1]
            subset = None
        else:
            embedding = segments[1]
            method = segments[-1]
            mid = segments[2:-1]
            subset = "/".join(mid) if mid else None
        return dataset, embedding, method, subset
    except Exception:
        return None


def _dig(obj: Dict[str, Any], *keys, default=None):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def map_prediction_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """将 prediction_summary.json 的结构映射到 METRIC_KEYS"""
    out = {}
    
    eval_metrics = data.get("evaluation_metrics", {})
    
    # Handle two possible structures for total_pairs/n_samples
    n_samples = data.get("total_pairs")
    if n_samples is None:
        n_samples = data.get("prediction_summary", {}).get("total_pairs")
    if n_samples is None:
        n_samples = eval_metrics.get("total_samples")

    # Confusion Matrix
    conf_matrix = eval_metrics.get("confusion_matrix", {})
    tp, fp, tn, fn = None, None, None, None
    if isinstance(conf_matrix, dict):
        tp = conf_matrix.get("tp")
        fp = conf_matrix.get("fp")
        tn = conf_matrix.get("tn")
        fn = conf_matrix.get("fn")
    elif isinstance(conf_matrix, list) and len(conf_matrix) == 2:
        # Standard format: [[TN, FP], [FN, TP]]
        try:
            tn = conf_matrix[0][0]
            fp = conf_matrix[0][1]
            fn = conf_matrix[1][0]
            tp = conf_matrix[1][1]
        except (IndexError, TypeError):
            pass

    out["true_positives"] = tp
    out["false_positives"] = fp
    out["true_negatives"] = tn
    out["false_negatives"] = fn

    # Mapping main metrics
    out["roc_auc"] = eval_metrics.get("auroc")
    out["pr_auc"] = eval_metrics.get("aupr")
    out["f1"] = eval_metrics.get("f1_binary") 
    out["precision"] = eval_metrics.get("precision")
    out["recall"] = eval_metrics.get("recall")
    out["accuracy"] = eval_metrics.get("accuracy")

    # Derived precision/recall if missing
    if out["precision"] is None and tp is not None and fp is not None:
        if (tp + fp) > 0:
            out["precision"] = tp / (tp + fp)
    if out["recall"] is None and tp is not None and fn is not None:
        if (tp + fn) > 0:
            out["recall"] = tp / (tp + fn)
    
    # Counts
    cls_report = eval_metrics.get("classification_report", {})
    pos_report = cls_report.get("Positive", {}) if isinstance(cls_report, dict) else {}
    neg_report = cls_report.get("Negative", {}) if isinstance(cls_report, dict) else {}
    
    n_pos = eval_metrics.get("positive_samples")
    if n_pos is None:
        n_pos = pos_report.get("support")
        
    n_neg = eval_metrics.get("negative_samples")
    if n_neg is None:
        n_neg = neg_report.get("support")
    
    out["n_samples"] = n_samples
    out["n_positive"] = n_pos
    out["n_negative"] = n_neg
    
    # derived
    if out["n_positive"] is not None and out["n_samples"]:
         out["positive_rate"] = float(out["n_positive"]) / n_samples
         
    return out
def extract_split_metrics(data: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    """提取某个数据拆分(c2/c3)的指标字典。
    优先级：顶层 -> data['results'][split] -> data['datasets'][split]
    如果对象内含 'metrics' 再展开一次。
    """
    cand = None
    # 1) 顶层直接包含
    if isinstance(_dig(data, split), dict):
        cand = data[split]
    # 2) results 下
    if cand is None and isinstance(_dig(data, "results", split), dict):
        cand = data["results"][split]
    # 3) datasets 下
    if cand is None and isinstance(_dig(data, "datasets", split), dict):
        cand = data["datasets"][split]
    # 4) 小写键
    if cand is None and isinstance(_dig(data, split.lower()), dict):
        cand = data[split.lower()]
    if cand is None and isinstance(_dig(data, "results", split.lower()), dict):
        cand = data["results"][split.lower()]
    if cand is None and isinstance(_dig(data, "datasets", split.lower()), dict):
        cand = data["datasets"][split.lower()]

    if not isinstance(cand, dict):
        return None

    # 如果内含 metrics，再下钻一层
    metrics = cand.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return cand


def row_from_metrics(dataset: str, embedding: str, method: str, subset: Optional[str],
                     m_c2: Optional[Dict[str, Any]], m_c3: Optional[Dict[str, Any]], 
                     m_pred: Optional[Dict[str, Any]], strict: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "embedding": embedding,
        "method": method,
        "subset": subset,
    }
    # 计算 c2/c3 的 pr_auc（aupr）平均值，若任一缺失则为 None
    def _num(v):
        try:
            return float(v)
        except Exception:
            return None

    v_c2_pr = _num(m_c2.get("pr_auc")) if isinstance(m_c2, dict) else None
    v_c3_pr = _num(m_c3.get("pr_auc")) if isinstance(m_c3, dict) else None
    v_pred_pr = _num(m_pred.get("pr_auc")) if isinstance(m_pred, dict) else None

    aupr_mean = None
    if v_c2_pr is not None and v_c3_pr is not None:
        aupr_mean = (v_c2_pr + v_c3_pr) / 2.0
    elif v_pred_pr is not None:
        aupr_mean = v_pred_pr

    row["aupr_mean"] = aupr_mean

    for key in METRIC_KEYS:
        row[f"c2_{key}"] = m_c2.get(key) if isinstance(m_c2, dict) else None
    for key in METRIC_KEYS:
        row[f"c3_{key}"] = m_c3.get(key) if isinstance(m_c3, dict) else None
    for key in METRIC_KEYS:
        row[f"pred_{key}"] = m_pred.get(key) if isinstance(m_pred, dict) else None

    if strict and (m_c2 is not None or m_c3 is not None):
        # 任一 split 缺字段即报错
        for split_name, m in (("c2", m_c2), ("c3", m_c3)):
            if not isinstance(m, dict):
                pass
            else:
                missing = [k for k in METRIC_KEYS if k not in m]
                if missing:
                    raise KeyError(f"{dataset}/{embedding}/{method}: {split_name} 缺少指标: {missing}")
    return row


def main():
    args = parse_args()
    setup_logging(args.verbose)

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_json_files(root)
    if not files:
        LOGGER.warning(f"未在 {root} 下找到任何 final_evaluation_summary.json 文件")

    rows: List[Dict[str, Any]] = []
    skipped: int = 0

    for fp in sorted(files):
        triplet = infer_triplet_from_path(fp)
        if not triplet:
            skipped += 1
            LOGGER.debug(f"跳过无法解析路径的文件: {fp}")
            continue
        dataset, embedding, method, subset = triplet

        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            skipped += 1
            LOGGER.warning(f"读取失败，跳过: {fp} ({e})")
            continue

        m_c2 = None
        m_c3 = None
        m_pred = None

        if fp.name == "final_evaluation_summary.json":
            m_c2 = extract_split_metrics(data, "c2")
            m_c3 = extract_split_metrics(data, "c3")
        elif fp.name == "prediction_summary.json":
            m_pred = map_prediction_metrics(data)

        row = row_from_metrics(dataset, embedding, method, subset, m_c2, m_c3, m_pred, args.strict)
        rows.append(row)

    # 列顺序
    fieldnames = ["dataset", "embedding", "method", "subset", "aupr_mean"]
    fieldnames += [f"c2_{k}" for k in METRIC_KEYS]
    fieldnames += [f"c3_{k}" for k in METRIC_KEYS]
    fieldnames += [f"pred_{k}" for k in METRIC_KEYS]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(f"已汇总 {len(rows)} 条记录，跳过 {skipped} 个文件 -> {out_path}")


if __name__ == "__main__":
    main()
