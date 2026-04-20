"""
Training runner utilities extracted from sepal-ppi.py to keep the entry script slim.

This module contains:
- ModelManager: manage best/last checkpoints and inference configs
- run_training_mode: end-to-end training entry
- train_sequence_model_with_validation: training loop with epoch validation
- run_epoch_validation: run validation on c2/c3 using InferenceEngine
- run_final_evaluation: final evaluation with best model

Note: We intentionally duplicate a few tiny helper extractors here to avoid
import cycles back to the entry script.
"""

from __future__ import annotations

import json
import time
import traceback
import math
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal project imports
from src.utils.helpers import set_random_seed, get_device
from src.utils.logger import format_memory
from src.data_processing import create_default_loaders
from src.models import create_avg_pool_mlp, create_yaml_model
from src.inference import InferenceEngine


# ----------------------- small local helpers (no-cycle) -----------------------
def _extract_pooling_type_from_config(model_config: Dict[str, Any]) -> str:
    """Best-effort extractor for pooling type from model config.

    Supports both modular model_config and legacy flat config. Defaults to 'avg'.
    """
    # modular config
    if isinstance(model_config, dict) and isinstance(model_config.get("model_config"), dict):
        mc = model_config["model_config"]
        # try nested places
        for key in ("pooling", "pooling_type"):
            if key in mc and isinstance(mc[key], (str, dict)):
                if isinstance(mc[key], str):
                    return mc[key]
                if isinstance(mc[key], dict):
                    return mc[key].get("type", "avg")

    # legacy flat config
    if isinstance(model_config, dict):
        if isinstance(model_config.get("pooling_type"), str):
            return model_config["pooling_type"]

    return "avg"


def _extract_interaction_type_from_config(model_config: Dict[str, Any]) -> str:
    """Best-effort extractor for interaction type; defaults to 'hadamard'."""
    # modular config
    if isinstance(model_config, dict) and isinstance(model_config.get("model_config"), dict):
        mc = model_config["model_config"]
        for key in ("interaction", "interaction_type"):
            if key in mc and isinstance(mc[key], (str, dict)):
                if isinstance(mc[key], str):
                    return mc[key]
                if isinstance(mc[key], dict):
                    return mc[key].get("type", "hadamard")

    # legacy flat config
    if isinstance(model_config, dict):
        if isinstance(model_config.get("interaction_type"), str):
            return model_config["interaction_type"]

    return "hadamard"


def _check_bucketed_lmdb(lmdb_path: str, logger) -> bool:
    """Robustly verify if an LMDB is bucketed; permissive on uncertainty.

    Strategy:
    - Accept directories containing data.mdb as valid LMDB; try to read metadata.
    - Accept directories containing multiple .lmdb shards (bucket_*.lmdb) as bucketed.
    - Accept single .lmdb files; try to read metadata.
    - If metadata cannot definitively confirm bucketed format, default to True to
      avoid blocking valid datasets (matches behavior prior to refactor).
    """
    try:
        from pathlib import Path as _P
        import lmdb  # local import to avoid top-level dependency when unused

        lmdb_path_obj = _P(lmdb_path)
        if not lmdb_path_obj.exists():
            logger.warning(f"LMDB path does not exist: {lmdb_path}")
            return False

        env = None
        # Case 1: Standard LMDB directory with data.mdb
        if lmdb_path_obj.is_dir() and (lmdb_path_obj / "data.mdb").exists():
            env = lmdb.open(str(lmdb_path_obj), readonly=True, lock=False, max_readers=1)
        # Case 2: Directory containing shard files (e.g., bucket_0.lmdb)
        elif lmdb_path_obj.is_dir():
            shard_candidates = [p for p in lmdb_path_obj.iterdir() if p.suffix == ".lmdb"]
            if shard_candidates:
                # Treat as bucketed; optionally open the first shard to probe metadata
                try:
                    env = lmdb.open(str(shard_candidates[0]), readonly=True, lock=False, max_readers=1)
                except Exception:
                    # Even if open fails, presence of shards is strong evidence of bucketing
                    return True
            else:
                logger.warning(f"No LMDB files found in directory: {lmdb_path}")
                # Cannot determine; be permissive
                return True
        # Case 3: Single LMDB file
        elif lmdb_path_obj.is_file() and lmdb_path_obj.suffix == ".lmdb":
            env = lmdb.open(str(lmdb_path_obj), readonly=True, lock=False, max_readers=1)
        else:
            logger.warning(f"Invalid LMDB path: {lmdb_path}")
            return False

        # Probe metadata/keys to infer bucketing
        try:
            if env is None:
                # If we couldn't open an env but evidence is strong, accept as bucketed
                return True

            with env.begin() as txn:
                # Common metadata keys used by bucketed preprocessors
                meta_keys = [
                    b"__BUCKET_META__",
                    b"__bucket_meta__",
                    b"bucket_meta",
                    b"__meta__",
                    b"meta:bucket_count",
                    b"bucket_count",
                    b"_BUCKETED_",
                    b"_bucketed",
                ]
                if any(txn.get(k) is not None for k in meta_keys):
                    return True

                # Scan a limited number of keys for bucket-like prefixes
                cur = txn.cursor()
                scanned = 0
                for k, _ in cur:
                    scanned += 1
                    if k.startswith(b"bucket_") or k.startswith(b"__bucket_"):
                        return True
                    if scanned >= 1024:
                        break

                # Could not confirm via metadata; be permissive to avoid false negatives
                logger.info("Could not determine LMDB bucketing from metadata; treating as bucketed for compatibility")
                return True
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
    except Exception as e:
        # Be permissive on unexpected errors to avoid blocking valid runs
        logger.warning(f"Failed to check LMDB bucketing status: {e}; assuming bucketed")
        return True


# ------------------------------ Model Manager -------------------------------
class ModelManager:
    """Manages model saving/loading with best/last tracking and configs."""

    def __init__(self, output_dir: str, logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.best_model_path = self.output_dir / "best_model.pth"
        self.last_model_path = self.output_dir / "last_model.pth"
        self.best_metric = 0.0
        self.best_epoch = 1
        self.inference_config_path = self.output_dir / "inference_config.yaml"

    def save_last_model(self, model, epoch: int, training_config: Dict, model_config: Dict):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "training_config": training_config,
            "model_config": model_config,
            "timestamp": time.time(),
        }
        torch.save(checkpoint, self.last_model_path)
        if self.logger:
            self.logger.debug(f"Saved latest model: {self.last_model_path}")

    def save_best_model(self, model, epoch: int, metric_value: float,
                        training_config: Dict, model_config: Dict) -> bool:
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch = epoch
            checkpoint_full = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "training_config": training_config,
                "model_config": model_config,
                "timestamp": time.time(),
            }

            # 检查是否启用分离保存
            separate = False
            input_cfg = None
            try:
                if isinstance(model_config, dict):
                    if "model_config" in model_config and isinstance(model_config["model_config"], dict):
                        input_cfg = (model_config["model_config"].get("input_data", {}) or {})
                        separate = bool(input_cfg.get("separate_save_models", False))
                    if not separate and "input_data" in model_config:
                        input_cfg = (model_config.get("input_data", {}) or {})
                        separate = bool(input_cfg.get("separate_save_models", False))
            except Exception:
                separate = False

            if separate:
                # Targets:
                # - complete_model.pth: full model (same as historical best_model.pth)
                # - input_layer.pth: input layer weights only
                # - best_model.pth: model weights with input_layer stripped

                # 1) 保存完整模型为 complete_model.pth
                complete_path = self.output_dir / "complete_model.pth"
                torch.save(checkpoint_full, complete_path)

                # 2) 提取输入层权重
                input_layer_path = self.output_dir / "input_layer.pth"
                meta = {
                    "epoch": epoch,
                    "timestamp": time.time(),
                    "source": "sepal_ppi.training",
                    "model_class": getattr(model, "__class__", type(model)).__name__,
                    "input_dim": None,
                    "embedding_dim": None,
                    "activation": None,
                    "projection_enabled": None,
                }
                try:
                    if input_cfg is None and "model_config" in model_config:
                        input_cfg = (model_config["model_config"].get("input_data", {}) or {})
                    meta["input_dim"] = (input_cfg or {}).get("input_dim")
                    meta["embedding_dim"] = (
                        model_config.get("embedding_dim") if isinstance(model_config, dict) else None
                    ) or (input_cfg or {}).get("embedding_dim")
                    proj_cfg = (input_cfg or {}).get("projection", {}) or {}
                    meta["activation"] = proj_cfg.get("activation", None)
                    meta["projection_enabled"] = bool(proj_cfg.get("enabled", False))
                except Exception:
                    pass

                input_state = None
                try:
                    input_layer = getattr(model, "input_layer", None)
                    if input_layer is not None and hasattr(input_layer, "state_dict"):
                        input_state = input_layer.state_dict()
                        torch.save({
                            "model_state_dict": input_state,
                            "meta": meta,
                            "model_config": model_config,
                        }, input_layer_path)
                        if self.logger:
                            self.logger.debug(f"Exported input layer weights: {input_layer_path}")
                    else:
                        if self.logger:
                            self.logger.warning("separate_save_models enabled but no input_layer found; skip exporting input layer.")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to export input layer weights: {e}")

                # 3) 生成去除 input_layer 的 best_model.pth
                try:
                    stripped_state = {
                        k: v for k, v in checkpoint_full["model_state_dict"].items() if not k.startswith("input_layer.")
                    }
                    checkpoint_stripped = dict(checkpoint_full)
                    checkpoint_stripped["model_state_dict"] = stripped_state
                    checkpoint_stripped["meta"] = {"stripped_input_layer": True}
                    torch.save(checkpoint_stripped, self.best_model_path)
                    if self.logger:
                        self.logger.debug(
                            f"Found better model (metric={metric_value:.6f}); saved split models: best_model(stripped), input_layer, complete_model"
                        )
                except Exception as e:
                    # Fallback: if stripping fails, save full model to traditional best_model.pth
                    torch.save(checkpoint_full, self.best_model_path)
                    if self.logger:
                        self.logger.warning(f"Failed to create stripped best_model; fell back to full model: {e}")
            else:
                # 传统路径：直接保存到 best_model.pth
                torch.save(checkpoint_full, self.best_model_path)
                if self.logger:
                    self.logger.debug(f"Found better model (metric={metric_value:.6f}); saved to {self.best_model_path}")
            return True
        return False

    def create_inference_config(self, original_config: Dict, model_path: str, best_epoch: int) -> Dict[str, Any]:
        # choose batch size: prefer inference.batch_size then training.batch_size
        inference_batch_size = None
        try:
            inference_batch_size = original_config.get("inference", {}).get("batch_size")
        except Exception:
            inference_batch_size = None
        if inference_batch_size is None:
            inference_batch_size = original_config.get("training", {}).get("batch_size", 32)

        data_config = dict(original_config.get("data", {}))
        test_files: Dict[str, str] = {}
        val_file = data_config.get("validation_file")
        if isinstance(val_file, str) and val_file:
            test_files["c2"] = val_file
        test_file = data_config.get("test_file")
        if isinstance(test_file, str) and test_file:
            test_files["c3"] = test_file
        data_config["test_files"] = test_files

        inference_config = {
            "mode": "inference",
            "model": {
                "model_path": model_path,
                "pooling_type": _extract_pooling_type_from_config(original_config["model"]),
                "interaction_type": _extract_interaction_type_from_config(original_config["model"]),
                "classifier_type": original_config["model"].get("classifier_type", "standard"),
                "embedding_dim": original_config["model"]["embedding_dim"],
            },
            "inference": {
                "batch_size": inference_batch_size,
                "threshold": 0.5,
                "best_epoch": best_epoch,
            },
            "data": data_config,
            "output": {
                "generate_confusion_matrix": True,
                "save_predictions": True,
                "output_dir": str(self.output_dir),
            },
        }

        # persist
        try:
            with open(self.inference_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(inference_config, f, allow_unicode=True, sort_keys=False)
            if self.logger:
                self.logger.debug(f"Wrote inference config: {self.inference_config_path}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write inference config: {e}")
        return inference_config

    def create_best_inference_config(self, original_config: Dict) -> Dict[str, Any]:
        best_config_path = self.output_dir / "best_config.yaml"
        inference_batch_size = None
        try:
            inference_batch_size = original_config.get("inference", {}).get("batch_size")
        except Exception:
            inference_batch_size = None
        if inference_batch_size is None:
            inference_batch_size = original_config.get("training", {}).get("batch_size", 32)

        data_config = dict(original_config.get("data", {}))
        test_files: Dict[str, str] = {}
        val_file = data_config.get("validation_file")
        if isinstance(val_file, str) and val_file:
            test_files["c2"] = val_file
        test_file = data_config.get("test_file")
        if isinstance(test_file, str) and test_file:
            test_files["c3"] = test_file
        data_config["test_files"] = test_files

        best_inference_config = {
            "mode": "inference",
            "model": {
                "model_path": str(self.best_model_path),
                "pooling_type": _extract_pooling_type_from_config(original_config["model"]),
                "interaction_type": _extract_interaction_type_from_config(original_config["model"]),
                "classifier_type": original_config["model"].get("classifier_type", "standard"),
                "embedding_dim": original_config["model"]["embedding_dim"],
            },
            "inference": {
                "batch_size": inference_batch_size,
                "threshold": 0.5,
                "best_epoch": self.best_epoch,
            },
            "data": data_config,
            "output": {
                "generate_confusion_matrix": True,
                "save_predictions": True,
                "output_dir": str(self.output_dir),
            },
        }

        try:
            with open(best_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(best_inference_config, f, allow_unicode=True, sort_keys=False)
            if self.logger:
                self.logger.info(f"Created best inference config: {best_config_path}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write best inference config: {e}")
        return best_inference_config

    def get_best_model_path(self) -> str:
        return str(self.best_model_path)

    def get_last_model_path(self) -> str:
        return str(self.last_model_path)

    def get_inference_config_path(self) -> str:
        return str(self.inference_config_path)


# ---------------------------- finetune utilities ----------------------------
def _freeze_model_modules(model: nn.Module, freeze_except: List[str], logger=None) -> Dict[str, int]:
    """Freeze all model submodules except those in freeze_except.

    Known top-level submodules on ConfigurableModel: preprocessing, prepairing, pooling, interaction, classifier
    Returns a dict with counts for logging.
    """
    known = ["input_layer", "preprocessing", "prepairing", "pooling", "interaction", "classifier"]
    keep = set(freeze_except or [])
    stats = {"frozen": 0, "trainable": 0}

    for name in known:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        trainable = name in keep
        for p in sub.parameters():
            p.requires_grad = trainable
            stats["trainable" if trainable else "frozen"] += p.numel()
        if logger:
            logger.info(f"Module {name}: {'trainable' if trainable else 'frozen'}")

    # Also ensure any parameters outside known modules follow freezing rules (rare)
    known_params = set()
    for name in known:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters(recurse=True):
            known_params.add(p)

    for p in model.parameters():
        if p in known_params:
            continue
        # default: freeze unless any keep requests global training
        trainable = len(keep) == 0
        p.requires_grad = trainable
        stats["trainable" if trainable else "frozen"] += p.numel()

    if logger:
        logger.info(f"Parameter stats -> trainable: {stats['trainable']:,} frozen: {stats['frozen']:,}")
    return stats


def _load_pretrained_weights(model: nn.Module, checkpoint_path: str, logger=None) -> Tuple[List[str], List[str]]:
    """
    加载预训练权重，自动跳过形状不匹配的参数。

    说明：PyTorch 的 load_state_dict 即便在 strict=False 下，遇到“同名但形状不一致”的参数仍会抛错。
    这里先根据当前模型的 state_dict 过滤一遍 checkpoint，只保留“键存在且形状一致”的权重，再进行加载。

    返回值：
      - missing: 当前模型缺失但未在 checkpoint 中出现的键（信息性）
      - unexpected: checkpoint 中存在但模型中不存在或被过滤掉的键（信息性）
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state = ckpt.get("model_state_dict", ckpt)

    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys: List[str] = []
    for k, v in raw_state.items():
        if k in model_state and isinstance(v, torch.Tensor):
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)
        else:
            # 保留不在模型中的键给 IncompatibleKeys 统计
            skipped_keys.append(k)

    # 实际加载
    result = model.load_state_dict(filtered_state, strict=False)

    # 汇总信息（把过滤掉的视作 unexpected）
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", [])) + skipped_keys

    if logger:
        logger.info(f"Loaded pretrained weights (shape-checked): {checkpoint_path}")
        if skipped_keys:
            # 只打印前若干条，避免日志过长
            head = 10
            logger.warning(
                f"Skipped {len(skipped_keys)} incompatible keys (shape mismatch or unused). "
                f"Examples: {skipped_keys[:head]}{' ...' if len(skipped_keys) > head else ''}"
            )
        if missing:
            logger.debug(f"Missing weights: {missing}")
        if unexpected:
            logger.debug(f"Unexpected/filtered weights: {unexpected}")

    return missing, unexpected


# ---------------------------- validation helpers ----------------------------
def run_epoch_validation(model_path: str, inference_config_path: str,
                         data_loaders: Dict, device, logger) -> Tuple[Dict, Dict]:
    """Run epoch-time validation on c2/c3 datasets using InferenceEngine."""
    try:
        with open(inference_config_path, "r", encoding="utf-8") as f:
            inference_config = yaml.safe_load(f)

        inference_engine = InferenceEngine(inference_config, logger)
        inference_engine.load_model(model_path, device=device)

        val_datasets = {}
        test_datasets = {}

        if "c2" in data_loaders and hasattr(data_loaders["c2"], "dataset"):
            val_datasets["c2"] = data_loaders["c2"].dataset
        if "c3" in data_loaders and hasattr(data_loaders["c3"], "dataset"):
            test_datasets["c3"] = data_loaders["c3"].dataset

        # Fallback: scan for keys ending with _dataset
        for key, loader in data_loaders.items():
            if isinstance(key, str) and key.endswith("_dataset") and hasattr(loader, "dataset"):
                name = key[:-8]
                if name == "c2" and "c2" not in val_datasets:
                    val_datasets["c2"] = loader.dataset
                if name == "c3" and "c3" not in test_datasets:
                    test_datasets["c3"] = loader.dataset

        logger.debug(f"Found validation datasets: {list(val_datasets.keys())}")
        logger.debug(f"Found test datasets: {list(test_datasets.keys())}")

        val_metrics: Dict[str, Any] = {}
        test_metrics: Dict[str, Any] = {}

        if val_datasets:
            val_results = inference_engine.evaluate_datasets(val_datasets, show_progress=True, save_results=False)
            if val_results and "c2" in val_results:
                val_metrics = val_results["c2"]

        if test_datasets:
            test_results = inference_engine.evaluate_datasets(test_datasets, show_progress=True, save_results=False)
            if test_results and "c3" in test_results:
                test_metrics = test_results["c3"]

        return val_metrics, test_metrics
    except Exception as e:
        logger.error(f"Epoch validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}, {}


def run_final_evaluation(model_path: str, inference_config_path: str,
                         data_loaders: Dict, output_dir: str, device, logger) -> Dict:
    """Run final evaluation with the best model and persist results if requested."""
    try:
        with open(inference_config_path, "r", encoding="utf-8") as f:
            inference_config = yaml.safe_load(f)

        inference_engine = InferenceEngine(inference_config, logger)
        inference_engine.load_model(model_path, device=device)

        # Prepare datasets for evaluation
        eval_datasets = {}
        if "c2" in data_loaders and hasattr(data_loaders["c2"], "dataset"):
            eval_datasets["c2"] = data_loaders["c2"].dataset
        if "c3" in data_loaders and hasattr(data_loaders["c3"], "dataset"):
            eval_datasets["c3"] = data_loaders["c3"].dataset

        results = inference_engine.evaluate_datasets(
            eval_datasets, show_progress=True, save_results=True
        ) if eval_datasets else {}

        # 导出门控统计 CSV（c2/c3 同表）
        try:
            gating_rows: List[Dict[str, Any]] = []
            for dataset_name, split_result in (results or {}).items():
                if not isinstance(split_result, dict):
                    continue
                gating_summary = split_result.get('feature_gating_summary') or {}
                feature_stats = gating_summary.get('features') if isinstance(gating_summary, dict) else None
                if not isinstance(feature_stats, dict):
                    continue

                for feature_name, stat in feature_stats.items():
                    if not isinstance(stat, dict):
                        continue
                    gating_rows.append({
                        'dataset': dataset_name,
                        'feature_name': feature_name,
                        'mean': float(stat.get('mean', 0.0)),
                        'var': float(stat.get('var', 0.0)),
                        'n': int(stat.get('count', 0)),
                    })

            if gating_rows:
                gating_csv_path = Path(output_dir) / "feature_gating.csv"
                pd.DataFrame(gating_rows).to_csv(gating_csv_path, index=False)
                logger.info(f"Feature gating summary written: {gating_csv_path}")
        except Exception as e:
            logger.warning(f"Failed to write feature_gating.csv: {e}")

        # Persist a compact summary
        try:
            summary_path = Path(output_dir) / "final_evaluation_summary.json"

            compact: Dict[str, Any] = {}
            for split_name, split_result in (results or {}).items():
                if not isinstance(split_result, dict):
                    continue

                # 基础指标：去掉大数组字段
                base = {
                    mk: mv
                    for mk, mv in split_result.items()
                    if mk not in ("predictions", "true_labels")
                }

                protein_ids = split_result.get("protein_ids")
                predictions = split_result.get("predictions")
                true_labels = split_result.get("true_labels")

                detailed_pairs = []
                try:
                    if protein_ids is not None and predictions is not None:
                        # 将预测结果与 protein_ids 对齐，生成包含概率与 logit 的结构
                        preds_list = list(predictions)
                        labels_list = list(true_labels) if true_labels is not None else None

                        n = min(len(protein_ids), len(preds_list))
                        for i in range(n):
                            pair = protein_ids[i]
                            try:
                                p_val = float(preds_list[i])
                            except Exception:
                                continue

                            # 计算 logit，避免数值溢出
                            p_clamped = min(max(p_val, 1e-7), 1.0 - 1e-7)
                            logit_val = math.log(p_clamped / (1.0 - p_clamped))

                            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                                p1_id, p2_id = pair[0], pair[1]
                            else:
                                # 回退：无法拆分成两个 ID 时，尽量保留原信息
                                p1_id, p2_id = str(pair), ""

                            item: Dict[str, Any] = {
                                "protein1_id": p1_id,
                                "protein2_id": p2_id,
                                "probability": p_val,
                                "logit": logit_val,
                            }

                            if labels_list is not None and i < len(labels_list):
                                try:
                                    item["label"] = int(labels_list[i])
                                except Exception:
                                    pass

                            detailed_pairs.append(item)
                except Exception as enrich_err:
                    logger.warning(f"Failed to enrich protein_ids with predictions: {enrich_err}")
                    detailed_pairs = []

                # 仅在成功构造详细结构时覆盖 protein_ids
                if detailed_pairs:
                    base["protein_ids"] = detailed_pairs

                compact[split_name] = base

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(compact, f, indent=2, ensure_ascii=False)
            logger.info(f"Final evaluation summary written: {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to write final evaluation summary: {e}")

        return results or {}
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}


def _collect_epoch_head_stats(model, data_loader, device, logger, max_batches: int = 0) -> Dict[str, Any]:
    """Collect epoch-level head-gating and head attention correlation stats on a loader."""
    if data_loader is None:
        return {}

    was_training = model.training
    model.eval()

    all_gate_weights = []
    per_head_values: List[List[np.ndarray]] = []
    num_heads_detected = None

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                protein1_seq = batch["protein1_seq"].to(device)
                protein1_mask = batch["protein1_mask"].to(device)
                protein2_seq = batch["protein2_seq"].to(device)
                protein2_mask = batch["protein2_mask"].to(device)

                protein_ids = None
                if "protein_ids" in batch:
                    protein1_ids = [pid[0] for pid in batch["protein_ids"]]
                    protein2_ids = [pid[1] for pid in batch["protein_ids"]]
                    protein_ids = (protein1_ids, protein2_ids)

                _ = model(protein1_seq, protein2_seq, protein1_mask, protein2_mask, protein_ids)

                gate_weights = None
                if hasattr(model, "get_head_gating_weights"):
                    gate_weights = model.get_head_gating_weights()
                if gate_weights is not None:
                    gate_np = gate_weights.detach().cpu().numpy()
                    all_gate_weights.append(gate_np)
                    if num_heads_detected is None:
                        num_heads_detected = gate_np.shape[1]

                attn = None
                if hasattr(model, "get_attention_weights"):
                    attn = model.get_attention_weights()
                if not attn:
                    continue

                for protein_key, mask_tensor in (("protein1", batch["protein1_mask"]), ("protein2", batch["protein2_mask"])):
                    attn_tensor = attn.get(protein_key)
                    if attn_tensor is None:
                        continue
                    # [B, L, H]
                    attn_np = attn_tensor.detach().cpu().numpy()
                    mask_np = mask_tensor.detach().cpu().numpy().astype(np.bool_)
                    bsz, _, heads = attn_np.shape
                    if num_heads_detected is None:
                        num_heads_detected = heads
                    if not per_head_values:
                        per_head_values = [[] for _ in range(heads)]

                    for b in range(bsz):
                        valid = mask_np[b]
                        if valid.sum() <= 1:
                            continue
                        sample_attn = attn_np[b, valid, :]  # [N_valid, H]
                        for h in range(sample_attn.shape[1]):
                            per_head_values[h].append(sample_attn[:, h].astype(np.float32))
    except Exception as e:
        logger.warning(f"Collect epoch head stats failed: {e}")
        return {}
    finally:
        if was_training:
            model.train()

    result: Dict[str, Any] = {}
    if all_gate_weights:
        gate_all = np.concatenate(all_gate_weights, axis=0)
        result["gate_mean"] = gate_all.mean(axis=0)
        result["gate_std"] = gate_all.std(axis=0)

    if per_head_values:
        head_series = []
        for h_values in per_head_values:
            if not h_values:
                head_series.append(np.array([0.0], dtype=np.float32))
            else:
                head_series.append(np.concatenate(h_values, axis=0))

        min_len = min(len(v) for v in head_series) if head_series else 0
        if min_len > 1:
            aligned = np.stack([v[:min_len] for v in head_series], axis=0)
            corr = np.corrcoef(aligned)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 1.0)
            result["head_corr"] = corr

    if num_heads_detected is not None:
        result["num_heads"] = int(num_heads_detected)

    return result


def _append_gate_stats_tsv(output_dir: str, epoch: int, gate_mean: np.ndarray, gate_std: np.ndarray):
    out_path = Path(output_dir) / "epoch_head_gating_stats.tsv"
    need_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("epoch\thead\tmean\tstd\n")
        for h in range(len(gate_mean)):
            f.write(f"{epoch}\t{h + 1}\t{gate_mean[h]:.4f}\t{gate_std[h]:.4f}\n")


def _append_head_corr_upper_tsv(output_dir: str, epoch: int, corr: np.ndarray):
    out_path = Path(output_dir) / "epoch_head_attention_correlation.tsv"
    with open(out_path, "a", encoding="utf-8") as f:
        n = corr.shape[0]
        f.write(f"epoch\t{epoch}\n")
        header = ["head"] + [str(i + 1) for i in range(n)]
        f.write("\t".join(header) + "\n")
        for i in range(n):
            row = [str(i + 1)]
            for j in range(n):
                if j < i:
                    row.append("")
                else:
                    row.append(f"{corr[i, j]:.4f}")
            f.write("\t".join(row) + "\n")
        f.write("\n")


def _reset_epoch_head_stats_files(output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("epoch_head_gating_stats.tsv", "epoch_head_attention_correlation.tsv"):
        file_path = out_dir / name
        if file_path.exists():
            file_path.unlink()


# ------------------------------- training api --------------------------------
def run_training_mode(config: Dict[str, Any], output_dir: str, logger) -> Dict[str, Any]:
    """Run training mode with epoch-by-epoch validation using inference engine."""

    def _collect_multimodal_feature_sources(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        def add_source(unit_name: str, unit_cfg: Dict[str, Any]):
            if not isinstance(unit_cfg, dict):
                return
            dp_cfg = unit_cfg.get("data_processing")
            if not isinstance(dp_cfg, dict):
                return
            feature_folder = dp_cfg.get("all_feature_folder")
            if not feature_folder or str(feature_folder).lower() in {"none", "null", "false"}:
                return
            lmdb_path = str(Path(str(feature_folder)) / "multimodal_features.lmdb")
            lmdb_exists = Path(lmdb_path).exists()
            sources.append(
                {
                    "unit": str(unit_name),
                    "feature_folder": str(feature_folder),
                    "feature_lmdb": lmdb_path,
                    "feature_lmdb_exists": lmdb_exists,
                }
            )

        def scan_model_unit(model_unit_cfg: Any):
            if not isinstance(model_unit_cfg, dict):
                return
            preproc_cfg = model_unit_cfg.get("preprocessing")
            if not isinstance(preproc_cfg, dict):
                return

            # 兼容两种结构：
            # 1) preprocessing: {unit_name: {data_processing: {...}}}
            # 2) preprocessing: {preprocessor: ..., data_processing: {...}, feature_files: ...}
            if isinstance(preproc_cfg.get("data_processing"), dict):
                unit_name = preproc_cfg.get("preprocessor") or preproc_cfg.get("name") or "preprocessing"
                add_source(str(unit_name), preproc_cfg)
                return

            for unit_name, unit_cfg in preproc_cfg.items():
                if not isinstance(unit_cfg, dict):
                    continue
                add_source(str(unit_name), unit_cfg)

        # 常见的配置落点：顶层 model_unit 或 model.model_unit
        scan_model_unit(cfg.get("model_unit"))
        scan_model_unit(cfg.get("model", {}).get("model_unit"))
        # 某些路径会把 model_unit 内嵌在 model_config 下
        scan_model_unit(cfg.get("model", {}).get("model_config", {}).get("model_unit"))
        return sources

    # Set random seed
    seed = config["training"]["seed"]
    set_random_seed(seed)
    logger.debug(f"Random seed set to: {seed}")

    # 每次训练前重置 epoch 统计文件，避免追加历史 run 数据
    _reset_epoch_head_stats_files(output_dir)

    # Get device
    device = get_device()
    logger.info(f"Compute device: {device}")

    # Log training configuration
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    target_precision = data_config.get("target_precision", "fp32")

    log_config = {
        "Embeddings": data_config.get("embedding_file", "N/A"),
        "Multimodal feature folder": "None",
        "Multimodal feature lmdb": "None",
        "Training": f"{training_config['epochs']} epochs, batch={training_config['batch_size']}, lr={training_config['learning_rate']}",
    }

    # 显式报告多模态特征路径与LMDB（若不存在则为“无”）
    try:
        sources = _collect_multimodal_feature_sources(config)
        if sources:
            log_config["Multimodal feature folder"] = "; ".join(
                [f"{s['unit']}={s['feature_folder']}" for s in sources]
            )
            log_config["Multimodal feature lmdb"] = "; ".join(
                [
                    f"{s['unit']}={s['feature_lmdb']}" if s.get("feature_lmdb_exists") else f"{s['unit']}=None"
                    for s in sources
                ]
            )
    except Exception as e:
        # 不让日志增强影响训练流程
        log_config["Multimodal feature folder"] = "None"
        log_config["Multimodal feature lmdb"] = "None"
        if hasattr(logger, "debug"):
            logger.debug(f"Failed to collect multimodal feature sources: {e}")

    if hasattr(logger, "log_config"):
        logger.log_config(log_config, "Training configuration")

    try:
        # Step 1: Check bucketing status and prepare data
        logger.debug("Checking bucketing status and loading data...")

        if data_config["use_sequence_data"]:
            embedding_file = data_config["embedding_file"]
            is_cis_data = data_config.get("cis_type", False)

            if not is_cis_data and not _check_bucketed_lmdb(embedding_file, logger):
                logger.error("Detected a non-bucketed LMDB file!")
                logger.error("Please bucket the LMDB using the preprocessing script first:")
                logger.error(
                    f"python emb_tools/preprocess_bucketed_lmdb.py --source-lmdb {embedding_file} --output-dir bucketed_embeddings/"
                )
                raise ValueError("A pre-bucketed LMDB file is required")
            elif is_cis_data:
                logger.info("Detected CIS-level data configuration, skipping bucketing check")

            if is_cis_data:
                from src.data_processing.cis_data_loader import create_cis_data_loaders

                cis_config = {
                    "embedding_file": embedding_file,
                    "train_file": data_config["train_file"],
                    "test_files": {"c2": data_config["validation_file"], "c3": data_config["test_file"]},
                    "batch_size": training_config["batch_size"],
                    "cache_size": data_config.get("cache_size", 8000),
                    # 以数据源embedding维度解析LMDB
                    "embedding_dim": data_config["embedding_dim"],
                    "target_precision": data_config.get("target_precision", "fp32"),
                    "seed": seed,
                }

                try:
                    cis_batch_data = create_cis_data_loaders(cis_config)
                    data_loaders = cis_batch_data["data_loaders"]
                    embedding_cache = cis_batch_data["embedding_cache"]  # noqa: F841 - for logging/debug
                except Exception as e:
                    logger.error(f"Failed to create CIS data loader: {e}")
                    logger.error("Please check:")
                    logger.error(f"  1. Whether the LMDB file exists: {embedding_file}")
                    logger.error(f"  2. Whether the train file exists: {data_config.get('train_file')}")
                    logger.error(f"  3. Whether the validation file exists: {data_config.get('validation_file')}")
                    logger.error(f"  4. Whether the test file exists: {data_config.get('test_file')}")
                    logger.error(f"  5. Whether embedding_dim is correctly configured: {model_config.get('embedding_dim')}")
                    raise RuntimeError(f"Failed to initialize CIS data loader: {e}") from e

                logger.info("Using CIS-specific data loader")
            else:
                from src.data_processing.smart_batch_loader import create_smart_batch_data_loaders

                compat_config = {
                    "embedding_file": embedding_file,
                    "train_file": data_config["train_file"],
                    "test_files": {"c2": data_config["validation_file"], "c3": data_config["test_file"]},
                    "fasta_file": data_config["fasta_file"],
                    "batch_size": training_config["batch_size"],
                    "cache_size": data_config.get("cache_size", 8000),
                    # Loader 使用数据源维度，模型通过 input_layer 对齐
                    "embedding_dim": data_config["embedding_dim"],
                    "max_length": data_config.get("max_length", 1024),
                    "pooling_type": _extract_pooling_type_from_config(model_config),
                    "target_precision": data_config.get("target_precision"),
                    "cis_type": data_config.get("cis_type", False),
                    "seed": seed,
                }

                smart_batch_data = create_smart_batch_data_loaders(compat_config)
                data_loaders = smart_batch_data["data_loaders"]
                embedding_cache = smart_batch_data["embedding_cache"]  # noqa: F841

            # Log cache stats if available
            try:
                cache_stats = (cis_batch_data if is_cis_data else smart_batch_data)["cache_stats"]
                logger.debug(
                    f"Cache initialized: {cache_stats['cache_size']}/{cache_stats['max_capacity']} (hit rate: {cache_stats['hit_rate']:.1%})"
                )
                logger.debug(
                    f"Pooling type: {(cis_batch_data if is_cis_data else smart_batch_data)['pooling_type']}"
                )
            except Exception:
                pass
        else:
            data_loaders = create_default_loaders()

        # Log dataset information (best-effort)
        dataset_info: Dict[str, Any] = {}
        if data_config["use_sequence_data"]:
            try:
                if "train" in data_loaders:
                    train_file = data_config.get("train_file")
                    if train_file and Path(train_file).exists():
                        with open(train_file, "r") as f:
                            train_count = sum(1 for line in f if line.strip())
                        dataset_info["train"] = {"n_samples": train_count}
                    else:
                        dataset_info["train"] = {"n_samples": "Unknown"}

                test_files = data_config.get("test_files", {})
                if isinstance(test_files, dict) and test_files:
                    for test_name, test_file in test_files.items():
                        if test_file and Path(test_file).exists():
                            with open(test_file, "r") as f:
                                test_count = sum(1 for line in f if line.strip())
                            dataset_info[test_name] = {"n_samples": test_count}
                        else:
                            dataset_info[test_name] = {"n_samples": "Unknown"}
                else:
                    val_file = data_config.get("validation_file")
                    if val_file and Path(val_file).exists():
                        with open(val_file, "r") as f:
                            val_count = sum(1 for line in f if line.strip())
                        dataset_info["c2"] = {"n_samples": val_count}

                    test_file = data_config.get("test_file")
                    if test_file and Path(test_file).exists():
                        with open(test_file, "r") as f:
                            test_count = sum(1 for line in f if line.strip())
                        dataset_info["c3"] = {"n_samples": test_count}
            except Exception as e:
                if hasattr(logger, "warning"):
                    logger.warning(f"Unable to get dataset statistics: {e}")
                dataset_info = {"error": f"Unable to load dataset info: {e}"}

        if hasattr(logger, "log_dataset_info"):
            logger.log_dataset_info(dataset_info)

        # Step 2: Create model
        if "model_config" in model_config and isinstance(model_config["model_config"], dict):
            from src.models.model_unit.model_factory import create_model_from_config

            model = create_model_from_config(
                {"model_config": model_config["model_config"], "embedding_dim": model_config.get("embedding_dim", data_config.get("embedding_dim", 1280))},
                device=device,
            )
            from src.models import print_model_architecture

            print_model_architecture(model, logger.logger if hasattr(logger, "logger") else None)
        elif "model_architecture_file" in model_config and model_config["model_architecture_file"]:
            architecture_file = model_config["model_architecture_file"]
            logger.info(f"Using YAML architecture config file: {architecture_file}")

            override_params = {"embedding_dim": model_config.get("embedding_dim", 1280)}
            model = create_yaml_model(config_path=architecture_file, device=device, **override_params)

            from src.models import print_model_architecture

            print_model_architecture(model, logger.logger if hasattr(logger, "logger") else None)
        else:
            required_params = ["embedding_dim", "pooling_type", "interaction_type", "classifier_type"]
            missing_params = [param for param in required_params if param not in model_config]
            if missing_params:
                raise ValueError(
                    f"Traditional model configuration lacks necessary parameters: {missing_params}. Please configure using YAML architecture or provide complete traditional configuration parameters."
                )

            model = create_avg_pool_mlp(
                embedding_dim=model_config["embedding_dim"],
                pooling_type=model_config["pooling_type"],
                interaction_type=model_config["interaction_type"],
                classifier_type=model_config["classifier_type"],
                device=device,
            )
            if hasattr(logger, "log_model_info"):
                logger.log_model_info(model)

        # Step 3: Initialize ModelManager and inference engine
        model_manager = ModelManager(output_dir, logger)
        inference_config = {
            "model": model_config,
            "inference": {"batch_size": training_config["batch_size"], "threshold": 0.5},
            "output": {"generate_confusion_matrix": False},
        }
        _ = InferenceEngine(inference_config, logger)  # constructed to mirror old path

        # Step 4: Train with validation
        logger.debug("Preparing to call training function")
        logger.debug(f"data_loaders type: {type(data_loaders)}")
        logger.debug(f"data_loaders keys: {list(data_loaders.keys()) if isinstance(data_loaders, dict) else 'Not a dict'}")

        epoch_callback = config.get("optuna", {}).get("epoch_callback", None)

        training_results = train_sequence_model_with_validation(
            model, data_loaders, config, device, logger, model_manager, _, epoch_callback
        )

        # Step 5: Final evaluation using best model
        final_evaluation = run_final_evaluation(
            model_manager.get_best_model_path(),
            model_manager.get_inference_config_path(),
            data_loaders,
            output_dir,
            device,
            logger,
        )

        # Step 6: Persist best config
        model_manager.create_best_inference_config(config)
        logger.info(f"Created best inference config file: {model_manager.output_dir}/best_config.yaml")

        # Step 7: Save training results (without model object)
        results_file = Path(output_dir) / "training_results.json"
        save_results = {
            "training_results": {k: v for k, v in training_results.items() if k != "model"},
            "final_evaluation": {
                k: {key: val for key, val in v.items() if key not in ["predictions", "true_labels"]}
                for k, v in (final_evaluation or {}).items()
            },
            "config": config,
            "best_epoch": model_manager.best_epoch,
            "best_metric": model_manager.best_metric,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Training finished! Results saved to: {output_dir}")

        return {
            "model": model,
            "device": device,
            "config": config,
            "training_results": training_results,
            "final_evaluation": final_evaluation,
            "data_loaders": data_loaders,
            "output_dir": output_dir,
            "logger": logger,
            "model_manager": model_manager,
        }
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def run_fusion_finetune_mode(config: Dict[str, Any], output_dir: str, logger) -> Dict[str, Any]:
    """Run finetune mode: load base checkpoint (trained without cross features),
    build target model (with feature_cross_transformer), load partial weights, freeze modules except specified, and train.

    Expects config['finetune'] = {
        'base_checkpoint': str,
        'freeze_except': [str, ...],
        'use_pretrained_embedding_dim': bool (default True),  # If False, use model.embedding_dim from config
    }
    """
    # Validate finetune section
    finetune_cfg = config.get("finetune", {})
    base_ckpt = finetune_cfg.get("base_checkpoint")
    if not base_ckpt or not Path(base_ckpt).exists():
        raise FileNotFoundError(f"Basic pre training weights/configurations do not exist: {base_ckpt}")
    freeze_except = finetune_cfg.get("freeze_except", ["preprocessing"])  # default
    use_pretrained_embedding_dim = finetune_cfg.get("use_pretrained_embedding_dim", True)

    # Determine embedding_dim: either from pretrained checkpoint or from current config
    if use_pretrained_embedding_dim:
        # Derive target embedding_dim from sibling best_config.yaml; fallback to resolved_config.yaml, then checkpoint
        try:
            base_path = Path(base_ckpt)
            if base_path.suffix.lower() in (".yml", ".yaml"):
                best_cfg_path = base_path
            else:
                best_cfg_path = base_path.with_name("best_config.yaml")
            alt_cfg_path = base_path.with_name("resolved_config.yaml")

            cfg_path_to_use = None
            if best_cfg_path.exists():
                cfg_path_to_use = best_cfg_path
            elif alt_cfg_path.exists():
                cfg_path_to_use = alt_cfg_path
                logger.info(f"best_config.yaml not found, using alternative config: {alt_cfg_path}")

            pretrained_embed_dim = None
            if cfg_path_to_use is not None:
                with open(cfg_path_to_use, "r", encoding="utf-8") as f:
                    _best_cfg = yaml.safe_load(f)
                pretrained_embed_dim = _best_cfg.get("model", {}).get("embedding_dim", None)
            else:
                # Fallback: try load embedding_dim from checkpoint's saved model_config
                try:
                    ckpt = torch.load(str(base_path), map_location="cpu")
                    if isinstance(ckpt, dict):
                        mc = ckpt.get("model_config", {}) or {}
                        pretrained_embed_dim = (
                            mc.get("embedding_dim")
                            or (mc.get("model_config", {}) or {}).get("embedding_dim")
                            or (mc.get("model", {}) or {}).get("embedding_dim")
                        )
                        if pretrained_embed_dim:
                            logger.info(f"Parsed embedding_dim from checkpoint metadata: {pretrained_embed_dim}")
                except Exception:
                    pass

            if not isinstance(pretrained_embed_dim, int):
                raise FileNotFoundError(
                    "Could not find best_config.yaml or resolved_config.yaml in the base model directory, and failed to parse embedding_dim from the checkpoint."
                    f" Tried paths: {best_cfg_path}, {alt_cfg_path}"
                )

            # Override current model embedding_dim to match pretrained model
            if "model" not in config:
                config["model"] = {}
            old_dim = config["model"].get("embedding_dim", None)
            config["model"]["embedding_dim"] = int(pretrained_embed_dim)
            logger.info(
                f"[Finetune] Using pretrained model target embedding_dim for current model: {old_dim} -> {pretrained_embed_dim} (source: {cfg_path_to_use or 'checkpoint'})"
            )
        except Exception as e:
            logger.error(f"Failed to parse pretrained best config to determine embedding_dim: {e}")
            raise
    else:
        # Use embedding_dim from current config (allows training input_layer with new output dimension)
        current_embed_dim = config.get("model", {}).get("embedding_dim")
        if current_embed_dim is None:
            raise ValueError(
                "[Finetune] use_pretrained_embedding_dim=False but model.embedding_dim is not set in config. "
                "Please specify model.embedding_dim in your config file."
            )
        logger.info(
            f"[Finetune] use_pretrained_embedding_dim=False, using config embedding_dim: {current_embed_dim} "
            "(input_layer will be reinitialized with new output dimension)"
        )

    # Set random seed
    seed = config["training"]["seed"]
    set_random_seed(seed)
    logger.debug(f"Random seed set to: {seed}")

    # Get device
    device = get_device()
    logger.info(f"Compute device: {device}")

    # Unpack configs
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    target_precision = data_config.get("target_precision", "fp32")
    log_config = {
        "Mode": "Fusion Finetune",
        "Embeddings": data_config.get("embedding_file", "N/A"),
        "Data": f"{target_precision}, gpu_cache={format_memory(data_config.get('cache_size', 10000) * model_config.get('embedding_dim', 1280) * 4)}",
        "Training": f"{training_config['epochs']} epochs, batch={training_config['batch_size']}, lr={training_config['learning_rate']}",
    }
    if hasattr(logger, "log_config"):
        logger.log_config(log_config, "Training configuration")

    # Step 1: Data loaders (reuse logic)
    logger.debug("[Finetune] Checking bucketing status and loading data...")
    if data_config["use_sequence_data"]:
        embedding_file = data_config["embedding_file"]
        is_cis_data = data_config.get("cis_type", False)

        if not is_cis_data and not _check_bucketed_lmdb(embedding_file, logger):
            logger.error("Detected a non-bucketed LMDB file!")
            logger.error("Please bucket the LMDB using the preprocessing script first:")
            logger.error(
                f"python emb_tools/preprocess_bucketed_lmdb.py --source-lmdb {embedding_file} --output-dir bucketed_embeddings/"
            )
            raise ValueError("A pre-bucketed LMDB file is required")
        elif is_cis_data:
            logger.info("Detected CIS-level data configuration, skipping bucketing check")

        if is_cis_data:
            from src.data_processing.cis_data_loader import create_cis_data_loaders

            cis_config = {
                "embedding_file": embedding_file,
                "train_file": data_config["train_file"],
                "test_files": {"c2": data_config["validation_file"], "c3": data_config["test_file"]},
                "batch_size": training_config["batch_size"],
                "cache_size": data_config.get("cache_size", 8000),
                # 使用数据源的嵌入维度来解析LMDB
                "embedding_dim": data_config["embedding_dim"],
                "target_precision": data_config.get("target_precision", "fp32"),
                "seed": seed,
            }

            cis_batch_data = create_cis_data_loaders(cis_config)
            data_loaders = cis_batch_data["data_loaders"]
        else:
            from src.data_processing.smart_batch_loader import create_smart_batch_data_loaders

            compat_config = {
                "embedding_file": embedding_file,
                "train_file": data_config["train_file"],
                "test_files": {"c2": data_config["validation_file"], "c3": data_config["test_file"]},
                "fasta_file": data_config["fasta_file"],
                "batch_size": training_config["batch_size"],
                "cache_size": data_config.get("cache_size", 8000),
                # Loader 以数据源维度解析LMDB，后续由 input_layer 对齐到模型维度
                "embedding_dim": data_config["embedding_dim"],
                "max_length": data_config.get("max_length", 1024),
                "pooling_type": _extract_pooling_type_from_config(model_config),
                "target_precision": data_config.get("target_precision"),
                "cis_type": data_config.get("cis_type", False),
                "seed": seed,
            }

            smart_batch_data = create_smart_batch_data_loaders(compat_config)
            data_loaders = smart_batch_data["data_loaders"]
    else:
        data_loaders = create_default_loaders()

    # Step 2: Build target model (with cross transformer as per YAML)
    if "model_config" in model_config and isinstance(model_config["model_config"], dict):
        from src.models.model_unit.model_factory import create_model_from_config
        model = create_model_from_config(
            {"model_config": model_config["model_config"], "embedding_dim": model_config.get("embedding_dim", data_config.get("embedding_dim", 1280))},
            device=device,
        )
        from src.models import print_model_architecture
        print_model_architecture(model, logger.logger if hasattr(logger, "logger") else None)
    elif "model_architecture_file" in model_config and model_config["model_architecture_file"]:
        architecture_file = model_config["model_architecture_file"]
        logger.info(f"[Finetune] Using YAML architecture config file: {architecture_file}")
        override_params = {"embedding_dim": model_config.get("embedding_dim", 1280)}
        model = create_yaml_model(config_path=architecture_file, device=device, **override_params)
        from src.models import print_model_architecture
        print_model_architecture(model, logger.logger if hasattr(logger, "logger") else None)
    else:
        raise ValueError("Finetune mode requires YAML/modular model configuration to inject feature_cross_transformer")

    # Step 3: Load pretrained weights partially and freeze
    _load_pretrained_weights(model, base_ckpt, logger)
    _freeze_model_modules(model, freeze_except=freeze_except, logger=logger)

    # Step 4: Initialize ModelManager and inference engine
    model_manager = ModelManager(output_dir, logger)
    inference_config = {
        "model": model_config,
        "inference": {"batch_size": training_config["batch_size"], "threshold": 0.5},
        "output": {"generate_confusion_matrix": False},
    }
    _ = InferenceEngine(inference_config, logger)

    # Step 5: Train with validation (reusing existing loop)
    logger.info("Starting fusion finetune training...")
    epoch_callback = config.get("optuna", {}).get("epoch_callback", None)
    training_results = train_sequence_model_with_validation(
        model, data_loaders, config, device, logger, model_manager, _, epoch_callback
    )

    # Step 6: Final evaluation
    final_evaluation = run_final_evaluation(
        model_manager.get_best_model_path(),
        model_manager.get_inference_config_path(),
        data_loaders,
        output_dir,
        device,
        logger,
    )

    # ---- 导出最佳模型的门控报告（若预处理支持） ----
    try:
        preproc = getattr(model, 'preprocessing', None)
        if preproc is not None and hasattr(preproc, 'compute_feature_gate_report'):
            # 从验证集采样一批 protein_ids 进行统计
            sample_loader = None
            if isinstance(data_loaders, dict):
                sample_loader = data_loaders.get('c2') or data_loaders.get('train') or next(iter(data_loaders.values()))
            protein_ids = []
            embeddings_batch = None
            attn_mask_batch = None
            if sample_loader is not None:
                for batch in sample_loader:
                    if 'protein_ids' in batch:
                        protein_ids = [pid[0] for pid in batch['protein_ids']]
                    # 可选：计算 e（输入到模型的投影前嵌入），这里直接取 batch 的 seq 作为近似
                    embeddings_batch = batch.get('protein1_seq', None)
                    attn_mask_batch = batch.get('protein1_mask', None)
                    if torch.is_tensor(embeddings_batch):
                        embeddings_batch = embeddings_batch.to(device)
                    if torch.is_tensor(attn_mask_batch):
                        attn_mask_batch = attn_mask_batch.to(device)
                    break
            try:
                report = preproc.compute_feature_gate_report(embeddings_batch, attn_mask_batch, protein_ids, max_samples=64)
                out_path = Path(output_dir) / 'best_gate_report.json'
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Gate report saved: {out_path}")
            except Exception as e:
                logger.warning(f"Gate report computation failed: {e}")
    except Exception:
        pass

    # ---- 融合微调：导出最佳模型的门控报告（若预处理支持） ----
    try:
        preproc = getattr(model, 'preprocessing', None)
        if preproc is not None and hasattr(preproc, 'compute_feature_gate_report'):
            sample_loader = None
            if isinstance(data_loaders, dict):
                sample_loader = data_loaders.get('c2') or data_loaders.get('train') or next(iter(data_loaders.values()))
            protein_ids = []
            embeddings_batch = None
            attn_mask_batch = None
            if sample_loader is not None:
                for batch in sample_loader:
                    if 'protein_ids' in batch:
                        protein_ids = [pid[0] for pid in batch['protein_ids']]
                    embeddings_batch = batch.get('protein1_seq', None)
                    attn_mask_batch = batch.get('protein1_mask', None)
                    if torch.is_tensor(embeddings_batch):
                        embeddings_batch = embeddings_batch.to(device)
                    if torch.is_tensor(attn_mask_batch):
                        attn_mask_batch = attn_mask_batch.to(device)
                    break
            try:
                report = preproc.compute_feature_gate_report(embeddings_batch, attn_mask_batch, protein_ids, max_samples=64)
                out_path = Path(output_dir) / 'best_gate_report.json'
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Gate report saved: {out_path}")
            except Exception as e:
                logger.warning(f"Gate report computation failed: {e}")
    except Exception:
        pass

    # Step 7: Persist best config
    model_manager.create_best_inference_config(config)
    logger.info(f"Created best inference config file: {model_manager.output_dir}/best_config.yaml")

    # Step 8: Save training results (without model object)
    results_file = Path(output_dir) / "training_results.json"
    save_results = {
        "training_results": {k: v for k, v in training_results.items() if k != "model"},
        "final_evaluation": {
            k: {key: val for key, val in v.items() if key not in ["predictions", "true_labels"]}
            for k, v in (final_evaluation or {}).items()
        },
        "config": config,
        "best_epoch": model_manager.best_epoch,
        "best_metric": model_manager.best_metric,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"Fusion finetune finished! Results saved to: {output_dir}")

    return {
        "model": model,
        "device": device,
        "config": config,
        "training_results": training_results,
        "final_evaluation": final_evaluation,
        "data_loaders": data_loaders,
        "output_dir": output_dir,
        "logger": logger,
        "model_manager": model_manager,
    }


def train_sequence_model_with_validation(
    model,
    data_loaders,
    config,
    device,
    logger,
    model_manager: ModelManager,
    inference_engine: InferenceEngine,  # unused, kept for signature compatibility
    epoch_callback=None,
):
    """Train model with epoch-by-epoch validation using the inference engine.

    epoch_callback: Optional callable(epoch:int, epoch_result:dict, best_metric:float) -> bool
    """
    logger.debug(f"data_loaders type: {type(data_loaders)}")
    logger.debug(f"data_loaders keys: {list(data_loaders.keys()) if isinstance(data_loaders, dict) else 'Not a dict'}")

    model.train()
    model.to(device)

    training_config = config["training"]

    # optimizer
    from src.models.train_model import create_optimizer

    optimizer_config = training_config.get("optimizer", {"type": "adam"})
    optimizer_type = optimizer_config.get("type", "adam")
    optimizer_kwargs = {k: v for k, v in optimizer_config.items() if k != "type"}
    optimizer = create_optimizer(
        model=model, optimizer_type=optimizer_type, learning_rate=training_config["learning_rate"], **optimizer_kwargs
    )

    # class imbalance handling
    class_imbalance_config = training_config.get("class_imbalance", {})
    if class_imbalance_config.get("enabled", False):
        if class_imbalance_config.get("weighted_bce", {}).get("enabled", False):
            weighted_bce_config = class_imbalance_config["weighted_bce"]
            if weighted_bce_config.get("auto_weight", False):
                total_samples = len(data_loaders["train"].dataset)
                positive_samples = 0
                import random

                random.seed(42)
                sample_indices = random.sample(range(total_samples), min(1000, total_samples))
                for i in sample_indices:
                    sample = data_loaders["train"].dataset[i]
                    if sample["label"] > 0.5:
                        positive_samples += 1
                positive_ratio = positive_samples / min(1000, total_samples)
                negative_ratio = 1 - positive_ratio
                positive_weight = negative_ratio / max(positive_ratio, 1e-6)
                logger.info(f"Auto-computed class weight: pos_ratio={positive_ratio:.3f}, pos_weight={positive_weight:.2f}")
                pos_weight = torch.tensor([positive_weight], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                positive_weight = class_imbalance_config["weighted_bce"].get("positive_weight", 10.0)
                pos_weight = torch.tensor([positive_weight], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                logger.info(f"Using manually set positive class weight: {positive_weight}")
        else:
            criterion = nn.BCELoss()
    else:
        criterion = nn.BCELoss()

    # gradient accumulation
    ga_cfg = training_config.get("gradient_accumulation", {})
    ga_enabled = ga_cfg.get("enabled", False)
    ga_steps = ga_cfg.get("steps", 1)

    # gradient clipping
    gc_cfg = training_config.get("gradient_clipping", {})
    gc_enabled = gc_cfg.get("enabled", False)
    max_grad_norm = gc_cfg.get("max_norm", 1.0)

    logger.info(f"Using optimizer: {optimizer_type}")
    if ga_enabled:
        logger.info(f"Gradient accumulation enabled: {ga_steps} steps")
        logger.info(f"Effective batch size: {training_config['batch_size'] * ga_steps}")
    if gc_enabled:
        logger.info(f"Gradient clipping enabled: max_norm={max_grad_norm}")

    # LR scheduler
    from src.models.train_model import create_lr_scheduler

    scheduler_config = training_config.get("lr_scheduler", {})
    if scheduler_config:
        scheduler_config["total_epochs"] = training_config["epochs"]
        scheduler = create_lr_scheduler(optimizer, scheduler_config)
        if scheduler:
            logger.info(f"LR scheduler: {scheduler_config.get('type', 'unknown')}")
            if scheduler_config.get("warmup", {}).get("enabled", False):
                warmup_epochs = scheduler_config["warmup"].get("epochs", 0)
                warmup_strategy = scheduler_config["warmup"].get("strategy", "exponential")
                logger.debug(f"  Warmup: {warmup_epochs} epochs, strategy: {warmup_strategy}")
        else:
            scheduler = None
            logger.warning("Failed to create LR scheduler; using fixed learning rate")
    else:
        scheduler = None
        logger.warning("No LR scheduler configured; using fixed learning rate")

    epochs = training_config["epochs"]
    verbose = config["logging"]["verbose"]
    logger.info(f"Starting training ({epochs} epochs)")

    epoch_losses = []
    epoch_results = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            data_loaders["train"], desc=f"Epoch {epoch + 1}/{epochs}", leave=False, disable=not verbose
        )

        batch_count = 0
        accumulation_count = 0
        optimizer.zero_grad()

        for batch in train_pbar:
            batch_count += 1
            accumulation_count += 1
            if batch_count <= 3:
                logger.debug(f"Processing batch {batch_count}")

            protein1_seq = batch["protein1_seq"].to(device)
            protein1_mask = batch["protein1_mask"].to(device)
            protein2_seq = batch["protein2_seq"].to(device)
            protein2_mask = batch["protein2_mask"].to(device)
            labels = batch["label"].to(device)

            protein_ids = None
            if "protein_ids" in batch:
                protein1_ids = [pid[0] for pid in batch["protein_ids"]]
                protein2_ids = [pid[1] for pid in batch["protein_ids"]]
                protein_ids = (protein1_ids, protein2_ids)

            predictions = model(
                protein1_seq, protein2_seq, protein1_mask, protein2_mask, protein_ids
            )

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(predictions.squeeze(), labels)
            else:
                loss = criterion(predictions.squeeze(), labels)

            # head-gating 熵正则（若模型支持）
            if hasattr(model, "get_head_entropy_regularization_loss"):
                gate_entropy_loss = model.get_head_entropy_regularization_loss()
                if gate_entropy_loss is not None:
                    loss = loss + gate_entropy_loss

            if ga_enabled:
                loss = loss / ga_steps

            loss.backward()

            if (not ga_enabled) or (accumulation_count >= ga_steps):
                if gc_enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count = 0

            total_loss += loss.item() * (ga_steps if ga_enabled else 1)
            num_batches += 1

            current_loss = total_loss / max(num_batches, 1)
            if scheduler is not None and hasattr(scheduler, "get_last_lr"):
                try:
                    _lrs = scheduler.get_last_lr()
                    current_lr = (
                        _lrs[0] if isinstance(_lrs, (list, tuple)) and len(_lrs) > 0 else optimizer.param_groups[0]["lr"]
                    )
                except Exception:
                    current_lr = optimizer.param_groups[0]["lr"]
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.6f}"})

        train_pbar.close()

        epoch_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(epoch_loss)

        # save last checkpoint
        model_manager.save_last_model(model, epoch + 1, training_config, config["model"])

        # epoch validation
        logger.debug(f"Starting epoch {epoch + 1} validation...")
        _ = model_manager.create_inference_config(config, model_manager.get_last_model_path(), epoch + 1)
        val_metrics, test_metrics = run_epoch_validation(
            model_manager.get_last_model_path(), model_manager.get_inference_config_path(), data_loaders, device, logger
        )

        current_val_metric = val_metrics.get("pr_auc", 0.0)
        is_best = model_manager.save_best_model(
            model, epoch + 1, current_val_metric, training_config, config["model"]
        )

        # scheduler step
        if scheduler is not None:
            try:
                if training_config.get("lr_scheduler", {}).get("type", "none") == "adaptive":
                    scheduler.step(current_val_metric)
                else:
                    scheduler.step()
            except Exception as _e:
                logger.warning(f"LR scheduler step failed: {_e}")
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug(f"Epoch {epoch + 1} LR: {current_lr:.6f}")

        if epoch == 0 and not Path(model_manager.get_best_model_path()).exists():
            logger.warning("First epoch validation failed; using training model as initial best model")
            model_manager.save_best_model(model, epoch + 1, 0.0, training_config, config["model"])

        epoch_time = time.time() - epoch_start
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            try:
                _lrs = scheduler.get_last_lr()
                current_lr = (
                    _lrs[0] if isinstance(_lrs, (list, tuple)) and len(_lrs) > 0 else optimizer.param_groups[0]["lr"]
                )
            except Exception:
                current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        if hasattr(logger, "log_epoch_results"):
            logger.log_epoch_results(
                epoch=epoch + 1,
                train_loss=epoch_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                learning_rate=current_lr,
                best_metric=model_manager.best_metric,
                best_epoch=model_manager.best_epoch,
                epoch_time=epoch_time,
            )

        current_epoch_result = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "epoch_time": epoch_time,
            "learning_rate": current_lr,
            "is_best": is_best,
        }
        epoch_results.append(current_epoch_result)

        # 每个 epoch 导出 head-gating 与注意力头相关性统计
        try:
            stats_loader = None
            if isinstance(data_loaders, dict):
                stats_loader = data_loaders.get("c2") or data_loaders.get("test") or data_loaders.get("train")
            if stats_loader is not None:
                epoch_stats = _collect_epoch_head_stats(model, stats_loader, device, logger)
                if epoch_stats:
                    if "gate_mean" in epoch_stats and "gate_std" in epoch_stats:
                        _append_gate_stats_tsv(
                            output_dir=str(model_manager.output_dir),
                            epoch=epoch + 1,
                            gate_mean=epoch_stats["gate_mean"],
                            gate_std=epoch_stats["gate_std"],
                        )
                    if "head_corr" in epoch_stats:
                        _append_head_corr_upper_tsv(
                            output_dir=str(model_manager.output_dir),
                            epoch=epoch + 1,
                            corr=epoch_stats["head_corr"],
                        )
        except Exception as e:
            logger.warning(f"Epoch head stats export failed at epoch {epoch + 1}: {e}")

        should_stop = False
        if epoch_callback:
            try:
                should_stop = epoch_callback(epoch + 1, current_epoch_result, model_manager.best_metric)
                if should_stop:
                    logger.info(f"Early stopping signal received; stopping at epoch {epoch + 1}")
                    break
            except Exception as e:
                logger.warning(f"Epoch callback execution failed: {e}")

    total_time = time.time() - start_time
    return {
        "epoch_losses": epoch_losses,
        "epoch_results": epoch_results,
        "total_training_time": total_time,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "best_val_metric": model_manager.best_metric,
        "best_epoch": model_manager.best_epoch,
        "model": model,
    }
