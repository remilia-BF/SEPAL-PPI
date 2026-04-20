"""
Ensemble runner utilities extracted from sepal-ppi.py to keep the entry script slim.

This module provides entry points:
- run_ensemble_inference_mode
- run_ensemble_predict_mode
- run_ensemble_training_mode

It depends on existing engines under src/ensemble and core utils.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json
import pickle

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# project imports
from src.utils.helpers import set_random_seed, get_device
from src.inference import InferenceEngine


def _extract_pooling_type_from_config(model_config: Dict[str, Any]) -> str:
    """Extract pooling type from model config, compatible with modular and traditional config.

    Returns: 'avg' | 'max' | 'attention'
    """
    # Prefer modular path: model_config.model_config.pooling.method
    try:
        if isinstance(model_config, dict) and isinstance(model_config.get("model_config"), dict):
            pooling_method = model_config["model_config"].get("pooling", {}).get("method", "average_pooling")
            mapping = {"average_pooling": "avg", "max_pooling": "max", "attention_pooling": "attention"}
            return mapping.get(pooling_method, "avg")
    except Exception:
        pass

    # Next: YAML architecture file (legacy path)
    try:
        if isinstance(model_config.get("model_architecture_file"), str) and model_config["model_architecture_file"]:
            import yaml

            config_path = Path(model_config["model_architecture_file"]).expanduser().resolve()
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f)
                method = y.get("model_config", {}).get("pooling", {}).get("method", "average_pooling")
                mapping = {"average_pooling": "avg", "max_pooling": "max", "attention_pooling": "attention"}
                return mapping.get(method, "avg")
    except Exception:
        pass

    # Fall back to traditional key
    return model_config.get("pooling_type", "avg")


def run_ensemble_inference_mode(config: Dict[str, Any], output_dir: str, logger) -> Dict[str, Any]:
    """Run ensemble inference mode using pre-trained meta-learner."""
    logger.info("=== Ensemble inference mode ===")

    # Get device
    device = get_device()
    logger.info(f"Compute device: {device}")

    # Extract ensemble configuration
    ensemble_config = config["ensemble"]
    meta_learner_path = ensemble_config["meta_learner_path"]
    models_config = ensemble_config["models"]

    # Load meta-learner
    logger.info(f"Loading meta-learner: {meta_learner_path}")
    with open(meta_learner_path, "rb") as f:
        meta_learner_data = pickle.load(f)

    meta_learner = meta_learner_data["meta_learner"]
    scaler = meta_learner_data["scaler"]
    model_names = meta_learner_data["model_names"]

    # Training mode of meta-learner
    training_mode = ensemble_config.get("meta_learner_info", {}).get("training_mode", "logits")
    logger.info(f"Meta-learner training mode: {training_mode}")
    logger.info(f"Meta-learner loaded, supported models: {model_names}")

    # Load individual models
    inference_engines = []
    for model_info in models_config:
        model_name = model_info["model_name"]
        model_path = model_info["model_path"]

        logger.info(f"Loading model {model_name}: {model_path}")
        temp_config = {
            "mode": "inference",
            "model": model_info["model_config"],
            "data": model_info["data_config"],
            "inference": config.get("inference", {}),
            "output": config.get("output", {}),
        }
        engine = InferenceEngine(temp_config, logger)
        engine.load_model(model_path, device=device)
        inference_engines.append(engine)
        logger.info(f"Model {model_name} loaded")

    logger.info(f"All {len(inference_engines)} models loaded")

    data_config = config["data"]
    logger.info("Creating datasets for each model")

    # Collect predictions from all models
    all_predictions: Dict[str, Dict[str, Any]] = {}

    available_test_files = dict(data_config.get("test_files", {}))
    if not available_test_files:
        if data_config.get("train_file"):
            available_test_files["c1"] = data_config["train_file"]
        if data_config.get("validation_file"):
            available_test_files["c2"] = data_config["validation_file"]
        if data_config.get("test_file"):
            available_test_files["c3"] = data_config["test_file"]

    for dataset_name in available_test_files.keys():
        logger.info(f"Processing dataset: {dataset_name}")
        all_predictions[dataset_name] = {}

        for (engine, model_info) in zip(inference_engines, models_config):
            model_name = model_info["model_name"]
            logger.info(f"  Running inference with model {model_name}...")

            model_data_config = model_info["data_config"]
            model_is_cis = model_data_config.get("cis_type", False)

            model_test_file = None
            if "test_files" in model_data_config and dataset_name in model_data_config["test_files"]:
                model_test_file = model_data_config["test_files"][dataset_name]
            elif dataset_name == "c2" and model_data_config.get("validation_file"):
                model_test_file = model_data_config["validation_file"]
            elif dataset_name == "c3" and model_data_config.get("test_file"):
                model_test_file = model_data_config["test_file"]
            else:
                model_test_file = available_test_files.get(dataset_name)

            if not model_test_file:
                logger.warning(f"Model {model_name} has no file path for dataset {dataset_name}")
                continue

            if model_is_cis:
                from src.data_processing.cis_data_loader import create_cis_data_loaders

                model_test_files = {dataset_name: model_test_file}
                model_cis_config = {
                    "embedding_file": model_data_config["embedding_file"],
                    "test_files": model_test_files,
                    "batch_size": config["inference"]["batch_size"],
                    "cache_size": model_data_config.get("cache_size", 8000),
                    # Parse LMDB with the model's source embedding dimension
                    "embedding_dim": model_data_config.get("embedding_dim", model_info["model_config"].get("embedding_dim", 1280)),
                    "target_precision": model_data_config.get("target_precision", "fp32"),
                }
                try:
                    model_cis_batch_data = create_cis_data_loaders(model_cis_config)
                    model_data_loaders = model_cis_batch_data["data_loaders"]
                except Exception as e:
                    logger.error(f"Failed to create CIS data loader for model {model_name} during ensemble inference: {e}")
                    continue

                if dataset_name in model_data_loaders and hasattr(model_data_loaders[dataset_name], "dataset"):
                    model_dataset = model_data_loaders[dataset_name].dataset
                else:
                    logger.warning(f"Unable to create dataset {dataset_name} for CIS model {model_name}")
                    continue
            else:
                from src.data_processing.smart_batch_loader import create_smart_batch_data_loaders

                model_test_files = {dataset_name: model_test_file}
                model_compat_config = {
                    "embedding_file": model_data_config["embedding_file"],
                    "test_files": model_test_files,
                    "fasta_file": model_data_config["fasta_file"],
                    "batch_size": config["inference"]["batch_size"],
                    "cache_size": model_data_config.get("cache_size", 8000),
                    # Parse LMDB with the model's source embedding dimension
                    "embedding_dim": model_data_config.get("embedding_dim", model_info["model_config"].get("embedding_dim", 1280)),
                    "max_length": model_data_config.get("max_length", 1024),
                    "pooling_type": _extract_pooling_type_from_config(model_info["model_config"]),
                    "target_precision": model_data_config.get("target_precision"),
                    "cis_type": model_data_config.get("cis_type", False),
                }
                model_smart_batch_data = create_smart_batch_data_loaders(model_compat_config)
                model_data_loaders = model_smart_batch_data["data_loaders"]

                if dataset_name in model_data_loaders and hasattr(model_data_loaders[dataset_name], "dataset"):
                    model_dataset = model_data_loaders[dataset_name].dataset
                else:
                    logger.warning(f"Unable to create dataset {dataset_name} for model {model_name}")
                    continue

            results = engine.evaluate_datasets({dataset_name: model_dataset}, show_progress=True, save_results=False, return_logits=True)
            if dataset_name in results:
                result = results[dataset_name]
                all_predictions[dataset_name][model_name] = {
                    "logits": result.get("predictions", []),
                    "labels": result.get("true_labels", []),
                    "protein_ids": result.get("protein_ids", []),
                }
                logger.info(f"    Collected predictions for {len(all_predictions[dataset_name][model_name]['logits'])} samples")

    # Ensemble prediction
    logger.info("Running ensemble prediction...")
    ensemble_results: Dict[str, pd.DataFrame] = {}

    for dataset_name, model_data in all_predictions.items():
        logger.info(f"Processing dataset: {dataset_name}")
        model_logits: List[np.ndarray] = []
        true_labels = None
        protein_ids = None

        for model_name in model_names:
            if model_name in model_data:
                logits = np.array(model_data[model_name]["logits"])
                model_logits.append(logits)
                if true_labels is None:
                    true_labels = np.array(model_data[model_name]["labels"])  # type: ignore
                    protein_ids = model_data[model_name]["protein_ids"]

        if len(model_logits) != len(model_names):
            logger.warning(f"Missing predictions from some models in dataset {dataset_name}; skipping")
            continue

        X_test = np.column_stack(model_logits)

        if X_test.shape[1] != scaler.n_features_in_:
            logger.warning(
                f"Feature dimension mismatch: current={X_test.shape[1]}, expected={scaler.n_features_in_}"
            )
            logger.warning(f"Meta-learner training mode: {training_mode}, inference uses logits mode")
            if training_mode == "features" and X_test.shape[1] < scaler.n_features_in_:
                padding = np.zeros((X_test.shape[0], scaler.n_features_in_ - X_test.shape[1]))
                X_test = np.column_stack([X_test, padding])
                logger.info(f"Feature dimension adjusted: {X_test.shape[1]}")
            elif X_test.shape[1] > scaler.n_features_in_:
                X_test = X_test[:, : scaler.n_features_in_]
                logger.info(f"Feature dimension adjusted: {X_test.shape[1]}")
            else:
                logger.error("Cannot match feature dimensions; check model configuration")
                continue

        X_test_scaled = scaler.transform(X_test)

        ensemble_proba = meta_learner.predict_proba(X_test_scaled)[:, 1]
        ensemble_pred = meta_learner.predict(X_test_scaled)

        # Weights (only applicable to linear models)
        weights = getattr(meta_learner, "coef_", [[0] * len(model_names)])[0]

        if protein_ids and len(protein_ids) > 0:
            protein1_list: List[str] = []
            protein2_list: List[str] = []
            for pid in protein_ids:
                if isinstance(pid, (list, tuple)) and len(pid) >= 2:
                    protein1_list.append(pid[0])
                    protein2_list.append(pid[1])
                else:
                    pid_str = str(pid)
                    if "_" in pid_str:
                        parts = pid_str.split("_", 1)
                        protein1_list.append(parts[0])
                        protein2_list.append(parts[1])
                    else:
                        protein1_list.append(f"protein1_{len(protein1_list)}")
                        protein2_list.append(f"protein2_{len(protein2_list)}")
        else:
            protein1_list = [f"protein1_{i}" for i in range(len(true_labels))]
            protein2_list = [f"protein2_{i}" for i in range(len(true_labels))]

        result_data = {
            "protein1": protein1_list,
            "protein2": protein2_list,
            "true_label": true_labels,
            "ensemble_probability": ensemble_proba,
            "ensemble_prediction": ensemble_pred,
        }
        for i, model_name in enumerate(model_names):
            model_proba = torch.sigmoid(torch.tensor(model_logits[i])).numpy()
            result_data[f"{model_name}_probability"] = model_proba
            result_data[f"{model_name}_weight"] = [weights[i]] * len(model_proba)

        ensemble_results[dataset_name] = pd.DataFrame(result_data)

        auc = roc_auc_score(true_labels, ensemble_proba)
        ap = average_precision_score(true_labels, ensemble_proba)
        acc = accuracy_score(true_labels, ensemble_pred)
        f1 = f1_score(true_labels, ensemble_pred)
        logger.info(f"{dataset_name} ensemble performance: AUC={auc:.4f}, AP={ap:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Save results
    logger.info("Saving ensemble inference results...")
    for dataset_name, df in ensemble_results.items():
        output_file = Path(output_dir) / f"{dataset_name}_ensemble_predictions.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {dataset_name} predictions to: {output_file}")

    return {"ensemble_results": ensemble_results, "config": config, "output_dir": output_dir}


def run_ensemble_predict_mode(config: Dict[str, Any], output_dir: str, logger) -> Dict[str, Any]:
    """Run ensemble predict mode for new protein pairs."""
    logger.info("=== Ensemble predict mode ===")
    from src.ensemble.ensemble_predict_engine import EnsemblePredictEngine

    engine = EnsemblePredictEngine(config, output_dir, logger)
    results = engine.run_ensemble_predict()
    return results


def run_ensemble_training_mode(
    config_paths: List[str],
    output_dir: str,
    logger,
    device,
    seed: int = 42,
    mode: str = "logits",
    train_model: str = "linear",
    model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run ensemble training mode."""
    from src.ensemble.ensemble_engine import EnsembleInferenceEngine
    import yaml

    # Set random seed
    set_random_seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save ensemble training configuration
    ensemble_config = {
        "config_paths": config_paths,
        "seed": seed,
        "mode": mode,
        "train_model": train_model,
        "model_params": model_params or {},
        "output_dir": str(output_path),
    }

    config_save_path = output_path / "ensemble_training_config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(ensemble_config, f, default_flow_style=False, allow_unicode=True, indent=2)

    logger.info(f"Ensemble training config saved to: {config_save_path}")

    try:
        # Initialize engine
        logger.info("Initializing ensemble inference engine...")
        ensemble_engine = EnsembleInferenceEngine(config_paths=config_paths, output_dir=str(output_path), logger=logger)

        logger.info("Ensemble inference engine will automatically handle data loading...")
        logger.info(f"Starting ensemble training (mode={mode}, model={train_model})...")

        # Collect features or logits
        all_data = ensemble_engine.collect_logits(mode=mode)

        # Train meta-learner
        ensemble_engine.train_meta_learner(all_data, mode=mode, model_type=train_model)

        # Ensemble prediction
        ensemble_results = ensemble_engine.ensemble_predict(all_data, mode=mode)

        results = {
            "ensemble_results": ensemble_results,
            "meta_learner_info": {
                "mode": mode,
                "train_model": train_model,
                "model_params": model_params or {},
            },
        }

        # 保存训练结果（序列化 DataFrame）
        results_path = output_path / "ensemble_training_results.json"
        serializable_results: Dict[str, Any] = {
            "meta_learner_info": results["meta_learner_info"],
            "ensemble_results": {},
        }
        for dataset_name, df in results["ensemble_results"].items():
            serializable_results["ensemble_results"][dataset_name] = {
                "ensemble_prediction": df["ensemble_prediction"].tolist(),
                "ensemble_label": df["ensemble_label"].tolist(),
                "true_label": df["true_label"].tolist(),
            }
            for col in df.columns:
                if col.endswith("_prediction") and col != "ensemble_prediction":
                    serializable_results["ensemble_results"][dataset_name][col] = df[col].tolist()

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Ensemble training results saved to: {results_path}")

        # Generate final_evaluation_summary.json for collect_metrics.py
        logger.info("Generating evaluation summary for metric collection...")
        evaluation_summary = {}
        
        for dataset_name, df in results["ensemble_results"].items():
            if "true_label" not in df.columns:
                continue
            
            y_true = df["true_label"].to_numpy()
            
            # Handle probability column names which differ between inference and training engines
            if "ensemble_probability" in df.columns:
                y_score = df["ensemble_probability"].to_numpy()
            elif "ensemble_prediction" in df.columns:
                # In ensemble_engine.py (used in training), 'ensemble_prediction' stores probabilities
                y_score = df["ensemble_prediction"].to_numpy()
            else:
                continue

            # Handle binary prediction column names
            if "ensemble_label" in df.columns:
                # In ensemble_engine.py (used in training), 'ensemble_label' stores binary predictions
                y_pred = df["ensemble_label"].to_numpy()
            elif "ensemble_prediction" in df.columns and "ensemble_probability" in df.columns:
                # In inference mode, 'ensemble_prediction' stores binary predictions
                y_pred = df["ensemble_prediction"].to_numpy()
            else:
                y_pred = (y_score >= 0.5).astype(int)
            
            # Check if we have valid labels (mix of 0 and 1)
            auc = None
            ap = None
            if len(np.unique(y_true)) >= 2:
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                    ap = float(average_precision_score(y_true, y_score))
                except Exception:
                    pass

            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except ValueError:
                # Handle cases where confusion matrix might not return 4 values (e.g. only 1 class predicted)
                tn, fp, fn, tp = 0, 0, 0, 0
                # Fallback if needed, though ravel typically works if both classes present in y_true/y_pred combined
            
            metrics = {
                "dataset_name": dataset_name,
                "roc_auc": auc,
                "pr_auc": ap,
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "threshold": 0.5,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "n_samples": int(len(y_true)),
                "n_positive": int(sum(y_true)),
                "n_negative": int(len(y_true) - sum(y_true)),
                "positive_rate": float(np.mean(y_true)),
                "protein_ids": []
            }

            # Populate protein_ids if columns exist
            if "protein1" in df.columns and "protein2" in df.columns:
                p1s = df["protein1"].tolist()
                p2s = df["protein2"].tolist()
                
                for i in range(len(y_true)):
                    prob = float(y_score[i])
                    # Calculate logit: log(p / (1-p)) - avoid division by zero
                    epsilon = 1e-15
                    clipped_prob = max(epsilon, min(1 - epsilon, prob))
                    logit = np.log(clipped_prob / (1 - clipped_prob))
                    
                    metrics["protein_ids"].append({
                        "protein1_id": str(p1s[i]),
                        "protein2_id": str(p2s[i]),
                        "probability": prob,
                        "logit": float(logit),
                        "label": int(y_true[i])
                    })
            
            evaluation_summary[dataset_name] = metrics

        summary_path = output_path / "final_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Final evaluation summary saved to: {summary_path}")

        # Generate inference config
        logger.info("Generating ensemble inference config file...")
        inference_config = ensemble_engine.generate_inference_config(training_mode=mode)
        inference_config_path = output_path / "ensemble_inference_config.yaml"
        with open(inference_config_path, "w", encoding="utf-8") as f:
            yaml.dump(inference_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        logger.info(f"Ensemble inference config generated: {inference_config_path}")

        # Summary
        logger.info("=== Ensemble training summary ===")
        logger.info(f"Number of models: {len(config_paths)}")
        logger.info(f"Meta-learner mode: {mode}")
        logger.info(f"Meta-learner type: {train_model}")

        if "cv_results" in results:
            cv_results = results["cv_results"]
            try:
                logger.info(f"Cross-val AUC: {cv_results.get('auc', float('nan')):.4f}")
                logger.info(f"Cross-val AP: {cv_results.get('ap', float('nan')):.4f}")
                logger.info(f"Cross-val F1: {cv_results.get('f1', float('nan')):.4f}")
            except Exception:
                pass

        if "test_results" in results:
            test_results = results["test_results"]
            try:
                logger.info(f"Test AUC: {test_results.get('auc', float('nan')):.4f}")
                logger.info(f"Test AP: {test_results.get('ap', float('nan')):.4f}")
                logger.info(f"Test F1: {test_results.get('f1', float('nan')):.4f}")
            except Exception:
                pass

        return results
    except Exception as e:
        logger.error(f"Ensemble training failed: {str(e)}")
        raise
