"""
Unified inference engine for SEPAL-PPI
Handles both validation during training and standalone inference
"""

import torch
import torch.utils.data
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy


class InferenceEngine:
    """Unified inference engine for model evaluation and prediction"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize inference engine
        
        Args:
            config: Inference configuration dictionary
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger
        self.device: Optional[torch.device] = None
        self.model: Optional[torch.nn.Module] = None
    
    def load_model(self, model_path: Optional[str] = None, model: Optional[torch.nn.Module] = None, device=None):
        """
        Load model for inference
        
        Args:
            model_path: Path to saved model file
            model: Pre-loaded model instance
            device: Computing device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is not None:
            # Use provided model (for validation during training)
            self.model = model
            self.model.to(self.device)
        elif model_path:
            # Load model from file (for standalone inference)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Check if checkpoint contains model config
            if 'model_config' in checkpoint:
                saved_model_config = checkpoint['model_config']
                if self.logger:
                    self.logger.debug(f"Read model config from checkpoint")
            else:
                saved_model_config = self.config.get('model', {})
                if self.logger:
                    self.logger.debug(f"Using model config from inference configuration")
            
            # 统一确定 embedding_dim：优先使用运行时配置，其次使用数据配置，最后回退到checkpoint
            def _resolve_embedding_dim(saved_cfg: Dict[str, Any]) -> int:
                runtime_model_dim = (self.config.get('model', {}) or {}).get('embedding_dim')
                runtime_data_dim = (self.config.get('data', {}) or {}).get('embedding_dim')
                saved_dim = saved_cfg.get('embedding_dim')
                # Priority: runtime model > runtime data > checkpoint
                chosen = runtime_model_dim or runtime_data_dim or saved_dim or 1280
                # Warn if there is a conflict
                if self.logger and saved_dim and runtime_model_dim and saved_dim != runtime_model_dim:
                    self.logger.warning(
                        f"Detected mismatch between checkpoint and runtime embedding_dim: ckpt={saved_dim}, runtime={runtime_model_dim}; using runtime value"
                    )
                return int(chosen)

            # 如果运行时配置中提供了 model_config.preprocessing.data_processing，
            # 则在不破坏 checkpoint 结构的前提下进行浅层覆盖（例如 all_feature_folder 覆盖）。
            try:
                runtime_model_cfg = self.config.get('model') or {}
                if isinstance(runtime_model_cfg, dict) and isinstance(runtime_model_cfg.get('model_config'), dict):
                    rt_preproc = runtime_model_cfg['model_config'].get('preprocessing')
                    if isinstance(rt_preproc, dict):
                        smc = saved_model_config.setdefault('model_config', {}) if isinstance(saved_model_config, dict) else None
                        if isinstance(smc, dict):
                            sm_preproc = smc.get('preprocessing', {}) or {}
                            # 仅对 data_processing 做键级别更新
                            rt_dp = rt_preproc.get('data_processing')
                            if isinstance(rt_dp, dict):
                                sm_dp = sm_preproc.get('data_processing', {}) or {}
                                sm_dp.update(rt_dp)
                                sm_preproc['data_processing'] = sm_dp
                            smc['preprocessing'] = sm_preproc
                            saved_model_config['model_config'] = smc
                            if self.logger:
                                self.logger.debug("Applied runtime preprocessing overrides to saved model_config")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to apply runtime preprocessing overrides: {e}")

            # 调试：在主 logger 上报告最终生效的多模态特征根目录，方便确认 Human / rice
            if self.logger and isinstance(saved_model_config, dict) and isinstance(saved_model_config.get('model_config'), dict):
                try:
                    final_dp = (
                        saved_model_config
                        .get('model_config', {})
                        .get('preprocessing', {})
                        .get('data_processing', {})
                    ) or {}
                    all_feature_folder = final_dp.get('all_feature_folder', None)
                    if all_feature_folder is not None:
                        self.logger.debug(
                            f"InferenceEngine using multimodal feature root directory all_feature_folder={all_feature_folder}"
                        )
                except Exception:
                    # 仅为调试输出，不影响正常推理
                    pass

            # 优先使用组合后的模块化配置
            if 'model_config' in saved_model_config and isinstance(saved_model_config['model_config'], dict):
                from src.models.model_unit.model_factory import create_model_from_config
                embedding_dim = _resolve_embedding_dim(saved_model_config)
                # Force align input_layer.input_dim with runtime data's embedding_dim to avoid 640/5120 mismatch
                runtime_input_dim = (self.config.get('data', {}) or {}).get('embedding_dim', embedding_dim)
                model_cfg = copy.deepcopy(saved_model_config['model_config'])
                if not isinstance(model_cfg.get('input_data'), dict):
                    model_cfg['input_data'] = {}
                prev_input_dim = model_cfg['input_data'].get('input_dim')
                model_cfg['input_data']['input_dim'] = int(runtime_input_dim)
                if self.logger and (prev_input_dim is None or int(prev_input_dim) != int(runtime_input_dim)):
                    self.logger.warning(
                        f"Override input_layer.input_dim: {prev_input_dim} -> {runtime_input_dim} to match runtime LMDB dimension"
                    )
                self.model = create_model_from_config({'model_config': model_cfg, 'embedding_dim': embedding_dim}, device=self.device)
            # 其次使用YAML架构配置
            elif 'model_architecture_file' in saved_model_config and saved_model_config['model_architecture_file']:
                # 使用YAML配置系统创建模型
                from src.models import create_yaml_model
                architecture_file = saved_model_config['model_architecture_file']
                
                if self.logger:
                    self.logger.debug(f"Create model using YAML architecture file: {architecture_file}")
                
                # 创建覆盖参数（优先使用运行时配置，以避免LMDB与权重维度不一致）
                embedding_dim = _resolve_embedding_dim(saved_model_config)
                override_params = {
                    'embedding_dim': embedding_dim
                }
                
                self.model = create_yaml_model(
                    config_path=architecture_file,
                    device=self.device,
                    **override_params
                )
            else:
                # 使用传统方法创建模型 (向后兼容)
                from src.models import create_avg_pool_mlp
                
                if self.logger:
                    self.logger.debug(f"Create model using legacy configuration")
                
                # 从保存的配置中提取参数（优先运行时配置），避免维度不一致
                embedding_dim = _resolve_embedding_dim(saved_model_config)
                self.model = create_avg_pool_mlp(
                    embedding_dim=embedding_dim,
                    pooling_type=saved_model_config.get('pooling_type', 'avg'),
                    interaction_type=saved_model_config.get('interaction_type', 'hadamard'),
                    classifier_type=saved_model_config.get('classifier_type', 'standard'),
                    device=self.device
                )
            
            # Load state dict（允许 best_model 去除 input_layer 的情况）
            state = checkpoint.get('model_state_dict', checkpoint)

            # 如果检测到去除了 input_layer，尝试加载同目录 input_layer.pth 并先注入到模型
            try:
                stripped_flag = False
                try:
                    stripped_flag = bool(checkpoint.get('meta', {}).get('stripped_input_layer', False))
                except Exception:
                    stripped_flag = False
                has_input_keys = any(str(k).startswith('input_layer.') for k in state.keys())
                if (stripped_flag or not has_input_keys) and hasattr(self, 'config'):
                    ckpt_path = Path(model_path)
                    input_layer_path = ckpt_path.with_name('input_layer.pth')
                    if input_layer_path.exists() and hasattr(self.model, 'input_layer'):
                        try:
                            # 显式设置 weights_only=False，保持兼容旧权重格式，并抑制 PyTorch 的 FutureWarning
                            il_ckpt = torch.load(str(input_layer_path), map_location=self.device, weights_only=False)
                            il_state = il_ckpt.get('model_state_dict', il_ckpt)
                            self.model.input_layer.load_state_dict(il_state, strict=False)
                            if self.logger:
                                self.logger.info(f"Loaded input layer weights from same directory: {input_layer_path}")
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Failed to load input layer weights: {e}")
            except Exception:
                pass

            # 加载时进行形状过滤，避免不同配置间的 shape mismatch 报错
            try:
                model_state = self.model.state_dict()
                filtered_state = {k: v for k, v in state.items() if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape}
                self.model.load_state_dict(filtered_state, strict=False)
            except TypeError:
                # 兼容老版本 PyTorch 接口
                self.model.load_state_dict(state)
            self.model.to(self.device)
            
            if self.logger:
                stripped = bool(checkpoint.get('meta', {}).get('stripped_input_layer', False)) if isinstance(checkpoint, dict) else False
                self.logger.debug(f"Loaded model: {model_path} (stripped_input_layer={stripped})")
        else:
            raise ValueError("必须提供model_path或model参数")
        
        # Verify model is loaded
        if self.model is None:
            raise RuntimeError("模型加载失败")
    
    def _get_pooling_type_from_model_config(self) -> str:
        """
        从模型配置中获取池化类型
        
        Returns:
            str: 池化类型 ('avg', 'max', 'attention')
        """
        model_config = self.config.get('model', {})
        
        # 检查是否使用YAML架构配置
        if 'model_architecture_file' in model_config and model_config['model_architecture_file']:
            # Read pooling config from YAML architecture file
            try:
                import yaml
                from pathlib import Path
                
                architecture_file = model_config['model_architecture_file']
                config_path = Path(architecture_file)
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                    
                    pooling_method = yaml_config.get('model_config', {}).get('pooling', {}).get('method', 'average_pooling')
                    
                    # Map YAML method to data loader pooling_type
                    method_mapping = {
                        'average_pooling': 'avg',
                        'max_pooling': 'max', 
                        'attention_pooling': 'attention'
                    }
                    
                    return method_mapping.get(pooling_method, 'avg')
                else:
                    if self.logger:
                        self.logger.warning(f"YAML architecture file not found: {architecture_file}, using default pooling type")
                    return 'avg'
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to read YAML architecture file: {e}, using default pooling type")
                return 'avg'
        else:
            # 使用传统配置格式
            return model_config.get('pooling_type', 'avg')

    def _get_feature_concat_preprocessor(self):
        """获取模型中的 FeatureConcat 预处理器（若存在）。"""
        model_ref = self.model
        if model_ref is None:
            return None

        # 兼容 DataParallel / DDP
        if hasattr(model_ref, 'module'):
            model_ref = model_ref.module

        preprocessor = getattr(model_ref, 'preprocessing', None)
        if preprocessor is None:
            return None

        if hasattr(preprocessor, 'reset_feature_gating_stats') and hasattr(preprocessor, 'get_feature_gating_stats'):
            return preprocessor
        return None
    
    def evaluate_datasets(self, datasets: Dict[str, Any], 
                         show_progress: bool = False, 
                         save_results: bool = True,
                         output_dir: Optional[str] = None,
                         return_logits: bool = False,
                         return_features: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on multiple datasets
        
        Args:
            datasets: Dictionary of datasets to evaluate
            show_progress: Whether to show progress bar
            save_results: Whether to save evaluation results
            output_dir: Output directory for results
            return_logits: Whether to return raw logits instead of probabilities
            return_features: Whether to return last layer features instead of logits
            
        Returns:
            Dictionary of evaluation results
        """
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用load_model方法")
        
        self.model.eval()
        evaluation_results = {}
        
        for dataset_name, dataset in datasets.items():
            if self.logger and not show_progress:
                self.logger.debug(f"Evaluating dataset: {dataset_name}")
            
            results = self._evaluate_single_dataset(
                dataset, dataset_name, show_progress, return_logits, return_features, save_results, output_dir
            )
            evaluation_results[dataset_name] = results
        
        # Save results if requested
        if save_results and output_dir:
            self._save_evaluation_results(evaluation_results, output_dir)
        
        return evaluation_results
    
    def _evaluate_single_dataset(self, dataset, dataset_name: str, 
                                show_progress: bool = False,
                                return_logits: bool = False,
                                return_features: bool = False,
                                save_results: bool = True,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate model on a single dataset"""
        gating_preprocessor = self._get_feature_concat_preprocessor()
        if gating_preprocessor is not None:
            try:
                gating_preprocessor.reset_feature_gating_stats(dataset_name=dataset_name)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to reset feature gating stats for {dataset_name}: {e}")

        all_predictions = []
        all_labels = []
        all_protein_ids = []
        all_features = [] if return_features else None
        
        # 为流式写入注意力权重准备文件
        attention_file_handle = None
        attention_sample_count = 0
        if save_results and self._should_collect_attention_weights():
            # 使用传递的output_dir参数，如果没有则使用配置中的
            if output_dir:
                attention_output_dir = Path(output_dir)
            else:
                attention_output_dir = Path(self.config.get('output', {}).get('output_dir', './results'))
            attention_file_path = attention_output_dir / f"{dataset_name}_attention_weights.jsonl"
            attention_file_handle = open(attention_file_path, 'w', encoding='utf-8')
        
        # Create DataLoader with appropriate collator
        batch_size = self.config.get('inference', {}).get('batch_size', 32)
        
        # Check if dataset uses smart batch format (has 'protein1_embedding' key)
        sample = dataset[0] if len(dataset) > 0 else {}
        if 'protein1_embedding' in sample:
            # Use smart collate function
            from src.data_processing.smart_batch_loader import smart_collate_fn
            
            def create_collate_fn():
                # 获取池化类型，使用与训练时相同的函数
                pooling_type = self._get_pooling_type_from_model_config()
                max_length = self.config.get('data', {}).get('max_length', 1024)
                
                def collate_fn(batch):
                    return smart_collate_fn(batch, pooling_type=pooling_type, max_length=max_length)
                return collate_fn
            
            eval_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=create_collate_fn(),
                num_workers=0, pin_memory=False
            )
        else:
            # Use traditional bucketed collator
            from src.data_processing.sequence_dataset import BucketedCollator
            if self.device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            collator = BucketedCollator(self.device)
            eval_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
            )
        
        # Evaluation loop
        with torch.no_grad():
            if show_progress:
                eval_pbar = tqdm(eval_loader, desc=f"Evaluating {dataset_name}", leave=False)
                iterator = eval_pbar
            else:
                iterator = eval_loader
                
            for batch in iterator:
                # Handle different batch formats
                if 'protein1_seq' in batch:
                    # Traditional format
                    protein1_seq = batch['protein1_seq']
                    protein1_mask = batch['protein1_mask']
                    protein2_seq = batch['protein2_seq']
                    protein2_mask = batch['protein2_mask']
                    labels = batch['label']
                elif 'protein1_embedding' in batch:
                    # Smart batch format
                    protein1_seq = batch['protein1_embedding']
                    protein1_mask = batch['protein1_mask']
                    protein2_seq = batch['protein2_embedding']
                    protein2_mask = batch['protein2_mask']
                    labels = batch['label']
                else:
                    if self.logger:
                        self.logger.error(f"Unrecognized batch format, available keys: {list(batch.keys())}")
                    continue
                
                # Extract protein IDs if available
                protein_ids = None
                batch_protein_ids = []
                
                if 'protein_ids' in batch:
                    # 传统格式：直接包含protein_ids
                    protein1_ids = [pid[0] for pid in batch['protein_ids']]
                    protein2_ids = [pid[1] for pid in batch['protein_ids']]
                    protein_ids = (protein1_ids, protein2_ids)
                    batch_protein_ids = batch['protein_ids']
                    all_protein_ids.extend(batch_protein_ids)
                elif 'metadata' in batch and 'protein1_ids' in batch['metadata']:
                    # 智能批次格式：protein_ids在metadata中
                    protein1_ids = batch['metadata']['protein1_ids']
                    protein2_ids = batch['metadata']['protein2_ids']
                    protein_ids = (protein1_ids, protein2_ids)
                    batch_protein_ids = [(p1, p2) for p1, p2 in zip(protein1_ids, protein2_ids)]
                    all_protein_ids.extend(batch_protein_ids)
                else:
                    # Create placeholder protein IDs if not available
                    batch_size = labels.size(0)
                    placeholder_ids = [(f"protein1_{i}", f"protein2_{i}") for i in range(batch_size)]
                    batch_protein_ids = placeholder_ids
                    all_protein_ids.extend(placeholder_ids)
                
                try:
                    # Ensure model is loaded
                    model_ref = self.model
                    if model_ref is None:
                        if self.logger:
                            self.logger.error("Model not loaded")
                        continue
                    
                    # 传递protein_ids以支持多模态特征获取
                    if return_features:
                        predictions, features = model_ref(protein1_seq, protein2_seq, protein1_mask, protein2_mask, protein_ids=protein_ids, return_features=True)
                    else:
                        predictions = model_ref(protein1_seq, protein2_seq, protein1_mask, protein2_mask, protein_ids=protein_ids)
                    
                    # Collect attention weights (if model supports and non-CIS mode and saving results)
                    if attention_file_handle is not None:
                        attention_weights = self._collect_attention_weights(model_ref, batch_protein_ids)
                        # Stream write attention weights per sample
                        for attention_data in attention_weights:
                            import json
                            json.dump(attention_data, attention_file_handle, ensure_ascii=False, separators=(',', ':'))
                            attention_file_handle.write('\n')
                            attention_sample_count += 1
                    
                    # Safe conversion: numpy to list, including scalar case
                    def _to_list_safe(np_array):
                        """Safely convert numpy arrays to lists, handling 0-D scalars"""
                        if np_array.ndim == 0:
                            return [np_array.item()]  # 标量转为单元素列表
                        else:
                            return np_array.tolist()  # 数组正常转列表
                    
                    if return_logits:
                        # Convert probabilities back to logits using logit function
                        # logit(p) = log(p / (1 - p))
                        # Clamp to avoid numerical issues
                        predictions_clamped = torch.clamp(predictions.squeeze(), 1e-7, 1 - 1e-7)
                        logits = torch.log(predictions_clamped / (1 - predictions_clamped))
                        all_predictions.extend(_to_list_safe(logits.cpu().numpy()))
                    else:
                        all_predictions.extend(_to_list_safe(predictions.squeeze().cpu().numpy()))
                    
                    if return_features and all_features is not None:
                        all_features.extend(_to_list_safe(features.squeeze().cpu().numpy()))
                    
                    all_labels.extend(_to_list_safe(labels.cpu().numpy()))
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"模型推理错误: {e}")
                    continue
            
            if show_progress:
                eval_pbar.close()
        
    # Close attention weights file and log stats
        if attention_file_handle is not None:
            attention_file_handle.close()
            if self.logger:
                self.logger.info(f"Attention weights saved to: {attention_file_path}")
                self.logger.debug(f"Contains attention weights for {attention_sample_count} protein pairs")
        
        # Calculate metrics
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        
        if return_logits:
            # For logits, convert back to probabilities for metric calculation
            predictions_for_metrics = 1 / (1 + np.exp(-predictions_array))  # sigmoid(logits)
        else:
            predictions_for_metrics = predictions_array
        
        try:
            roc_auc = roc_auc_score(labels_array, predictions_for_metrics)
            pr_auc = average_precision_score(labels_array, predictions_for_metrics)
        except ValueError:
            roc_auc = 0.0
            pr_auc = 0.0
        
        # Calculate detailed metrics using create_prediction_report
        threshold = self.config.get('inference', {}).get('threshold', 0.5)
        from ..models.predict_model import create_prediction_report
        
        detailed_report = create_prediction_report(
            predictions_for_metrics, labels_array, threshold=threshold
        )

        feature_gating_summary = {}
        if gating_preprocessor is not None:
            try:
                feature_gating_summary = gating_preprocessor.get_feature_gating_stats()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to collect feature gating stats for {dataset_name}: {e}")
                feature_gating_summary = {}
        
        results = {
            'dataset_name': dataset_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': detailed_report['f1_score'],
            'precision': detailed_report['precision'],
            'recall': detailed_report['recall'],
            'accuracy': detailed_report['accuracy'],
            'threshold': threshold,
            'true_positives': detailed_report['true_positives'],
            'false_positives': detailed_report['false_positives'],
            'true_negatives': detailed_report['true_negatives'],
            'false_negatives': detailed_report['false_negatives'],
            'n_samples': len(labels_array),
            'n_positive': int(np.sum(labels_array)),
            'n_negative': int(len(labels_array) - np.sum(labels_array)),
            'positive_rate': float(np.mean(labels_array)),
            'predictions': predictions_array,
            'true_labels': labels_array,
            'protein_ids': all_protein_ids if all_protein_ids else None,
            'feature_gating_summary': feature_gating_summary,
            'attention_weights_file': f"{dataset_name}_attention_weights.jsonl" if attention_sample_count > 0 else None,
            'attention_sample_count': attention_sample_count
        }
        
        if return_features:
            results['features'] = np.array(all_features)
        
        return results
    
    def _should_collect_attention_weights(self) -> bool:
        """
        Determine whether to collect attention weights
        Only collect in non-CIS mode and when the model supports attention
        """
        # 检查是否为CIS模式
        is_cis_mode = self.config.get('data', {}).get('cis_type', False)
        if is_cis_mode:
            return False
        
        # 检查模型是否支持注意力权重
        if self.model is None:
            return False
        
        return hasattr(self.model, 'get_attention_weights')
    
    def _collect_attention_weights(self, model, batch_protein_ids) -> list:
        """
        Collect attention weights for the current batch
        """
        try:
            attention_weights_dict = model.get_attention_weights()
            if not attention_weights_dict:
                return []
            
            batch_attention_data = []
            
            # 获取protein1和protein2的注意力权重
            protein1_weights = attention_weights_dict.get('protein1')
            protein2_weights = attention_weights_dict.get('protein2')
            
            if self.logger:
                self.logger.debug(f"Collecting attention weights: protein1_weights={protein1_weights is not None}, "
                                f"protein2_weights={protein2_weights is not None}, "
                                f"batch_size={len(batch_protein_ids)}")
            
            for i, protein_pair in enumerate(batch_protein_ids):
                protein1_id, protein2_id = protein_pair
                
                # 获取蛋白质序列（从FASTA文件）
                protein1_seq, protein2_seq = self._get_protein_sequences(protein1_id, protein2_id)
                
                # 处理注意力权重数据
                attention_data = self._process_protein_pair_attention_v2(
                    protein1_id, protein2_id, protein1_seq, protein2_seq, 
                    protein1_weights, protein2_weights, i
                )
                
                if attention_data:
                    batch_attention_data.append(attention_data)
            
            return batch_attention_data
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error collecting attention weights: {e}")
            return []
    
    def _process_protein_pair_attention_v2(self, protein1_id: str, protein2_id: str, 
                                         protein1_seq: str, protein2_seq: str, 
                                         protein1_weights: torch.Tensor, 
                                         protein2_weights: torch.Tensor,
                                         batch_idx: int) -> dict:
        """
        Process attention weights for a protein pair (v2)
        """
        try:
            attention_data = {
                "protein_pair_id": f"{protein1_id}_{protein2_id}",
                "protein1_id": protein1_id,
                "protein2_id": protein2_id,
                "protein1_length": len(protein1_seq),
                "protein2_length": len(protein2_seq),
            }
            
            # 处理protein1的注意力权重
            if protein1_weights is not None and batch_idx < protein1_weights.size(0):
                sample_p1_attention = protein1_weights[batch_idx]
                _, attention_num = sample_p1_attention.shape
                attention_data["attention_heads"] = attention_num
                attention_data["protein1_attention"] = self._process_single_protein_attention(
                    protein1_id, protein1_seq, sample_p1_attention
                )
            else:
                attention_data["protein1_attention"] = {}
            
            # 处理protein2的注意力权重
            if protein2_weights is not None and batch_idx < protein2_weights.size(0):
                sample_p2_attention = protein2_weights[batch_idx]
                attention_data["protein2_attention"] = self._process_single_protein_attention(
                    protein2_id, protein2_seq, sample_p2_attention
                )
            else:
                attention_data["protein2_attention"] = {}
            
            return attention_data
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error processing protein pair attention weights: {e}")
            return {}
    
    def _process_single_protein_attention(self, protein_id: str, protein_seq: str, 
                                        attention_weights: torch.Tensor) -> dict:
        """
        Process attention weights for a single protein
        """
        try:
            seq_len, attention_num = attention_weights.shape
            
            # 调整序列长度以匹配注意力权重
            effective_len = min(seq_len, len(protein_seq))
            
            protein_heads = {}
            for head_idx in range(attention_num):
                head_weights = attention_weights[:effective_len, head_idx]
                
                # 归一化到0.001-1范围并保留4位小数
                if head_weights.numel() > 0:
                    head_weights_normalized = self._normalize_attention_weights(head_weights)
                    
                    # 直接存储权重值列表，不包含position和amino_acid
                    weights_list = []
                    for pos, weight in enumerate(head_weights_normalized):
                        if pos < len(protein_seq):
                            weights_list.append(round(float(weight), 4))
                    
                    protein_heads[f"head_{head_idx+1}"] = weights_list
                else:
                    protein_heads[f"head_{head_idx+1}"] = []
            
            return protein_heads
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error processing single protein attention weights: {e}")
            return {}
    
    def _get_protein_sequences(self, protein1_id: str, protein2_id: str) -> tuple:
        """
        Get protein sequences from FASTA
        """
        try:
            fasta_file = self.config.get('data', {}).get('fasta_file')
            if not fasta_file:
                return "", ""
            
            # 使用已有的FASTA解析器
            from ..data_processing.fasta_parser import parse_fasta_file
            
            # 缓存FASTA数据避免重复解析
            if not hasattr(self, '_fasta_cache'):
                self._fasta_cache = parse_fasta_file(fasta_file)
            
            protein1_seq = self._fasta_cache.get(protein1_id, "")
            protein2_seq = self._fasta_cache.get(protein2_id, "")
            
            return protein1_seq, protein2_seq
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error retrieving protein sequences: {e}")
            return "", ""
    

    
    def _normalize_attention_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize attention weights to the range 0.001-1.0
        """
        if weights.numel() == 0:
            return weights
        
        # 先进行softmax归一化（如果还没有）
        weights_softmax = torch.softmax(weights, dim=0)
        
        # 映射到0.001-1.0范围
        min_val = 0.001
        max_val = 1.0
        
        weights_min = weights_softmax.min()
        weights_max = weights_softmax.max()
        
        if weights_max > weights_min:
            # 线性映射到目标范围
            normalized = (weights_softmax - weights_min) / (weights_max - weights_min)
            normalized = normalized * (max_val - min_val) + min_val
        else:
            # 如果所有权重相同，设为中间值
            normalized = torch.full_like(weights_softmax, (max_val + min_val) / 2)
        
        return normalized
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_only = {}
        for name, result in results.items():
            metrics_only[name] = {
                'roc_auc': result['roc_auc'],
                'pr_auc': result['pr_auc'],
                'f1': result.get('f1', 0.0),
                'precision': result.get('precision', 0.0),
                'recall': result.get('recall', 0.0),
                'accuracy': result.get('accuracy', 0.0),
                'threshold': result.get('threshold', 0.5),
                'true_positives': result.get('true_positives', 0),
                'false_positives': result.get('false_positives', 0),
                'true_negatives': result.get('true_negatives', 0),
                'false_negatives': result.get('false_negatives', 0),
                'n_samples': result['n_samples'],
                'n_positive': result['n_positive'],
                'n_negative': result['n_negative'],
                'positive_rate': result['positive_rate']
            }
        
        with open(output_path / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics_only, f, indent=2)
        
        # 注意力权重已在评估过程中流式写入到JSONL文件
        
        # Generate confusion matrices if enabled
        if self.config.get('output', {}).get('generate_confusion_matrix', True):
            self._generate_confusion_matrices(results, output_path)
    

    def _generate_confusion_matrices(self, results: Dict[str, Any], output_dir: Path):
        """Generate and save confusion matrices"""
        try:
            plt.switch_backend('Agg')
            
            threshold = self.config.get('inference', {}).get('threshold', 0.5)
            
            for dataset_name, result in results.items():
                if 'predictions' in result and 'true_labels' in result:
                    predictions = result['predictions']
                    true_labels = result['true_labels']
                    
                    # Convert predictions to binary using threshold
                    pred_binary = (predictions > threshold).astype(int)
                    
                    # Generate confusion matrix
                    cm = confusion_matrix(true_labels, pred_binary)
                    
                    # Create figure
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=['Negative', 'Positive'],
                               yticklabels=['Negative', 'Positive'])
                    plt.title(f'Confusion Matrix - {dataset_name.upper()}')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    
                    # Save figure
                    cm_file = output_dir / f"confusion_matrix_{dataset_name}.png"
                    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Save confusion matrix data as JSON
                    cm_data = {
                        'confusion_matrix': cm.tolist(),
                        'true_negatives': int(cm[0, 0]),
                        'false_positives': int(cm[0, 1]),
                        'false_negatives': int(cm[1, 0]),
                        'true_positives': int(cm[1, 1]),
                        'threshold': threshold
                    }
                    
                    cm_json_file = output_dir / f"confusion_matrix_{dataset_name}.json"
                    with open(cm_json_file, 'w') as f:
                        json.dump(cm_data, f, indent=2)
            
            if self.logger:
                self.logger.debug("Confusion matrices generated")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error generating confusion matrices: {str(e)}")
    
    def predict_interactions(self, protein_pairs, return_probabilities: bool = True):
        """
        Predict interactions for given protein pairs
        
        Args:
            protein_pairs: List or dataset of protein pairs
            return_probabilities: Whether to return probabilities or binary predictions
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用load_model方法")
        
        self.model.eval()
        predictions = []
        
        # Implementation depends on the format of protein_pairs
        # This is a placeholder for the actual prediction logic
        with torch.no_grad():
            # Process protein pairs and generate predictions
            pass
        
        threshold = self.config.get('inference', {}).get('threshold', 0.5)
        predictions_array = np.array(predictions)
        
        if return_probabilities:
            return predictions_array
        else:
            return (predictions_array > threshold).astype(int)