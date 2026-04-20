#!/usr/bin/env python3
"""
Pooling_lmdb_creat.py - 池化嵌入生成脚本

从ESM嵌入生成注意力池化后的蛋白嵌入LMDB和单蛋白注意力权重JSONL文件。
支持两种模式：
1. 带多模态特征模式 (--pdbfeature-folder): 使用 feature_concat 预处理
2. 仅序列模式 (不指定 --pdbfeature-folder): 使用 nonefeature 模式

Usage:
    # 仅序列模式
    python Pooling_lmdb_creat.py \
        --input-layer-ckpt results/.../input_layer.pth \
        --model-pretrain-ckpt results/.../complete_model.pth \
        --fasta dataset/Strings_plant50/protein.fasta \
        --output-dir outputs/pooled_embeddings

    # 带多模态特征模式
    python Pooling_lmdb_creat.py \
        --input-layer-ckpt results/.../input_layer.pth \
        --model-pretrain-ckpt results/.../complete_model.pth \
        --fasta dataset/Strings_plant50/protein.fasta \
        --pdbfeature-folder mutifeature/Strings_plant50 \
        --output-dir outputs/pooled_embeddings

    # 启用MLP分割
    python Pooling_lmdb_creat.py \
        --input-layer-ckpt results/.../input_layer.pth \
        --model-pretrain-ckpt results/.../complete_model.pth \
        --fasta dataset/Strings_plant50/protein.fasta \
        --output-dir outputs/pooled_embeddings \
        --segmentation-model
"""

import os
import sys
import json
import time
import shutil
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from Bio import SeqIO


# =============================================
# 工具函数
# =============================================

def _safe_torch_load(path: str, map_location: torch.device):
    """安全加载PyTorch检查点"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)
        return torch.load(path, map_location=map_location)


def find_resolved_config(model_dir: Path) -> Optional[Path]:
    """在模型目录中查找 resolved_config.yaml"""
    config_path = model_dir / "resolved_config.yaml"
    if config_path.exists():
        return config_path
    
    # 尝试查找 best_config.yaml
    best_config_path = model_dir / "best_config.yaml"
    if best_config_path.exists():
        return best_config_path
    
    return None


def read_fasta_sequences(fasta_file_path: str) -> Dict[str, str]:
    """读取FASTA文件，返回 {seq_id: sequence}"""
    sequences = {}
    try:
        with open(fasta_file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq)
    except FileNotFoundError:
        print(f"错误: 文件 {fasta_file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"读取FASTA文件时出错: {e}")
        sys.exit(1)
    return sequences


# =============================================
# ESM 模型加载
# =============================================

# Lazy imports
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

HF_MODEL_NAME_MAP: Dict[str, str] = {
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b": "facebook/esm2_t36_3B_UR50D",
    "esm2_15b": "facebook/esm2_t48_15B_UR50D",
    "esm1b_650m": "facebook/esm1b_t33_650M_UR50S",
}

MODEL_DIM_MAP: Dict[str, int] = {
    "esm2_15b": 5120,
    "esm2_3b": 2560,
    "esm2_650m": 1280,
    "esm2_150m": 640,
    "esm1b_650m": 1280,
}


def load_esm_model(model_name: str, device: torch.device):
    """加载ESM模型"""
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("Transformers 未安装，请先安装: pip install transformers")
    
    if model_name not in HF_MODEL_NAME_MAP:
        raise ValueError(f"不支持的模型: {model_name}. 支持: {list(HF_MODEL_NAME_MAP.keys())}")
    
    hf_name = HF_MODEL_NAME_MAP[model_name]
    print(f"加载ESM模型: {hf_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    
    hidden_size = getattr(model.config, "hidden_size", MODEL_DIM_MAP.get(model_name))
    print(f"模型加载完成, hidden_size={hidden_size}")
    
    return model, tokenizer, hidden_size


def get_esm_embeddings(model, tokenizer, device: torch.device, sequence: str) -> Optional[np.ndarray]:
    """获取ESM残基级嵌入 (不含CLS和EOS)"""
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(**inputs)
            emb = outputs.last_hidden_state  # [1, L+2, H]
            # 去除CLS和EOS token
            middle_embeddings = emb[0, 1:-1].float().cpu().numpy()
            return middle_embeddings
    except Exception as e:
        print(f"ESM嵌入计算错误: {e}")
        return None


# =============================================
# 模型组件
# =============================================

class InputLayerProjector(nn.Module):
    """输入层投影器，支持MLP投影
    
    有两种模式：
    1. 单层线性投影 (hidden_dim=None): 使用 self.proj
    2. MLP投影 (hidden_dim!=None): 使用 self.proj1, self.proj2, self.ln 以匹配训练时的 MLPProjectionInputLayer
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None,
                 activation: str = 'gelu', residual: bool = True, layernorm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.residual = residual and (input_dim == output_dim)
        self._use_mlp = hidden_dim is not None
        
        if hidden_dim is None:
            # 单层线性投影
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            # MLP投影 - 使用与训练时相同的结构
            self.proj1 = nn.Linear(input_dim, hidden_dim)
            self.proj2 = nn.Linear(hidden_dim, output_dim)
            
            if activation.lower() == 'gelu':
                self.act = nn.GELU()
            elif activation.lower() == 'relu':
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()
        
        # LayerNorm (MLP模式时为 self.ln，单层模式时为 self.layernorm)
        if layernorm:
            if hidden_dim is not None:
                self.ln = nn.LayerNorm(output_dim)
            else:
                self.layernorm = nn.LayerNorm(output_dim)
        else:
            self.ln = None
            self.layernorm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mlp:
            # MLP路径
            out = self.proj1(x)
            out = self.act(out)
            out = self.proj2(out)
            # 残差连接 (在LayerNorm之前)
            if self.residual:
                out = out + x
            # LayerNorm (在残差之后)
            if self.ln is not None:
                out = self.ln(out)
        else:
            # 单层路径
            out = self.proj(x)
            # 残差连接 (在LayerNorm之前)
            if self.residual:
                out = out + x
            # LayerNorm (在残差之后)
            if self.layernorm is not None:
                out = self.layernorm(out)
        
        return out
    
    @staticmethod
    def from_checkpoint(ckpt_path: str, device: torch.device) -> 'InputLayerProjector':
        """从检查点加载"""
        ckpt = _safe_torch_load(ckpt_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        meta = ckpt.get('meta', {})
        model_config = ckpt.get('model_config', {})
        
        # 提取配置
        input_dim = meta.get('input_dim')
        embedding_dim = meta.get('embedding_dim')
        activation = meta.get('activation', 'gelu')
        
        # 从model_config中获取更详细的投影配置
        projection_config = {}
        if 'model_config' in model_config:
            input_data = model_config['model_config'].get('input_data', {})
            projection_config = input_data.get('projection', {})
        
        hidden_dim = projection_config.get('hidden_dim')
        residual = projection_config.get('residual', True)
        layernorm = projection_config.get('layernorm', True)
        
        # 从权重推断维度
        weight_key = 'proj.weight' if 'proj.weight' in state else 'weight'
        if weight_key in state:
            weight = state[weight_key]
            if hidden_dim is not None:
                # MLP: 第一层权重 [hidden_dim, input_dim]
                if input_dim is None:
                    input_dim = weight.shape[1]
            else:
                # 单层线性: [output_dim, input_dim]
                if input_dim is None:
                    input_dim = weight.shape[1]
                if embedding_dim is None:
                    embedding_dim = weight.shape[0]
        
        if input_dim is None or embedding_dim is None:
            raise RuntimeError(f"无法从检查点推断维度: input_dim={input_dim}, embedding_dim={embedding_dim}")
        
        # 创建模型
        projector = InputLayerProjector(
            input_dim=input_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=residual,
            layernorm=layernorm
        )
        
        # 加载权重
        projector.load_state_dict(state, strict=False)
        projector.to(device)
        projector.eval()
        
        print(f"[InputLayer] 加载完成: {input_dim} -> {embedding_dim}, hidden={hidden_dim}")
        return projector


class AttentionPoolingUnit(nn.Module):
    """注意力池化单元"""
    
    def __init__(self, embedding_dim: int, attention_num: int = 1, dropout: float = 0.1,
                 save_attention: bool = True, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_num = attention_num
        self.save_attention = save_attention
        self.temperature = temperature
        
        self.attention_weights = nn.Linear(embedding_dim, attention_num)
        self.dropout = nn.Dropout(dropout)
        
        # 存储注意力权重
        self.last_attention_weights = None
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            embeddings: [batch_size, seq_len, embedding_dim] 或 [seq_len, embedding_dim]
            attention_mask: [batch_size, seq_len] 或 [seq_len]
        Returns:
            pooled: [batch_size, embedding_dim] 或 [embedding_dim]
        """
        squeeze_batch = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            squeeze_batch = True
        
        batch_size, seq_len, _ = embeddings.shape
        
        # 计算注意力分数 [batch_size, seq_len, attention_num]
        attention_scores = self.attention_weights(embeddings)
        attention_scores = self.dropout(attention_scores)
        
        # 应用掩码
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            attention_scores = attention_scores + (1.0 - mask) * (-1e9)
        
        # 温度缩放 + softmax
        scaled_scores = attention_scores / max(self.temperature, 1e-4)
        attention_weights = F.softmax(scaled_scores, dim=1)
        
        # 保存注意力权重
        if self.save_attention:
            self.last_attention_weights = attention_weights.detach().cpu()
        
        # 加权平均
        if self.attention_num == 1:
            weighted_embeddings = embeddings * attention_weights
            pooled = weighted_embeddings.sum(dim=1)
        else:
            pooled_heads = []
            for head_idx in range(self.attention_num):
                head_weights = attention_weights[:, :, head_idx:head_idx+1]
                weighted_embeddings = embeddings * head_weights
                head_pooled = weighted_embeddings.sum(dim=1)
                pooled_heads.append(head_pooled)
            pooled = torch.stack(pooled_heads, dim=0).mean(dim=0)
        
        if squeeze_batch:
            pooled = pooled.squeeze(0)
        
        return pooled
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取最后一次前向传播的注意力权重"""
        return self.last_attention_weights
    
    @staticmethod
    def from_config(pooling_config: Dict[str, Any], device: torch.device) -> 'AttentionPoolingUnit':
        """从配置创建"""
        embedding_dim = pooling_config.get('embedding_dim', 1280)
        attention_num = pooling_config.get('attention_num', 1)
        dropout = pooling_config.get('dropout', 0.1)
        save_attention = pooling_config.get('save_attention', True)
        
        attention_config = pooling_config.get('attention_config', {})
        temperature = attention_config.get('temperature', 1.0)
        
        return AttentionPoolingUnit(
            embedding_dim=embedding_dim,
            attention_num=attention_num,
            dropout=dropout,
            save_attention=save_attention,
            temperature=temperature
        ).to(device)


# =============================================
# 注意力权重处理
# =============================================

def normalize_attention_weights(weights: torch.Tensor, min_val: float = 0.001, max_val: float = 1.0) -> torch.Tensor:
    """归一化注意力权重到指定范围"""
    if weights.numel() == 0:
        return weights
    
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-8:
        return torch.full_like(weights, (min_val + max_val) / 2)
    
    normalized = (weights - w_min) / (w_max - w_min)
    normalized = normalized * (max_val - min_val) + min_val
    return normalized


def _normalize_attention_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    将注意力权重归一化到 0.001-1.0 范围
    与 inference_engine.py 中的 _normalize_attention_weights 保持一致
    """
    if weights.numel() == 0:
        return weights
    
    # 先进行softmax归一化（与 inference_engine 一致）
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


def format_single_protein_attention(protein_id: str, sequence: str, 
                                    attention_weights: torch.Tensor) -> Dict[str, Any]:
    """
    格式化单蛋白注意力权重
    
    Args:
        protein_id: 蛋白质ID
        sequence: 蛋白质序列
        attention_weights: [seq_len, attention_num]
    
    Returns:
        单蛋白注意力数据字典
    """
    seq_len, attention_num = attention_weights.shape
    effective_len = min(seq_len, len(sequence))
    
    attention_data = {
        "protein_id": protein_id,
        "length": len(sequence),
        "attention_heads": attention_num,
        "attention": {}
    }
    
    # 直接保存softmax输出（不做额外的线性归一化变换），保留原始相对比例
    # 与 inference_engine.py 保持一致：softmax → 线性映射到 [0.001, 1.0]
    for head_idx in range(attention_num):
        head_weights = attention_weights[:effective_len, head_idx]
        # 应用与 inference_engine 相同的归一化
        head_weights_normalized = _normalize_attention_weights(head_weights)
        # 转为浮点并限制小数位以减少文件体积与量化差异
        weights_list = [round(float(w), 4) for w in head_weights_normalized]
        attention_data["attention"][f"head_{head_idx + 1}"] = weights_list
    
    return attention_data


# =============================================
# 模型分割功能
# =============================================

def extract_classifier_weights(complete_model_path: str, output_dir: Path, 
                               resolved_config: Dict[str, Any], device: torch.device):
    """
    提取分类器权重并生成配置文件
    
    Args:
        complete_model_path: complete_model.pth 路径
        output_dir: 输出目录
        resolved_config: 解析后的配置
        device: 设备
    """
    print("\n[Segmentation] 开始分割MLP分类器...")
    
    # 加载完整模型
    ckpt = _safe_torch_load(complete_model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model_config = ckpt.get('model_config', resolved_config.get('model', {}))
    
    # 提取 classifier.* 权重
    classifier_state = {}
    for key, value in state.items():
        if key.startswith('classifier.'):
            # 移除 'classifier.' 前缀
            new_key = key[len('classifier.'):]
            classifier_state[new_key] = value
    
    if not classifier_state:
        print("[Segmentation] 警告: 未找到 classifier.* 权重")
        return
    
    # 保存分类器权重
    classifier_path = output_dir / "classifier.pth"
    torch.save({
        'model_state_dict': classifier_state,
        'meta': {
            'source': 'Pooling_lmdb_creat.py',
            'timestamp': time.time(),
        }
    }, classifier_path)
    print(f"[Segmentation] 分类器权重已保存: {classifier_path}")
    
    # 生成配置文件 - 仅包含 learning_architecture 和 output
    model_config_dict = model_config.get('model_config', model_config)
    
    classifier_config = {
        'description': '从预训练模型分割的MLP分类器配置',
        'note': '此配置仅包含分类器部分，pooling/post_pooling/preprocessing/input_data已在预处理阶段完成',
        'learning_architecture': model_config_dict.get('learning_architecture', {}),
        'output': model_config_dict.get('output', {}),
    }
    
    # 更新 input_dim 为 embedding_dim (因为池化后输入是单个向量)
    embedding_dim = model_config_dict.get('pooling', {}).get('embedding_dim', 1280)
    if 'learning_architecture' in classifier_config:
        classifier_config['learning_architecture']['input_dim'] = embedding_dim
        classifier_config['learning_architecture']['description'] = (
            f"输入维度为{embedding_dim} (Hadamard乘积后的池化嵌入)"
        )
    
    config_path = output_dir / "classifier_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(classifier_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"[Segmentation] 分类器配置已保存: {config_path}")


# =============================================
# 主处理流程
# =============================================

def process_single_protein(
    protein_id: str,
    sequence: str,
    esm_model,
    esm_tokenizer,
    input_layer: InputLayerProjector,
    internal_projector: Optional[InputLayerProjector],
    pooling_layer: AttentionPoolingUnit,
    preprocessing_layer: Optional[nn.Module],
    feature_folder: Optional[str],
    device: torch.device,
    batch_size: int = 1  # 预留batch接口，当前不使用
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    处理单个蛋白质
    
    Args:
        protein_id: 蛋白质ID
        sequence: 蛋白质序列
        esm_model: ESM模型
        esm_tokenizer: ESM分词器
        input_layer: 输入投影层 (ESM维度 -> embedding_dim)
        internal_projector: 内部投影层 (embedding_dim -> hidden -> embedding_dim, 可选)
        pooling_layer: 注意力池化层
        preprocessing_layer: 预处理层 (可选，用于多模态特征)
        feature_folder: 多模态特征文件夹 (可选)
        device: 计算设备
        batch_size: 批次大小 (预留接口)
    
    Returns:
        (pooled_embedding_bytes, attention_data): 池化嵌入字节 (bf16 via int16 view) 和注意力数据
    """
    # 1. 获取ESM嵌入
    raw_embeddings = get_esm_embeddings(esm_model, esm_tokenizer, device, sequence)
    if raw_embeddings is None:
        return None, None
    
    # 转为tensor
    embeddings = torch.from_numpy(raw_embeddings).to(device)  # [seq_len, esm_dim]
    
    # 2. 输入层投影 (ESM维度 -> embedding_dim)
    with torch.no_grad():
        projected = input_layer(embeddings)  # [seq_len, embedding_dim]
        
        # 2.5 内部投影层 (embedding_dim -> hidden -> embedding_dim)
        if internal_projector is not None:
            projected = internal_projector(projected)  # [seq_len, embedding_dim]
        
        # 3. 预处理 (如果有多模态特征)
        if preprocessing_layer is not None and feature_folder is not None:
            # 添加batch维度
            projected = projected.unsqueeze(0)  # [1, seq_len, embedding_dim]
            projected = preprocessing_layer(projected, protein_ids=[protein_id])
            projected = projected.squeeze(0)  # [seq_len, embedding_dim]
        
        # 4. 注意力池化
        pooled = pooling_layer(projected, attention_mask=None)  # [embedding_dim]
        
        # 获取注意力权重
        attention_weights = pooling_layer.get_attention_weights()
    
    # 将 pooled 转为 fp32 numpy 并以 bytes 保存（与 creatlmdb.py 保持一致）
    pooled_fp32 = pooled.cpu().float().contiguous().numpy().astype(np.float32)
    pooled_bytes = pooled_fp32
    
    # 格式化注意力数据
    attention_data = None
    if attention_weights is not None:
        # attention_weights: [1, seq_len, attention_num] -> [seq_len, attention_num]
        weights = attention_weights.squeeze(0) if attention_weights.dim() == 3 else attention_weights
        attention_data = format_single_protein_attention(protein_id, sequence, weights)
    
    return pooled_bytes, attention_data


def process_and_save_pooled_embeddings(
    sequences: Dict[str, str],
    esm_model,
    esm_tokenizer,
    input_layer: InputLayerProjector,
    internal_projector: Optional[InputLayerProjector],
    pooling_layer: AttentionPoolingUnit,
    preprocessing_layer: Optional[nn.Module],
    feature_folder: Optional[str],
    output_dir: Path,
    device: torch.device,
    commit_interval: int = 100,
    force: bool = False,
    append: bool = False
):
    """
    处理所有蛋白质并保存到LMDB和JSONL
    """
    total_sequences = len(sequences)
    processed_count = 0
    start_time = time.time()
    
    lmdb_path = output_dir / "pooled_embeddings.lmdb"
    attention_path = output_dir / "attention_weights.jsonl"

    # 写入精度元数据，指示保存的嵌入精度
    try:
        precision_meta = output_dir / "pooled_precision.json"
        if not precision_meta.exists():
            with open(precision_meta, 'w', encoding='utf-8') as pm:
                json.dump({'precision': 'fp32'}, pm)
    except Exception:
        pass
    
    # 处理已存在的文件
    if lmdb_path.exists() and not append:
        if force:
            if lmdb_path.is_dir():
                shutil.rmtree(lmdb_path)
            else:
                lmdb_path.unlink()
            print("已删除旧的LMDB数据库")
        else:
            print(f"LMDB文件已存在: {lmdb_path}")
            choice = input("是否删除并重建 (1) 或追加 (2)? ")
            if choice == '1':
                if lmdb_path.is_dir():
                    shutil.rmtree(lmdb_path)
                else:
                    lmdb_path.unlink()
            else:
                append = True
    
    # 检查已存在的key
    existing_keys = set()
    if append and lmdb_path.exists():
        try:
            with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        existing_keys.add(key.decode())
            print(f"发现 {len(existing_keys)} 个已存在的序列")
        except Exception as e:
            print(f"读取现有LMDB失败: {e}")
    
    # 过滤待处理序列
    sequences_to_process = {k: v for k, v in sequences.items() if k not in existing_keys}
    if not sequences_to_process:
        print("所有序列已处理完成")
        return
    
    print(f"待处理序列: {len(sequences_to_process)}/{total_sequences}")
    
    # 打开文件
    attention_mode = 'a' if append else 'w'
    
    try:
        with lmdb.open(str(lmdb_path), map_size=int(1e12)) as env, \
             open(attention_path, attention_mode, encoding='utf-8') as attention_file:
            
            buffer = []
            progress = tqdm(sequences_to_process.items(), desc="处理序列", total=len(sequences_to_process))
            
            for protein_id, sequence in progress:
                try:
                    pooled, attention_data = process_single_protein(
                        protein_id=protein_id,
                        sequence=sequence,
                        esm_model=esm_model,
                        esm_tokenizer=esm_tokenizer,
                        input_layer=input_layer,
                        internal_projector=internal_projector,
                        pooling_layer=pooling_layer,
                        preprocessing_layer=preprocessing_layer,
                        feature_folder=feature_folder,
                        device=device
                    )
                    
                    if pooled is None:
                        continue
                    
                    # 添加到buffer
                    buffer.append((protein_id.encode(), pooled.tobytes()))
                    
                    # 写入注意力权重
                    if attention_data is not None:
                        json.dump(attention_data, attention_file, ensure_ascii=False, separators=(',', ':'))
                        attention_file.write('\n')
                    
                    processed_count += 1
                    
                    # 批量提交
                    if len(buffer) >= commit_interval:
                        with env.begin(write=True) as txn:
                            for key, value in buffer:
                                txn.put(key, value)
                        buffer.clear()
                        torch.cuda.empty_cache()
                    
                    # 更新进度
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (len(sequences_to_process) - processed_count) / rate if rate > 0 else 0
                    progress.set_postfix_str(f"速度: {rate:.2f}/s, 剩余: {remaining/60:.1f}min")
                    
                except Exception as e:
                    print(f"\n处理 {protein_id} 时出错: {e}")
                    continue
            
            # 提交剩余buffer
            if buffer:
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)
    
    except KeyboardInterrupt:
        print("\n检测到中断，正在保存已处理数据...")
        if buffer:
            with lmdb.open(str(lmdb_path), map_size=int(1e12)) as env:
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)
    
    print(f"\n处理完成!")
    print(f"  池化嵌入: {lmdb_path}")
    print(f"  注意力权重: {attention_path}")
    print(f"  处理序列数: {processed_count}")


# =============================================
# 命令行接口
# =============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="生成注意力池化蛋白嵌入LMDB和注意力权重JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅序列模式
  python Pooling_lmdb_creat.py \\
      --input-layer-ckpt results/.../input_layer.pth \\
      --model-pretrain-ckpt results/.../complete_model.pth \\
      --fasta dataset/Strings_plant50/protein.fasta \\
      --output-dir outputs/pooled_embeddings

  # 带多模态特征模式
  python Pooling_lmdb_creat.py \\
      --input-layer-ckpt results/.../input_layer.pth \\
      --model-pretrain-ckpt results/.../complete_model.pth \\
      --fasta dataset/Strings_plant50/protein.fasta \\
      --pdbfeature-folder mutifeature/Strings_plant50 \\
      --output-dir outputs/pooled_embeddings
        """
    )
    
    # 必需参数
    parser.add_argument("--input-layer-ckpt", type=str, required=True,
                        help="输入投影层权重路径 (input_layer.pth)")
    parser.add_argument("--model-pretrain-ckpt", type=str, required=True,
                        help="完整模型权重路径 (complete_model.pth)")
    parser.add_argument("--fasta", "-f", type=str, required=True,
                        help="输入FASTA文件路径")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="输出目录")
    
    # 可选参数
    parser.add_argument("--pdbfeature-folder", type=str, default=None,
                        help="多模态特征根目录 (如 mutifeature/Strings_plant50)，不指定则使用仅序列模式")
    parser.add_argument("--model", "-m", type=str, default="esm2_15b",
                        choices=list(HF_MODEL_NAME_MAP.keys()),
                        help="ESM模型名称 (默认: esm2_15b)")
    parser.add_argument("--segmentation-model", action="store_true",
                        help="启用MLP分类器分割，生成 classifier.pth 和 classifier_config.yaml")
    
    # 处理选项
    parser.add_argument("--commit-interval", type=int, default=100,
                        help="LMDB批量提交间隔 (默认: 100)")
    parser.add_argument("--force", action="store_true",
                        help="强制删除已存在的输出文件")
    parser.add_argument("--append", action="store_true",
                        help="追加模式，跳过已处理的序列")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 验证输入文件
    if not os.path.isfile(args.input_layer_ckpt):
        print(f"错误: 输入层检查点不存在: {args.input_layer_ckpt}")
        sys.exit(1)
    
    if not os.path.isfile(args.model_pretrain_ckpt):
        print(f"错误: 模型检查点不存在: {args.model_pretrain_ckpt}")
        sys.exit(1)
    
    if not os.path.isfile(args.fasta):
        print(f"错误: FASTA文件不存在: {args.fasta}")
        sys.exit(1)
    
    # 查找 resolved_config.yaml
    model_dir = Path(args.model_pretrain_ckpt).parent
    config_path = find_resolved_config(model_dir)
    if config_path is None:
        print(f"错误: 在 {model_dir} 中未找到 resolved_config.yaml 或 best_config.yaml")
        sys.exit(1)
    
    print(f"使用配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        resolved_config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载ESM模型
    esm_model, esm_tokenizer, esm_hidden_size = load_esm_model(args.model, device)
    
    # 加载输入层
    input_layer = InputLayerProjector.from_checkpoint(args.input_layer_ckpt, device)
    
    # 从配置中提取池化配置
    model_config = resolved_config.get('model', {}).get('model_config', {})
    pooling_config = model_config.get('pooling', {})
    
    if not pooling_config:
        print("警告: 配置中未找到池化配置，使用默认值")
        pooling_config = {
            'embedding_dim': 1280,
            'attention_num': 2,
            'dropout': 0.05,
            'save_attention': True,
            'attention_config': {'temperature': 0.3}
        }
    
    # 创建池化层
    pooling_layer = AttentionPoolingUnit.from_config(pooling_config, device)
    
    # 加载池化层权重
    ckpt = _safe_torch_load(args.model_pretrain_ckpt, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    
    pooling_state = {}
    for key, value in state.items():
        if key.startswith('pooling.'):
            new_key = key[len('pooling.'):]
            pooling_state[new_key] = value
    
    if pooling_state:
        pooling_layer.load_state_dict(pooling_state, strict=False)
        print(f"[Pooling] 加载了 {len(pooling_state)} 个权重")
    else:
        print("[Pooling] 警告: 未找到池化层权重")
    
    pooling_layer.eval()
    
    # 加载内部投影层 (从 complete_model.pth 的 input_layer.* 权重)
    internal_projector = None
    internal_state = {}
    for key, value in state.items():
        if key.startswith('input_layer.'):
            new_key = key[len('input_layer.'):]
            internal_state[new_key] = value
    
    if internal_state:
        # 从权重推断维度
        input_data_config = model_config.get('input_data', {})
        projection_config = input_data_config.get('projection', {})
        
        embedding_dim = input_data_config.get('embedding_dim', 1280)
        hidden_dim = projection_config.get('hidden_dim')
        activation = projection_config.get('activation', 'gelu')
        residual = projection_config.get('residual', True)
        layernorm = projection_config.get('layernorm', True)
        
        if hidden_dim is not None:
            internal_projector = InputLayerProjector(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                residual=residual,
                layernorm=layernorm
            )
            internal_projector.load_state_dict(internal_state, strict=False)
            internal_projector.to(device)
            internal_projector.eval()
            print(f"[InternalProjector] 加载完成: {embedding_dim} -> {hidden_dim} -> {embedding_dim}")
        else:
            print("[InternalProjector] 未配置 hidden_dim，跳过内部投影层")
    else:
        print("[InternalProjector] 未找到 input_layer.* 权重，跳过内部投影层")
    
    # 预处理层 (多模态特征模式)
    preprocessing_layer = None
    feature_folder = args.pdbfeature_folder
    
    if feature_folder is not None:
        if not os.path.isdir(feature_folder):
            print(f"错误: 特征文件夹不存在: {feature_folder}")
            sys.exit(1)
        
        preprocessing_config = model_config.get('preprocessing', {})
        if preprocessing_config.get('preprocessor') == 'feature_concat':
            print(f"[Preprocessing] 加载多模态特征: {feature_folder}")
            
            # 动态导入 FeatureConcatUnit
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from src.models.model_unit.preprocessing.FeatureConcatUnit import FeatureConcatUnit
                
                # 更新配置中的特征路径
                preprocessing_config['data_processing']['all_feature_folder'] = feature_folder
                
                preprocessing_layer = FeatureConcatUnit(
                    embedding_dim=pooling_config.get('embedding_dim', 1280),
                    **{k: v for k, v in preprocessing_config.items() if k not in ['preprocessor', 'description', 'embedding_dim']}
                )
                
                # 加载预处理层权重
                preprocessing_state = {}
                for key, value in state.items():
                    if key.startswith('preprocessing.'):
                        new_key = key[len('preprocessing.'):]
                        preprocessing_state[new_key] = value
                
                if preprocessing_state:
                    preprocessing_layer.load_state_dict(preprocessing_state, strict=False)
                    print(f"[Preprocessing] 加载了 {len(preprocessing_state)} 个权重")
                
                preprocessing_layer.to(device)
                preprocessing_layer.eval()
                
            except ImportError as e:
                print(f"警告: 无法加载 FeatureConcatUnit: {e}")
                print("将使用仅序列模式")
                preprocessing_layer = None
                feature_folder = None
        else:
            print(f"[Preprocessing] 配置为 {preprocessing_config.get('preprocessor', 'none')}，跳过多模态特征")
            feature_folder = None
    
    # 读取序列
    sequences = read_fasta_sequences(args.fasta)
    print(f"加载了 {len(sequences)} 个序列")
    
    # 处理序列
    process_and_save_pooled_embeddings(
        sequences=sequences,
        esm_model=esm_model,
        esm_tokenizer=esm_tokenizer,
        input_layer=input_layer,
        internal_projector=internal_projector,
        pooling_layer=pooling_layer,
        preprocessing_layer=preprocessing_layer,
        feature_folder=feature_folder,
        output_dir=output_dir,
        device=device,
        commit_interval=args.commit_interval,
        force=args.force,
        append=args.append
    )
    
    # 分割MLP分类器
    if args.segmentation_model:
        extract_classifier_weights(
            complete_model_path=args.model_pretrain_ckpt,
            output_dir=output_dir,
            resolved_config=resolved_config,
            device=device
        )
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()
