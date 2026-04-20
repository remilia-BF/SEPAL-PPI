"""
分类器单元模块
包含各种分类器的可配置实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class StandardMLPUnit(nn.Module):
    """
    标准MLP分类器单元
    
    多层感知机分类器，包含可配置的层数、激活函数、dropout等
    """
    
    def __init__(self,
                 input_dim: int,
                 layers: List[int] = None,
                 activation: str = "ReLU",
                 dropout: float = 0.0,
                 output_activation: str = "sigmoid",
                 **kwargs):
        """
        初始化标准MLP分类器单元
        
        Args:
            input_dim (int): 输入特征维度
            layers (List[int]): 隐藏层维度列表，默认为[1024, 512, 128, 16]
                              每个数字代表一个隐藏层的神经元数量
            activation (str): 激活函数，默认为"ReLU"
                            支持: ReLU, GELU, SiLU, Tanh, LeakyReLU
            dropout (float): dropout率，默认为0.0
                           用于防止过拟合，在每个隐藏层后应用
            output_activation (str): 输出激活函数，默认为"sigmoid"
                                   支持: sigmoid, softmax, none
            **kwargs: 额外参数
        """
        super(StandardMLPUnit, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout
        
        if layers is None:
            layers = [1024, 512, 128, 16]
        
        self.layers = layers
        
        # 选择激活函数
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 选择输出激活函数
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"不支持的输出激活函数: {output_activation}")
        
        # 构建网络层
        network_layers = []
        prev_size = input_dim
        
        # 添加隐藏层
        for hidden_size in layers:
            network_layers.append(nn.Linear(prev_size, hidden_size))
            network_layers.append(self.activation)
            if dropout > 0:
                network_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 添加输出层
        network_layers.append(nn.Linear(prev_size, 1))  # 二分类输出
        network_layers.append(self.output_activation)
        
        self.network = nn.Sequential(*network_layers)
        
        logger.debug(f"创建标准MLP分类器: input_dim={input_dim}, layers={layers}, "
                    f"activation={activation}, dropout={dropout}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [batch_size, input_dim]
            return_features (bool): 是否返回倒数第二层特征
        
        Returns:
            torch.Tensor: 分类预测 [batch_size, 1]
            torch.Tensor: 倒数第二层特征 [batch_size, feature_dim] (如果return_features=True)
        """
        if return_features:
            # 获取倒数第二层的特征
            features = None
            current_input = x
            
            # 手动遍历网络层，找到倒数第二层的输出
            for i, layer in enumerate(self.network):
                current_input = layer(current_input)
                # 倒数第二层是倒数第三个层（排除最后的线性层和激活函数）
                if i == len(self.network) - 3:
                    features = current_input
                    break
            
            # 继续完成剩余的前向传播
            for i in range(len(self.network) - 2, len(self.network)):
                current_input = self.network[i](current_input)
            
            return current_input, features
        else:
            return self.network(x)


class ResidualMLPUnit(nn.Module):
    """
    残差MLP分类器单元
    
    带残差连接的MLP分类器，有助于深层网络的训练
    """
    
    def __init__(self,
                 input_dim: int,
                 layers: List[int] = None,
                 activation: str = "ReLU",
                 dropout: float = 0.0,
                 output_activation: str = "sigmoid",
                 residual_frequency: int = 2,
                 **kwargs):
        """
        初始化残差MLP分类器单元
        
        Args:
            input_dim (int): 输入特征维度
            layers (List[int]): 隐藏层维度列表，默认为[1024, 512, 128, 16]
            activation (str): 激活函数，默认为"ReLU"
            dropout (float): dropout率，默认为0.0
            output_activation (str): 输出激活函数，默认为"sigmoid"
            residual_frequency (int): 残差连接频率，默认为2
                                    即每2层添加一个残差连接
            **kwargs: 额外参数
        """
        super(ResidualMLPUnit, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout
        self.residual_frequency = residual_frequency
        
        if layers is None:
            layers = [1024, 512, 128, 16]
        
        self.layers = layers
        
        # 选择激活函数
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 选择输出激活函数
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"不支持的输出激活函数: {output_activation}")
        
        # 构建网络层（使用ModuleList以便于残差连接）
        self.linear_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()  # 用于维度匹配的线性层
        
        prev_size = input_dim
        
        # 添加隐藏层
        for i, hidden_size in enumerate(layers):
            self.linear_layers.append(nn.Linear(prev_size, hidden_size))
            if dropout > 0:
                self.dropout_layers.append(nn.Dropout(dropout))
            else:
                self.dropout_layers.append(nn.Identity())
            
            # 如果需要残差连接且维度不匹配，添加维度匹配层
            if (i + 1) % residual_frequency == 0 and prev_size != hidden_size:
                self.residual_layers.append(nn.Linear(prev_size, hidden_size))
            else:
                self.residual_layers.append(nn.Identity())
            
            prev_size = hidden_size
        
        # 输出层
        self.output_layer = nn.Linear(prev_size, 1)
        
        logger.debug(f"创建残差MLP分类器: input_dim={input_dim}, layers={layers}, "
                    f"activation={activation}, dropout={dropout}, "
                    f"residual_frequency={residual_frequency}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [batch_size, input_dim]
            return_features (bool): 是否返回倒数第二层特征
        
        Returns:
            torch.Tensor: 分类预测 [batch_size, 1]
            torch.Tensor: 倒数第二层特征 [batch_size, feature_dim] (如果return_features=True)
        """
        residual_input = x
        features = None
        
        for i, (linear, dropout, residual_proj) in enumerate(zip(
            self.linear_layers, self.dropout_layers, self.residual_layers
        )):
            # 线性变换 + 激活
            x = self.activation(linear(x))
            x = dropout(x)
            
            # 每residual_frequency层添加残差连接
            if (i + 1) % self.residual_frequency == 0:
                # 应用残差连接
                residual_projected = residual_proj(residual_input)
                x = x + residual_projected
                residual_input = x  # 更新残差输入
            
            # 保存倒数第二层的特征
            if i == len(self.linear_layers) - 2:  # 倒数第二层
                features = x
        
        # 输出层
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        if return_features:
            return x, features
        else:
            return x


class AttentionMLPUnit(nn.Module):
    """
    注意力MLP分类器单元
    
    在特征上应用注意力机制，然后进行MLP分类
    """
    
    def __init__(self,
                 input_dim: int,
                 layers: List[int] = None,
                 activation: str = "ReLU",
                 dropout: float = 0.0,
                 output_activation: str = "sigmoid",
                 attention_heads: int = 4,
                 attention_dropout: float = 0.1,
                 **kwargs):
        """
        初始化注意力MLP分类器单元
        
        Args:
            input_dim (int): 输入特征维度
            layers (List[int]): 隐藏层维度列表，默认为[512, 128, 16]
            activation (str): 激活函数，默认为"ReLU"
            dropout (float): dropout率，默认为0.0
            output_activation (str): 输出激活函数，默认为"sigmoid"
            attention_heads (int): 注意力头数，默认为4
                                 用于对输入特征进行多头注意力处理
            attention_dropout (float): 注意力dropout率，默认为0.1
            **kwargs: 额外参数
        """
        super(AttentionMLPUnit, self).__init__()
        self.input_dim = input_dim
        self.attention_heads = attention_heads
        
        if layers is None:
            layers = [512, 128, 16]  # 注意力版本使用较小的网络
        
        # 多头注意力层（将输入特征视为单个序列元素）
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 标准MLP层
        self.mlp = StandardMLPUnit(
            input_dim=input_dim,
            layers=layers,
            activation=activation,
            dropout=dropout,
            output_activation=output_activation
        )
        
        logger.debug(f"创建注意力MLP分类器: input_dim={input_dim}, layers={layers}, "
                    f"attention_heads={attention_heads}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [batch_size, input_dim]
            return_features (bool): 是否返回倒数第二层特征
        
        Returns:
            torch.Tensor: 分类预测 [batch_size, 1]
            torch.Tensor: 倒数第二层特征 [batch_size, feature_dim] (如果return_features=True)
        """
        # 为注意力机制添加序列维度
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 自注意力（query=key=value）
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        
        # 层归一化 + 残差连接
        x_attended = self.layer_norm(attn_output + x_seq)
        
        # 移除序列维度
        x_attended = x_attended.squeeze(1)  # [batch_size, input_dim]
        
        # MLP分类
        if return_features:
            return self.mlp(x_attended, return_features=True)
        else:
            return self.mlp(x_attended)


class MutiMLPUnit(nn.Module):
    """
    多专家 MLP 分类器

    输入期望为 [batch_size, head_num, input_dim]。
    每个 head 进入独立 MLP 专家，随后通过 head-gating 融合为最终预测。
    """

    def __init__(
        self,
        input_dim: int,
        head: int = 8,
        layers: List[int] = None,
        activation: str = "GELU",
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        gate_hidden_dim: int = 64,
        gate_dropout: float = 0.1,
        gate_entropy_weight: float = 0.0,
        **kwargs,
    ):
        super(MutiMLPUnit, self).__init__()
        self.input_dim = int(input_dim)
        self.head = int(head)
        self.output_activation_type = output_activation
        self.gate_entropy_weight = float(gate_entropy_weight)

        if layers is None:
            layers = [1024, 512, 128, 16]
        self.layers = list(layers)
        self.activation_name = activation

        self.activation = self._build_activation(activation)
        self.dropout_rate = float(dropout)

        # 每个 head 一个专家网络（输出 logit）
        self.experts = nn.ModuleList([
            self._build_expert_network(self.input_dim, self.layers, self.activation_name, self.dropout_rate)
            for _ in range(self.head)
        ])

        gate_input_dim = self.head * self.input_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            self._build_activation(activation),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, self.head)
        )

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        elif output_activation == "softmax":
            # 二分类场景下退化为 sigmoid 的语义
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的输出激活函数: {output_activation}")

        self._last_gate_weights = None
        self._last_gate_entropy_loss = None

        logger.debug(
            f"创建多专家MLP分类器: input_dim={input_dim}, head={head}, layers={layers}, "
            f"activation={activation}, dropout={dropout}, gate_hidden_dim={gate_hidden_dim}, "
            f"gate_entropy_weight={self.gate_entropy_weight}"
        )

    def _build_activation(self, name: str) -> nn.Module:
        if name == "ReLU":
            return nn.ReLU()
        if name == "GELU":
            return nn.GELU()
        if name == "SiLU":
            return nn.SiLU()
        if name == "Tanh":
            return nn.Tanh()
        if name == "LeakyReLU":
            return nn.LeakyReLU()
        raise ValueError(f"不支持的激活函数: {name}")

    def _build_expert_network(self, input_dim: int, layers: List[int], activation_name: str, dropout: float) -> nn.Sequential:
        net_layers = []
        prev_size = input_dim
        for hidden_size in layers:
            net_layers.append(nn.Linear(prev_size, hidden_size))
            net_layers.append(self._build_activation(activation_name))
            if dropout > 0:
                net_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        net_layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*net_layers)

    def _expert_hidden(self, x_head: torch.Tensor, expert: nn.Sequential) -> torch.Tensor:
        if len(expert) <= 1:
            return x_head
        hidden = x_head
        for layer in expert[:-1]:
            hidden = layer(hidden)
        return hidden

    def get_last_gate_weights(self) -> Optional[torch.Tensor]:
        if self._last_gate_weights is None:
            return None
        return self._last_gate_weights.detach().cpu()

    def get_gate_entropy_regularization_loss(self) -> Optional[torch.Tensor]:
        if self._last_gate_entropy_loss is None:
            return None
        return self._last_gate_entropy_loss

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, H, D] 或 [B, D]
        """
        if x.dim() == 2:
            # 兼容错误输入：复制到所有 head
            x = x.unsqueeze(1).expand(-1, self.head, -1)
        elif x.dim() != 3:
            raise ValueError(f"MutiMLPUnit expects input shape [B,H,D] or [B,D], got {tuple(x.shape)}")

        batch_size, in_heads, in_dim = x.shape
        if in_dim != self.input_dim:
            raise ValueError(f"MutiMLPUnit input_dim mismatch: expected {self.input_dim}, got {in_dim}")

        if in_heads < self.head:
            pad = torch.zeros((batch_size, self.head - in_heads, in_dim), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif in_heads > self.head:
            x = x[:, :self.head, :]

        gate_logits = self.gate_network(x.reshape(batch_size, -1))
        gate_weights = F.softmax(gate_logits, dim=1)
        self._last_gate_weights = gate_weights

        # 熵正则：鼓励 gate 分布更均匀（最小化 KL(gate || Uniform)）
        if self.head > 1:
            gate_clamped = torch.clamp(gate_weights, min=1e-8)
            entropy = -(gate_clamped * torch.log(gate_clamped)).sum(dim=1)  # [B]
            max_entropy = torch.log(torch.tensor(float(self.head), device=gate_weights.device, dtype=gate_weights.dtype))
            kl_to_uniform = max_entropy - entropy
            self._last_gate_entropy_loss = kl_to_uniform.mean() * self.gate_entropy_weight
        else:
            self._last_gate_entropy_loss = torch.zeros((), device=gate_weights.device, dtype=gate_weights.dtype)

        expert_logits = []
        expert_hidden = []
        for h in range(self.head):
            x_h = x[:, h, :]
            expert = self.experts[h]
            logit_h = expert(x_h)
            expert_logits.append(logit_h)
            if return_features:
                expert_hidden.append(self._expert_hidden(x_h, expert))

        expert_logits = torch.stack(expert_logits, dim=1).squeeze(-1)  # [B, H]
        fused_logit = (expert_logits * gate_weights).sum(dim=1, keepdim=True)
        prediction = self.output_activation(fused_logit)

        if return_features:
            if expert_hidden:
                hidden_stack = torch.stack(expert_hidden, dim=1)  # [B, H, D_hidden]
                fused_hidden = (hidden_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
            else:
                fused_hidden = x.mean(dim=1)
            return prediction, fused_hidden
        return prediction


def create_classifier_unit(classifier_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建分类器单元
    
    Args:
        classifier_config (Dict): 分类器配置，包含type和其他参数
                                必须包含: type (str) - 分类器类型
                                可选包含: input_dim, layers, activation等
    
    Returns:
        nn.Module: 分类器单元实例
    
    Raises:
        ValueError: 当分类器类型不支持时
    
    Examples:
        >>> config = {
        ...     'type': 'MLP',
        ...     'input_dim': 1280,
        ...     'layers': [1024, 512, 128, 16],
        ...     'activation': 'ReLU'
        ... }
        >>> classifier = create_classifier_unit(config)
        
        >>> config = {
        ...     'type': 'residual_MLP',
        ...     'input_dim': 1280,
        ...     'layers': [1024, 512, 128, 16],
        ...     'residual_frequency': 2
        ... }
        >>> classifier = create_classifier_unit(config)
    """
    classifier_type = classifier_config.get('type', 'MLP')
    input_dim = classifier_config.get('input_dim', None)
    
    if input_dim is None:
        raise ValueError("分类器单元需要指定input_dim参数")
    
    # 移除type和input_dim，将其余参数传递给具体的分类器单元
    kwargs = {k: v for k, v in classifier_config.items() if k not in ['type', 'input_dim']}
    
    if classifier_type == 'MLP':
        return StandardMLPUnit(input_dim=input_dim, **kwargs)
    elif classifier_type == 'residual_MLP':
        return ResidualMLPUnit(input_dim=input_dim, **kwargs)
    elif classifier_type == 'attention_MLP':
        return AttentionMLPUnit(input_dim=input_dim, **kwargs)
    elif classifier_type == 'muti_MLP':
        return MutiMLPUnit(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}. "
                        f"支持的类型: MLP, residual_MLP, attention_MLP, muti_MLP")