import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class IdentityInputLayer(nn.Module):
	"""无投影，直接输出输入张量。"""

	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x


class LinearProjectionInputLayer(nn.Module):
	"""
	将残基级嵌入从 input_dim 线性投影到 embedding_dim。

	预期输入形状: [batch, L, input_dim]
	输出形状: [batch, L, embedding_dim]
	"""

	def __init__(
		self,
		input_dim: int,
		embedding_dim: int,
		dropout: float = 0.0,
		bias: bool = True,
		activation: Optional[str] = None,
	):
		super().__init__()
		self.input_dim = int(input_dim)
		self.embedding_dim = int(embedding_dim)
		self.proj = nn.Linear(self.input_dim, self.embedding_dim, bias=bias)
		self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
		if activation is None or activation == 'none':
			self.act = nn.Identity()
		elif activation.lower() == 'relu':
			self.act = nn.ReLU()
		elif activation.lower() == 'gelu':
			self.act = nn.GELU()
		elif activation.lower() == 'silu' or activation.lower() == 'swish':
			self.act = nn.SiLU()
		else:
			# 未知激活则回退为 Identity
			self.act = nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, L, C_in]
		B, L, C = x.shape
		if C != self.input_dim:
			# 允许动态输入，但不匹配时仍会线性映射（需维度对齐）；
			# 如果完全不匹配，nn.Linear 会抛错，便于及早发现问题。
			pass
		y = self.proj(x)
		y = self.act(y)
		y = self.dropout(y)
		return y



class MLPProjectionInputLayer(nn.Module):
	"""
	MLP Projector with optional Residual and LayerNorm.
	Structure: Linear(in, hidden) -> Act -> Dropout -> Linear(hidden, out)
	"""

	def __init__(
		self,
		input_dim: int,
		embedding_dim: int,
		hidden_dim: int,
		dropout: float = 0.0,
		bias: bool = True,
		activation: Optional[str] = "relu",
		residual: bool = False,
		layernorm: bool = False,
	):
		super().__init__()
		self.input_dim = int(input_dim)
		self.embedding_dim = int(embedding_dim)
		self.hidden_dim = int(hidden_dim)
		self.residual = residual
		self.layernorm = layernorm

		self.proj1 = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
		self.proj2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=bias)

		self.dropout = (
			nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
		)

		if activation is None or activation == "none":
			self.act = nn.Identity()
		elif activation.lower() == "relu":
			self.act = nn.ReLU()
		elif activation.lower() == "gelu":
			self.act = nn.GELU()
		elif activation.lower() == "silu" or activation.lower() == "swish":
			self.act = nn.SiLU()
		else:
			self.act = nn.Identity()

		if self.residual and self.input_dim != self.embedding_dim:
			# Shortcut projection
			self.shortcut = nn.Linear(self.input_dim, self.embedding_dim, bias=bias)
		else:
			self.shortcut = nn.Identity()

		if self.layernorm:
			self.ln = nn.LayerNorm(self.embedding_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, L, C_in]

		# Main path
		y = self.proj1(x)
		y = self.act(y)
		y = self.dropout(y)
		y = self.proj2(y)

		# Residual
		if self.residual:
			if self.input_dim == self.embedding_dim:
				y = y + x
			else:
				y = y + self.shortcut(x)

		# LayerNorm
		if self.layernorm:
			y = self.ln(y)

		return y


def create_input_layer(config: Dict[str, Any]) -> nn.Module:
	"""
	创建输入层。

	配置字段:
	  - input_dim: 上游嵌入维度 (来自数据/LMDB)。默认与 embedding_dim 相同。
	  - embedding_dim: 模型目标维度（必需）。
	  - projection:
	      type: 'linear' | 'mlp'
	      enabled: bool
	      dropout: float
	      bias: bool
	      activation: str
	      hidden_dim: int (MLP only)
	      residual: bool (MLP only)
	      layernorm: bool (MLP only)

	当 enabled=false 且 input_dim==embedding_dim 时，返回 Identity。
	当 enabled=true 或 input_dim!=embedding_dim 时，返回选定的投影层。
	"""
	embedding_dim = int(config.get('embedding_dim'))
	input_dim = int(config.get('input_dim', embedding_dim))
	proj_cfg = config.get('projection', {}) or {}
	
	enabled = bool(proj_cfg.get('enabled', False))
	dropout = float(proj_cfg.get('dropout', 0.0))
	bias = bool(proj_cfg.get('bias', True))
	activation = proj_cfg.get('activation', None)

	# MLP specific config
	proj_type = proj_cfg.get('type', 'linear')
	hidden_dim = int(proj_cfg.get('hidden_dim', 2560))
	residual = bool(proj_cfg.get('residual', False))
	layernorm = bool(proj_cfg.get('layernorm', False))

	# 启用条件：
	# - 若 projection.enabled 为 True，强制启用
	# - 若 input_dim 与 embedding_dim 不一致，自动启用
	force_enable = enabled
	auto_enable = (input_dim != embedding_dim)
	if not (force_enable or auto_enable):
		return IdentityInputLayer()
	
	if proj_type == 'mlp':
		return MLPProjectionInputLayer(
			input_dim=input_dim,
			embedding_dim=embedding_dim,
			hidden_dim=hidden_dim,
			dropout=dropout,
			bias=bias,
			activation=activation,
			residual=residual,
			layernorm=layernorm,
		)
	else:
		return LinearProjectionInputLayer(
			input_dim=input_dim,
			embedding_dim=embedding_dim,
			dropout=dropout,
			bias=bias,
			activation=activation,
		)
