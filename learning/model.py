"""
In-Context Dynamics Transformer for parafoil systems.

Architecture:
  - Input projection MLP: token_dim -> d_model
  - Transformer encoder with learnable positional encoding
  - Per-horizon output heads: d_model -> target_dim x H

The model takes a context window of K past (state, action) tokens
and predicts H future state deltas.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer dynamics model configuration."""
    token_dim: int = 23       # input token dimension
    target_dim: int = 17      # prediction target dimension per step
    d_model: int = 128        # transformer hidden dimension
    n_heads: int = 4          # attention heads
    n_layers: int = 4         # transformer encoder layers
    d_ff: int = 512           # feedforward hidden dimension
    dropout: float = 0.1
    max_context_length: int = 100
    prediction_horizon: int = 20
    # Input/output projection
    proj_hidden: int = 128    # hidden dim in input projection MLP
    shared_output_head: bool = False  # if True, share head across horizons


class InputProjection(nn.Module):
    """2-layer MLP with LayerNorm to project tokens into d_model space."""

    def __init__(self, token_dim: int, d_model: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.net(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence positions."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return self.dropout(x + self.pos_embed[:, :T, :])


class OutputHead(nn.Module):
    """MLP output head for a single prediction horizon step."""

    def __init__(self, d_model: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        return self.net(x)


class InContextDynamicsTransformer(nn.Module):
    """
    Transformer-based in-context dynamics model.

    Input:  context tokens (B, K, token_dim)
    Output: predicted state deltas (B, H, target_dim)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = InputProjection(
            config.token_dim, config.d_model,
            config.proj_hidden, config.dropout
        )

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(
            config.max_context_length, config.d_model, config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Output heads
        if config.shared_output_head:
            self.output_heads = nn.ModuleList([
                OutputHead(config.d_model, config.target_dim, config.d_ff // 2)
            ])
        else:
            self.output_heads = nn.ModuleList([
                OutputHead(config.d_model, config.target_dim, config.d_ff // 2)
                for _ in range(config.prediction_horizon)
            ])

        # Horizon query embeddings (learned queries for each future step)
        self.horizon_queries = nn.Parameter(
            torch.randn(1, config.prediction_horizon, config.d_model) * 0.02
        )

        # Cross-attention from horizon queries to encoder output
        self.cross_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.cross_ff_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, context: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context: (B, K, token_dim) - context window
            padding_mask: (B, K) - True for padded positions

        Returns:
            predictions: (B, H, target_dim) - predicted state deltas
        """
        B = context.size(0)
        H = self.config.prediction_horizon

        # Project and add positional encoding
        x = self.input_proj(context)         # (B, K, d_model)
        x = self.pos_enc(x)                  # (B, K, d_model)

        # Transformer encoding
        memory = self.transformer(x, src_key_padding_mask=padding_mask)
        # memory: (B, K, d_model)

        # Cross-attention: horizon queries attend to encoder memory
        queries = self.horizon_queries.expand(B, -1, -1)  # (B, H, d_model)

        attn_out, _ = self.cross_attn(queries, memory, memory,
                                       key_padding_mask=padding_mask)
        queries = self.cross_norm(queries + attn_out)
        queries = self.cross_ff_norm(queries + self.cross_ff(queries))
        # queries: (B, H, d_model)

        # Per-step output heads
        predictions = torch.zeros(B, H, self.config.target_dim,
                                  device=context.device, dtype=context.dtype)
        for i in range(H):
            head_idx = 0 if self.config.shared_output_head else i
            predictions[:, i, :] = self.output_heads[head_idx](queries[:, i, :])

        return predictions

    def predict_single_step(self, context: torch.Tensor,
                            padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict only the next single step (used by rollout loss).

        Args:
            context: (B, K, token_dim)
        Returns:
            delta: (B, target_dim) — predicted state delta for step 1 only
        """
        B = context.size(0)

        x = self.input_proj(context)
        x = self.pos_enc(x)
        memory = self.transformer(x, src_key_padding_mask=padding_mask)

        query = self.horizon_queries[:, :1, :].expand(B, -1, -1)  # (B, 1, d_model)
        attn_out, _ = self.cross_attn(query, memory, memory,
                                       key_padding_mask=padding_mask)
        query = self.cross_norm(query + attn_out)
        query = self.cross_ff_norm(query + self.cross_ff(query))

        head_idx = 0 if self.config.shared_output_head else 0
        return self.output_heads[head_idx](query[:, 0, :])  # (B, target_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
#           Baseline: simple MLP dynamics model
# ============================================================

class MLPDynamicsModel(nn.Module):
    """
    Single-step MLP baseline: (state, action) -> delta_state.
    No context / no in-context adaptation.
    """

    def __init__(self, state_dim: int = 23, action_dim: int = 2,
                 target_dim: int = 17, hidden_dims=(256, 256, 256),
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
        Returns:
            delta:  (B, target_dim)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
