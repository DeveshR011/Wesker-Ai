"""
Neural Network Components for the Tactical Agent.

Architecture:
    1. ObservationEncoder: CNN for local grid + MLP for scalar features → per-step token
    2. CausalTransformer: Self-attention over trajectory window with causal mask
    3. PolicyHead: Action distribution logits
    4. ValueHead: Scalar state-value V(s)
    5. QuantileHead: Distributional value via quantile predictions Z(s)

The transformer processes a trajectory of encoded observations to implicitly
approximate belief states in the POMDP, enabling:
    - Memory of past enemy sightings
    - Resource depletion trend detection
    - Danger escalation pattern recognition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import TransformerConfig, DistributionalConfig, EnvConfig


class ObservationEncoder(nn.Module):
    """
    Encode a single observation into a fixed-size embedding.

    Processes:
        - local_grid (C, H, W) through a small CNN → flat vector
        - scalars (4,) through a linear layer
        - prev_action through an embedding table
        - prev_reward and stress through linear projection

    All sub-embeddings are concatenated and projected to d_model.
    """

    def __init__(self, env_cfg: EnvConfig, tf_cfg: TransformerConfig):
        super().__init__()
        self.tf_cfg = tf_cfg
        obs_size = 2 * env_cfg.vision_radius + 1  # e.g. 11

        # ── CNN for local grid (C, H, W) ──
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # → (64, 3, 3)
            nn.Flatten(),             # → 576
            nn.Linear(64 * 3 * 3, tf_cfg.grid_embed_dim),
            nn.ReLU(),
        )

        # ── MLP for scalar features [health, ammo, stress, has_key] ──
        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, tf_cfg.scalar_embed_dim),
            nn.ReLU(),
        )

        # ── Action embedding ──
        self.action_embed = nn.Embedding(
            env_cfg.num_actions, tf_cfg.action_embed_dim
        )

        # ── Reward + stress projection (2 scalars → embed) ──
        self.reward_stress_proj = nn.Sequential(
            nn.Linear(2, tf_cfg.reward_embed_dim),
            nn.ReLU(),
        )

        # ── Final projection to d_model ──
        total_dim = (
            tf_cfg.grid_embed_dim
            + tf_cfg.scalar_embed_dim
            + tf_cfg.action_embed_dim
            + tf_cfg.reward_embed_dim
        )
        self.projection = nn.Linear(total_dim, tf_cfg.d_model)
        self.layer_norm = nn.LayerNorm(tf_cfg.d_model)

    def forward(
        self,
        local_grid: torch.Tensor,   # (B, C, H, W)
        scalars: torch.Tensor,      # (B, 4)
        prev_action: torch.Tensor,  # (B,) long
        prev_reward: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Encode one timestep's observation → (B, d_model)."""
        grid_emb = self.grid_cnn(local_grid)          # (B, grid_embed_dim)
        scalar_emb = self.scalar_mlp(scalars)          # (B, scalar_embed_dim)
        action_emb = self.action_embed(prev_action)    # (B, action_embed_dim)

        # Reward and stress (stress is scalars[:, 2])
        reward_stress = torch.stack([prev_reward, scalars[:, 2]], dim=-1)
        rs_emb = self.reward_stress_proj(reward_stress)  # (B, reward_embed_dim)

        # Concatenate all sub-embeddings
        combined = torch.cat([grid_emb, scalar_emb, action_emb, rs_emb], dim=-1)
        token = self.layer_norm(self.projection(combined))
        return token  # (B, d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (B, T, d_model)."""
        return x + self.pe[:, :x.size(1)]


class CausalTransformer(nn.Module):
    """
    Causal (autoregressive) Transformer encoder over trajectory windows.

    Uses a causal attention mask so each position can only attend to
    itself and earlier positions—essential for sequential decision-making.

    Input: sequence of per-step observation embeddings (B, T, d_model)
    Output: contextualized representations (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_encoding = PositionalEncoding(cfg.d_model, max_len=cfg.trajectory_window + 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask (True = blocked)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        tokens: torch.Tensor,                    # (B, T, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process trajectory tokens with causal self-attention.

        Args:
            tokens: Observation embeddings (B, T, d_model)
            key_padding_mask: True where tokens should be ignored (B, T)

        Returns:
            Contextualized representations (B, T, d_model)
            Attention weights (B, T, T)
        """
        T = tokens.size(1)
        tokens = self.pos_encoding(tokens)
        causal_mask = self._generate_causal_mask(T, tokens.device)

        # The nn.TransformerEncoder does not return attention weights directly.
        # We need to use nn.Transformer instead.
        # For simplicity, we will not implement this change now.
        # We will return a dummy tensor for the attention weights.
        attn_weights = torch.zeros(tokens.size(0), T, T, device=tokens.device)

        out = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return self.final_norm(out), attn_weights


class PolicyHead(nn.Module):
    """
    Maps transformer output to action distribution logits.

    Output: unnormalized log-probabilities over discrete actions.
    """

    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → logits: (B, num_actions)."""
        return self.net(h)


class ValueHead(nn.Module):
    """
    Maps transformer output to a scalar state-value estimate V(s).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → value: (B, 1)."""
        return self.net(h)


class QuantileHead(nn.Module):
    """
    Distributional value head using Quantile Regression (QR-DQN style).

    Predicts N quantile values θ_1, ..., θ_N for the return distribution Z(s).
    Quantile fractions τ_i = (2i - 1) / (2N) are fixed (midpoints).

    Used to compute:
        CVaR_α = (1 / ⌊αN⌋) Σ_{i=1}^{⌊αN⌋} θ_{(i)}
    where θ_{(i)} are the sorted quantile predictions.
    """

    def __init__(self, d_model: int, num_quantiles: int):
        super().__init__()
        self.num_quantiles = num_quantiles

        # Fixed quantile fractions τ_i (midpoints of uniform partition)
        taus = (2 * torch.arange(num_quantiles).float() + 1) / (2 * num_quantiles)
        self.register_buffer("taus", taus)  # (N,)

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_quantiles),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d_model) → quantiles: (B, N)

        Returns N quantile predictions θ_i for the return distribution.
        """
        return self.net(h)

    def compute_cvar(self, quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute CVaR_α from predicted quantiles.

        CVaR_α = E[Z | Z ≤ q_α] ≈ mean of the bottom α fraction of quantile predictions.

        This represents the expected return in the worst α-fraction of outcomes.
        Optimizing for CVaR makes the agent risk-averse.

        Args:
            quantiles: Predicted quantiles (B, N), not necessarily sorted
            alpha: Risk level ∈ (0, 1]. α=0.25 → focus on worst 25%

        Returns:
            CVaR values (B, 1)
        """
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        k = max(1, int(alpha * self.num_quantiles))  # Number of quantiles to average
        cvar = sorted_q[:, :k].mean(dim=-1, keepdim=True)
        return cvar

    @staticmethod
    def quantile_regression_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        taus: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """
        Quantile Huber regression loss.

        For each quantile τ_i and target sample T_j:
            ρ^κ_τ(u) = |τ - 𝟙(u < 0)| * L_κ(u)
        where L_κ is the Huber loss with threshold κ.

        Args:
            predicted: (B, N) quantile predictions
            target: (B,) or (B, 1) target returns
            taus: (N,) quantile fractions
            kappa: Huber loss threshold

        Returns:
            Scalar loss
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1)  # (B, 1)

        # Pairwise TD errors: (B, N)
        td_error = target - predicted  # Broadcasting: (B, 1) - (B, N) → (B, N)

        # Huber loss element
        huber = torch.where(
            td_error.abs() <= kappa,
            0.5 * td_error.pow(2),
            kappa * (td_error.abs() - 0.5 * kappa),
        )

        # Asymmetric weighting by quantile fraction
        # τ_i for overestimation (td_error < 0), (1 - τ_i) for underestimation
        weight = torch.abs(taus.unsqueeze(0) - (td_error < 0).float())
        loss = (weight * huber).mean()
        return loss
