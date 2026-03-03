"""
Tactical Agent — the core decision-making module.

Combines:
    - ObservationEncoder: per-step observation → token
    - CausalTransformer: trajectory of tokens → contextualized representations
    - PolicyHead: action distribution
    - ValueHead: scalar V(s) estimate
    - QuantileHead: distributional value Z(s)

The agent processes a trajectory window of observations and outputs:
    1. Action distribution π(a|h_t) for policy gradient training
    2. State value V(s_t) for advantage computation
    3. Quantile predictions θ_i(s_t) for CVaR-based risk-sensitive training

When CVaR is enabled (ablation flag), the advantage uses:
    A_t = G_t − CVaR_α(s_t)  instead of  A_t = G_t − V(s_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, NamedTuple

from .config import Config
from .networks import (
    ObservationEncoder,
    CausalTransformer,
    PolicyHead,
    ValueHead,
    QuantileHead,
)


class AgentOutput(NamedTuple):
    """Structured output from the tactical agent's forward pass."""
    policy_logits: torch.Tensor    # (B, num_actions)
    value: torch.Tensor            # (B, 1)
    quantiles: Optional[torch.Tensor]  # (B, N_quantiles) or None
    cvar: Optional[torch.Tensor]       # (B, 1) or None
    attn_weights: torch.Tensor     # (B, T, T)


class TacticalAgent(nn.Module):
    """
    Transformer-based tactical agent for survival decision-making.

    Processes a sliding window of trajectory data through a causal transformer
    to produce policy, value, and distributional value outputs.

    The agent implicitly approximates the POMDP belief state through
    self-attention over observation history.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        env_cfg = config.env
        tf_cfg = config.transformer
        dist_cfg = config.distributional

        # ── Observation encoder: obs → per-step token ──
        self.obs_encoder = ObservationEncoder(env_cfg, tf_cfg)

        # ── Causal transformer: trajectory tokens → contextualized repr ──
        self.transformer = CausalTransformer(tf_cfg)

        # ── Output heads ──
        self.policy_head = PolicyHead(tf_cfg.d_model, env_cfg.num_actions)
        self.value_head = ValueHead(tf_cfg.d_model)

        # Distributional head (optional via ablation)
        self.quantile_head = None
        if config.ablation.use_distributional_value:
            self.quantile_head = QuantileHead(tf_cfg.d_model, dist_cfg.num_quantiles)

    def forward(
        self,
        local_grids: torch.Tensor,    # (B, T, C, H, W)
        scalars: torch.Tensor,        # (B, T, 4)  [health, ammo, stress, has_key]
        prev_actions: torch.Tensor,   # (B, T) long
        prev_rewards: torch.Tensor,   # (B, T)
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> AgentOutput:
        """
        Full forward pass over a trajectory window.

        Args:
            local_grids: Observation grids for each timestep
            scalars: [health, ammo, stress] for each timestep
            prev_actions: Previous action indices
            prev_rewards: Previous step rewards
            padding_mask: True where timesteps are padding (B, T)

        Returns:
            AgentOutput with policy logits, value, and optional quantiles/cvar
            (all computed at the LAST valid timestep in the sequence)
        """
        B, T = local_grids.shape[:2]

        # ── Encode each timestep independently ──
        # Reshape to batch all timesteps: (B*T, ...)
        grids_flat = local_grids.reshape(B * T, *local_grids.shape[2:])
        scalars_flat = scalars.reshape(B * T, -1)
        actions_flat = prev_actions.reshape(B * T)
        rewards_flat = prev_rewards.reshape(B * T)

        tokens = self.obs_encoder(
            grids_flat, scalars_flat, actions_flat, rewards_flat
        )  # (B*T, d_model)
        tokens = tokens.reshape(B, T, -1)  # (B, T, d_model)

        # ── Causal transformer ──
        h, attn_weights = self.transformer(tokens, key_padding_mask=padding_mask)  # (B, T, d_model)

        # ── Extract last valid timestep representation ──
        if padding_mask is not None:
            # Find last non-padded position for each batch element
            lengths = (~padding_mask).long().sum(dim=1) - 1  # (B,)
            lengths = lengths.clamp(min=0)
            h_last = h[torch.arange(B, device=h.device), lengths]  # (B, d_model)
        else:
            h_last = h[:, -1]  # (B, d_model)

        # ── Output heads ──
        policy_logits = self.policy_head(h_last)  # (B, num_actions)
        value = self.value_head(h_last)            # (B, 1)

        quantiles = None
        cvar = None
        if self.quantile_head is not None:
            quantiles = self.quantile_head(h_last)  # (B, N)
            cvar = self.quantile_head.compute_cvar(
                quantiles, self.config.distributional.cvar_alpha
            )  # (B, 1)

        return AgentOutput(
            policy_logits=policy_logits,
            value=value,
            quantiles=quantiles,
            cvar=cvar,
            attn_weights=attn_weights,
        )

    def forward_sequence(
        self,
        local_grids: torch.Tensor,
        scalars: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass returning outputs at ALL timesteps (for training).

        Returns:
            policy_logits: (B, T, num_actions)
            values: (B, T, 1)
            quantiles: (B, T, N) or None
            attn_weights: (B, T, T)
        """
        B, T = local_grids.shape[:2]

        # Encode all timesteps
        grids_flat = local_grids.reshape(B * T, *local_grids.shape[2:])
        scalars_flat = scalars.reshape(B * T, -1)
        actions_flat = prev_actions.reshape(B * T)
        rewards_flat = prev_rewards.reshape(B * T)

        tokens = self.obs_encoder(
            grids_flat, scalars_flat, actions_flat, rewards_flat
        ).reshape(B, T, -1)

        h, attn_weights = self.transformer(tokens, key_padding_mask=padding_mask)  # (B, T, d_model)

        # Compute heads at all timesteps
        h_flat = h.reshape(B * T, -1)
        policy_logits = self.policy_head(h_flat).reshape(B, T, -1)
        values = self.value_head(h_flat).reshape(B, T, 1)

        quantiles = None
        if self.quantile_head is not None:
            quantiles = self.quantile_head(h_flat).reshape(B, T, -1)

        return policy_logits, values, quantiles, attn_weights

    @torch.no_grad()
    def select_action(
        self,
        local_grids: torch.Tensor,
        scalars: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given the current trajectory window.

        Args:
            local_grids: (1, T, C, H, W)
            scalars: (1, T, 3)
            prev_actions: (1, T) long
            prev_rewards: (1, T)
            deterministic: If True, take argmax instead of sampling

        Returns:
            (action_int, log_prob, value)
        """
        output = self.forward(local_grids, scalars, prev_actions, prev_rewards)
        dist = Categorical(logits=output.policy_logits)

        if deterministic:
            action = output.policy_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(), output.value.squeeze()

    def get_value_for_advantage(self, output: AgentOutput) -> torch.Tensor:
        """
        Get the value estimate to use for advantage computation.

        If CVaR is enabled, uses CVaR_α(s) instead of V(s).
        This makes advantages risk-sensitive—penalizing states where
        the tail risk is high even if the mean value is acceptable.

        Returns:
            Value estimate (B, 1)
        """
        if (self.config.ablation.use_cvar_objective
                and output.cvar is not None):
            return output.cvar
        return output.value
