"""
World Model — Latent State Space Model for Imagination-Based Planning.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Observation (grid + scalars)                            │
    │         │                                                │
    │  ┌──────▼───────┐                                        │
    │  │ ObsEncoder   │  CNN + MLP → [μ, logσ] → z ∈ ℝ^L     │
    │  └──────────────┘                                        │
    │         │ z_t                                            │
    │  ┌──────▼─────────────┐                                  │
    │  │ LatentTransition   │  GRU: (z_t, a_t) → z_{t+1}     │
    │  └──────────────────┘                                    │
    │         │ z_{t+1}                                        │
    │  ┌──────▼──────────┐                                     │
    │  │ RewardPredictor │  MLP: (z_t, a_t) → r̂_t            │
    │  └─────────────────┘                                     │
    │  ┌──────────────────┐                                    │
    │  │  ObsDecoder      │  z → reconstructed obs (training) │
    │  └──────────────────┘                                    │
    └──────────────────────────────────────────────────────────┘

Training losses:
    L = w_recon · L_recon + β · L_KL + w_reward · L_reward + w_cons · L_cons

Usage for MCTS:
    z0 = world_model.encode(obs)
    for action in candidate_actions:
        z1 = world_model.transition(z0, action)
        r  = world_model.predict_reward(z0, action)
        # recurse deeper for lookahead...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .config import Config, WorldModelConfig, EnvConfig


# ─── Observation Encoder (VAE-style) ─────────────────────────────────────────

class ObsVAEEncoder(nn.Module):
    """
    Encode a raw environment observation → latent distribution (μ, logσ).

    Input:
        grid:    (B, C, H, W) local observation grid
        scalars: (B, 4) [health_norm, ammo_norm, stress, has_key]

    Output:
        mu:     (B, latent_dim)
        logvar: (B, latent_dim)
    """

    def __init__(self, env_cfg: EnvConfig, wm_cfg: WorldModelConfig):
        super().__init__()
        obs_size = 2 * env_cfg.vision_radius + 1

        # CNN backbone for local grid
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(3),   # → (64, 3, 3)
            nn.Flatten(),              # → 576
            nn.Linear(64 * 3 * 3, wm_cfg.hidden_dim),
            nn.ELU(),
        )

        # Scalar encoder
        self.scalar_enc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
        )

        # Project to latent distribution parameters
        combined = wm_cfg.hidden_dim + 32
        self.mu_head = nn.Linear(combined, wm_cfg.latent_dim)
        self.logvar_head = nn.Linear(combined, wm_cfg.latent_dim)

    def forward(
        self,
        grid: torch.Tensor,     # (B, C, H, W)
        scalars: torch.Tensor,  # (B, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar), each (B, latent_dim)."""
        cnn_feat = self.cnn(grid)              # (B, hidden_dim)
        sc_feat = self.scalar_enc(scalars)     # (B, 32)
        combined = torch.cat([cnn_feat, sc_feat], dim=-1)
        mu = self.mu_head(combined)
        logvar = self.logvar_head(combined)
        return mu, logvar


# ─── Observation Decoder ─────────────────────────────────────────────────────

class ObsDecoder(nn.Module):
    """
    Decode latent z back to observation grid for reconstruction loss.

    z: (B, latent_dim) → reconstructed_grid: (B, C, H, W)
    """

    def __init__(self, env_cfg: EnvConfig, wm_cfg: WorldModelConfig):
        super().__init__()
        obs_size = 2 * env_cfg.vision_radius + 1
        C = env_cfg.num_obs_channels

        # Decode latent to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(wm_cfg.latent_dim, wm_cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(wm_cfg.hidden_dim, 64 * 3 * 3),
            nn.ELU(),
        )

        # Transposed conv to reconstruct grid
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 3, 3)),           # (B, 64, 3, 3)
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ELU(),
            nn.Upsample(size=(obs_size, obs_size), mode="bilinear", align_corners=False),
            nn.Conv2d(32, C, kernel_size=1),
            nn.Sigmoid(),                           # Pixel values ∈ [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, C, H, W)."""
        feat = self.fc(z)
        return self.deconv(feat)


# ─── Latent Transition Model ──────────────────────────────────────────────────

class LatentTransitionModel(nn.Module):
    """
    Predict next latent state from (z_t, a_t).

    z_{t+1} = f(z_t, a_t)

    Implemented as a GRU cell so it can carry recurrent dynamics.
    The deterministic transition serves as the backbone for MCTS rollouts.

    A "consistency loss" penalizes the transition output from drifting far
    from the next-step encoder output, incentivizing accurate imagination.
    """

    def __init__(self, wm_cfg: WorldModelConfig, num_actions: int):
        super().__init__()
        self.latent_dim = wm_cfg.latent_dim
        self.action_embed = nn.Embedding(num_actions, 16)

        # GRU-based transition: treat z as hidden state, action embedding as input
        self.gru = nn.GRUCell(
            input_size=16,          # action embedding
            hidden_size=wm_cfg.latent_dim,
        )

        # Optional refinement layer
        self.refine = nn.Sequential(
            nn.Linear(wm_cfg.latent_dim, wm_cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(wm_cfg.hidden_dim, wm_cfg.latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z:      (B, latent_dim)
        action: (B,) int64 action indices

        Returns:
            z_next: (B, latent_dim) predicted next latent state
        """
        a_emb = self.action_embed(action)   # (B, 16)
        z_next_raw = self.gru(a_emb, z)     # (B, latent_dim)
        z_next = self.refine(z_next_raw)    # Non-linear refinement
        return z_next


# ─── Reward Predictor ─────────────────────────────────────────────────────────

class RewardPredictor(nn.Module):
    """
    Predict immediate reward from (z_t, a_t).

    r̂_t = g(z_t, a_t)

    Used during MCTS rollouts to evaluate imagined trajectories without
    querying the real environment.
    """

    def __init__(self, wm_cfg: WorldModelConfig, num_actions: int):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 16)

        self.net = nn.Sequential(
            nn.Linear(wm_cfg.latent_dim + 16, wm_cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(wm_cfg.hidden_dim, wm_cfg.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(wm_cfg.hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z:      (B, latent_dim)
        action: (B,) int64

        Returns:
            reward: (B, 1) predicted scalar reward
        """
        a_emb = self.action_embed(action)               # (B, 16)
        combined = torch.cat([z, a_emb], dim=-1)
        return self.net(combined)                        # (B, 1)


# ─── World Model (Composite) ──────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Complete latent world model for imagination-based planning.

    Components:
        encoder:          obs → (μ, logσ) → z  (VAE)
        decoder:          z → obs reconstruction
        transition:       (z, a) → z_next
        reward_predictor: (z, a) → r̂

    Public API for MCTS:
        encode(grid, scalars) → z
        transition(z, action) → z_next
        predict_reward(z, action) → r̂
        imagine_trajectory(z0, action_seq) → (z_seq, r_seq)
        compute_loss(batch) → total_loss, loss_dict
    """

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config.world_model
        self.env_cfg = config.env

        self.encoder = ObsVAEEncoder(config.env, config.world_model)
        self.decoder = ObsDecoder(config.env, config.world_model)
        self.transition_model = LatentTransitionModel(
            config.world_model, config.env.num_actions
        )
        self.reward_predictor = RewardPredictor(
            config.world_model, config.env.num_actions
        )

    # ── Encoding ──

    def encode(
        self,
        grid: torch.Tensor,     # (B, C, H, W)
        scalars: torch.Tensor,  # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent representation using reparametrization trick.

        Returns:
            z:      (B, latent_dim) sampled latent (train) or μ (eval)
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encoder(grid, scalars)

        if self.training:
            # Reparametrize: z = μ + σ · ε, ε ∼ N(0, I)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            # At inference, use mean (deterministic)
            z = mu

        return z, mu, logvar

    # ── Transition & Reward ──

    def transition(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state: (z, action) → z_next."""
        return self.transition_model(z, action)

    def predict_reward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward: (z, action) → r̂."""
        return self.reward_predictor(z, action)

    # ── Imagination Rollout ──

    @torch.no_grad()
    def imagine_trajectory(
        self,
        z0: torch.Tensor,      # (B, latent_dim) initial latent
        action_seq: List[int], # Sequence of actions to imagine
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Roll out an imagined trajectory from z0 given a fixed action sequence.

        Used by MCTS for latent space planning — the agent can "dream"
        about future trajectories without stepping the real environment.

        Returns:
            z_seq:  List of latent states [z0, z1, ..., z_T]
            r_seq:  (T,) predicted rewards at each step
        """
        was_training = self.training
        self.eval()

        z = z0
        z_seq = [z]
        rewards = []

        for a in action_seq:
            action_t = torch.tensor(
                [a] * z.size(0), dtype=torch.long, device=z.device
            )
            r = self.predict_reward(z, action_t)       # (B, 1)
            z = self.transition(z, action_t)            # (B, L)
            z_seq.append(z)
            rewards.append(r.squeeze(-1))               # (B,)

        rewards_tensor = torch.stack(rewards, dim=0)    # (T, B)

        if was_training:
            self.train()

        return z_seq, rewards_tensor

    # ── Loss Computation ──

    def compute_loss(
        self,
        grid: torch.Tensor,       # (B, C, H, W)
        scalars: torch.Tensor,    # (B, 3)
        next_grid: torch.Tensor,  # (B, C, H, W)
        next_scalars: torch.Tensor,
        action: torch.Tensor,     # (B,) int64
        reward: torch.Tensor,     # (B,) actual reward
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite world model training loss.

        L_total = w_recon · L_recon + β · L_KL + w_reward · L_reward + w_cons · L_cons

        Components:
            L_recon:  MSE between reconstructed and actual current obs grid
            L_KL:     KL(q(z|o) ‖ p(z)) = -½ Σ(1 + logσ² - μ² - σ²)
            L_reward: MSE between predicted and actual reward
            L_cons:   Latent consistency — ||z_{t+1}^trans - z_{t+1}^enc||²

        Returns:
            total_loss: scalar
            loss_dict: breakdown for logging
        """
        # ── Encode current obs ──
        z, mu, logvar = self.encode(grid, scalars)

        # ── Reconstruct current obs ──
        recon = self.decoder(z)
        L_recon = F.mse_loss(recon, grid)

        # ── KL divergence: KL(q(z|o) ‖ N(0, I)) ──
        # = -0.5 * Σ(1 + log σ² - μ² - σ²)
        L_KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # ── Predict reward ──
        r_hat = self.predict_reward(z, action).squeeze(-1)
        L_reward = F.mse_loss(r_hat, reward)

        # ── Latent consistency: imagined z_{t+1} vs. encoded z_{t+1} ──
        z_next_trans = self.transition(z.detach(), action)
        with torch.no_grad():
            z_next_enc, _, _ = self.encode(next_grid, next_scalars)
        L_cons = F.mse_loss(z_next_trans, z_next_enc.detach())

        # ── Weighted sum ──
        cfg = self.cfg
        total = (
            cfg.recon_weight * L_recon
            + cfg.kl_beta * L_KL
            + cfg.reward_pred_weight * L_reward
            + cfg.consistency_weight * L_cons
        )

        loss_dict = {
            "wm_total": total.item(),
            "wm_recon": L_recon.item(),
            "wm_kl": L_KL.item(),
            "wm_reward": L_reward.item(),
            "wm_consistency": L_cons.item(),
        }

        return total, loss_dict
