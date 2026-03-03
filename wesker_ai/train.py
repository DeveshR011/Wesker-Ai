"""
Training Loop — End-to-End PPO + Distributional RL + World Model + MCTS.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  Trainer                                                      │
    │    ├── SurvivalEnv(s)      ── environment simulation         │
    │    ├── TacticalAgent       ── transformer policy + value      │
    │    ├── WorldModel          ── latent imagination              │
    │    ├── StressModel         ── cognitive load dynamics         │
    │    ├── RolloutBuffer       ── stores transitions              │
    │    └── Optimizers          ── separate for agent + wm         │
    └──────────────────────────────────────────────────────────────┘

Training loop flow:
    1. collect_rollout()         — run env for K steps with agent
    2. update_agent()            — PPO update (CVaR or E[R])
    3. update_world_model()      — VAE + transition + reward losses
    4. log()                     — TensorBoard / console metrics
    5. evaluate() every N steps  — run EvaluationMetrics

Ablation flags (config.ablation):
    use_stress_model     — enable/disable stress dynamics
    use_cvar_objective   — CVaR vs E[R] policy gradient
    use_distributional   — train quantile value head
    use_world_model      — train and use world model
    use_mcts             — MCTS action selection vs. pure policy
    use_dominance_reward — include map control dominance reward
    use_action_delay     — stress-induced execution delay
    use_logit_noise      — stress-induced policy noise
"""

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import Config
from .environment import SurvivalEnv
from .agent import TacticalAgent
from .stress import StressModel
from .world_model import WorldModel
from .risk import (
    quantile_regression_loss,
    compute_gae,
    ppo_policy_loss,
    value_loss,
    entropy_bonus,
    expected_value_objective,
    cvar_policy_gradient,
    compute_objectives_comparison,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─── Rollout Buffer ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    """Single environment transition."""
    grid: np.ndarray         # (C, H, W)
    scalars: np.ndarray      # (3,)
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool
    stress: float
    map_control: float
    # Quantile predictions (if distributional)
    quantiles: Optional[np.ndarray] = None   # (N,)


class RolloutBuffer:
    """
    Fixed-length rollout buffer for PPO. Memory-optimized version.

    Stores transitions in pre-allocated numpy arrays instead of a list of
    dataclasses. This is significantly more memory efficient.
    """
    def __init__(self, rollout_length: int, cfg: Config):
        self.rollout_length = rollout_length
        self.obs_grid_shape = cfg.obs_grid_shape
        self.num_quantiles = cfg.distributional.num_quantiles
        self.num_scalar_features = cfg.num_scalar_features
        self.pos = 0
        self.full = False

        self.grids = np.zeros((rollout_length, *self.obs_grid_shape), dtype=np.uint8)
        self.scalars = np.zeros((rollout_length, self.num_scalar_features), dtype=np.float32)
        self.actions = np.zeros(rollout_length, dtype=np.int64)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.bool_)
        self.stresses = np.zeros(rollout_length, dtype=np.float32)
        self.map_controls = np.zeros(rollout_length, dtype=np.float32)
        self.quantiles = np.zeros((rollout_length, self.num_quantiles), dtype=np.float32)

    def add(self, t: Transition):
        self.grids[self.pos] = t.grid
        self.scalars[self.pos] = t.scalars
        self.actions[self.pos] = t.action
        self.log_probs[self.pos] = t.log_prob
        self.rewards[self.pos] = t.reward
        self.values[self.pos] = t.value
        self.dones[self.pos] = t.done
        self.stresses[self.pos] = t.stress
        self.map_controls[self.pos] = t.map_control
        if t.quantiles is not None:
            self.quantiles[self.pos] = t.quantiles
        
        self.pos += 1
        if self.pos == self.rollout_length:
            self.full = True
            self.pos = 0
    
    def clear(self):
        self.pos = 0
        self.full = False

    def is_full(self) -> bool:
        return self.full

    def compute_returns_and_advantages(
        self,
        bootstrap_value: float,
        gamma: float,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and lambda-returns for the stored rollout.
        """
        advantages, returns = compute_gae(
            self.rewards, self.values, self.dones, bootstrap_value, gamma, lam
        )
        return advantages, returns

    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Returns all stored data as a dictionary of torch tensors,
        moved to the specified device.
        """
        return {
            "grids": torch.tensor(self.grids, dtype=torch.float32).to(device),
            "scalars": torch.tensor(self.scalars, dtype=torch.float32).to(device),
            "actions": torch.tensor(self.actions, dtype=torch.long).to(device),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32).to(device),
            "old_values": torch.tensor(self.values, dtype=torch.float32).to(device),
            "old_quantiles": torch.tensor(self.quantiles, dtype=torch.float32).to(device),
        }


# ─── Trajectory Window Builder ───────────────────────────────────────────────

class TrajectoryWindowBuilder:
    """
    Maintain a sliding window of observations for the Transformer.

    The Transformer expects (B=1, T, ...) input where T = trajectory_window.
    This class buffers observations and pads at the start of an episode.
    """

    def __init__(self, window_size: int, obs_shape: tuple, device: torch.device):
        self.window_size = window_size
        self.obs_shape = obs_shape      # (C, H, W)
        self.device = device
        self._grids   = deque(maxlen=window_size)
        self._scalars = deque(maxlen=window_size)
        self._actions = deque(maxlen=window_size)
        self._rewards = deque(maxlen=window_size)

    def reset(self):
        """Clear window. Padding will be zeros (inserted by _pad_to_window)."""
        self._grids.clear()
        self._scalars.clear()
        self._actions.clear()
        self._rewards.clear()

    def push(self, grid: np.ndarray, scalars: np.ndarray, action: int, reward: float):
        self._grids.append(grid.copy())
        self._scalars.append(scalars.copy())
        self._actions.append(action)
        self._rewards.append(reward)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Return padded window as torch tensors on device.

        Returns:
            local_grids:  (1, T, C, H, W)
            scalars:      (1, T, 3)
            prev_actions: (1, T) long
            prev_rewards: (1, T)
        """
        T = self.window_size
        C, H, W = self.obs_shape

        pad_n = T - len(self._grids)
        grids   = np.zeros((T, C, H, W), dtype=np.float32)
        scalars = np.zeros((T, 4), dtype=np.float32)  # health, ammo, stress, has_key
        actions = np.zeros(T, dtype=np.int64)
        rewards = np.zeros(T, dtype=np.float32)

        for i, (g, s, a, r) in enumerate(
            zip(self._grids, self._scalars, self._actions, self._rewards)
        ):
            grids[pad_n + i]   = g
            scalars[pad_n + i] = s
            actions[pad_n + i] = a
            rewards[pad_n + i] = r

        return {
            "local_grids":  torch.from_numpy(grids).unsqueeze(0).to(self.device),   # (1, T, C, H, W)
            "scalars":      torch.from_numpy(scalars).unsqueeze(0).to(self.device), # (1, T, 3)
            "prev_actions": torch.from_numpy(actions).unsqueeze(0).to(self.device), # (1, T)
            "prev_rewards": torch.from_numpy(rewards).unsqueeze(0).to(self.device), # (1, T)
        }


# ─── Batch Sampler ───────────────────────────────────────────────────────────

def make_mini_batches(
    data: Dict[str, torch.Tensor],
    batch_size: int,
):
    """
    Yield mini-batches from flat training data.

    Args:
        data:       Dict of tensors, first dim = T (rollout length)
        batch_size: Mini-batch size
    """
    T = data[next(iter(data))].size(0)
    indices = torch.randperm(T)
    for start in range(0, T, batch_size):
        idx = indices[start : start + batch_size]
        yield {k: v[idx] for k, v in data.items()}


# ─── World Model Trainer ──────────────────────────────────────────────────────

class WorldModelTrainer:
    """
    Wraps world model training steps. Maintains its own replay buffer.
    """

    def __init__(self, world_model: WorldModel, lr: float, device: torch.device):
        self.wm = world_model
        self.optimizer = optim.Adam(world_model.parameters(), lr=lr)
        self.device = device
        # A small replay buffer of (grid, scalars, next_grid, next_scalars, action, reward)
        self.replay: List[tuple] = []
        self.max_replay = 5000

    def store(self, grid, scalars, next_grid, next_scalars, action, reward):
        self.replay.append((grid, scalars, next_grid, next_scalars, action, reward))
        if len(self.replay) > self.max_replay:
            self.replay.pop(0)

    def update(self, batch_size: int = 64) -> dict:
        """Sample a mini-batch and update world model parameters."""
        if len(self.replay) < batch_size:
            return {}

        indices = np.random.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[i] for i in indices]
        grids    = torch.stack([torch.tensor(b[0]) for b in batch]).to(self.device)
        scalars  = torch.stack([torch.tensor(b[1]) for b in batch]).to(self.device)
        n_grids  = torch.stack([torch.tensor(b[2]) for b in batch]).to(self.device)
        n_scalars= torch.stack([torch.tensor(b[3]) for b in batch]).to(self.device)
        actions  = torch.tensor([b[4] for b in batch], dtype=torch.long, device=self.device)
        rewards  = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        loss, loss_dict = self.wm.compute_loss(grids, scalars, n_grids, n_scalars, actions, rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.optimizer.step()
        return loss_dict


# ─── Main Trainer ─────────────────────────────────────────────────────────────

class Trainer:
    """
    End-to-end hierarchical survival AI trainer.

    Manages:
        - Environment rollout collection
        - Stress dynamics (logit noise + action delay)
        - Agent PPO updates (CVaR or E[R])
        - World model joint training
        - MCTS integration
        - TensorBoard logging
        - Checkpoint saving
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        config.device = str(self.device)  # Update device in config

        # ── Build components ──
        self.env = SurvivalEnv(config.env, seed=config.seed)
        self.agent = TacticalAgent(config).to(self.device)
        self.stress_model = StressModel(config.stress)

        # Observation shape for window builder
        self.obs_shape = (
            config.env.num_obs_channels,
            2 * config.env.vision_radius + 1,
            2 * config.env.vision_radius + 1,
        )
        self.window_builder = TrajectoryWindowBuilder(
            config.transformer.trajectory_window,
            self.obs_shape,
            self.device,
        )

        # ── World model (optional) ──
        self.world_model: Optional[WorldModel] = None
        self.wm_trainer: Optional[WorldModelTrainer] = None
        if config.ablation.use_world_model:
            self.world_model = WorldModel(config).to(self.device)
            self.wm_trainer = WorldModelTrainer(
                self.world_model, config.training.world_model_lr, self.device
            )

        # ── Optimizers ──
        self.agent_optimizer = optim.Adam(
            self.agent.parameters(), lr=config.training.learning_rate
        )

        # ── Rollout buffer ──
        self.rollout_buffer = RolloutBuffer(config.training.rollout_length, config)

        # ── Logging ──
        os.makedirs(config.training.log_dir, exist_ok=True)
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

        self.writer: Optional[object] = None
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.training.log_dir, f"seed{config.seed}")
            )
        if config.training.use_wandb and HAS_WANDB:
            wandb.init(
                project="wesker_ai",
                config=config,
                name=f"seed_{config.seed}",
            )

        # ── Training state ──
        self.global_step = 0
        self.update_count = 0
        self.episode_count = 0
        self.episode_returns: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.stress_collapse_count = 0
        self.total_steps_in_window = 0

        # Additional metrics
        self.metrics_ammo_at_death = deque(maxlen=100)
        self.metrics_time_near_enemies = deque(maxlen=100)
        self.metrics_dominance_contributions = deque(maxlen=100)
        self.metrics_mean_stress = deque(maxlen=100)
        self.metrics_cvar_returns = deque(maxlen=100)
        self.heatmap = np.zeros((config.env.grid_size, config.env.grid_size))
        
        # Stability guards
        self.target_kl = 0.05
        self.clip_vloss = True
        self.clip_range_vf = 0.2
        
        # Entropy annealing
        self.initial_entropy_coef = config.training.entropy_coef
        self.entropy_floor_coef = 0.005

    # ─── Main Training Loop ───────────────────────────────────────

    def train(self):
        """
        Main training loop.

        Alternates between rollout collection and updates until
        total_steps is reached.
        """
        cfg = self.cfg.training
        print(f"\n{'='*60}")
        print(f"  Wesker AI Training — Device: {self.device}")
        print(f"  Total Steps: {cfg.total_steps:,}")
        print(f"  Ablations: {self.cfg.ablation}")
        print(f"{'='*60}\n")

        obs = self.env.reset()
        self.stress_model.reset()
        self.window_builder.reset()

        episode_return = 0.0
        episode_steps = 0
        start_time = time.time()

        while self.global_step < cfg.total_steps:
            # ── Rollout collection ──
            self.rollout_buffer.clear()
            prev_obs = obs
            delayed_action = None  # For stress delay tracking

            for step in range(cfg.rollout_length):
                # Update stress
                dists = self.env.get_enemy_distances()
                sigma = 0.0
                if self.cfg.ablation.use_stress_model:
                    sigma = self.stress_model.update(
                        health=self.env.health,
                        max_health=self.cfg.env.max_health,
                        ammo=self.env.ammo,
                        max_ammo=self.cfg.env.max_ammo,
                        enemy_distances=dists,
                        num_enemies_in_view=len(dists),
                        episode_step=episode_steps,
                        max_episode_steps=self.cfg.env.max_steps,
                    )
                    self.env.stress = sigma

                # Build observation context window
                self.window_builder.push(
                    obs["local_grid"],
                    obs["scalars"],
                    int(obs["prev_action"]),
                    float(obs["prev_reward"]),
                )
                obs_ctx = self.window_builder.get_tensors()

                # Select action
                with torch.no_grad():
                    agent_out = self.agent.forward(
                        obs_ctx["local_grids"],
                        obs_ctx["scalars"],
                        obs_ctx["prev_actions"],
                        obs_ctx["prev_rewards"],
                    )

                logits = agent_out.policy_logits

                # Stress: inject logit noise
                if self.cfg.ablation.use_logit_noise:
                    logits = self.stress_model.apply_logit_noise(logits)

                from torch.distributions import Categorical
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                action = int(action_t.item())
                log_prob = float(dist.log_prob(action_t).item())
                val = float(agent_out.value.item())
                quants = (
                    agent_out.quantiles.cpu().numpy().squeeze(0)
                    if agent_out.quantiles is not None else None
                )

                # Stress: action delay
                if self.cfg.ablation.use_action_delay:
                    action = self.stress_model.check_action_delay(action)
                    if self.stress_model.is_critical:
                        self.stress_collapse_count += 1
                        self.total_steps_in_window += 1

                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                self.total_steps_in_window += 1

                # Dominance reward component
                if self.cfg.ablation.use_dominance_reward:
                    reward += self.cfg.training.dominance_lambda * info.get("map_control", 0.0)

                # Store world model training sample
                if self.wm_trainer is not None:
                    self.wm_trainer.store(
                        obs["local_grid"], obs["scalars"],
                        next_obs["local_grid"], next_obs["scalars"],
                        action, reward,
                    )

                # Store transition
                self.rollout_buffer.add(Transition(
                    grid=obs["local_grid"],
                    scalars=obs["scalars"],
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=val,
                    done=done,
                    stress=sigma,
                    map_control=info.get("map_control", 0.0),
                    quantiles=quants,
                ))

                episode_return += reward
                episode_steps += 1
                self.global_step += 1

                obs = next_obs

                self.heatmap[self.env.agent_y, self.env.agent_x] += 1

                if done:
                    self.episode_count += 1
                    self.episode_returns.append(episode_return)
                    self.episode_lengths.append(episode_steps)
                    self.metrics_ammo_at_death.append(info.get("ammo", 0))
                    
                    time_near = info.get("time_near_enemies", 0)
                    self.metrics_time_near_enemies.append(time_near / max(1, episode_steps))
                    
                    dom_rew = info.get("dominance_reward_sum", 0.0)
                    surv_rew = info.get("survival_reward_sum", 0.0)
                    total_rew = dom_rew + surv_rew
                    self.metrics_dominance_contributions.append(dom_rew / max(total_rew, 1e-5))
                    
                    mean_e_stress = info.get("stress_sum", 0.0) / max(1, episode_steps)
                    self.metrics_mean_stress.append(mean_e_stress)

                    # Compute CVaR of this episode return empirically via metric tracker
                    # For simplicity, we just keep the returns array
                    self.metrics_cvar_returns.append(episode_return)

                    episode_return = 0.0
                    episode_steps = 0

                    obs = self.env.reset()
                    self.stress_model.reset()
                    self.window_builder.reset()

                if self.global_step >= cfg.total_steps:
                    break

            # ── Bootstrap value for last state ──
            with torch.no_grad():
                last_obs_ctx = self.window_builder.get_tensors()
                last_out = self.agent.forward(
                    last_obs_ctx["local_grids"],
                    last_obs_ctx["scalars"],
                    last_obs_ctx["prev_actions"],
                    last_obs_ctx["prev_rewards"],
                )
                bootstrap_val = float(last_out.value.item())

            # ── PPO update ──
            agent_losses, attn_weights = self._update_agent(bootstrap_val)

            # ── World model update ──
            wm_losses = {}
            if self.wm_trainer is not None:
                wm_losses = self.wm_trainer.update(cfg.batch_size)

            self.update_count += 1

            # ── Logging ──
            if self.update_count % cfg.log_interval == 0:
                self._log_metrics(agent_losses, wm_losses, start_time, attn_weights)

            # ── Checkpoint ──
            if self.update_count % cfg.save_interval == 0:
                self._save_checkpoint()

    # ─── Agent Update (PPO) ───────────────────────────────────────

    def _update_agent(self, bootstrap_value: float) -> Tuple[dict, torch.Tensor]:
        """
        PPO update pass over collected rollout.

        For each mini-epoch:
            1. Recompute log-probs and values for all transitions
            2. Compute PPO loss (policy + value + entropy)
                - If use_cvar_objective: use CVaR advantage
                - Else:                  use standard E[R] advantage
            3. Optionally train quantile head (QR loss)
        """
        cfg = self.cfg.training
        buf = self.rollout_buffer

        # ── Compute GAE advantages ──
        advantages, returns = buf.compute_returns_and_advantages(
            bootstrap_value, cfg.gamma, cfg.gae_lambda
        )

        # Normalize and clip advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std
        norm_advantages = np.clip(norm_advantages, -5.0, 5.0)  # Stop exploding advantages

        # ── Build flat tensors for the rollout ──
        data = buf.get_tensors(self.device)
        data["returns"] = torch.tensor(returns, dtype=torch.float32).to(self.device)
        data["advantages"] = torch.tensor(norm_advantages, dtype=torch.float32).to(self.device)

        # ── PPO update epochs ──
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_quant_loss  = 0.0
        total_entropy     = 0.0
        agent_losses      = {}
        num_batches = 0
        attn_weights_sum = None


        # Anneal entropy slowly to floor vs linear
        progress = min(1.0, self.global_step / cfg.total_steps)
        current_entropy_coef = max(
            self.entropy_floor_coef, 
            self.initial_entropy_coef * (1.0 - progress * 0.9)
        )
        
        for j in range(cfg.num_epochs):
            # For early stopping on KL divergence
            epoch_kl = 0.0
            epoch_batches = 0
            
            for batch in make_mini_batches(data, cfg.batch_size):                # Forward pass: single-step (T=1 window)
                bg = batch["grids"].unsqueeze(1)   # (B, 1, C, H, W)
                bs = batch["scalars"].unsqueeze(1) # (B, 1, 3)
                ba = batch["actions"].unsqueeze(1) # (B, 1)
                br = torch.zeros_like(ba).float()  # (B, 1) dummy prev_reward for flat

                logits_seq, values_seq, quantiles_seq, attn_weights = self.agent.forward_sequence(
                    bg, bs, ba, br
                )

                if attn_weights_sum is None:
                    attn_weights_sum = attn_weights.detach().clone()
                else:
                    attn_weights_sum += attn_weights.detach()

                logits   = logits_seq[:, -1]    # (B, A)
                values_b = values_seq[:, -1, 0] # (B,)
                quants_b = quantiles_seq[:, -1] if quantiles_seq is not None else None

                # Action log-probs
                from torch.distributions import Categorical
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch["actions"])

                # KL penalty definition
                kl_div = (batch["old_log_probs"] - new_log_probs).mean()
                epoch_kl += kl_div.item()
                epoch_batches += 1
                
                kl_penalty = (batch["old_log_probs"] - new_log_probs).pow(2).mean() * 0.5
                kl_coef = self.cfg.training.kl_coef

                # Entropy
                ent = entropy_bonus(logits)

                # ── Policy loss ──
                if (self.cfg.ablation.use_cvar_objective
                        and quants_b is not None):
                    # CVaR advantage: advantage = G_t - CVaR_α(s_t)
                    from .risk import compute_cvar
                    cvar_b = compute_cvar(
                        quants_b, self.cfg.distributional.cvar_alpha
                    ).squeeze(-1)  # (B,)
                    pol_loss = cvar_policy_gradient(
                        new_log_probs, batch["returns"], cvar_b
                    )
                else:
                    # Standard E[R] PPO objective
                    pol_loss = ppo_policy_loss(
                        new_log_probs,
                        batch["old_log_probs"],
                        batch["advantages"],
                        clip_eps=cfg.ppo_clip,
                    )

                # ── Value loss ──
                # Clip returns to prevent crazy Q(s,a) / V(s) hallucination
                # Environment normally max around 50-100 total. Max returns clipped at 200.
                clipped_returns = torch.clamp(batch["returns"], -50.0, 200.0)
                
                val_loss = value_loss(
                    values_b, clipped_returns,
                    clip_val=self.clip_range_vf if self.clip_vloss else None,
                    old_values=batch["old_values"],
                )

                # ── Quantile regression loss ──
                quant_loss = torch.tensor(0.0, device=self.device)
                if quants_b is not None and self.cfg.ablation.use_distributional_value:
                    quant_loss = quantile_regression_loss(
                        quants_b,
                        clipped_returns,
                        self.agent.quantile_head.taus,
                        kappa=self.cfg.distributional.huber_kappa,
                    )

                # ── Total loss ──
                loss = (
                    pol_loss
                    + cfg.value_loss_coef * val_loss
                    + cfg.quantile_loss_coef * quant_loss
                    + kl_coef * kl_penalty
                    - current_entropy_coef * ent
                )

                self.agent_optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.agent.parameters(), cfg.max_grad_norm
                )
                self.agent_optimizer.step()

                total_policy_loss += pol_loss.item()
                total_value_loss  += val_loss.item()
                total_quant_loss  += quant_loss.item()
                total_entropy     += ent.item()
                agent_losses["grad_norm"] = agent_losses.get("grad_norm", 0) + grad_norm.item()
                agent_losses["kl_penalty"] = agent_losses.get("kl_penalty", 0) + kl_penalty.item()
                num_batches += 1
                
            # Check early stopping at epoch boundary
            mean_kl = epoch_kl / max(1, epoch_batches)
            if mean_kl > self.target_kl * 1.5:
                # print(f"  [PPO] Early stopping at epoch {j+1}/{cfg.num_epochs} (KL {mean_kl:.4f} > {self.target_kl*1.5:.4f})")
                break

        nb = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / nb,
            "value_loss":  total_value_loss  / nb,
            "quant_loss":  total_quant_loss  / nb,
            "entropy":     total_entropy     / nb,
            "grad_norm":   agent_losses.get("grad_norm", 0) / nb,
            "kl":          agent_losses.get("kl_penalty", 0) / nb,
        }, attn_weights_sum / nb if attn_weights_sum is not None else None

    # ─── Logging ─────────────────────────────────────────────────

    def _log_metrics(self, agent_losses: dict, wm_losses: dict, start_time: float, attn_weights: torch.Tensor):
        """Log metrics to console and TensorBoard."""
        step = self.global_step
        ep   = self.episode_count
        fps  = step / max(time.time() - start_time, 1.0)

        mean_ret = np.mean(self.episode_returns) if self.episode_returns else 0.0
        mean_len = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        collapse_rate = (
            self.stress_collapse_count / max(self.total_steps_in_window, 1)
        )

        line = (
            f"[Step {step:>7d}] ep={ep:>5d} | "
            f"ret={mean_ret:>7.2f} | len={mean_len:>5.0f} | "
            f"P={agent_losses.get('policy_loss',0):.3f} "
            f"V={agent_losses.get('value_loss',0):.3f} "
            f"H={agent_losses.get('entropy',0):.3f} | "
            f"collapse={collapse_rate:.3f} | fps={fps:.0f}"
        )
        print(line)

        if self.writer is not None:
            for k, v in agent_losses.items():
                self.writer.add_scalar(f"train/{k}", v, step)
            for k, v in wm_losses.items():
                self.writer.add_scalar(f"world_model/{k}", v, step)
            self.writer.add_scalar("train/mean_return", mean_ret, step)
            self.writer.add_scalar("train/mean_ep_len", mean_len, step)
            self.writer.add_scalar("train/stress_collapse_rate", collapse_rate, step)
            self.writer.add_scalar("train/fps", fps, step)
            
            # Log attention weights
            if attn_weights is not None:
                # Visualize the attention from the last token to all previous tokens
                last_token_attn = attn_weights[0, -1, :].cpu().numpy()
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.imshow(last_token_attn[np.newaxis, :], cmap='viridis', aspect='auto')
                ax.set_yticks([])
                ax.set_xlabel("Time")
                ax.set_title("Attention from Last Token")
                self.writer.add_figure("train/attention", fig, step)
            
            # Log heatmap
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(self.heatmap, cmap='hot', interpolation='nearest')
            ax.set_title("Agent Position Heatmap")
            self.writer.add_figure("train/heatmap", fig, step)

            # Additional detailed metrics logging
            if len(self.metrics_ammo_at_death) > 0:
                self.writer.add_scalar("metrics/avg_ammo_at_death", np.mean(self.metrics_ammo_at_death), step)
                self.writer.add_scalar("metrics/time_near_enemies_pct", np.mean(self.metrics_time_near_enemies), step)
                self.writer.add_scalar("metrics/dominance_contrib_pct", np.mean(self.metrics_dominance_contributions), step)
                self.writer.add_scalar("metrics/avg_stress", np.mean(self.metrics_mean_stress), step)
                
                # Plot CVaR 25% tail return vs Mean return
                returns = np.array(self.metrics_cvar_returns)
                cvar_tail = np.mean(np.sort(returns)[:max(1, int(len(returns)*0.25))])
                self.writer.add_scalar("metrics/cvar_tail_return", cvar_tail, step)
                
            self.writer.flush()

    # ─── Checkpoint ──────────────────────────────────────────────

    def _save_checkpoint(self):
        """Save agent and world model parameters."""
        path = os.path.join(
            self.cfg.training.checkpoint_dir,
            f"step_{self.global_step:08d}.pt",
        )
        ckpt = {
            "global_step": self.global_step,
            "update_count": self.update_count,
            "agent": self.agent.state_dict(),
            "agent_optimizer": self.agent_optimizer.state_dict(),
        }
        if self.world_model is not None:
            ckpt["world_model"] = self.world_model.state_dict()
            if self.wm_trainer is not None:
                ckpt["wm_optimizer"] = self.wm_trainer.optimizer.state_dict()
        torch.save(ckpt, path)
        print(f"  ✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load agent and world model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self.agent_optimizer.load_state_dict(ckpt["agent_optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.update_count = ckpt.get("update_count", 0)
        if self.world_model is not None and "world_model" in ckpt:
            self.world_model.load_state_dict(ckpt["world_model"])
        print(f"  ✓ Loaded checkpoint: {path} (step {self.global_step})")
