"""
Stress Model for Bounded Rationality.

Defines a stress variable σ_t that models cognitive load under threat.

Update rule:
    σ_{t+1} = momentum * σ_t + (1 - momentum) * (w1 * danger + w2 * scarcity) - decay

Where:
    danger  = Σ_i 1 / (dist_i + ε)    for each enemy within range
    scarcity = 1/(ammo + ε) + 1/(health + ε), normalized to [0, 1]

Effects on policy:
    1. Logit noise injection:  π'(a) = softmax(z + N(0, σ_t²))
       Higher stress → noisier, less precise decisions.
    2. Action execution delay: if σ_t > threshold, agent repeats previous action
       for k steps → causes stress spirals and collapse cascades.
"""

import torch
import numpy as np
from typing import List, Optional

from .config import StressConfig


class StressModel:
    """
    Tracks and applies cognitive stress dynamics.

    The stress variable σ_t ∈ [0, 1] modulates the agent's policy quality,
    implementing bounded rationality under pressure.
    """

    def __init__(self, config: StressConfig):
        self.cfg = config
        self.sigma: float = 0.0           # Current stress level
        self.delay_counter: int = 0        # Remaining delay steps
        self.last_action: int = 0          # Cached action during delay

    def reset(self):
        """Reset stress state for a new episode."""
        self.sigma = 0.0
        self.delay_counter = 0
        self.last_action = 0

    def update(
        self,
        health: float,
        max_health: float,
        ammo: int,
        max_ammo: int,
        enemy_distances: List[float],
        num_enemies_in_view: int,
        episode_step: int,
        max_episode_steps: int,
    ) -> float:
        """
        Update stress σ_t based on current danger and resource scarcity.

        Args:
            health: Current agent health
            max_health: Maximum possible health
            ammo: Current ammo count
            max_ammo: Maximum possible ammo
            enemy_distances: Manhattan distances to all visible enemies
            num_enemies_in_view: Number of enemies in the agent's field of view
            episode_step: Current step in the episode
            max_episode_steps: Maximum number of steps in an episode

        Returns:
            Updated σ_t value
        """
        eps = 1e-6

        # ── Danger component: inverse distance to enemies ──
        # Closer enemies → higher danger
        if enemy_distances:
            danger = sum(1.0 / (d + eps) for d in enemy_distances)
            # Normalize: max danger ≈ 1.0 when enemy is adjacent (dist=1)
            danger = min(danger, 5.0) / 5.0
        else:
            danger = 0.0

        # ── Scarcity component: low resources → high scarcity ──
        health_scarcity = 1.0 - (health / (max_health + eps))  # 0 at full, 1 at empty
        ammo_scarcity = 1.0 - (ammo / (max_ammo + eps))
        scarcity = 0.5 * health_scarcity + 0.5 * ammo_scarcity

        # ── Information Overload component: more enemies in view → higher stress ──
        info_overload = num_enemies_in_view / (self.cfg.max_enemies_in_view + eps)

        # ── Time Pressure component: stress increases as the episode progresses ──
        time_pressure = episode_step / max_episode_steps

        # ── Stress update with EMA smoothing ──
        raw_stress = (
            self.cfg.danger_weight * danger
            + self.cfg.scarcity_weight * scarcity
            + self.cfg.information_overload_weight * info_overload
            + self.cfg.time_pressure_weight * time_pressure
        )
        self.sigma = (
            self.cfg.momentum * self.sigma
            + (1.0 - self.cfg.momentum) * raw_stress
            - self.cfg.decay_rate
        )
        self.sigma = float(np.clip(self.sigma, self.cfg.min_stress, self.cfg.max_stress))

        return self.sigma

    def apply_logit_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Inject Gaussian noise into policy logits proportional to stress.

        π'(a) = softmax(z + N(0, (scale * σ_t)²))

        Higher stress → larger variance → more random actions.

        Args:
            logits: Raw policy logits, shape (*, num_actions)

        Returns:
            Noised logits (same shape)
        """
        if self.sigma < 1e-6:
            return logits

        noise_std = self.cfg.noise_scale * self.sigma
        noise = torch.randn_like(logits) * noise_std
        return logits + noise

    def check_action_delay(self, proposed_action: int) -> int:
        """
        Check if stress-induced delay should override the proposed action.

        If σ_t exceeds the delay threshold, the agent repeats its previous
        action for `delay_steps` timesteps—simulating cognitive freeze.

        Args:
            proposed_action: Action the policy would normally take

        Returns:
            Actual action to execute (may be the cached previous action)
        """
        if self.delay_counter > 0:
            # Still in delay—repeat last action
            self.delay_counter -= 1
            return self.last_action

        if self.sigma > self.cfg.delay_threshold:
            # Trigger new delay episode
            self.delay_counter = self.cfg.delay_steps
            # Return the last action (freeze)
            return self.last_action

        # No delay—use proposed action
        self.last_action = proposed_action
        return proposed_action

    @property
    def is_in_delay(self) -> bool:
        """Whether the agent is currently in a stress-induced delay."""
        return self.delay_counter > 0

    @property
    def is_critical(self) -> bool:
        """Whether stress is above the delay threshold."""
        return self.sigma > self.cfg.delay_threshold

    def get_state(self) -> dict:
        """Return stress state for logging."""
        return {
            "stress_sigma": self.sigma,
            "delay_counter": self.delay_counter,
            "is_critical": self.is_critical,
        }
