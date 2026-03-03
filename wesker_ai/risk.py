"""
Risk-Sensitive Training Module — Distributional RL + CVaR.

This module implements the core mathematical machinery for risk-sensitive
policy optimization using quantile regression and CVaR.

──────────────────────────────────────────────────────────────────────────
BACKGROUND
──────────────────────────────────────────────────────────────────────────

Standard RL optimizes:  J(π) = E[Z(s,a)]
Distributional RL:       π models the full distribution Z(s,a)
Risk-sensitive RL:       J(π) = CVaR_α(Z(s,a))

CVaR_α (Conditional Value-at-Risk at level α):
    CVaR_α = E[Z | Z ≤ q_α]

where q_α is the α-quantile of Z.

For α = 0.25: optimize the expected return of the WORST 25% of episodes.
This steers the agent to be catastrophe-averse — avoiding situations
where even bad outcomes are survivable.

──────────────────────────────────────────────────────────────────────────
MODULES
──────────────────────────────────────────────────────────────────────────

1. quantile_regression_loss  — Huber quantile loss ρ^κ_τ
2. compute_cvar               — CVaR from sorted quantile predictions
3. expected_value_objective   — baseline E[R] policy gradient
4. cvar_policy_gradient       — CVaR-adjusted policy gradient
5. ppo_policy_loss            — PPO clipped surrogate objective
6. value_loss                 — Bellman MSE baseline
7. entropy_bonus              — Entropy regularization
8. compute_gae                — Generalized Advantage Estimation
9. RiskMetrics                — Stateful metrics tracker
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ─── 1. Quantile Regression Loss ─────────────────────────────────────────────

def quantile_regression_loss(
    predicted_quantiles: torch.Tensor,
    target_returns: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Huber quantile regression loss (QR-DQN / IQN style).

    For each quantile fraction τ_i and target return T_j, computes:
        ρ^κ_τ(u) = |τ - 𝟙(u < 0)| · L_κ(u)

    where u = T_j - θ_i is the TD error and L_κ is the Huber loss.

    This loss is asymmetric: overestimates of quantile τ are penalized
    by (1-τ), underestimates by τ. The model thus learns the τ-quantile
    of the return distribution.

    Args:
        predicted_quantiles: (B, N) — N quantile predictions per sample
        target_returns:       (B,) or (B, 1) — bootstrapped return targets
        taus:                 (N,) — quantile fractions ∈ (0, 1)
        kappa:                Huber loss threshold (1.0 = standard)

    Returns:
        Scalar loss
    """
    if target_returns.dim() == 1:
        target_returns = target_returns.unsqueeze(-1)   # (B, 1)

    # TD error: (B, 1) - (B, N) → broadcast to (B, N)
    td_error = target_returns - predicted_quantiles    # (B, N)

    # Huber element
    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )  # (B, N)

    # Asymmetric quantile weighting
    # For a given τ_i: penalize under-prediction by τ_i, over-prediction by (1 - τ_i)
    indicator = (td_error < 0).float()                 # 𝟙(u < 0)
    weight = (taus.unsqueeze(0) - indicator).abs()     # (B, N)

    loss = (weight * huber).mean()
    return loss


# ─── 2. CVaR Computation ─────────────────────────────────────────────────────

def compute_cvar(quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute CVaR_α from a set of quantile predictions.

    Mathematical formulation:
        CVaR_α = (1 / ⌊αN⌋) Σ_{i=1}^{⌊αN⌋} θ_{(i)}

    where θ_{(i)} are the sorted quantile predictions in ascending order.
    This is the sample mean of the bottom α-fraction of the predicted
    return distribution.

    Intuition:
        α = 1.0  → E[Z]  (standard expected value)
        α = 0.25 → E[Z | Z ≤ q_{0.25}]  (mean of worst 25%)
        α → 0   → min(Z)  (pure worst-case)

    Args:
        quantiles: (B, N) predicted quantile values for return distribution
        alpha:     Risk level ∈ (0, 1]. Smaller → more risk-averse.

    Returns:
        CVaR values (B, 1)
    """
    N = quantiles.size(-1)
    # For small alpha, torch.topk with largest=False may be faster than sort.
    # However, N is typically small (e.g., 51), so sort is fine.
    sorted_q, _ = torch.sort(quantiles, dim=-1)          # (B, N) ascending
    k = max(1, int(alpha * N))                            # Number of tail quantiles
    cvar = sorted_q[:, :k].mean(dim=-1, keepdim=True)   # (B, 1)
    return cvar


# ─── 3. Expected Value Objective (baseline) ───────────────────────────────────

def expected_value_objective(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    Standard policy gradient objective: E[log π(a|s) · A(s,a)].

    This is the baseline E[R] objective. Maximizing this corresponds to
    standard PPO / REINFORCE — no explicit tail-risk aversion.

    Args:
        log_probs:  (B,) log-probabilities of taken actions
        advantages: (B,) advantage estimates A(s,a) = G_t - V(s_t)

    Returns:
        Scalar policy loss (negated for gradient descent)
    """
    return -(log_probs * advantages.detach()).mean()


# ─── 4. CVaR Policy Gradient ──────────────────────────────────────────────────

def cvar_policy_gradient(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    cvar_values: torch.Tensor,
) -> torch.Tensor:
    """
    CVaR-adjusted policy gradient objective.

    Instead of A_t = G_t - V(s_t), we compute:
        A_t^CVaR = G_t - CVaR_α(s_t)

    The advantage measure now reflects excess return over the **tail risk
    baseline** (CVaR) rather than the mean value. This penalizes actions
    that might appear profitable in expectation but lead to catastrophic
    tail outcomes.

    Behaviorally:
        - Agent prefers high-floor moves over high-ceiling risky ones
        - Combat is avoided unless the tail risk is already low
        - Ammo hoarding is incentivized (reduces scarcity stress)

    Args:
        log_probs:    (B,) log-probabilities of taken actions
        returns:      (B,) Monte Carlo returns G_t
        cvar_values:  (B,) or (B, 1) CVaR_α(s_t) baseline

    Returns:
        Scalar policy loss (negated)
    """
    if cvar_values.dim() == 2:
        cvar_values = cvar_values.squeeze(-1)
    advantages = (returns - cvar_values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return -(log_probs * advantages).mean()


# ─── 5. PPO Clipped Surrogate Loss ───────────────────────────────────────────

def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped surrogate objective.

    L^CLIP = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]

    where r_t = π(a|s) / π_old(a|s) is the probability ratio.

    This prevents excessively large policy updates by clipping the
    ratio, ensuring training stability in the presence of noisy
    advantage estimates.

    Args:
        log_probs:     (B,) current policy log-probs
        old_log_probs: (B,) old policy log-probs (behaviour policy)
        advantages:    (B,) advantage estimates (normalized externally)
        clip_eps:      PPO clipping range ε

    Returns:
        Scalar policy loss (negated for gradient descent)
    """
    ratio = torch.exp(log_probs - old_log_probs.detach())
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.detach()
    loss = -torch.min(surr1, surr2).mean()
    return loss


# ─── 6. Value Loss ───────────────────────────────────────────────────────────

def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_val: Optional[float] = None,
    old_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Value function MSE loss with optional PPO clipping.

    L^V = E[(V(s_t) - G_t)^2]

    When clip_val is set, we additionally constrain the value update
    to not move too far from old_values, mirroring PPO-style stability.

    Args:
        values:     (B,) or (B, 1) current value predictions
        returns:    (B,) discounted returns G_t
        clip_val:   Optional PPO value clipping range
        old_values: (B,) old value predictions (needed if clip_val set)

    Returns:
        Scalar value loss
    """
    values = values.squeeze(-1)
    loss_unclipped = F.mse_loss(values, returns)

    if clip_val is not None and old_values is not None:
        old_values = old_values.squeeze(-1)
        values_clipped = old_values + torch.clamp(
            values - old_values, -clip_val, clip_val
        )
        loss_clipped = F.mse_loss(values_clipped, returns)
        return torch.max(loss_unclipped, loss_clipped)

    return loss_unclipped


# ─── 7. Entropy Bonus ────────────────────────────────────────────────────────

def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """
    Entropy of the categorical policy distribution.

    H(π) = -Σ_a π(a|s) log π(a|s)

    Maximizing entropy encourages exploration and prevents premature
    collapse to deterministic policies — critical in sparse reward
    survival environments.

    Args:
        logits: (B, num_actions) raw policy logits

    Returns:
        Scalar mean entropy
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
    return entropy.mean()


# ─── 8. Generalized Advantage Estimation ─────────────────────────────────────

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE-λ).

    A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

    where δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)

    GAE interpolates between:
        λ = 0: TD(0) advantages — low variance, high bias
        λ = 1: Monte Carlo returns — high variance, zero bias

    The λ parameter controls this bias-variance trade-off.

    Args:
        rewards:         (T,) step rewards
        values:          (T,) state value estimates V(s_t)
        dones:           (T,) episode termination flags (1.0 = done)
        bootstrap_value: V(s_{T+1}) for bootstrap
        gamma:           Discount factor
        lam:             GAE λ parameter

    Returns:
        advantages: (T,) GAE advantage estimates
        returns:    (T,) lambda-returns G_t (= advantages + values)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else bootstrap_value
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ─── 9. Risk Metrics Tracker ─────────────────────────────────────────────────

class RiskMetrics:
    """
    Stateful tracker for risk-sensitive evaluation metrics.

    Accumulated over episodes and reset per evaluation window.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.episode_returns: list = []
        self.cvar_values: list = []

    def record_episode(self, episode_return: float):
        """Record a completed episode's return."""
        self.episode_returns.append(episode_return)

    def tail_risk_probability(self, threshold: Optional[float] = None) -> float:
        """
        P(Z ≤ threshold) — probability of very low return episode.

        If threshold is None, uses the α-quantile of recorded returns.
        """
        if not self.episode_returns:
            return 0.0
        arr = np.array(self.episode_returns)
        if threshold is None:
            threshold = np.quantile(arr, self.alpha)
        return float((arr <= threshold).mean())

    def cvar(self) -> float:
        """CVaR_α of empirical episode returns."""
        if not self.episode_returns:
            return 0.0
        arr = np.array(self.episode_returns)
        threshold = np.quantile(arr, self.alpha)
        tail = arr[arr <= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(arr.min())

    def expected_return(self) -> float:
        """Mean episode return (standard objective baseline)."""
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))

    def reset(self):
        """Clear accumulated data."""
        self.episode_returns.clear()
        self.cvar_values.clear()


# ─── Objective Comparison Helper ─────────────────────────────────────────────

def compute_objectives_comparison(
    quantiles: torch.Tensor,
    alpha: float,
) -> dict:
    """
    Compute both E[R] and CVaR_α from quantile predictions.

    Used for logging and ablation comparison.

    Returns dict:
        {
            'expected_value': E[Z] (α=1 CVaR),
            'cvar':           CVaR_α(Z),
            'cvar_gap':       E[Z] - CVaR_α (risk premium measure)
        }
    """
    ev = quantiles.mean(dim=-1, keepdim=True)   # (B, 1)
    cvar = compute_cvar(quantiles, alpha)         # (B, 1)
    gap = (ev - cvar).mean().item()
    return {
        "expected_value": ev.mean().item(),
        "cvar": cvar.mean().item(),
        "cvar_gap": gap,
    }
