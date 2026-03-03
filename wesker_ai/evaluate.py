"""
Evaluation Module — Research Metrics for Wesker AI.

Computes all 6 key evaluation metrics:
    1. Survival time         — mean steps alive per episode
    2. Death rate            — fraction of episodes ending in death
    3. CVaR return           — mean return of worst-α fraction of episodes
    4. Tail risk probability — P(episode_return ≤ q_{0.1})
    5. Stress-collapse freq  — fraction of steps where σ_t > delay_threshold
    6. Map control score     — mean map dominance at episode end

Also provides:
    - compare_objectives() — ablation comparison: CVaR vs E[R] agent
    - print_metrics_table() — formatted console output
    - EvaluationMetrics dataclass for programmatic access
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict

from .config import Config
from .environment import SurvivalEnv
from .agent import TacticalAgent
from .stress import StressModel
from .train import TrajectoryWindowBuilder


# ─── Metrics Dataclass ───────────────────────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """
    Container for all evaluation metrics.

    All values are means over n_episodes.
    """
    # ── Core survival ──
    survival_time: float = 0.0         # Mean steps survived
    death_rate: float = 0.0            # Fraction of episodes ending in death
    mean_return: float = 0.0           # E[G_t] — standard objective

    # ── Risk-sensitive ──
    cvar_return: float = 0.0           # CVaR_α of episode returns
    tail_risk_probability: float = 0.0 # P(return ≤ q_{0.1})

    # ── Stress ──
    mean_stress: float = 0.0           # Mean σ_t during episodes
    stress_collapse_frequency: float = 0.0  # Fraction steps with σ_t > threshold

    # ── Dominance ──
    map_control_score: float = 0.0     # Mean final map control ratio
    enemies_killed: float = 0.0        # Mean kills per episode

    # ── Return distribution ──
    return_std: float = 0.0
    return_min: float = 0.0
    return_max: float = 0.0
    return_p10: float = 0.0            # 10th percentile
    return_p90: float = 0.0            # 90th percentile

    # ── Raw data (excluded from repr) ──
    _episode_returns: List[float] = field(default_factory=list, repr=False)
    _episode_lengths: List[int]   = field(default_factory=list, repr=False)

    def __repr__(self):
        return (
            f"EvaluationMetrics(\n"
            f"  survival_time       = {self.survival_time:.1f} steps\n"
            f"  death_rate          = {self.death_rate:.3f}\n"
            f"  mean_return         = {self.mean_return:.3f}\n"
            f"  cvar_return         = {self.cvar_return:.3f}\n"
            f"  tail_risk_prob      = {self.tail_risk_probability:.3f}\n"
            f"  mean_stress         = {self.mean_stress:.3f}\n"
            f"  stress_collapse_freq= {self.stress_collapse_frequency:.3f}\n"
            f"  map_control_score   = {self.map_control_score:.3f}\n"
            f"  enemies_killed      = {self.enemies_killed:.1f}\n"
            f")"
        )


# ─── Main Evaluation Function ─────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    agent: TacticalAgent,
    env: SurvivalEnv,
    stress_model: StressModel,
    config: Config,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> EvaluationMetrics:
    """
    Run n_episodes of evaluation and compute all 6 research metrics.

    Args:
        agent:        Trained TacticalAgent
        env:          SurvivalEnv instance
        stress_model: StressModel instance
        config:       Master config
        n_episodes:   Number of evaluation episodes
        deterministic: If True, use greedy action selection

    Returns:
        EvaluationMetrics with all metrics populated
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    agent.eval()

    obs_shape = (
        config.env.num_obs_channels,
        2 * config.env.vision_radius + 1,
        2 * config.env.vision_radius + 1,
    )
    window = TrajectoryWindowBuilder(
        config.transformer.trajectory_window,
        obs_shape,
        device,
    )

    # ── Per-episode accumulators ──
    episode_returns    = []
    episode_lengths    = []
    deaths             = []
    stress_collapses   = []   # Collapse steps per episode
    total_steps_ep     = []   # Total steps per episode
    map_controls       = []
    kills_list         = []
    stress_sums        = []

    for ep_idx in range(n_episodes):
        # Re-seed env for reproducibility across episodes
        env.rng.seed(config.training.eval_seed_offset + ep_idx)
        obs = env.reset()
        stress_model.reset()
        window.reset()

        ep_return       = 0.0
        ep_steps        = 0
        ep_collapses    = 0
        ep_stress_sum   = 0.0
        ep_done_by_death = False

        while True:
            # ── Stress update ──
            dists = env.get_enemy_distances()
            sigma = 0.0
            if config.ablation.use_stress_model:
                sigma = stress_model.update(
                    health=env.health,
                    max_health=config.env.max_health,
                    ammo=env.ammo,
                    max_ammo=config.env.max_ammo,
                    enemy_distances=dists,
                    num_enemies_in_view=len(dists),
                    episode_step=ep_steps,
                    max_episode_steps=config.env.max_steps,
                )
                env.stress = sigma

            ep_stress_sum += sigma
            if sigma > config.stress.delay_threshold:
                ep_collapses += 1

            # ── Build window and get action ──
            window.push(
                obs["local_grid"], obs["scalars"],
                int(obs["prev_action"]), float(obs["prev_reward"])
            )
            obs_ctx = window.get_tensors()
            action, _, _ = agent.select_action(
                obs_ctx["local_grids"],
                obs_ctx["scalars"],
                obs_ctx["prev_actions"],
                obs_ctx["prev_rewards"],
                deterministic=deterministic,
            )

            # ── Apply stress effects during eval ──
            if config.ablation.use_action_delay:
                action = stress_model.check_action_delay(action)

            # ── Step env ──
            obs, reward, done, info = env.step(action)

            if config.ablation.use_dominance_reward:
                reward += config.training.dominance_lambda * info.get("map_control", 0.0)

            ep_return += reward
            ep_steps  += 1

            if done:
                ep_done_by_death = (env.health <= 0)
                break

            if ep_steps >= config.env.max_steps:
                break

        # ── Record episode stats ──
        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)
        deaths.append(1.0 if ep_done_by_death else 0.0)
        stress_collapses.append(ep_collapses)
        total_steps_ep.append(ep_steps)
        map_controls.append(env._compute_map_control())
        kills_list.append(float(env.total_enemies_killed))
        stress_sums.append(ep_stress_sum / max(ep_steps, 1))

    # ── Compute aggregate metrics ──
    returns_arr = np.array(episode_returns, dtype=np.float64)
    lengths_arr = np.array(episode_lengths, dtype=np.float64)

    alpha = config.distributional.cvar_alpha
    q_alpha = np.quantile(returns_arr, alpha)
    tail = returns_arr[returns_arr <= q_alpha]
    cvar = float(tail.mean()) if len(tail) > 0 else float(returns_arr.min())

    tail_risk = float((returns_arr <= np.quantile(returns_arr, 0.1)).mean())

    collapse_arr = np.array(stress_collapses, dtype=np.float64)
    steps_arr    = np.array(total_steps_ep, dtype=np.float64)
    collapse_freq = float(collapse_arr.sum() / max(steps_arr.sum(), 1))

    metrics = EvaluationMetrics(
        # Core survival
        survival_time       = float(lengths_arr.mean()),
        death_rate          = float(np.mean(deaths)),
        mean_return         = float(returns_arr.mean()),
        # Risk-sensitive
        cvar_return         = cvar,
        tail_risk_probability = tail_risk,
        # Stress
        mean_stress         = float(np.mean(stress_sums)),
        stress_collapse_frequency = collapse_freq,
        # Dominance
        map_control_score   = float(np.mean(map_controls)),
        enemies_killed      = float(np.mean(kills_list)),
        # Distribution
        return_std          = float(returns_arr.std()),
        return_min          = float(returns_arr.min()),
        return_max          = float(returns_arr.max()),
        return_p10          = float(np.percentile(returns_arr, 10)),
        return_p90          = float(np.percentile(returns_arr, 90)),
        _episode_returns    = episode_returns,
        _episode_lengths    = episode_lengths,
    )

    agent.train()
    return metrics


# ─── Print Table ────────────────────────────────────────────────────────────

def print_metrics_table(metrics: EvaluationMetrics, label: str = "Agent"):
    """Print a formatted research metrics table to console."""
    w = 50
    print(f"\n{'─'*w}")
    print(f"  Evaluation Results — {label}")
    print(f"{'─'*w}")
    rows = [
        ("Survival Time",          f"{metrics.survival_time:.1f} steps"),
        ("Death Rate",             f"{metrics.death_rate:.3f}"),
        ("Mean Return  E[G]",      f"{metrics.mean_return:.3f}"),
        ("CVaR Return  (risk)",    f"{metrics.cvar_return:.3f}"),
        ("Tail Risk Prob",         f"{metrics.tail_risk_probability:.3f}"),
        ("Mean Stress σ",          f"{metrics.mean_stress:.3f}"),
        ("Collapse Frequency",     f"{metrics.stress_collapse_frequency:.3f}"),
        ("Map Control",            f"{metrics.map_control_score:.3f}"),
        ("Enemies Killed",         f"{metrics.enemies_killed:.1f}"),
        ("Return  std",            f"{metrics.return_std:.3f}"),
        ("Return  p10 / p90",      f"{metrics.return_p10:.1f} / {metrics.return_p90:.1f}"),
    ]
    for name, val in rows:
        print(f"  {name:<26} {val:>16}")
    print(f"{'─'*w}\n")


# ─── Ablation Comparison ─────────────────────────────────────────────────────

def compare_objectives(
    cvar_metrics: EvaluationMetrics,
    ev_metrics: EvaluationMetrics,
) -> Dict[str, float]:
    """
    Side-by-side comparison of CVaR vs E[R] agents.

    Highlights where risk-sensitive training improves tail outcomes
    at the potential cost of mean performance.

    Returns:
        Dict with percentage differences (CVaR vs E[R])
    """
    def pct_diff(a, b):
        return ((a - b) / (abs(b) + 1e-9)) * 100

    comparison = {
        "survival_time_pct_diff":    pct_diff(cvar_metrics.survival_time,
                                               ev_metrics.survival_time),
        "death_rate_pct_diff":       pct_diff(cvar_metrics.death_rate,
                                               ev_metrics.death_rate),
        "cvar_return_pct_diff":      pct_diff(cvar_metrics.cvar_return,
                                               ev_metrics.cvar_return),
        "tail_risk_pct_diff":        pct_diff(cvar_metrics.tail_risk_probability,
                                               ev_metrics.tail_risk_probability),
        "collapse_freq_pct_diff":    pct_diff(cvar_metrics.stress_collapse_frequency,
                                               ev_metrics.stress_collapse_frequency),
        "map_control_pct_diff":      pct_diff(cvar_metrics.map_control_score,
                                               ev_metrics.map_control_score),
    }

    print("\n" + "="*60)
    print("  ABLATION: CVaR vs. Expected Value Objective")
    print("="*60)
    print(f"  {'Metric':<28} {'CVaR':>8} {'E[R]':>8} {'Δ%':>8}")
    print(f"  {'-'*56}")

    metric_pairs = [
        ("survival_time",          cvar_metrics.survival_time,          ev_metrics.survival_time),
        ("death_rate",             cvar_metrics.death_rate,             ev_metrics.death_rate),
        ("mean_return",            cvar_metrics.mean_return,            ev_metrics.mean_return),
        ("cvar_return",            cvar_metrics.cvar_return,            ev_metrics.cvar_return),
        ("tail_risk_prob",         cvar_metrics.tail_risk_probability,  ev_metrics.tail_risk_probability),
        ("stress_collapse_freq",   cvar_metrics.stress_collapse_frequency, ev_metrics.stress_collapse_frequency),
        ("map_control_score",      cvar_metrics.map_control_score,      ev_metrics.map_control_score),
    ]
    for name, cv, ev in metric_pairs:
        delta = pct_diff(cv, ev)
        print(f"  {name:<28} {cv:>8.3f} {ev:>8.3f} {delta:>+7.1f}%")
    print("="*60 + "\n")

    return comparison


# ─── Log Metrics to TensorBoard ──────────────────────────────────────────────

def log_eval_metrics(
    metrics: EvaluationMetrics,
    writer,
    global_step: int,
    prefix: str = "eval",
):
    """Write evaluation metrics to a TensorBoard SummaryWriter."""
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/survival_time",          metrics.survival_time,          global_.step)
    writer.add_scalar(f"{prefix}/death_rate",             metrics.death_rate,             global_step)
    writer.add_scalar(f"{prefix}/mean_return",            metrics.mean_return,            global_step)
    writer.add_scalar(f"{prefix}/cvar_return",            metrics.cvar_return,            global_step)
    writer.add_scalar(f"{prefix}/tail_risk_probability",  metrics.tail_risk_probability,  global_step)
    writer.add_scalar(f"{prefix}/mean_stress",            metrics.mean_stress,            global_step)
    writer.add_scalar(f"{prefix}/stress_collapse_freq",   metrics.stress_collapse_frequency, global_step)
    writer.add_scalar(f"{prefix}/map_control_score",      metrics.map_control_score,      global_step)
    writer.add_scalar(f"{prefix}/enemies_killed",         metrics.enemies_killed,         global_step)
    writer.add_scalar(f"{prefix}/return_std",             metrics.return_std,             global_step)
    writer.flush()
