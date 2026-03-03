"""
Configuration module for Wesker AI.
All hyperparameters, ablation flags, and environment settings are centralized here.
Uses dataclasses for clean, type-annotated configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment parameters for the 2D POMDP gridworld."""
    grid_size: int = 20                    # NxN grid
    vision_radius: int = 5                 # Agent's partial observation radius
    max_health: float = 100.0
    max_ammo: int = 50
    num_initial_enemies: int = 3           # Enemies at episode start
    max_enemies: int = 10                  # Cap on simultaneous enemies
    enemy_spawn_rate: float = 0.03         # P(spawn) per step
    health_pack_spawn_rate: float = 0.015  # P(health pack) per step
    ammo_pack_spawn_rate: float = 0.015    # P(ammo pack) per step
    shield_spawn_rate: float = 0.01        # P(shield) per step
    speed_boost_spawn_rate: float = 0.01   # P(speed boost) per step
    max_health_packs: int = 3              # Cap on simultaneous health packs
    max_ammo_packs: int = 3                # Cap on simultaneous ammo packs
    max_shields: int = 1                   # Cap on simultaneous shields
    max_speed_boosts: int = 1              # Cap on simultaneous speed boosts
    num_doors: int = 3                     # Number of doors in the environment
    num_keys: int = 1                      # Number of keys in the environment
    enemy_base_aggression: float = 0.5     # Base P(move toward agent)
    enemy_aggression_variance: float = 0.2 # Stochastic aggression range
    enemy_damage: float = 10.0             # Damage per enemy attack
    shoot_damage: float = 25.0             # Damage agent deals per shot
    shoot_range: int = 5                   # Max shooting range
    enemy_shoot_range: int = 3             # Max shooting range for enemies
    enemy_shoot_cooldown: int = 5          # Steps between enemy shots
    enemy_projectile_speed: int = 1        # Steps per projectile move
    health_pack_heal: float = 30.0
    ammo_pack_amount: int = 10
    shield_duration: int = 50              # Steps the shield lasts
    shield_damage_reduction: float = 0.5   # Damage reduction from shield
    speed_boost_duration: int = 50         # Steps the speed boost lasts
    speed_boost_multiplier: int = 2        # Agent movement speed multiplier
    enemy_detection_noise: float = 0.1     # P(false negative) for enemy obs
    max_steps: int = 1000                  # Episode length cap
    # Observation dimensions (derived)
    num_obs_channels: int = 12             # 10 used (walls,enemies,health,ammo,self,projectiles,shields,speed_boosts,doors,keys) + 2 reserved
    num_actions: int = 9                   # 4 move + 4 shoot + 1 stay


@dataclass
class StressConfig:
    """Stress model σ_t parameters."""
    danger_weight: float = 0.15            # w1: weight for enemy proximity
    scarcity_weight: float = 0.08          # w2: weight for resource scarcity
    information_overload_weight: float = 0.1 # w3: weight for number of enemies in view
    time_pressure_weight: float = 0.001    # w4: weight for episode progress
    max_enemies_in_view: int = 5           # Max number of enemies for normalization
    decay_rate: float = 0.03               # Natural stress decay per step
    noise_scale: float = 1.0               # Multiplier for logit noise σ_t
    delay_threshold: float = 0.7           # σ_t above this → action delay
    delay_steps: int = 1                   # Steps of delay when triggered
    max_stress: float = 1.0                # Clamp upper bound
    min_stress: float = 0.0                # Clamp lower bound
    momentum: float = 0.9                  # EMA smoothing for σ_t


@dataclass
class TransformerConfig:
    """Causal Transformer architecture parameters."""
    d_model: int = 128                     # Embedding dimension
    n_heads: int = 4                       # Attention heads
    n_layers: int = 4                      # Transformer layers
    d_ff: int = 256                        # Feed-forward hidden dim
    dropout: float = 0.1
    trajectory_window: int = 32            # Context length T
    # Sub-embedding dimensions
    grid_embed_dim: int = 64               # CNN output dim for local grid
    scalar_embed_dim: int = 16             # MLP output for scalar features
    action_embed_dim: int = 16             # Action embedding dim
    reward_embed_dim: int = 16             # Reward/stress embedding dim


@dataclass
class DistributionalConfig:
    """Distributional RL / quantile regression parameters."""
    num_quantiles: int = 32                # Number of quantile fractions τ_i
    cvar_alpha: float = 0.25               # CVaR confidence level
    huber_kappa: float = 1.0               # Huber loss threshold for QR


@dataclass
class WorldModelConfig:
    """World model (VAE + transition + reward predictor) parameters."""
    latent_dim: int = 64                   # Latent state dimension
    hidden_dim: int = 128                  # Hidden layers in transition/reward
    kl_beta: float = 0.1                   # KL divergence weight
    recon_weight: float = 1.0              # Reconstruction loss weight
    reward_pred_weight: float = 1.0        # Reward prediction loss weight
    consistency_weight: float = 0.5        # Latent consistency regularization


@dataclass
class MCTSConfig:
    """MCTS planning parameters."""
    num_simulations: int = 50              # MCTS rollouts per action selection
    c_puct: float = 1.5                    # Exploration constant
    max_depth: int = 10                    # Max tree depth
    discount: float = 0.99                # Discount for tree backup
    temperature: float = 1.0               # Action selection temperature
    dirichlet_alpha: float = 0.3           # Root exploration noise
    dirichlet_epsilon: float = 0.25        # Fraction of noise at root


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    total_steps: int = 1_000_000           # Total environment steps
    rollout_length: int = 256              # Steps per rollout before update
    num_envs: int = 4                      # Parallel environments
    batch_size: int = 64                   # Minibatch size for updates
    num_epochs: int = 4                    # PPO epochs per update
    learning_rate: float = 3e-4
    world_model_lr: float = 1e-3
    gamma: float = 0.99                    # Discount factor
    gae_lambda: float = 0.95              # GAE lambda
    ppo_clip: float = 0.2                 # PPO clipping epsilon
    value_loss_coef: float = 0.5          # Value loss weight
    entropy_coef: float = 0.01            # Entropy bonus weight
    quantile_loss_coef: float = 0.5       # Quantile regression loss weight
    max_grad_norm: float = 0.5            # Gradient clipping
    dominance_lambda: float = 0.1          # Weight for dominance reward
    # Logging
    log_interval: int = 10                 # Log every N updates
    eval_interval: int = 50                # Evaluate every N updates
    eval_episodes: int = 20                # Episodes per evaluation
    save_interval: int = 100               # Checkpoint every N updates
    log_dir: str = "runs"                  # TensorBoard log directory
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False                # Enable Weights & Biases logging
    kl_coef: float = 0.1                   # KL penalty scaling
    eval_seed_offset: int = 1000           # Seed offset for evaluation episodes


@dataclass
class AblationFlags:
    """Toggle switches for ablation studies."""
    use_stress_model: bool = True          # Enable stress dynamics
    use_cvar_objective: bool = True        # CVaR vs expected value
    use_distributional_value: bool = True  # Quantile value head
    use_world_model: bool = True           # World model training
    use_mcts: bool = True                  # MCTS planning
    use_dominance_reward: bool = True      # Dominance reward component
    use_action_delay: bool = True          # Stress-induced action delay
    use_logit_noise: bool = True           # Stress-induced policy noise


@dataclass
class AdvancedEnvTrainingConfig:
    """Training parameters specific to the 3D-like AdvancedSurvivalEnv."""
    use_advanced_env: bool = False          # Enable AdvancedSurvivalEnv instead of base
    # Override transformer input dims for the larger obs
    advanced_obs_channels: int = 20        # AdvancedSurvivalEnv observation channels
    advanced_scalar_features: int = 6      # [health, std_ammo, heavy_ammo, exp_ammo, stress, has_key]
    advanced_num_actions: int = 19         # Extended action space


@dataclass
class MultiAgentTrainingConfig:
    """Training parameters for multi-agent scenarios."""
    enabled: bool = False                  # Enable multi-agent training
    num_agents: int = 4
    mode: str = "competitive"              # "competitive" | "cooperative" | "mixed"
    team_size: int = 2                     # Used when mode == "mixed"
    shared_params: bool = False            # Share network parameters across agents
    centralized_critic: bool = False       # Use global state for centralized value fn
    friendly_fire: bool = False
    shared_reward_weight: float = 0.5


@dataclass
class HITLTrainingConfig:
    """Configuration for human-in-the-loop training."""
    enabled: bool = False                  # Enable HITL wrapper
    guidance_mode: str = "suggest"         # "suggest" | "override" | "approve"
    trigger: str = "on_entropy"            # "always"|"on_entropy"|"on_danger"|"periodic"|"on_request"
    entropy_threshold: float = 1.8
    danger_health_threshold: float = 0.3
    periodic_interval: int = 20
    max_human_reward: float = 2.0
    min_human_reward: float = -2.0
    suggest_logit_bias: float = 3.0
    feedback_decay: float = 0.95
    interactive: bool = False              # Use CLI interface; False = NonInteractive (testing)


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    env: EnvConfig = field(default_factory=EnvConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationFlags = field(default_factory=AblationFlags)
    # ── New feature configs ──
    advanced_env: AdvancedEnvTrainingConfig = field(default_factory=AdvancedEnvTrainingConfig)
    multi_agent: MultiAgentTrainingConfig = field(default_factory=MultiAgentTrainingConfig)
    hitl: HITLTrainingConfig = field(default_factory=HITLTrainingConfig)
    seed: int = 42
    device: str = "cuda"                   # "cuda" or "cpu"

    @property
    def obs_grid_shape(self):
        """Shape of the local observation grid: (C, H, W)."""
        if self.advanced_env.use_advanced_env:
            s = 2 * self.env.vision_radius + 1
            return (self.advanced_env.advanced_obs_channels, s, s)
        s = 2 * self.env.vision_radius + 1
        return (self.env.num_obs_channels, s, s)

    @property
    def num_scalar_features(self):
        """Number of scalar observation features."""
        if self.advanced_env.use_advanced_env:
            return self.advanced_env.advanced_scalar_features
        return 4  # health, ammo, stress, has_key

    @property
    def effective_num_actions(self) -> int:
        """Number of discrete actions for the active environment."""
        if self.advanced_env.use_advanced_env:
            return self.advanced_env.advanced_num_actions
        return self.env.num_actions
