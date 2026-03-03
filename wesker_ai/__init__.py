"""
Wesker AI — Hierarchical Survival Optimization Package.

A research-grade prototype combining:
    - POMDP Survival Gridworld Environment
    - Transformer-based Tactical Policy
    - Distributional RL with CVaR risk-sensitivity
    - Stress-based Bounded Rationality
    - World Model (VAE + latent transition)
    - MCTS Planning in Latent Space
    - 3D-Like Advanced Survival Environment
    - Multi-Agent Scenarios (competitive / cooperative / mixed)
    - Human-in-the-Loop (HITL) guidance and reward shaping
"""

from .config import (
    Config, EnvConfig, StressConfig, TransformerConfig,
    DistributionalConfig, WorldModelConfig, MCTSConfig,
    TrainingConfig, AblationFlags,
    AdvancedEnvTrainingConfig, MultiAgentTrainingConfig, HITLTrainingConfig,
)
from .environment import SurvivalEnv, Enemy
from .agent import TacticalAgent, AgentOutput
from .stress import StressModel
from .networks import (
    ObservationEncoder, CausalTransformer,
    PolicyHead, ValueHead, QuantileHead,
)
from .risk import (
    quantile_regression_loss,
    compute_cvar,
    compute_gae,
    ppo_policy_loss,
    value_loss,
    entropy_bonus,
    cvar_policy_gradient,
    expected_value_objective,
    RiskMetrics,
)
from .world_model import WorldModel
from .mcts import MCTS, MCTSNode, select_mcts_action
from .train import Trainer, RolloutBuffer
from .evaluate import EvaluationMetrics, run_evaluation, print_metrics_table, compare_objectives

# ── New features ──
from .complex_environment import (
    AdvancedSurvivalEnv, AdvancedEnvConfig,
    AdvancedEnemy, EnemyType, Action as AdvancedAction,
)
from .multi_agent_env import (
    MultiAgentEnv, MultiAgentConfig, MultiAgentMode,
    AgentState, MAObservation,
)
from .human_in_loop import (
    HumanInTheLoopWrapper, HITLConfig,
    GuidanceMode, InterventionTrigger,
    HumanFeedbackBuffer, HumanFeedbackEvent,
    HumanInterface, CLIHumanInterface, NonInteractiveInterface,
    make_hitl_env,
)

__version__ = "2.0.0"
__all__ = [
    # Config
    "Config", "EnvConfig", "StressConfig", "TransformerConfig",
    "DistributionalConfig", "WorldModelConfig", "MCTSConfig",
    "TrainingConfig", "AblationFlags",
    "AdvancedEnvTrainingConfig", "MultiAgentTrainingConfig", "HITLTrainingConfig",
    # Core
    "SurvivalEnv", "Enemy",
    "TacticalAgent", "AgentOutput",
    "StressModel",
    # Networks
    "ObservationEncoder", "CausalTransformer",
    "PolicyHead", "ValueHead", "QuantileHead",
    # Risk
    "quantile_regression_loss", "compute_cvar", "compute_gae",
    "ppo_policy_loss", "value_loss", "entropy_bonus",
    "cvar_policy_gradient", "expected_value_objective", "RiskMetrics",
    # World Model + MCTS
    "WorldModel",
    "MCTS", "MCTSNode", "select_mcts_action",
    # Training + Eval
    "Trainer", "RolloutBuffer",
    "EvaluationMetrics", "run_evaluation", "print_metrics_table", "compare_objectives",
    # Advanced 3D-Like Environment
    "AdvancedSurvivalEnv", "AdvancedEnvConfig",
    "AdvancedEnemy", "EnemyType", "AdvancedAction",
    # Multi-Agent
    "MultiAgentEnv", "MultiAgentConfig", "MultiAgentMode",
    "AgentState", "MAObservation",
    # Human-in-the-Loop
    "HumanInTheLoopWrapper", "HITLConfig",
    "GuidanceMode", "InterventionTrigger",
    "HumanFeedbackBuffer", "HumanFeedbackEvent",
    "HumanInterface", "CLIHumanInterface", "NonInteractiveInterface",
    "make_hitl_env",
]
