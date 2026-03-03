"""
Human-in-the-Loop (HITL) Module.

Provides mechanisms for a human operator to interact with, guide, and give
feedback to the Wesker AI agent during both training and evaluation.

Three interaction modalities
────────────────────────────
1. Action Guidance
   The human can suggest or override the agent's action at any timestep.
   Guidance modes:
       SUGGEST  — human action is added as a soft prior to the policy logits
       OVERRIDE — human action fully replaces the agent's sampled action
       APPROVE  — agent proposes action; human accepts or vetoes it

2. Reward Shaping (Human Feedback)
   The human provides real-time scalar feedback (+1 good, -1 bad, 0 neutral)
   that is added to the environment reward signal. This implements
   a simplified form of RLHF (Reinforcement Learning from Human Feedback).

3. Uncertainty-based Queries
   When the agent's policy entropy exceeds a threshold, the system
   automatically queries the human for guidance. This focuses human
   attention on the situations where the agent is most uncertain,
   minimising the operator's cognitive burden.

Integration
───────────
Wrap any SurvivalEnv (or AdvancedSurvivalEnv / MultiAgentEnv) with
HumanInTheLoopWrapper:

    env = SurvivalEnv(config.env)
    hitl_env = HumanInTheLoopWrapper(env, hitl_config)
    obs = hitl_env.reset()
    obs, rew, done, info = hitl_env.step(agent_action, policy_logits=logits)

The wrapper intercepts each step, checks for pending human input, and
combines agent + human signals transparently.

For non-interactive scripted testing use HumanFeedbackBuffer to replay
pre-recorded human feedback trajectories.
"""

import time
import threading
import queue
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from enum import Enum

from .environment import SurvivalEnv, ACTION_STAY


# ─── Enums & Config ───────────────────────────────────────────────────────────

class GuidanceMode(Enum):
    """How human action guidance is blended with agent actions."""
    SUGGEST  = "suggest"   # Bias policy logits toward human action (soft)
    OVERRIDE = "override"  # Replace agent action entirely
    APPROVE  = "approve"   # Agent proposes; human accepts or vetoes


class InterventionTrigger(Enum):
    """When to query the human."""
    ALWAYS       = "always"        # Query every step (demo/teaching mode)
    ON_ENTROPY   = "on_entropy"    # Query when policy entropy > threshold
    ON_DANGER    = "on_danger"     # Query when health < threshold
    ON_REQUEST   = "on_request"    # Only when human explicitly requests
    PERIODIC     = "periodic"      # Query every N steps


@dataclass
class HITLConfig:
    """Configuration for the Human-in-the-Loop wrapper."""
    guidance_mode: GuidanceMode = GuidanceMode.SUGGEST
    intervention_trigger: InterventionTrigger = InterventionTrigger.ON_ENTROPY

    # Entropy threshold for ON_ENTROPY trigger
    entropy_threshold: float = 1.8    # Bits — high entropy = uncertain agent

    # Health threshold for ON_DANGER trigger (fraction of max health)
    danger_health_threshold: float = 0.3

    # Periodic query interval
    periodic_interval: int = 20

    # Reward shaping magnitude clamp
    max_human_reward: float = 2.0
    min_human_reward: float = -2.0

    # Logit bias strength when mode == SUGGEST
    suggest_logit_bias: float = 3.0

    # Approval timeout (seconds); if no response, agent action is used
    approve_timeout: float = 5.0

    # Whether to log all HITL interactions
    log_interactions: bool = True

    # Discount for human feedback (older feedback is downweighted)
    feedback_decay: float = 0.95

    # Maximum number of actions in the action label map
    num_actions: int = 9

    # Human-readable labels for the 9 base actions
    action_labels: List[str] = field(default_factory=lambda: [
        "STAY", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "SHOOT_N", "SHOOT_S", "SHOOT_E", "SHOOT_W",
    ])


# ─── Feedback buffer ──────────────────────────────────────────────────────────

@dataclass
class HumanFeedbackEvent:
    """A single human feedback or guidance event."""
    step: int
    event_type: str                 # "reward", "action_suggestion", "override", "veto"
    value: Any                      # float for reward; int for action
    entropy: float = 0.0
    health_norm: float = 1.0
    timestamp: float = field(default_factory=time.time)


class HumanFeedbackBuffer:
    """
    Ring-buffer that stores human feedback events during a session.

    Can be:
        - Written to during live interaction
        - Replayed for offline analysis or imitation learning pre-training
        - Serialised to disk for persistence across sessions
    """

    def __init__(self, maxlen: int = 10_000):
        self._events: List[HumanFeedbackEvent] = []
        self.maxlen = maxlen

    def record(self, event: HumanFeedbackEvent):
        self._events.append(event)
        if len(self._events) > self.maxlen:
            self._events.pop(0)

    def get_action_suggestions(self) -> List[HumanFeedbackEvent]:
        return [e for e in self._events if e.event_type in ("action_suggestion", "override")]

    def get_reward_signals(self) -> List[HumanFeedbackEvent]:
        return [e for e in self._events if e.event_type == "reward"]

    def replay(self) -> List[HumanFeedbackEvent]:
        return list(self._events)

    def clear(self):
        self._events.clear()

    def save(self, path: str):
        """Save feedback buffer to a numpy .npz file."""
        if not self._events:
            return
        steps   = np.array([e.step for e in self._events], dtype=np.int32)
        types   = np.array([e.event_type for e in self._events])
        values  = np.array([float(e.value) for e in self._events], dtype=np.float32)
        entropy = np.array([e.entropy for e in self._events], dtype=np.float32)
        health  = np.array([e.health_norm for e in self._events], dtype=np.float32)
        np.savez(path, steps=steps, types=types, values=values,
                 entropy=entropy, health=health)

    def load(self, path: str):
        """Load feedback buffer from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self._events = []
        for step, typ, val, ent, hlth in zip(
            data["steps"], data["types"], data["values"],
            data["entropy"], data["health"]
        ):
            self._events.append(HumanFeedbackEvent(
                step=int(step), event_type=str(typ),
                value=float(val), entropy=float(ent), health_norm=float(hlth),
            ))

    def __len__(self) -> int:
        return len(self._events)


# ─── Interaction Interface ────────────────────────────────────────────────────

class HumanInterface:
    """
    Abstract base class for human interaction backends.
    Subclass to implement different frontends (CLI, GUI, web socket, etc.).
    """

    def query_action(
        self,
        obs_summary: dict,
        agent_action: int,
        action_labels: List[str],
        timeout: float = 5.0,
    ) -> Optional[int]:
        """
        Ask the human to choose an action.

        Args:
            obs_summary: dict with health, ammo, stress, num_enemies
            agent_action: What the agent would do (shown for reference)
            action_labels: Human-readable action names
            timeout: Seconds to wait; returns None on timeout

        Returns:
            Human-chosen action index, or None to accept agent's action
        """
        raise NotImplementedError

    def query_feedback(
        self,
        obs_summary: dict,
        agent_action: int,
        reward: float,
    ) -> float:
        """
        Ask the human for a scalar reward signal for the current step.

        Returns:
            float in [-2, 2]  (0 = no feedback)
        """
        raise NotImplementedError

    def notify(self, message: str):
        """Display an informational message to the human."""
        raise NotImplementedError


class CLIHumanInterface(HumanInterface):
    """
    Command-line interface for human-in-the-loop interaction.

    Prompts the operator via stdin/stdout. Suitable for debugging and
    scripted evaluation. For real-time operation the approve_timeout
    should be generous (≥ 10 s).
    """

    def query_action(
        self,
        obs_summary: dict,
        agent_action: int,
        action_labels: List[str],
        timeout: float = 5.0,
    ) -> Optional[int]:
        print("\n─── Human Query ───────────────────────────────")
        print(f"  Health: {obs_summary.get('health', '?'):.1f}  "
              f"Ammo: {obs_summary.get('ammo', '?')}  "
              f"Stress: {obs_summary.get('stress', 0):.2f}  "
              f"Enemies: {obs_summary.get('num_enemies', '?')}")
        print(f"  Agent wants to: [{agent_action}] {action_labels[agent_action]}")
        print("  Enter action number to override, or press Enter to accept:")
        for idx, label in enumerate(action_labels):
            marker = " ← agent" if idx == agent_action else ""
            print(f"    {idx}: {label}{marker}")

        result: List[Optional[int]] = [None]

        def _read():
            try:
                raw = input("  Your choice (or Enter): ").strip()
                if raw == "":
                    result[0] = None
                else:
                    v = int(raw)
                    if 0 <= v < len(action_labels):
                        result[0] = v
            except (ValueError, EOFError):
                result[0] = None

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def query_feedback(
        self,
        obs_summary: dict,
        agent_action: int,
        reward: float,
    ) -> float:
        print(f"\n  [Feedback] Action={agent_action}, EnvReward={reward:.2f}")
        try:
            raw = input("  Rate this action (+1 good / -1 bad / 0 neutral): ").strip()
            return float(raw) if raw else 0.0
        except (ValueError, EOFError):
            return 0.0

    def notify(self, message: str):
        print(f"  [HITL] {message}")


class NonInteractiveInterface(HumanInterface):
    """
    Non-interactive stub — returns no feedback.
    Use this in automated tests or when running in headless environments.
    Can optionally replay a pre-recorded feedback buffer.
    """

    def __init__(self, replay_buffer: Optional[HumanFeedbackBuffer] = None):
        self._buffer = replay_buffer
        self._replay_idx = 0

    def query_action(self, obs_summary, agent_action, action_labels, timeout=5.0):
        if self._buffer:
            events = self._buffer.get_action_suggestions()
            if self._replay_idx < len(events):
                ev = events[self._replay_idx]
                self._replay_idx += 1
                return int(ev.value)
        return None  # Accept agent's action

    def query_feedback(self, obs_summary, agent_action, reward):
        return 0.0  # No feedback

    def notify(self, message: str):
        pass  # Silent


# ─── HITL Wrapper ─────────────────────────────────────────────────────────────

class HumanInTheLoopWrapper:
    """
    Wraps any environment (SurvivalEnv, AdvancedSurvivalEnv, MultiAgentEnv)
    to inject human guidance and reward feedback into the training loop.

    The wrapper is transparent to the training code: it exposes the same
    reset() / step() interface as the underlying environment, with two
    additional optional parameters on step():

        step(action, policy_logits=None, agent_value=None)

    If policy_logits is given, entropy is computed to trigger ON_ENTROPY
    interventions. Otherwise entropy is estimated as 0 (no trigger).

    Human Feedback Integration
    ──────────────────────────
    Shaped reward at step t:
        r_shaped = r_env + α * r_human_t + β * Σ_{k<t} γ^{t-k} * r_human_k

    where r_human_t is the human scalar signal at step t, α = 1.0,
    β = 0.1, and γ = cfg.feedback_decay.

    The cumulative term lets past positive/negative feedback continue to
    influence learning for several subsequent steps.
    """

    def __init__(
        self,
        env: SurvivalEnv,
        config: Optional[HITLConfig] = None,
        interface: Optional[HumanInterface] = None,
    ):
        self.env = env
        self.cfg = config or HITLConfig()
        self.interface = interface or CLIHumanInterface()
        self.feedback_buffer = HumanFeedbackBuffer()

        self._step_count: int = 0
        self._pending_feedback: float = 0.0    # Accumulated past feedback
        self._last_intervention_step: int = -1
        self._episode_interactions: int = 0
        self._episode_human_reward_sum: float = 0.0

        # Interaction statistics
        self.total_interventions: int = 0
        self.total_overrides: int = 0
        self.total_vetoes: int = 0
        self.total_feedback_given: int = 0

    # ─── Core API ────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset underlying env and HITL state."""
        self._step_count = 0
        self._pending_feedback = 0.0
        self._episode_interactions = 0
        self._episode_human_reward_sum = 0.0
        self.interface.notify("New episode started.")
        return self.env.reset()

    def step(
        self,
        agent_action: int,
        policy_logits: Optional[torch.Tensor] = None,
        agent_value: Optional[float] = None,
    ) -> Tuple[dict, float, bool, dict]:
        """
        Step with optional human guidance and feedback.

        Args:
            agent_action: Action sampled by the agent policy
            policy_logits: Raw logits for entropy computation (optional)
            agent_value: Agent's value estimate for the current state (optional)

        Returns:
            Standard (obs, reward, done, info) tuple with shaped reward
        """
        self._step_count += 1

        # Compute entropy if logits provided
        entropy = 0.0
        if policy_logits is not None:
            p = torch.softmax(policy_logits, dim=-1)
            entropy = float(-(p * torch.log(p + 1e-8)).sum().item())

        # Build observation summary for human interface
        obs_summary = self._build_obs_summary()

        # 1. Determine if we should query the human
        should_query = self._check_trigger(entropy, obs_summary)

        executed_action = agent_action
        human_guidance_reward = 0.0

        if should_query:
            self.total_interventions += 1
            self._episode_interactions += 1
            self._last_intervention_step = self._step_count

            if self.cfg.guidance_mode == GuidanceMode.OVERRIDE:
                human_action = self.interface.query_action(
                    obs_summary, agent_action,
                    self.cfg.action_labels[:self.cfg.num_actions],
                    self.cfg.approve_timeout,
                )
                if human_action is not None:
                    executed_action = human_action
                    self.total_overrides += 1
                    if self.cfg.log_interactions:
                        self.feedback_buffer.record(HumanFeedbackEvent(
                            step=self._step_count,
                            event_type="override",
                            value=human_action,
                            entropy=entropy,
                            health_norm=obs_summary.get("health", 1.0),
                        ))

            elif self.cfg.guidance_mode == GuidanceMode.SUGGEST:
                human_action = self.interface.query_action(
                    obs_summary, agent_action,
                    self.cfg.action_labels[:self.cfg.num_actions],
                    self.cfg.approve_timeout,
                )
                if human_action is not None and policy_logits is not None:
                    # Bias logits toward the human suggestion
                    biased = policy_logits.clone()
                    biased[0, human_action] += self.cfg.suggest_logit_bias
                    executed_action = int(torch.argmax(biased, dim=-1).item())
                    if self.cfg.log_interactions:
                        self.feedback_buffer.record(HumanFeedbackEvent(
                            step=self._step_count,
                            event_type="action_suggestion",
                            value=human_action,
                            entropy=entropy,
                            health_norm=obs_summary.get("health", 1.0),
                        ))

            elif self.cfg.guidance_mode == GuidanceMode.APPROVE:
                # Show agent's action; human can veto
                human_action = self.interface.query_action(
                    obs_summary, agent_action,
                    self.cfg.action_labels[:self.cfg.num_actions],
                    self.cfg.approve_timeout,
                )
                if human_action is not None and human_action != agent_action:
                    # Human vetoed — use their action
                    executed_action = human_action
                    self.total_vetoes += 1
                    if self.cfg.log_interactions:
                        self.feedback_buffer.record(HumanFeedbackEvent(
                            step=self._step_count,
                            event_type="veto",
                            value=human_action,
                            entropy=entropy,
                            health_norm=obs_summary.get("health", 1.0),
                        ))

        # 2. Execute in environment
        obs, env_reward, done, info = self.env.step(executed_action)

        # 3. Collect human reward feedback
        human_fb = 0.0
        feedback_trigger = (
            should_query or
            self.cfg.intervention_trigger == InterventionTrigger.ALWAYS
        )
        if feedback_trigger:
            raw_fb = self.interface.query_feedback(obs_summary, executed_action, env_reward)
            human_fb = float(np.clip(raw_fb, self.cfg.min_human_reward, self.cfg.max_human_reward))
            if human_fb != 0.0:
                self.total_feedback_given += 1
                self._episode_human_reward_sum += human_fb
                if self.cfg.log_interactions:
                    self.feedback_buffer.record(HumanFeedbackEvent(
                        step=self._step_count,
                        event_type="reward",
                        value=human_fb,
                        entropy=entropy,
                        health_norm=obs_summary.get("health", 1.0),
                    ))

        # 4. Compute shaped reward with feedback decay
        self._pending_feedback = (
            self.cfg.feedback_decay * self._pending_feedback + human_fb
        )
        shaped_reward = env_reward + human_fb + 0.1 * self._pending_feedback

        # 5. Add HITL metadata to info
        info.update({
            "hitl_intervened": should_query,
            "hitl_executed_action": executed_action,
            "hitl_agent_action": agent_action,
            "hitl_human_feedback": human_fb,
            "hitl_shaped_reward": shaped_reward,
            "hitl_entropy": entropy,
            "hitl_episode_interactions": self._episode_interactions,
            "hitl_total_interventions": self.total_interventions,
            "hitl_human_reward_sum": self._episode_human_reward_sum,
        })

        return obs, shaped_reward, done, info

    # ─── Trigger logic ────────────────────────────────────────────

    def _check_trigger(self, entropy: float, obs_summary: dict) -> bool:
        """Determine whether to query the human this step."""
        trigger = self.cfg.intervention_trigger

        if trigger == InterventionTrigger.ALWAYS:
            return True

        if trigger == InterventionTrigger.ON_ENTROPY:
            return entropy > self.cfg.entropy_threshold

        if trigger == InterventionTrigger.ON_DANGER:
            health_norm = obs_summary.get("health", 1.0)
            return health_norm < self.cfg.danger_health_threshold

        if trigger == InterventionTrigger.PERIODIC:
            return (self._step_count % self.cfg.periodic_interval) == 0

        if trigger == InterventionTrigger.ON_REQUEST:
            return False  # Only triggered externally via request_intervention()

        return False

    # ─── External API ─────────────────────────────────────────────

    def request_intervention(self):
        """
        Externally trigger a human intervention on the next step.
        Call this from training code when the agent encounters a novel state
        it hasn't been trained on.
        """
        self._force_next_intervention = True

    def set_interface(self, interface: HumanInterface):
        """Swap out the human interface at runtime."""
        self.interface = interface

    def get_session_stats(self) -> dict:
        """Summary statistics for the current session."""
        return {
            "total_steps": self._step_count,
            "total_interventions": self.total_interventions,
            "intervention_rate": self.total_interventions / max(self._step_count, 1),
            "total_overrides": self.total_overrides,
            "total_vetoes": self.total_vetoes,
            "total_feedback_given": self.total_feedback_given,
            "feedback_buffer_size": len(self.feedback_buffer),
        }

    def get_imitation_dataset(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a behavioural-cloning dataset from recorded human overrides.

        Returns a dict with 'actions' (int32) suitable for imitation-learning
        pre-training, paired with the step indices where overrides occurred.
        Returns None if no overrides have been recorded.
        """
        overrides = [
            e for e in self.feedback_buffer.replay()
            if e.event_type in ("override", "veto")
        ]
        if not overrides:
            return None
        return {
            "steps": np.array([e.step for e in overrides], dtype=np.int32),
            "human_actions": np.array([int(e.value) for e in overrides], dtype=np.int32),
            "entropies": np.array([e.entropy for e in overrides], dtype=np.float32),
            "health_norms": np.array([e.health_norm for e in overrides], dtype=np.float32),
        }

    # ─── Helpers ─────────────────────────────────────────────────

    def _build_obs_summary(self) -> dict:
        """Extract a lightweight summary from the underlying env for display."""
        env = self.env
        return {
            "health": getattr(env, "health", 100.0) / getattr(env.cfg, "max_health", 100.0),
            "ammo": getattr(env, "ammo", 0),
            "stress": getattr(env, "stress", 0.0),
            "num_enemies": getattr(env, "alive_enemy_count", 0),
            "step": self._step_count,
        }

    # ─── Delegation ──────────────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate unknown attribute access to the wrapped environment."""
        return getattr(self.env, name)


# ─── Convenience factory ──────────────────────────────────────────────────────

def make_hitl_env(
    env: SurvivalEnv,
    guidance_mode: GuidanceMode = GuidanceMode.SUGGEST,
    trigger: InterventionTrigger = InterventionTrigger.ON_ENTROPY,
    interactive: bool = True,
    replay_buffer: Optional[HumanFeedbackBuffer] = None,
) -> HumanInTheLoopWrapper:
    """
    Convenience factory for creating a HITL-wrapped environment.

    Args:
        env:            Any SurvivalEnv-compatible environment
        guidance_mode:  How human actions are applied (SUGGEST/OVERRIDE/APPROVE)
        trigger:        When to query the human
        interactive:    If False, uses NonInteractiveInterface (for testing)
        replay_buffer:  Optional pre-recorded feedback to replay

    Returns:
        HumanInTheLoopWrapper ready to use in place of the base env
    """
    cfg = HITLConfig(guidance_mode=guidance_mode, intervention_trigger=trigger)
    interface: HumanInterface = (
        CLIHumanInterface() if interactive
        else NonInteractiveInterface(replay_buffer)
    )
    return HumanInTheLoopWrapper(env, cfg, interface)
