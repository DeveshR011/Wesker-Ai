"""
Multi-Agent Survival Environment.

Extends the base SurvivalEnv to support N competing or cooperating agents
sharing the same gridworld. Supports three interaction modes:

    COMPETITIVE  — Last agent standing wins; agents can shoot each other.
    COOPERATIVE  — All agents must collectively survive; shared reward.
    MIXED        — Teams compete; team-mates share reward, enemies fight.

Architecture
────────────
    MultiAgentEnv wraps a single GridWorld (shared spatial state).
    Each agent has its own:
        - Health, ammo, stress
        - Partial observation (centered on that agent)
        - Position + action history

    The environment produces per-agent observations and rewards each step,
    following the standard MARL (obs_n, reward_n, done_n, info_n) API.

Multi-agent training notes
──────────────────────────
    For COMPETITIVE mode, use independent PPO with separate agent networks.
    For COOPERATIVE mode, parameter sharing across agents is supported.
    For MIXED mode, each team shares parameters within the team.

    Centralized-critic training is supported via the `get_global_state()`
    method which returns a flattened full-grid view (not partial) for use
    in a centralized value function (MAPPO / QMIX style).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import copy

from .config import EnvConfig
from .environment import (
    SurvivalEnv, Enemy, Projectile,
    ACTION_STAY, ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W,
    ACTION_SHOOT_N, ACTION_SHOOT_S, ACTION_SHOOT_E, ACTION_SHOOT_W,
    DIRECTION_DELTAS,
)


# ─── Mode definitions ─────────────────────────────────────────────────────────

class MultiAgentMode(Enum):
    COMPETITIVE = "competitive"   # Last agent standing wins
    COOPERATIVE = "cooperative"   # All agents share reward; survive together
    MIXED       = "mixed"         # Team-based: intra-team cooperative, inter-competitive


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent scenarios."""
    num_agents: int = 4                      # Total number of agents
    mode: MultiAgentMode = MultiAgentMode.COMPETITIVE

    # Team assignments (only used in MIXED mode)
    # List of team IDs per agent, e.g. [0, 0, 1, 1] for two 2-agent teams
    team_assignments: Optional[List[int]] = None

    # Shooting between agents
    friendly_fire: bool = False              # In MIXED mode, allow team-mate damage?
    agent_shoot_damage: float = 20.0         # Damage agents deal to each other

    # Cooperative specifics
    shared_reward_weight: float = 0.5        # Fraction of team-mate rewards to add
    spawn_min_dist_between_agents: int = 4   # Min distance between agent spawns

    # Competitive victory
    survival_bonus: float = 5.0             # Bonus for outlasting another agent
    last_survivor_bonus: float = 20.0       # Bonus for being the last agent alive

    # Observation
    show_team_mates_in_obs: bool = True      # Whether team-mates appear in observation
    # Channel for other agents in observation (added on top of base channels)
    # obs_channels = base channels + 2 (friendly agents ch, enemy agents ch)
    agent_obs_extra_channels: int = 2


# ─── Per-agent state ──────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Per-agent runtime state within the shared environment."""
    agent_id: int
    team_id: int
    y: int
    x: int
    health: float
    ammo: int
    stress: float = 0.0
    prev_action: int = ACTION_STAY
    prev_reward: float = 0.0
    alive: bool = True
    kills: int = 0                # Agent-kills (other agents)
    enemy_kills: int = 0          # Environment enemy kills
    steps_survived: int = 0
    shield_timer: int = 0
    speed_boost_timer: int = 0
    has_key: bool = False


# ─── Multi-Agent Observation ──────────────────────────────────────────────────

class MAObservation(NamedTuple):
    """Per-agent observation bundle."""
    local_grid: np.ndarray    # (C+2, H, W)
    scalars: np.ndarray       # (5,) [health, ammo, stress, has_key, num_alive_allies]
    prev_action: int
    prev_reward: float
    agent_id: int


# ─── Multi-Agent Environment ──────────────────────────────────────────────────

class MultiAgentEnv:
    """
    Multi-Agent Survival Gridworld.

    Supports COMPETITIVE, COOPERATIVE, and MIXED (team-based) scenarios.
    Wraps a single SurvivalEnv for the shared world state (enemies, items, map)
    while managing per-agent positions, health, ammo, and partial observations.

    Usage
    ─────
        env = MultiAgentEnv(env_config, ma_config, seed=42)
        obs_n = env.reset()                  # Dict[agent_id → MAObservation]
        obs_n, rew_n, done_n, info_n = env.step(actions)   # actions: Dict[agent_id → int]
        global_state = env.get_global_state()  # For centralized critic

    Termination
    ───────────
        COMPETITIVE: episode ends when only one (or zero) agents remain.
        COOPERATIVE: episode ends when ALL agents die or max_steps reached.
        MIXED: episode ends when one team is fully eliminated.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        ma_config: Optional[MultiAgentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.env_cfg = env_config
        self.ma_cfg = ma_config or MultiAgentConfig()
        self.rng = np.random.RandomState(seed)
        self.grid_size = env_config.grid_size
        self.vision_radius = env_config.vision_radius
        self.obs_size = 2 * self.vision_radius + 1
        self.num_agents = self.ma_cfg.num_agents

        # Validate and set team assignments
        if self.ma_cfg.mode == MultiAgentMode.MIXED:
            if self.ma_cfg.team_assignments is None:
                # Default: split agents into two teams
                half = self.num_agents // 2
                self.team_assignments = [0] * half + [1] * (self.num_agents - half)
            else:
                self.team_assignments = self.ma_cfg.team_assignments
        else:
            # In COMPETITIVE mode each agent is its own team; COOP all same team
            if self.ma_cfg.mode == MultiAgentMode.COMPETITIVE:
                self.team_assignments = list(range(self.num_agents))
            else:
                self.team_assignments = [0] * self.num_agents

        # Build the shared world (enemies, resources, walls)
        self._shared_env = SurvivalEnv(env_config, seed=seed)

        # Per-agent states
        self.agent_states: List[AgentState] = []
        self.step_count: int = 0

        # Track which agents are done (but episode may continue for others)
        self.agent_done: List[bool] = [False] * self.num_agents
        self.episode_done: bool = False

    # ─── Core API ────────────────────────────────────────────────

    @property
    def num_obs_channels(self) -> int:
        """Total observation channels = base + 2 extra (allies/enemies)."""
        return self.env_cfg.num_obs_channels + self.ma_cfg.agent_obs_extra_channels

    def reset(self) -> Dict[int, MAObservation]:
        """Reset episode. Returns dict of initial observations per agent."""
        self.step_count = 0
        self.agent_done = [False] * self.num_agents
        self.episode_done = False

        # Reset shared world (enemies, items, walls)
        self._shared_env.reset()

        # Spawn agents at spread-out positions
        self.agent_states = []
        occupied = set()
        for i in range(self.num_agents):
            y, x = self._find_spawn(occupied)
            occupied.add((y, x))
            self.agent_states.append(AgentState(
                agent_id=i,
                team_id=self.team_assignments[i],
                y=y,
                x=x,
                health=self.env_cfg.max_health,
                ammo=self.env_cfg.max_ammo // 2,
            ))

        return {i: self._observe(i) for i in range(self.num_agents)}

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[Dict[int, MAObservation], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        """
        Execute one step for all agents simultaneously.

        Args:
            actions: Dict mapping agent_id → action_int

        Returns:
            (obs_n, reward_n, done_n, info_n) — all dicts keyed by agent_id
        """
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.num_agents)}

        # ── 1. Shared world step (enemies, resources, spawning) ──
        # Use a dummy action for the shared env (we override agent position)
        # The shared env only handles NPC enemies and resource management
        self._shared_env.step_count = self.step_count

        # ── 2. Agent actions ──
        for i, ag in enumerate(self.agent_states):
            if not ag.alive:
                continue
            action = actions.get(i, ACTION_STAY)
            ag.prev_action = action
            rew = self._execute_agent_action(i, action)
            rewards[i] += rew

        # ── 3. Agent-vs-agent shooting resolution ──
        for i, ag in enumerate(self.agent_states):
            if not ag.alive:
                continue
            action = actions.get(i, ACTION_STAY)
            if action in (ACTION_SHOOT_N, ACTION_SHOOT_S, ACTION_SHOOT_E, ACTION_SHOOT_W):
                rew = self._agent_shoot_agents(i, action)
                rewards[i] += rew

        # ── 4. Enemy step in shared world ──
        self._shared_env._enemy_step()
        # Apply enemy damage to each agent
        for i, ag in enumerate(self.agent_states):
            if not ag.alive:
                continue
            self._shared_env.agent_y = ag.y
            self._shared_env.agent_x = ag.x
            self._shared_env.health = ag.health
            self._shared_env.shield_timer = ag.shield_timer
            enemy_rew = self._compute_enemy_damage(i)
            rewards[i] += enemy_rew

        # ── 5. Resource pickups per agent ──
        for i, ag in enumerate(self.agent_states):
            if not ag.alive:
                continue
            rew = self._pickup_resources(i)
            rewards[i] += rew

        # ── 6. Survival rewards ──
        for i, ag in enumerate(self.agent_states):
            if ag.alive:
                rewards[i] += 1.0
                ag.steps_survived += 1

        # ── 7. Stochastic spawns in shared world ──
        self._shared_env._stochastic_spawns()

        # ── 8. Cooperative reward sharing ──
        if self.ma_cfg.mode in (MultiAgentMode.COOPERATIVE, MultiAgentMode.MIXED):
            rewards = self._apply_cooperative_sharing(rewards)

        # ── 9. Check agent deaths ──
        new_deaths = []
        for i, ag in enumerate(self.agent_states):
            if ag.alive and ag.health <= 0:
                ag.alive = False
                ag.health = 0.0
                self.agent_done[i] = True
                rewards[i] -= 10.0
                new_deaths.append(i)

        # ── 10. Competitive survival bonus ──
        if self.ma_cfg.mode in (MultiAgentMode.COMPETITIVE, MultiAgentMode.MIXED):
            for dead_id in new_deaths:
                for i, ag in enumerate(self.agent_states):
                    if ag.alive and self.team_assignments[i] != self.team_assignments[dead_id]:
                        rewards[i] += self.ma_cfg.survival_bonus

        # ── 11. Check episode termination ──
        alive_agents = [i for i, ag in enumerate(self.agent_states) if ag.alive]
        self.episode_done = self._check_episode_done(alive_agents)

        if self.episode_done:
            # Last survivor bonus (competitive)
            if len(alive_agents) == 1 and self.ma_cfg.mode == MultiAgentMode.COMPETITIVE:
                rewards[alive_agents[0]] += self.ma_cfg.last_survivor_bonus
            # Max steps
            if self.step_count >= self.env_cfg.max_steps:
                for i in range(self.num_agents):
                    self.agent_done[i] = True

        # Update prev rewards
        for i, ag in enumerate(self.agent_states):
            ag.prev_reward = rewards[i]

        obs_n = {i: self._observe(i) for i in range(self.num_agents)}
        done_n = {i: self.agent_done[i] or self.episode_done for i in range(self.num_agents)}
        info_n = {i: self._agent_info(i) for i in range(self.num_agents)}

        return obs_n, rewards, done_n, info_n

    # ─── Observation ─────────────────────────────────────────────

    def _observe(self, agent_id: int) -> MAObservation:
        """Build partial observation for a specific agent."""
        ag = self.agent_states[agent_id]
        R = self.vision_radius
        C = self.num_obs_channels
        local_grid = np.zeros((C, self.obs_size, self.obs_size), dtype=np.float32)

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                gy, gx = ag.y + dy, ag.x + dx
                ly, lx = dy + R, dx + R

                # Base channels (reuse shared env logic)
                if gy < 0 or gy >= self.grid_size or gx < 0 or gx >= self.grid_size:
                    local_grid[0, ly, lx] = 1.0
                    continue

                local_grid[0, ly, lx] = self._shared_env.walls[gy, gx]
                pos = (gy, gx)

                # Enemies (NPC)
                for e in self._shared_env.enemies:
                    if e.alive and e.y == gy and e.x == gx:
                        if self.rng.random() > self.env_cfg.enemy_detection_noise:
                            local_grid[1, ly, lx] = 1.0

                if pos in self._shared_env.health_packs:
                    local_grid[2, ly, lx] = 1.0
                if pos in self._shared_env.ammo_packs:
                    local_grid[3, ly, lx] = 1.0
                for p in self._shared_env.projectiles:
                    if p.y == gy and p.x == gx:
                        local_grid[5, ly, lx] = 1.0
                if pos in self._shared_env.shields:
                    local_grid[6, ly, lx] = 1.0
                if pos in self._shared_env.speed_boosts:
                    local_grid[7, ly, lx] = 1.0
                if pos in self._shared_env.doors:
                    local_grid[8, ly, lx] = 1.0
                if pos in self._shared_env.keys:
                    local_grid[9, ly, lx] = 1.0

                # Extra channel 10: ally agents
                if self.ma_cfg.show_team_mates_in_obs:
                    for j, other in enumerate(self.agent_states):
                        if j == agent_id or not other.alive:
                            continue
                        if other.y == gy and other.x == gx:
                            if self.team_assignments[j] == self.team_assignments[agent_id]:
                                local_grid[self.env_cfg.num_obs_channels, ly, lx] = 1.0
                            else:
                                local_grid[self.env_cfg.num_obs_channels + 1, ly, lx] = 1.0

        # Self marker at center
        local_grid[4, R, R] = 1.0

        # Scalar features
        num_alive_allies = sum(
            1 for j, other in enumerate(self.agent_states)
            if j != agent_id and other.alive
            and self.team_assignments[j] == self.team_assignments[agent_id]
        )
        scalars = np.array([
            ag.health / self.env_cfg.max_health,
            ag.ammo / self.env_cfg.max_ammo,
            ag.stress,
            1.0 if ag.has_key else 0.0,
            num_alive_allies / max(self.num_agents - 1, 1),
        ], dtype=np.float32)

        return MAObservation(
            local_grid=local_grid,
            scalars=scalars,
            prev_action=ag.prev_action,
            prev_reward=np.float32(ag.prev_reward),
            agent_id=agent_id,
        )

    # ─── Global State (for centralized critic) ────────────────────

    def get_global_state(self) -> np.ndarray:
        """
        Return a flattened global state observation for use in a
        centralized critic (MAPPO / QMIX style).

        Includes:
            - Full wall grid (flattened)
            - All agent positions + health + ammo (normalized)
            - All enemy positions + health (normalized)
            - Resource positions (binary grid)

        Returns:
            1-D float32 numpy array
        """
        G = self.grid_size
        state_parts = []

        # Walls (flattened)
        state_parts.append(self._shared_env.walls.flatten())

        # Agent positions + stats
        agent_grid = np.zeros((3, G, G), dtype=np.float32)  # ch: position, health, ammo
        for ag in self.agent_states:
            if ag.alive:
                agent_grid[0, ag.y, ag.x] = 1.0
                agent_grid[1, ag.y, ag.x] = ag.health / self.env_cfg.max_health
                agent_grid[2, ag.y, ag.x] = ag.ammo / self.env_cfg.max_ammo
        state_parts.append(agent_grid.flatten())

        # Enemy positions + health
        enemy_grid = np.zeros((2, G, G), dtype=np.float32)
        for e in self._shared_env.enemies:
            if e.alive:
                enemy_grid[0, e.y, e.x] = 1.0
                enemy_grid[1, e.y, e.x] = e.health / 50.0
        state_parts.append(enemy_grid.flatten())

        # Resource grids
        res_grid = np.zeros((2, G, G), dtype=np.float32)
        for (hy, hx) in self._shared_env.health_packs:
            res_grid[0, hy, hx] = 1.0
        for (ay, ax) in self._shared_env.ammo_packs:
            res_grid[1, ay, ax] = 1.0
        state_parts.append(res_grid.flatten())

        return np.concatenate(state_parts, axis=0)

    @property
    def global_state_size(self) -> int:
        """Dimension of the global state vector."""
        G = self.grid_size
        return G * G + 3 * G * G + 2 * G * G + 2 * G * G

    # ─── Action execution ─────────────────────────────────────────

    def _execute_agent_action(self, agent_id: int, action: int) -> float:
        """Execute movement and shooting vs. NPC enemies for a single agent."""
        ag = self.agent_states[agent_id]

        if action == ACTION_STAY:
            return 0.0

        if action in (ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W):
            dy, dx = DIRECTION_DELTAS[action]
            ny, nx = ag.y + dy, ag.x + dx
            if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size
                    and self._shared_env.walls[ny, nx] == 0):
                # Don't walk into another agent's cell
                occupied = {(other.y, other.x) for j, other in enumerate(self.agent_states)
                            if j != agent_id and other.alive}
                if (ny, nx) not in occupied:
                    ag.y, ag.x = ny, nx
            return 0.0

        if action in (ACTION_SHOOT_N, ACTION_SHOOT_S, ACTION_SHOOT_E, ACTION_SHOOT_W):
            # NPC enemy shooting
            if ag.ammo <= 0:
                return -0.1
            ag.ammo -= 1
            dy, dx = DIRECTION_DELTAS[action]
            for dist in range(1, self.env_cfg.shoot_range + 1):
                ty, tx = ag.y + dy * dist, ag.x + dx * dist
                if ty < 0 or ty >= self.grid_size or tx < 0 or tx >= self.grid_size:
                    break
                if self._shared_env.walls[ty, tx] == 1:
                    break
                for e in self._shared_env.enemies:
                    if e.alive and e.y == ty and e.x == tx:
                        e.health -= self.env_cfg.shoot_damage
                        if not e.alive:
                            ag.enemy_kills += 1
                            self._shared_env.total_enemies_killed += 1
                            return 2.0
                        return 0.5
            return 0.0

        return 0.0

    def _agent_shoot_agents(self, shooter_id: int, action: int) -> float:
        """
        Resolve shooting between agents.

        In COMPETITIVE mode, all agents can shoot each other.
        In COOPERATIVE mode, no agent-vs-agent shooting.
        In MIXED mode, only enemy-team members can be shot (if friendly_fire=False).
        """
        if self.ma_cfg.mode == MultiAgentMode.COOPERATIVE:
            return 0.0

        shooter = self.agent_states[shooter_id]
        if not shooter.alive:
            return 0.0

        dy, dx = DIRECTION_DELTAS[action]
        reward = 0.0

        for dist in range(1, self.env_cfg.shoot_range + 1):
            ty, tx = shooter.y + dy * dist, shooter.x + dx * dist
            if ty < 0 or ty >= self.grid_size or tx < 0 or tx >= self.grid_size:
                break
            if self._shared_env.walls[ty, tx] == 1:
                break
            for j, target in enumerate(self.agent_states):
                if j == shooter_id or not target.alive:
                    continue
                if target.y == ty and target.x == tx:
                    # Check friendly fire rules
                    same_team = (self.team_assignments[shooter_id] == self.team_assignments[j])
                    if same_team and not self.ma_cfg.friendly_fire:
                        return 0.0   # Can't shoot team-mates
                    target.health -= self.ma_cfg.agent_shoot_damage
                    if target.health <= 0:
                        target.alive = False
                        target.health = 0.0
                        self.agent_done[j] = True
                        shooter.kills += 1
                        reward += 5.0   # Agent kill reward
                    else:
                        reward += 1.0   # Hit reward
                    return reward
        return reward

    def _compute_enemy_damage(self, agent_id: int) -> float:
        """Apply NPC enemy melee damage to agent if adjacent."""
        ag = self.agent_states[agent_id]
        reward = 0.0
        for e in self._shared_env.enemies:
            if not e.alive:
                continue
            if abs(e.y - ag.y) <= 1 and abs(e.x - ag.x) <= 1:
                dmg = self.env_cfg.enemy_damage
                if ag.shield_timer > 0:
                    from .config import EnvConfig as _EC
                    dmg *= (1.0 - self.env_cfg.shield_damage_reduction)
                ag.health -= dmg
                reward -= 0.5
        return reward

    def _pickup_resources(self, agent_id: int) -> float:
        """Auto-collect resources at agent's current position."""
        ag = self.agent_states[agent_id]
        pos = (ag.y, ag.x)
        reward = 0.0
        shared = self._shared_env

        if pos in shared.health_packs:
            shared.health_packs.remove(pos)
            heal = min(self.env_cfg.health_pack_heal, self.env_cfg.max_health - ag.health)
            ag.health += heal
            reward += 1.0

        if pos in shared.ammo_packs:
            shared.ammo_packs.remove(pos)
            gain = min(self.env_cfg.ammo_pack_amount, self.env_cfg.max_ammo - ag.ammo)
            ag.ammo += gain
            reward += 0.5

        if pos in shared.shields:
            shared.shields.remove(pos)
            ag.shield_timer = self.env_cfg.shield_duration
            reward += 1.0

        if pos in shared.keys:
            shared.keys.remove(pos)
            ag.has_key = True
            reward += 1.0

        return reward

    # ─── Cooperative sharing ──────────────────────────────────────

    def _apply_cooperative_sharing(self, rewards: Dict[int, float]) -> Dict[int, float]:
        """
        In cooperative/mixed mode, add a fraction of team-mates' rewards
        to each agent's reward to encourage team coordination.
        """
        new_rewards = dict(rewards)
        for i in range(self.num_agents):
            if not self.agent_states[i].alive:
                continue
            team_bonus = 0.0
            team_count = 0
            for j in range(self.num_agents):
                if j == i or not self.agent_states[j].alive:
                    continue
                if self.team_assignments[j] == self.team_assignments[i]:
                    team_bonus += rewards[j]
                    team_count += 1
            if team_count > 0:
                new_rewards[i] += self.ma_cfg.shared_reward_weight * (team_bonus / team_count)
        return new_rewards

    # ─── Termination ─────────────────────────────────────────────

    def _check_episode_done(self, alive_agents: List[int]) -> bool:
        if self.step_count >= self.env_cfg.max_steps:
            return True

        if self.ma_cfg.mode == MultiAgentMode.COMPETITIVE:
            return len(alive_agents) <= 1

        if self.ma_cfg.mode == MultiAgentMode.COOPERATIVE:
            return len(alive_agents) == 0

        if self.ma_cfg.mode == MultiAgentMode.MIXED:
            # Check if any team has been fully eliminated
            teams = set(self.team_assignments)
            for team in teams:
                team_alive = any(
                    self.agent_states[i].alive
                    for i in range(self.num_agents)
                    if self.team_assignments[i] == team
                )
                if not team_alive:
                    return True
            return False

        return False

    # ─── Helpers ─────────────────────────────────────────────────

    def _find_spawn(self, occupied: set) -> Tuple[int, int]:
        """Find a valid spawn position away from occupied cells."""
        min_dist = self.ma_cfg.spawn_min_dist_between_agents
        for _ in range(200):
            y = self.rng.randint(1, self.grid_size - 1)
            x = self.rng.randint(1, self.grid_size - 1)
            if self._shared_env.walls[y, x] != 0:
                continue
            ok = all(
                abs(y - oy) + abs(x - ox) >= min_dist
                for (oy, ox) in occupied
            )
            if ok:
                return (y, x)
        return (self.grid_size // 2, self.grid_size // 2)

    def _agent_info(self, agent_id: int) -> dict:
        ag = self.agent_states[agent_id]
        return {
            "agent_id": agent_id,
            "team_id": ag.team_id,
            "health": ag.health,
            "ammo": ag.ammo,
            "stress": ag.stress,
            "alive": ag.alive,
            "kills": ag.kills,
            "enemy_kills": ag.enemy_kills,
            "steps_survived": ag.steps_survived,
            "num_enemies": self._shared_env.alive_enemy_count,
            "step": self.step_count,
        }

    # ─── Utility ─────────────────────────────────────────────────

    def get_enemy_distances(self, agent_id: int) -> List[float]:
        """Manhattan distances from the specified agent to all alive NPC enemies."""
        ag = self.agent_states[agent_id]
        return [
            float(abs(e.y - ag.y) + abs(e.x - ag.x))
            for e in self._shared_env.enemies if e.alive
        ]

    def get_agent_distances(self, agent_id: int) -> Dict[int, float]:
        """Manhattan distances from the specified agent to all other alive agents."""
        ag = self.agent_states[agent_id]
        return {
            j: float(abs(other.y - ag.y) + abs(other.x - ag.x))
            for j, other in enumerate(self.agent_states)
            if j != agent_id and other.alive
        }

    @property
    def num_alive(self) -> int:
        return sum(1 for ag in self.agent_states if ag.alive)

    @property
    def alive_agent_ids(self) -> List[int]:
        return [i for i, ag in enumerate(self.agent_states) if ag.alive]

    def get_team_stats(self) -> Dict[int, dict]:
        """Per-team aggregate statistics."""
        teams: Dict[int, dict] = {}
        for i, ag in enumerate(self.agent_states):
            tid = ag.team_id
            if tid not in teams:
                teams[tid] = {"alive": 0, "total_kills": 0, "total_health": 0.0}
            if ag.alive:
                teams[tid]["alive"] += 1
                teams[tid]["total_health"] += ag.health
            teams[tid]["total_kills"] += ag.kills + ag.enemy_kills
        return teams
