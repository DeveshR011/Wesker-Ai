"""
Advanced 3D-Like Survival Environment (POMDP).

Extends the base 2D gridworld with richer mechanics that approximate
challenges found in first-person shooter and robotics simulation domains:

New Features vs. Base Environment:
    * Multi-floor dungeons  — portal/staircase transitions between floors
    * Enemy archetypes      — Grunt, Sniper, Tank with distinct AI behaviours
    * Environmental hazards — fire zones, acid pools, radiation fields
    * Cover objects         — solid pillars/crates that block projectiles
    * Ammo types            — Standard, Heavy, Explosive with blast radius
    * Depth channel         — simulated distance-to-nearest-wall per ray
    * 20-channel observation grid providing richer situational awareness
    * Expanded action space — 4 move + 4 shoot(std) + 4 shoot(heavy) +
                              4 shoot(explosive) + 2 interact + 1 stay = 19

Observation channels (20 total):
    0 : walls / out-of-bounds
    1 : enemies (with detection noise)
    2 : health packs
    3 : standard ammo packs
    4 : self marker (center)
    5 : enemy projectiles
    6 : shields
    7 : speed boosts
    8 : doors
    9 : keys
    10: fire hazard zones
    11: acid pool zones
    12: radiation field zones
    13: cover objects (block projectiles, not movement)
    14: portal / staircase markers
    15: heavy ammo packs
    16: explosive ammo packs
    17: tank enemies (high health)
    18: sniper enemies (long range)
    19: simulated depth / distance-to-wall

Scalar features (6):
    health_norm, std_ammo_norm, heavy_ammo_norm, explosive_ammo_norm, stress, has_key
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum

from .config import EnvConfig


# ─── Action definitions ────────────────────────────────────────────────────────

class Action(IntEnum):
    STAY          = 0
    MOVE_N        = 1
    MOVE_S        = 2
    MOVE_E        = 3
    MOVE_W        = 4
    SHOOT_STD_N   = 5
    SHOOT_STD_S   = 6
    SHOOT_STD_E   = 7
    SHOOT_STD_W   = 8
    SHOOT_HVY_N   = 9
    SHOOT_HVY_S   = 10
    SHOOT_HVY_E   = 11
    SHOOT_HVY_W   = 12
    SHOOT_EXP_N   = 13
    SHOOT_EXP_S   = 14
    SHOOT_EXP_E   = 15
    SHOOT_EXP_W   = 16
    INTERACT      = 17   # Open door / activate portal
    CROUCH_TOGGLE = 18   # Toggle crouch (reduces movement but improves accuracy)

NUM_ADVANCED_ACTIONS = 19

MOVE_ACTIONS   = {Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W}
STD_SHOOT_ACTIONS = {Action.SHOOT_STD_N, Action.SHOOT_STD_S, Action.SHOOT_STD_E, Action.SHOOT_STD_W}
HVY_SHOOT_ACTIONS = {Action.SHOOT_HVY_N, Action.SHOOT_HVY_S, Action.SHOOT_HVY_E, Action.SHOOT_HVY_W}
EXP_SHOOT_ACTIONS = {Action.SHOOT_EXP_N, Action.SHOOT_EXP_S, Action.SHOOT_EXP_E, Action.SHOOT_EXP_W}

DIRECTION_DELTAS: Dict[int, Tuple[int, int]] = {
    Action.MOVE_N: (-1, 0),     Action.MOVE_S: (1, 0),
    Action.MOVE_E: (0, 1),      Action.MOVE_W: (0, -1),
    Action.SHOOT_STD_N: (-1, 0), Action.SHOOT_STD_S: (1, 0),
    Action.SHOOT_STD_E: (0, 1),  Action.SHOOT_STD_W: (0, -1),
    Action.SHOOT_HVY_N: (-1, 0), Action.SHOOT_HVY_S: (1, 0),
    Action.SHOOT_HVY_E: (0, 1),  Action.SHOOT_HVY_W: (0, -1),
    Action.SHOOT_EXP_N: (-1, 0), Action.SHOOT_EXP_S: (1, 0),
    Action.SHOOT_EXP_E: (0, 1),  Action.SHOOT_EXP_W: (0, -1),
}


# ─── Enemy Archetypes ─────────────────────────────────────────────────────────

class EnemyType(IntEnum):
    GRUNT   = 0   # Standard, medium health, medium aggression
    SNIPER  = 1   # Long-range, low health, stays distant
    TANK    = 2   # High health, slow, high damage melee


@dataclass
class AdvancedEnemy:
    y: int
    x: int
    kind: EnemyType = EnemyType.GRUNT
    health: float = 50.0
    aggression: float = 0.5
    shoot_cooldown: int = 0
    alert_level: float = 0.0   # 0=unaware, 1=fully alerted

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.y, self.x)

    @property
    def max_health(self) -> float:
        return {EnemyType.GRUNT: 50.0, EnemyType.SNIPER: 30.0, EnemyType.TANK: 150.0}[self.kind]

    @property
    def shoot_range(self) -> int:
        return {EnemyType.GRUNT: 3, EnemyType.SNIPER: 8, EnemyType.TANK: 2}[self.kind]

    @property
    def damage(self) -> float:
        return {EnemyType.GRUNT: 10.0, EnemyType.SNIPER: 15.0, EnemyType.TANK: 25.0}[self.kind]

    @property
    def base_cooldown(self) -> int:
        return {EnemyType.GRUNT: 5, EnemyType.SNIPER: 8, EnemyType.TANK: 10}[self.kind]

    @property
    def move_speed(self) -> int:
        """Steps between movements (higher = slower)."""
        return {EnemyType.GRUNT: 1, EnemyType.SNIPER: 2, EnemyType.TANK: 2}[self.kind]


@dataclass
class AdvancedProjectile:
    y: int
    x: int
    dy: int
    dx: int
    speed: int
    steps_until_move: int
    damage: float = 10.0
    is_explosive: bool = False
    blast_radius: int = 1


# ─── Advanced Environment Config ─────────────────────────────────────────────

@dataclass
class AdvancedEnvConfig:
    """Configuration for the 3D-like advanced survival environment."""
    grid_size: int = 30
    vision_radius: int = 7
    num_floors: int = 3                    # Simulated floors via portals
    max_health: float = 100.0
    max_std_ammo: int = 50
    max_heavy_ammo: int = 20
    max_explosive_ammo: int = 10
    num_initial_enemies: int = 5
    max_enemies: int = 15
    enemy_spawn_rate: float = 0.02
    health_pack_spawn_rate: float = 0.01
    std_ammo_spawn_rate: float = 0.01
    heavy_ammo_spawn_rate: float = 0.005
    explosive_ammo_spawn_rate: float = 0.003
    max_health_packs: int = 3
    max_std_ammo_packs: int = 3
    max_heavy_ammo_packs: int = 2
    max_explosive_ammo_packs: int = 2
    max_shields: int = 1
    num_doors: int = 5
    num_keys: int = 2
    num_portals: int = 2                   # Floor transitions
    num_cover_objects: int = 8             # Projectile-blocking covers
    num_fire_zones: int = 4
    num_acid_zones: int = 3
    num_radiation_zones: int = 2
    fire_damage_per_step: float = 3.0
    acid_damage_per_step: float = 5.0
    radiation_damage_per_step: float = 1.5
    enemy_base_aggression: float = 0.5
    enemy_aggression_variance: float = 0.2
    shoot_damage_std: float = 20.0
    shoot_damage_heavy: float = 40.0
    shoot_damage_explosive: float = 35.0
    explosive_blast_radius: int = 2
    shoot_range_std: int = 5
    shoot_range_heavy: int = 7
    shoot_range_explosive: int = 6
    enemy_detection_noise: float = 0.08
    shield_duration: int = 50
    shield_damage_reduction: float = 0.5
    max_steps: int = 1500
    # Observation
    num_obs_channels: int = 20
    num_actions: int = NUM_ADVANCED_ACTIONS
    num_scalar_features: int = 6   # health, std_ammo, heavy_ammo, exp_ammo, stress, has_key


# ─── Advanced Survival Environment ────────────────────────────────────────────

class AdvancedSurvivalEnv:
    """
    3D-Like Survival Environment with multi-floor dungeons, enemy archetypes,
    environmental hazards, cover mechanics, and multiple ammo types.

    This provides significantly richer challenges than the base 2D gridworld,
    approximating aspects of first-person-shooter and robotics simulation settings.

    Observation:
        local_grid: (20, H, W) — 20 channel observation grid
        scalars:    (6,)       — [health, std_ammo, heavy_ammo, exp_ammo, stress, has_key]
        prev_action: int
        prev_reward: float
        current_floor: int

    Action Space: 19 discrete actions (see Action enum above)
    """

    NUM_OBS_CHANNELS = 20

    def __init__(self, config: Optional[AdvancedEnvConfig] = None, seed: Optional[int] = None):
        self.cfg = config or AdvancedEnvConfig()
        self.rng = np.random.RandomState(seed)
        self.grid_size = self.cfg.grid_size
        self.vision_radius = self.cfg.vision_radius
        self.obs_size = 2 * self.vision_radius + 1

        # Per-floor wall layouts (initialized in reset)
        self.floors: List[np.ndarray] = []
        self.current_floor: int = 0

        # Episode state
        self.agent_y: int = 0
        self.agent_x: int = 0
        self.health: float = 0.0
        self.std_ammo: int = 0
        self.heavy_ammo: int = 0
        self.explosive_ammo: int = 0
        self.enemies: List[AdvancedEnemy] = []
        self.projectiles: List[AdvancedProjectile] = []
        self.health_packs: List[Tuple[int, int]] = []
        self.std_ammo_packs: List[Tuple[int, int]] = []
        self.heavy_ammo_packs: List[Tuple[int, int]] = []
        self.explosive_ammo_packs: List[Tuple[int, int]] = []
        self.shields: List[Tuple[int, int]] = []
        self.doors: List[Tuple[int, int]] = []
        self.keys: List[Tuple[int, int]] = []
        self.portals: List[Tuple[int, int]] = []      # portal positions on current floor
        self.cover_objects: List[Tuple[int, int]] = []
        self.fire_zones: List[Tuple[int, int]] = []
        self.acid_zones: List[Tuple[int, int]] = []
        self.radiation_zones: List[Tuple[int, int]] = []
        self.has_key: bool = False
        self.shield_timer: int = 0
        self.is_crouching: bool = False
        self.step_count: int = 0
        self.total_enemies_killed: int = 0
        self.total_enemies_spawned: int = 0
        self.prev_action: int = int(Action.STAY)
        self.prev_reward: float = 0.0
        self.stress: float = 0.0
        self.done: bool = False
        self.survival_reward_sum: float = 0.0
        self.dominance_reward_sum: float = 0.0
        self.stress_sum: float = 0.0
        self.time_near_enemies: int = 0

    # ─── Properties ───────────────────────────────────────────────

    @property
    def walls(self) -> np.ndarray:
        """Active floor's wall grid."""
        return self.floors[self.current_floor]

    # ─── Core API ────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset episode. Returns initial observation."""
        self.step_count = 0
        self.total_enemies_killed = 0
        self.total_enemies_spawned = 0
        self.done = False
        self.current_floor = 0
        self.prev_action = int(Action.STAY)
        self.prev_reward = 0.0
        self.stress = 0.0
        self.shield_timer = 0
        self.is_crouching = False
        self.has_key = False
        self.time_near_enemies = 0
        self.survival_reward_sum = 0.0
        self.dominance_reward_sum = 0.0
        self.stress_sum = 0.0
        self.projectiles = []

        # Generate multi-floor map
        self._generate_all_floors()

        # Place agent in floor 0
        self.agent_y, self.agent_x = self._random_open_cell(min_dist_from=(0, 0))
        self.health = self.cfg.max_health
        self.std_ammo = self.cfg.max_std_ammo // 2
        self.heavy_ammo = self.cfg.max_heavy_ammo // 4
        self.explosive_ammo = self.cfg.max_explosive_ammo // 4

        # Spawn enemies
        self.enemies = []
        for _ in range(self.cfg.num_initial_enemies):
            self._spawn_enemy(min_dist=6)
        self.total_enemies_spawned = len(self.enemies)

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one step. Returns (obs, reward, done, info)."""
        assert 0 <= action < self.cfg.num_actions, f"Invalid action {action}"
        self.step_count += 1
        reward = 0.0

        # Timers
        if self.shield_timer > 0:
            self.shield_timer -= 1

        action_e = Action(action)

        # 1. Agent action
        if action_e == Action.STAY:
            pass
        elif action_e in MOVE_ACTIONS:
            reward += self._move_agent(action_e)
        elif action_e in STD_SHOOT_ACTIONS:
            reward += self._shoot(action_e, ammo_type="std")
        elif action_e in HVY_SHOOT_ACTIONS:
            reward += self._shoot(action_e, ammo_type="heavy")
        elif action_e in EXP_SHOOT_ACTIONS:
            reward += self._shoot(action_e, ammo_type="explosive")
        elif action_e == Action.INTERACT:
            reward += self._interact()
        elif action_e == Action.CROUCH_TOGGLE:
            self.is_crouching = not self.is_crouching

        # 2. Pick up resources
        reward += self._pickup_resources()

        # 3. Hazard damage
        reward += self._apply_hazard_damage()

        # 4. Enemy step
        reward += self._enemy_step()

        # 5. Move projectiles
        reward += self._move_projectiles()

        # 6. Stochastic spawns
        self._stochastic_spawns()

        # 7. Survival reward
        surv_rew = 1.0
        reward += surv_rew
        self.survival_reward_sum += surv_rew

        # 8. Map control / dominance
        map_control = self._compute_map_control()
        dom_rew = map_control * 0.5
        reward += dom_rew
        self.dominance_reward_sum += dom_rew

        # Track stress proximity
        dists = self.get_enemy_distances()
        if dists and min(dists) <= 2:
            self.time_near_enemies += 1
        self.stress_sum += self.stress

        # 9. Termination
        if self.health <= 0:
            self.health = 0.0
            reward -= 10.0
            self.done = True
        elif self.step_count >= self.cfg.max_steps:
            self.done = True

        self.prev_action = action
        self.prev_reward = reward
        obs = self._get_observation()

        info = {
            "health": self.health,
            "std_ammo": self.std_ammo,
            "heavy_ammo": self.heavy_ammo,
            "explosive_ammo": self.explosive_ammo,
            "num_enemies": sum(1 for e in self.enemies if e.alive),
            "step": self.step_count,
            "enemies_killed": self.total_enemies_killed,
            "stress": self.stress,
            "map_control": map_control,
            "current_floor": self.current_floor,
            "time_near_enemies": self.time_near_enemies,
            "survival_reward_sum": self.survival_reward_sum,
            "dominance_reward_sum": self.dominance_reward_sum,
            "stress_sum": self.stress_sum,
        }
        return obs, reward, self.done, info

    # ─── Observation ─────────────────────────────────────────────

    def _get_observation(self) -> Dict[str, np.ndarray]:
        R = self.vision_radius
        local_grid = np.zeros((self.NUM_OBS_CHANNELS, self.obs_size, self.obs_size), dtype=np.float32)

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                gy, gx = self.agent_y + dy, self.agent_x + dx
                ly, lx = dy + R, dx + R

                # Channel 0: walls / OOB
                if gy < 0 or gy >= self.grid_size or gx < 0 or gx >= self.grid_size:
                    local_grid[0, ly, lx] = 1.0
                    continue

                local_grid[0, ly, lx] = self.walls[gy, gx]
                pos = (gy, gx)

                # Cover objects also appear as opaque (channel 13)
                if pos in self.cover_objects:
                    local_grid[13, ly, lx] = 1.0

                # Channel 1: enemies (with detection noise)
                for e in self.enemies:
                    if e.alive and e.y == gy and e.x == gx:
                        if self.rng.random() > self.cfg.enemy_detection_noise:
                            local_grid[1, ly, lx] = 1.0
                        # Specific type markers
                        if e.kind == EnemyType.TANK and self.rng.random() > self.cfg.enemy_detection_noise:
                            local_grid[17, ly, lx] = 1.0
                        if e.kind == EnemyType.SNIPER and self.rng.random() > self.cfg.enemy_detection_noise:
                            local_grid[18, ly, lx] = 1.0

                if pos in self.health_packs:
                    local_grid[2, ly, lx] = 1.0
                if pos in self.std_ammo_packs:
                    local_grid[3, ly, lx] = 1.0
                for p in self.projectiles:
                    if p.y == gy and p.x == gx:
                        local_grid[5, ly, lx] = 1.0
                if pos in self.shields:
                    local_grid[6, ly, lx] = 1.0
                if pos in self.doors:
                    local_grid[8, ly, lx] = 1.0
                if pos in self.keys:
                    local_grid[9, ly, lx] = 1.0
                if pos in self.fire_zones:
                    local_grid[10, ly, lx] = 1.0
                if pos in self.acid_zones:
                    local_grid[11, ly, lx] = 1.0
                if pos in self.radiation_zones:
                    local_grid[12, ly, lx] = 1.0
                if pos in self.portals:
                    local_grid[14, ly, lx] = 1.0
                if pos in self.heavy_ammo_packs:
                    local_grid[15, ly, lx] = 1.0
                if pos in self.explosive_ammo_packs:
                    local_grid[16, ly, lx] = 1.0

        # Channel 4: self marker
        local_grid[4, R, R] = 1.0

        # Channel 19: depth (simulated ray-cast distance to wall per direction)
        depth_map = self._compute_depth_channel()
        local_grid[19] = depth_map

        scalars = np.array([
            self.health / self.cfg.max_health,
            self.std_ammo / self.cfg.max_std_ammo,
            self.heavy_ammo / self.cfg.max_heavy_ammo,
            self.explosive_ammo / max(self.cfg.max_explosive_ammo, 1),
            self.stress,
            1.0 if self.has_key else 0.0,
        ], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "scalars": scalars,
            "prev_action": self.prev_action,
            "prev_reward": np.float32(self.prev_reward),
            "current_floor": np.int32(self.current_floor),
        }

    def _compute_depth_channel(self) -> np.ndarray:
        """
        Simulate depth perception: for each cell in the observation window,
        compute the distance from the agent to the nearest wall along the
        ray from agent to that cell (normalized to [0, 1]).
        """
        R = self.vision_radius
        depth = np.ones((self.obs_size, self.obs_size), dtype=np.float32)
        max_dist = float(R * np.sqrt(2))

        for dy_i, dy in enumerate(range(-R, R + 1)):
            for dx_i, dx in enumerate(range(-R, R + 1)):
                dist = np.sqrt(dy ** 2 + dx ** 2)
                if dist == 0:
                    depth[dy_i, dx_i] = 1.0
                    continue
                steps = int(dist)
                blocked = False
                for s in range(1, steps + 1):
                    gy = int(round(self.agent_y + dy * s / dist))
                    gx = int(round(self.agent_x + dx * s / dist))
                    if (gy < 0 or gy >= self.grid_size
                            or gx < 0 or gx >= self.grid_size
                            or self.walls[gy, gx] == 1
                            or (gy, gx) in self.cover_objects):
                        depth[dy_i, dx_i] = (s - 1) / max(max_dist, 1)
                        blocked = True
                        break
                if not blocked:
                    depth[dy_i, dx_i] = dist / max(max_dist, 1)
        return depth

    # ─── Agent Actions ────────────────────────────────────────────

    def _move_agent(self, action: Action) -> float:
        dy, dx = DIRECTION_DELTAS[int(action)]
        # Crouching halves movement
        if self.is_crouching and self.rng.random() < 0.5:
            return 0.0
        ny, nx = self.agent_y + dy, self.agent_x + dx
        if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
            return 0.0
        if (ny, nx) in self.doors and not self.has_key:
            return -0.1
        if (ny, nx) in self.doors and self.has_key:
            self.doors.remove((ny, nx))
            self.has_key = False
        if (ny, nx) in self.cover_objects:
            return -0.05   # Cover is passable but slows agent
        if self.walls[ny, nx] == 0:
            self.agent_y, self.agent_x = ny, nx
        return 0.0

    def _shoot(self, action: Action, ammo_type: str = "std") -> float:
        """Fire in a direction with specified ammo type."""
        ammo_attr = {"std": "std_ammo", "heavy": "heavy_ammo", "explosive": "explosive_ammo"}
        range_map = {
            "std": self.cfg.shoot_range_std,
            "heavy": self.cfg.shoot_range_heavy,
            "explosive": self.cfg.shoot_range_explosive,
        }
        damage_map = {
            "std": self.cfg.shoot_damage_std,
            "heavy": self.cfg.shoot_damage_heavy,
            "explosive": self.cfg.shoot_damage_explosive,
        }

        if getattr(self, ammo_attr[ammo_type]) <= 0:
            return -0.1
        setattr(self, ammo_attr[ammo_type], getattr(self, ammo_attr[ammo_type]) - 1)

        dy, dx = DIRECTION_DELTAS[int(action)]
        reward = 0.0
        dmg = damage_map[ammo_type]
        shoot_range = range_map[ammo_type]

        for dist in range(1, shoot_range + 1):
            ty, tx = self.agent_y + dy * dist, self.agent_x + dx * dist
            if ty < 0 or ty >= self.grid_size or tx < 0 or tx >= self.grid_size:
                break
            if self.walls[ty, tx] == 1:
                break
            # Cover blocks projectiles
            if (ty, tx) in self.cover_objects:
                break

            if ammo_type == "explosive":
                # Check for enemies in blast radius
                hits = False
                for e in self.enemies:
                    if not e.alive:
                        continue
                    if abs(e.y - ty) <= self.cfg.explosive_blast_radius and abs(e.x - tx) <= self.cfg.explosive_blast_radius:
                        e.health -= dmg
                        hits = True
                        if not e.alive:
                            self.total_enemies_killed += 1
                            reward += 3.0
                        else:
                            reward += 0.8
                if hits:
                    return reward
            else:
                for e in self.enemies:
                    if e.alive and e.y == ty and e.x == tx:
                        e.health -= dmg
                        if not e.alive:
                            self.total_enemies_killed += 1
                            reward += 2.0 + (1.0 if e.kind == EnemyType.TANK else 0.0)
                        else:
                            reward += 0.5
                        return reward
        return reward

    def _interact(self) -> float:
        """Interact with adjacent portal/staircase to change floor."""
        pos = (self.agent_y, self.agent_x)
        # Check if on a portal
        if pos in self.portals:
            old_floor = self.current_floor
            self.current_floor = (self.current_floor + 1) % self.cfg.num_floors
            # Reset hazards/projectiles for new floor
            self.projectiles = []
            self.fire_zones = []
            self.acid_zones = []
            self.radiation_zones = []
            # Re-generate hazards for this floor
            self._populate_floor_hazards()
            # Respawn agent in new floor
            self.agent_y, self.agent_x = self._random_open_cell()
            # Spawn new enemies on new floor
            for _ in range(self.cfg.num_initial_enemies // 2):
                self._spawn_enemy(min_dist=5)
            return 2.0   # Bonus for ascending a floor
        return 0.0

    # ─── Resource Pickup ─────────────────────────────────────────

    def _pickup_resources(self) -> float:
        reward = 0.0
        pos = (self.agent_y, self.agent_x)

        if pos in self.health_packs:
            self.health_packs.remove(pos)
            heal = min(30.0, self.cfg.max_health - self.health)
            self.health += heal
            reward += 1.0

        if pos in self.std_ammo_packs:
            self.std_ammo_packs.remove(pos)
            gain = min(10, self.cfg.max_std_ammo - self.std_ammo)
            self.std_ammo += gain
            reward += 0.4

        if pos in self.heavy_ammo_packs:
            self.heavy_ammo_packs.remove(pos)
            gain = min(5, self.cfg.max_heavy_ammo - self.heavy_ammo)
            self.heavy_ammo += gain
            reward += 0.7

        if pos in self.explosive_ammo_packs:
            self.explosive_ammo_packs.remove(pos)
            gain = min(3, self.cfg.max_explosive_ammo - self.explosive_ammo)
            self.explosive_ammo += gain
            reward += 1.0

        if pos in self.shields:
            self.shields.remove(pos)
            self.shield_timer = self.cfg.shield_duration
            reward += 1.5

        if pos in self.keys:
            self.keys.remove(pos)
            self.has_key = True
            reward += 1.0

        return reward

    # ─── Hazards ──────────────────────────────────────────────────

    def _apply_hazard_damage(self) -> float:
        pos = (self.agent_y, self.agent_x)
        dmg = 0.0

        if pos in self.fire_zones:
            dmg += self.cfg.fire_damage_per_step
        if pos in self.acid_zones:
            dmg += self.cfg.acid_damage_per_step
        if pos in self.radiation_zones:
            dmg += self.cfg.radiation_damage_per_step

        if dmg > 0:
            if self.shield_timer > 0:
                dmg *= (1.0 - self.cfg.shield_damage_reduction)
            self.health -= dmg
            return -dmg / self.cfg.max_health

        return 0.0

    # ─── Enemy AI ────────────────────────────────────────────────

    def _enemy_step(self) -> float:
        reward = 0.0
        for e in self.enemies:
            if not e.alive:
                continue

            if e.shoot_cooldown > 0:
                e.shoot_cooldown -= 1

            # Update alert level
            dist_to_agent = abs(e.y - self.agent_y) + abs(e.x - self.agent_x)
            if dist_to_agent <= self.vision_radius:
                e.alert_level = min(1.0, e.alert_level + 0.2)
            else:
                e.alert_level = max(0.0, e.alert_level - 0.05)

            # Archetype-specific AI
            if e.kind == EnemyType.SNIPER:
                # Sniper: keep distance, shoot from range
                if dist_to_agent < 4:
                    # Retreat
                    dy = -int(np.sign(self.agent_y - e.y))
                    dx = -int(np.sign(self.agent_x - e.x))
                    self._try_move_enemy(e, dy, dx)
                elif e.alert_level > 0.3 and e.shoot_cooldown == 0:
                    if self._has_line_of_sight((e.y, e.x), (self.agent_y, self.agent_x)):
                        self._enemy_fire(e)
            elif e.kind == EnemyType.TANK:
                # Tank: bulldoze towards agent, heavy melee
                if e.alert_level > 0.2:
                    dy = int(np.sign(self.agent_y - e.y))
                    dx = int(np.sign(self.agent_x - e.x))
                    self._try_move_enemy(e, dy, dx)
                if dist_to_agent <= 1:
                    dmg = e.damage
                    if self.shield_timer > 0:
                        dmg *= (1.0 - self.cfg.shield_damage_reduction)
                    self.health -= dmg
                    reward -= 0.8
            else:
                # Grunt: stochastic aggression
                aggression = np.clip(
                    e.aggression + self.rng.uniform(-self.cfg.enemy_aggression_variance,
                                                     self.cfg.enemy_aggression_variance),
                    0.0, 1.0
                )
                if self.rng.random() < aggression * e.alert_level:
                    dy = int(np.sign(self.agent_y - e.y))
                    dx = int(np.sign(self.agent_x - e.x))
                    self._try_move_enemy(e, dy, dx)
                else:
                    move = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][self.rng.randint(5)]
                    self._try_move_enemy(e, move[0], move[1])

                if dist_to_agent <= 1:
                    dmg = e.damage
                    if self.shield_timer > 0:
                        dmg *= (1.0 - self.cfg.shield_damage_reduction)
                    self.health -= dmg
                    reward -= 0.5

                if e.shoot_cooldown == 0 and dist_to_agent <= e.shoot_range:
                    if self._has_line_of_sight((e.y, e.x), (self.agent_y, self.agent_x)):
                        self._enemy_fire(e)

        return reward

    def _try_move_enemy(self, e: AdvancedEnemy, dy: int, dx: int):
        ny, nx = e.y + dy, e.x + dx
        if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size
                and self.walls[ny, nx] == 0
                and (ny, nx) not in self.cover_objects):
            e.y, e.x = ny, nx

    def _enemy_fire(self, e: AdvancedEnemy):
        e.shoot_cooldown = e.base_cooldown
        dy = int(np.sign(self.agent_y - e.y))
        dx = int(np.sign(self.agent_x - e.x))
        self.projectiles.append(AdvancedProjectile(
            y=e.y, x=e.x, dy=dy, dx=dx,
            speed=1, steps_until_move=1,
            damage=e.damage,
        ))

    def _move_projectiles(self) -> float:
        reward = 0.0
        for p in self.projectiles[:]:
            p.steps_until_move -= 1
            if p.steps_until_move == 0:
                p.steps_until_move = p.speed
                ny, nx = p.y + p.dy, p.x + p.dx
                # Blocked by wall or cover
                if (ny < 0 or ny >= self.grid_size or nx < 0 or nx >= self.grid_size
                        or self.walls[ny, nx] == 1
                        or (ny, nx) in self.cover_objects):
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    continue
                p.y, p.x = ny, nx
                if p.y == self.agent_y and p.x == self.agent_x:
                    dmg = p.damage
                    if self.shield_timer > 0:
                        dmg *= (1.0 - self.cfg.shield_damage_reduction)
                    self.health -= dmg
                    reward -= 0.5
                    if p in self.projectiles:
                        self.projectiles.remove(p)
        return reward

    # ─── Utility ─────────────────────────────────────────────────

    def _has_line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        y1, x1 = p1
        y2, x2 = p2
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True
        x_inc, y_inc = dx / steps, dy / steps
        x, y = float(x1), float(y1)
        for _ in range(steps):
            ry, rx = round(y), round(x)
            if self.walls[ry, rx] == 1 or (ry, rx) in self.cover_objects:
                return False
            x += x_inc
            y += y_inc
        return True

    def _random_open_cell(self, min_dist_from: Optional[Tuple[int, int]] = None,
                           min_dist: int = 0) -> Tuple[int, int]:
        for _ in range(200):
            y = self.rng.randint(1, self.grid_size - 1)
            x = self.rng.randint(1, self.grid_size - 1)
            if self.walls[y, x] != 0 or (y, x) in self.cover_objects:
                continue
            if min_dist_from is not None:
                d = abs(y - min_dist_from[0]) + abs(x - min_dist_from[1])
                if d < min_dist:
                    continue
            return (y, x)
        return (self.grid_size // 2, self.grid_size // 2)

    # ─── Spawning ────────────────────────────────────────────────

    def _generate_all_floors(self):
        """Generate wall layouts and items for all floors."""
        self.floors = []
        for floor_idx in range(self.cfg.num_floors):
            floor_walls = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            floor_walls[0, :] = 1.0
            floor_walls[-1, :] = 1.0
            floor_walls[:, 0] = 1.0
            floor_walls[:, -1] = 1.0
            # Random rooms
            for _ in range(6 + floor_idx):
                w = self.rng.randint(3, 8)
                h = self.rng.randint(3, 8)
                x = self.rng.randint(1, self.grid_size - w - 1)
                y = self.rng.randint(1, self.grid_size - h - 1)
                floor_walls[y:y+h, x] = 1.0
                floor_walls[y:y+h, x+w-1] = 1.0
                floor_walls[y, x:x+w] = 1.0
                floor_walls[y+h-1, x:x+w] = 1.0
            self.floors.append(floor_walls)

        # Populate floor 0 items
        self._populate_floor_items()
        self._populate_floor_hazards()

    def _populate_floor_items(self):
        """Place resources, doors, keys, portals, and cover on current floor."""
        self.doors = []
        self.keys = []
        self.portals = []
        self.cover_objects = []
        self.health_packs = []
        self.std_ammo_packs = []
        self.heavy_ammo_packs = []
        self.explosive_ammo_packs = []
        self.shields = []

        # Doors in walls
        for _ in range(self.cfg.num_doors):
            for attempt in range(50):
                y = self.rng.randint(1, self.grid_size - 2)
                x = self.rng.randint(1, self.grid_size - 2)
                if (self.walls[y, x] == 1
                        and self.walls[y+1, x] == 0 and self.walls[y-1, x] == 0):
                    self.walls[y, x] = 0
                    self.doors.append((y, x))
                    break

        # Keys
        for _ in range(self.cfg.num_keys):
            pos = self._random_open_cell()
            if pos:
                self.keys.append(pos)

        # Portals (floor transitions)
        for _ in range(self.cfg.num_portals):
            pos = self._random_open_cell()
            if pos:
                self.portals.append(pos)

        # Cover objects
        for _ in range(self.cfg.num_cover_objects):
            pos = self._random_open_cell()
            if pos:
                self.cover_objects.append(pos)

        # Resources
        for _ in range(self.cfg.max_health_packs):
            pos = self._random_open_cell()
            if pos:
                self.health_packs.append(pos)
        for _ in range(self.cfg.max_std_ammo_packs):
            pos = self._random_open_cell()
            if pos:
                self.std_ammo_packs.append(pos)
        for _ in range(self.cfg.max_heavy_ammo_packs):
            pos = self._random_open_cell()
            if pos:
                self.heavy_ammo_packs.append(pos)
        for _ in range(self.cfg.max_explosive_ammo_packs):
            pos = self._random_open_cell()
            if pos:
                self.explosive_ammo_packs.append(pos)
        for _ in range(self.cfg.max_shields):
            pos = self._random_open_cell()
            if pos:
                self.shields.append(pos)

    def _populate_floor_hazards(self):
        """Place environmental hazard zones on the current floor."""
        self.fire_zones = []
        self.acid_zones = []
        self.radiation_zones = []

        for _ in range(self.cfg.num_fire_zones):
            pos = self._random_open_cell(min_dist_from=(self.agent_y, self.agent_x), min_dist=3)
            if pos:
                self.fire_zones.append(pos)
                # Spread hazard to adjacent cells
                for adj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = pos[0] + adj[0], pos[1] + adj[1]
                    if (0 < ny < self.grid_size - 1 and 0 < nx < self.grid_size - 1
                            and self.walls[ny, nx] == 0):
                        self.fire_zones.append((ny, nx))

        for _ in range(self.cfg.num_acid_zones):
            pos = self._random_open_cell(min_dist_from=(self.agent_y, self.agent_x), min_dist=3)
            if pos:
                self.acid_zones.append(pos)

        for _ in range(self.cfg.num_radiation_zones):
            pos = self._random_open_cell(min_dist_from=(self.agent_y, self.agent_x), min_dist=5)
            if pos:
                self.radiation_zones.append(pos)
                for adj in [(0, 1), (1, 0), (0, -1), (-1, 0),
                             (0, 2), (2, 0), (0, -2), (-2, 0)]:
                    ny, nx = pos[0] + adj[0], pos[1] + adj[1]
                    if (0 < ny < self.grid_size - 1 and 0 < nx < self.grid_size - 1
                            and self.walls[ny, nx] == 0):
                        self.radiation_zones.append((ny, nx))

    def _spawn_enemy(self, min_dist: int = 3):
        for _ in range(50):
            y = self.rng.randint(1, self.grid_size - 1)
            x = self.rng.randint(1, self.grid_size - 1)
            if self.walls[y, x] != 0:
                continue
            dist = abs(y - self.agent_y) + abs(x - self.agent_x)
            if dist >= min_dist:
                kind_roll = self.rng.random()
                if kind_roll < 0.15:
                    kind = EnemyType.TANK
                    hp = 150.0
                elif kind_roll < 0.35:
                    kind = EnemyType.SNIPER
                    hp = 30.0
                else:
                    kind = EnemyType.GRUNT
                    hp = 50.0
                aggression = np.clip(
                    self.cfg.enemy_base_aggression + self.rng.uniform(-0.2, 0.2),
                    0.1, 0.9,
                )
                self.enemies.append(AdvancedEnemy(y=y, x=x, kind=kind, health=hp, aggression=aggression))
                return

    def _stochastic_spawns(self):
        alive = sum(1 for e in self.enemies if e.alive)
        if alive < self.cfg.max_enemies and self.rng.random() < self.cfg.enemy_spawn_rate:
            self._spawn_enemy(min_dist=4)
            self.total_enemies_spawned += 1

        if len(self.health_packs) < self.cfg.max_health_packs and self.rng.random() < self.cfg.health_pack_spawn_rate:
            pos = self._random_open_cell()
            if pos:
                self.health_packs.append(pos)
        if len(self.std_ammo_packs) < self.cfg.max_std_ammo_packs and self.rng.random() < self.cfg.std_ammo_spawn_rate:
            pos = self._random_open_cell()
            if pos:
                self.std_ammo_packs.append(pos)
        if len(self.heavy_ammo_packs) < self.cfg.max_heavy_ammo_packs and self.rng.random() < self.cfg.heavy_ammo_spawn_rate:
            pos = self._random_open_cell()
            if pos:
                self.heavy_ammo_packs.append(pos)
        if len(self.explosive_ammo_packs) < self.cfg.max_explosive_ammo_packs and self.rng.random() < self.cfg.explosive_ammo_spawn_rate:
            pos = self._random_open_cell()
            if pos:
                self.explosive_ammo_packs.append(pos)

    # ─── Metrics ─────────────────────────────────────────────────

    def _compute_map_control(self) -> float:
        alive = sum(1 for e in self.enemies if e.alive)
        control = 1.0 - (alive / max(self.cfg.max_enemies, 1))
        if self.total_enemies_spawned > 0:
            kill_ratio = self.total_enemies_killed / self.total_enemies_spawned
            control = 0.5 * control + 0.5 * kill_ratio
        # Bonus for higher floors
        control += self.current_floor * 0.05
        return float(np.clip(control, 0.0, 1.0))

    def get_enemy_distances(self) -> List[float]:
        return [
            float(abs(e.y - self.agent_y) + abs(e.x - self.agent_x))
            for e in self.enemies if e.alive
        ]

    @property
    def alive_enemy_count(self) -> int:
        return sum(1 for e in self.enemies if e.alive)
