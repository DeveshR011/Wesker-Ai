"""
2D Gridworld Survival Simulator (POMDP).

State space: agent health, ammo, position; enemy positions and aggression; resources.
Observation: partial view within vision radius with noisy enemy detection.
Actions: 4 move (NESW) + 4 shoot (NESW) + 1 stay = 9 discrete actions.
Episode terminates on agent death (health <= 0) or max_steps.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

from .config import EnvConfig


# ─── Action definitions ───
ACTION_STAY = 0
ACTION_MOVE_N = 1
ACTION_MOVE_S = 2
ACTION_MOVE_E = 3
ACTION_MOVE_W = 4
ACTION_SHOOT_N = 5
ACTION_SHOOT_S = 6
ACTION_SHOOT_E = 7
ACTION_SHOOT_W = 8

# Direction vectors: (dy, dx) — row, col
DIRECTION_DELTAS = {
    ACTION_MOVE_N: (-1, 0), ACTION_MOVE_S: (1, 0),
    ACTION_MOVE_E: (0, 1),  ACTION_MOVE_W: (0, -1),
    ACTION_SHOOT_N: (-1, 0), ACTION_SHOOT_S: (1, 0),
    ACTION_SHOOT_E: (0, 1),  ACTION_SHOOT_W: (0, -1),
}


@dataclass
class Enemy:
    """Represents an enemy entity with position and individual aggression level."""
    y: int
    x: int
    health: float = 50.0
    aggression: float = 0.5  # P(move toward agent)
    shoot_cooldown: int = 0

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.y, self.x)


@dataclass
class Projectile:
    """Represents a projectile fired by an enemy."""
    y: int
    x: int
    dy: int
    dx: int
    speed: int
    steps_until_move: int


class SurvivalEnv:
    """
    POMDP Survival Gridworld.

    The agent navigates a grid, collecting scarce resources (health packs,
    ammo packs) while avoiding/fighting stochastically aggressive enemies.
    The agent only observes a local patch with noisy enemy detection.

    Observation dict:
        local_grid: np.ndarray (C, H, W) with C=10 channels
            ch0: walls/boundaries
            ch1: enemies (with detection noise)
            ch2: health packs
            ch3: ammo packs
            ch4: self marker (center)
            ch5: enemy projectiles
            ch6: shields
            ch7: speed boosts
            ch8: doors
            ch9: keys
        scalars: np.ndarray (4,) — [health_norm, ammo_norm, stress, has_key]
        prev_action: int
        prev_reward: float
    """

    def __init__(self, config: EnvConfig, seed: Optional[int] = None):
        self.cfg = config
        self.rng = np.random.RandomState(seed)
        self.grid_size = config.grid_size
        self.vision_radius = config.vision_radius
        self.obs_size = 2 * self.vision_radius + 1

        # Walls: border cells are walls
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.walls[0, :] = 1.0
        self.walls[-1, :] = 1.0
        self.walls[:, 0] = 1.0
        self.walls[:, -1] = 1.0

        # Episode state (initialized in reset)
        self.agent_y: int = 0
        self.agent_x: int = 0
        self.health: float = 0.0
        self.ammo: int = 0
        self.enemies: List[Enemy] = []
        self.projectiles: List[Projectile] = []
        self.health_packs: List[Tuple[int, int]] = []
        self.ammo_packs: List[Tuple[int, int]] = []
        self.shields: List[Tuple[int, int]] = []
        self.speed_boosts: List[Tuple[int, int]] = []
        self.doors: List[Tuple[int, int]] = []
        self.keys: List[Tuple[int, int]] = []
        self.has_key: bool = False
        self.shield_timer: int = 0
        self.speed_boost_timer: int = 0
        self.step_count: int = 0
        self.total_enemies_spawned: int = 0
        self.total_enemies_killed: int = 0
        self.prev_action: int = ACTION_STAY
        self.prev_reward: float = 0.0
        self.stress: float = 0.0  # Set externally by stress model
        self.time_near_enemies: int = 0
        self.survival_reward_sum: float = 0.0
        self.dominance_reward_sum: float = 0.0
        self.stress_sum: float = 0.0
        self.done: bool = False

    # ─── Core API ────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset episode. Returns initial observation."""
        self.step_count = 0
        self.total_enemies_spawned = 0
        self.total_enemies_killed = 0
        self.done = False
        self.prev_action = ACTION_STAY
        self.prev_reward = 0.0
        self.stress = 0.0
        self.time_near_enemies = 0
        self.survival_reward_sum = 0.0
        self.dominance_reward_sum = 0.0
        self.stress_sum = 0.0
        self.projectiles = []
        self.shields = []
        self.speed_boosts = []
        self.doors = []
        self.keys = []
        self.has_key = False
        self.shield_timer = 0
        self.speed_boost_timer = 0

        # Place agent at random interior cell
        self.agent_y = self.rng.randint(1, self.grid_size - 1)
        self.agent_x = self.rng.randint(1, self.grid_size - 1)
        self.health = self.cfg.max_health
        self.ammo = self.cfg.max_ammo // 2  # Start with half ammo

        # Reset walls to border only before generating new map
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.walls[0, :] = 1.0
        self.walls[-1, :] = 1.0
        self.walls[:, 0] = 1.0
        self.walls[:, -1] = 1.0

        # Generate map layout
        self._generate_map()

        # Spawn initial enemies away from agent
        self.enemies = []
        for _ in range(self.cfg.num_initial_enemies):
            self._spawn_enemy(min_dist_to_agent=5)
        self.total_enemies_spawned = len(self.enemies)

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one environment step.

        Returns: (observation, reward, done, info)
        """
        assert 0 <= action < self.cfg.num_actions, f"Invalid action {action}"
        self.step_count += 1
        reward = 0.0
        info = {}

        # ── 0. Decrement item timers ──
        if self.shield_timer > 0:
            self.shield_timer -= 1
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1

        # ── 1. Agent action execution ──
        if action == ACTION_STAY:
            pass  # No-op
        elif action in (ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W):
            reward += self._move_agent(action)
        elif action in (ACTION_SHOOT_N, ACTION_SHOOT_S, ACTION_SHOOT_E, ACTION_SHOOT_W):
            reward += self._shoot(action)

        # ── 2. Auto-pickup resources at agent position ──
        reward += self._pickup_resources()

        # ── 3. Enemy behavior: move + attack ──
        reward += self._enemy_step()

        # ── 3.5. Move projectiles ──
        reward += self._move_projectiles()

        # ── 4. Stochastic spawning of enemies and resources ──
        self._stochastic_spawns()

        # ── 5. Survival reward (alive bonus) ──
        surv_rew = 1.0
        reward += surv_rew
        self.survival_reward_sum += surv_rew

        # ── 6. Dominance reward component ──
        map_control = self._compute_map_control()
        dom_rew = map_control * 1.0  # Assume lambda=1.0 for calculation
        if self.cfg.ablation.use_dominance_reward if hasattr(self.cfg, 'ablation') else True:
            reward += dom_rew
            self.dominance_reward_sum += dom_rew
        info["map_control"] = map_control

        # Track time near enemies and stress
        dists = self.get_enemy_distances()
        if dists and min(dists) <= 2:
            self.time_near_enemies += 1
        self.stress_sum += self.stress

        # ── 7. Check termination ──
        if self.health <= 0:
            self.health = 0
            death_pen = -10.0
            reward += death_pen
            self.survival_reward_sum += death_pen  # Death penalty counts as survival component
            self.done = True
        elif self.step_count >= self.cfg.max_steps:
            self.done = True

        self.prev_action = action
        self.prev_reward = reward
        obs = self._get_observation()

        info.update({
            "health": self.health,
            "ammo": self.ammo,
            "num_enemies": sum(1 for e in self.enemies if e.alive),
            "step": self.step_count,
            "enemies_killed": self.total_enemies_killed,
            "stress": self.stress,
            "time_near_enemies": self.time_near_enemies,
            "survival_reward_sum": self.survival_reward_sum,
            "dominance_reward_sum": self.dominance_reward_sum,
            "stress_sum": self.stress_sum,
        })

        return obs, reward, self.done, info

    # ─── Observation construction ────────────────────────────────

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct partial observation within vision radius.
        Returns dict with local_grid, scalars, prev_action, prev_reward.
        """
        R = self.vision_radius
        local_grid = np.zeros((self.cfg.num_obs_channels, self.obs_size, self.obs_size),
                              dtype=np.float32)

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                gy, gx = self.agent_y + dy, self.agent_x + dx
                ly, lx = dy + R, dx + R  # Local grid coords

                # Channel 0: walls / out-of-bounds
                if gy < 0 or gy >= self.grid_size or gx < 0 or gx >= self.grid_size:
                    local_grid[0, ly, lx] = 1.0
                else:
                    local_grid[0, ly, lx] = self.walls[gy, gx]

                    # Channel 1: enemies (with detection noise → false negatives)
                    for e in self.enemies:
                        if e.alive and e.y == gy and e.x == gx:
                            if self.rng.random() > self.cfg.enemy_detection_noise:
                                local_grid[1, ly, lx] = 1.0

                    # Channel 2: health packs
                    if (gy, gx) in self.health_packs:
                        local_grid[2, ly, lx] = 1.0

                    # Channel 3: ammo packs
                    if (gy, gx) in self.ammo_packs:
                        local_grid[3, ly, lx] = 1.0

                    # Channel 5: enemy projectiles
                    for p in self.projectiles:
                        if p.y == gy and p.x == gx:
                            local_grid[5, ly, lx] = 1.0

                    # Channel 6: shields
                    if (gy, gx) in self.shields:
                        local_grid[6, ly, lx] = 1.0
                    
                    # Channel 7: speed boosts
                    if (gy, gx) in self.speed_boosts:
                        local_grid[7, ly, lx] = 1.0
                    
                    # Channel 8: doors
                    if (gy, gx) in self.doors:
                        local_grid[8, ly, lx] = 1.0

                    # Channel 9: keys
                    if (gy, gx) in self.keys:
                        local_grid[9, ly, lx] = 1.0

        # Channel 4: self marker at center
        local_grid[4, R, R] = 1.0

        # Scalar features normalized to [0, 1]
        scalars = np.array([
            self.health / self.cfg.max_health,
            self.ammo / self.cfg.max_ammo,
            self.stress,
            1.0 if self.has_key else 0.0,
        ], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "scalars": scalars,
            "prev_action": self.prev_action,
            "prev_reward": np.float32(self.prev_reward),
        }

    # ─── Agent actions ───────────────────────────────────────────

    def _move_agent(self, action: int) -> float:
        """Move agent in the specified direction. Returns reward delta."""
        dy, dx = DIRECTION_DELTAS[action]
        if self.speed_boost_timer > 0:
            dy *= self.cfg.speed_boost_multiplier
            dx *= self.cfg.speed_boost_multiplier
        ny, nx = self.agent_y + dy, self.agent_x + dx
        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
            if (ny, nx) in self.doors and not self.has_key:
                return -0.1 # Penalty for trying to open a door without a key
            if (ny, nx) in self.doors and self.has_key:
                self.doors.remove((ny, nx))
                self.has_key = False
            if self.walls[ny, nx] == 0:
                self.agent_y, self.agent_x = ny, nx
        return 0.0

    def _shoot(self, action: int) -> float:
        """Fire in a direction. Hits first enemy in line within range. Costs 1 ammo."""
        if self.ammo <= 0:
            return -0.1  # Penalty for empty click
        self.ammo -= 1
        dy, dx = DIRECTION_DELTAS[action]
        reward = 0.0
        for dist in range(1, self.cfg.shoot_range + 1):
            ty, tx = self.agent_y + dy * dist, self.agent_x + dx * dist
            if ty < 0 or ty >= self.grid_size or tx < 0 or tx >= self.grid_size:
                break
            if self.walls[ty, tx] == 1:
                break
            for e in self.enemies:
                if e.alive and e.y == ty and e.x == tx:
                    e.health -= self.cfg.shoot_damage
                    if not e.alive:
                        self.total_enemies_killed += 1
                        reward += 2.0  # Kill reward
                    else:
                        reward += 0.5  # Hit reward
                    return reward  # Only hit first enemy in line
        return reward

    def _pickup_resources(self) -> float:
        """Auto-collect resources at agent's position."""
        reward = 0.0
        pos = (self.agent_y, self.agent_x)

        if pos in self.health_packs:
            self.health_packs.remove(pos)
            heal = min(self.cfg.health_pack_heal, self.cfg.max_health - self.health)
            self.health += heal
            reward += 1.0

        if pos in self.ammo_packs:
            self.ammo_packs.remove(pos)
            gain = min(self.cfg.ammo_pack_amount, self.cfg.max_ammo - self.ammo)
            self.ammo += gain
            reward += 0.5

        if pos in self.shields:
            self.shields.remove(pos)
            self.shield_timer = self.cfg.shield_duration
            reward += 1.0

        if pos in self.speed_boosts:
            self.speed_boosts.remove(pos)
            self.speed_boost_timer = self.cfg.speed_boost_duration
            reward += 1.0
        
        if pos in self.keys:
            self.keys.remove(pos)
            self.has_key = True
            reward += 1.0

        return reward

    # ─── Enemy AI ────────────────────────────────────────────────

    def _enemy_step(self) -> float:
        """Move enemies and handle attacks. Returns reward delta (negative for damage)."""
        reward = 0.0
        for e in self.enemies:
            if not e.alive:
                continue

            # Cooldown for shooting
            if e.shoot_cooldown > 0:
                e.shoot_cooldown -= 1

            # Stochastic aggression: move toward agent or random
            aggression = np.clip(
                e.aggression + self.rng.uniform(-self.cfg.enemy_aggression_variance,
                                                 self.cfg.enemy_aggression_variance),
                0.0, 1.0
            )

            if self.rng.random() < aggression:
                # Move toward agent (greedy)
                dy = np.sign(self.agent_y - e.y)
                dx = np.sign(self.agent_x - e.x)
                # Prefer axis with larger distance
                if abs(self.agent_y - e.y) >= abs(self.agent_x - e.x):
                    move = (int(dy), 0)
                else:
                    move = (0, int(dx))
            else:
                # Random movement
                move = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][self.rng.randint(5)]

            ny, nx = e.y + move[0], e.x + move[1]
            if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size
                    and self.walls[ny, nx] == 0):
                e.y, e.x = ny, nx

            # Attack if adjacent to agent
            if abs(e.y - self.agent_y) <= 1 and abs(e.x - self.agent_x) <= 1:
                damage = self.cfg.enemy_damage
                if self.shield_timer > 0:
                    damage *= (1.0 - self.cfg.shield_damage_reduction)
                self.health -= damage
                reward -= 0.5  # Damage penalty

            # Shoot if has line of sight and cooldown is over
            if e.shoot_cooldown == 0 and self.has_line_of_sight(e.pos, (self.agent_y, self.agent_x)):
                self._enemy_shoot(e)

        return reward

    def _move_projectiles(self) -> float:
        """Move projectiles and check for collisions. Returns reward delta."""
        reward = 0.0
        for p in self.projectiles[:]:
            p.steps_until_move -= 1
            if p.steps_until_move == 0:
                p.steps_until_move = p.speed
                ny, nx = p.y + p.dy, p.x + p.dx
                if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size
                        and self.walls[ny, nx] == 0):
                    p.y, p.x = ny, nx
                    if p.y == self.agent_y and p.x == self.agent_x:
                        damage = self.cfg.enemy_damage
                        if self.shield_timer > 0:
                            damage *= (1.0 - self.cfg.shield_damage_reduction)
                        self.health -= damage
                        reward -= 0.5  # Damage penalty
                        self.projectiles.remove(p)
                else:
                    self.projectiles.remove(p)
        return reward

    def _enemy_shoot(self, enemy: Enemy):
        """Enemy fires a projectile at the agent."""
        enemy.shoot_cooldown = self.cfg.enemy_shoot_cooldown
        dy = np.sign(self.agent_y - enemy.y)
        dx = np.sign(self.agent_x - enemy.x)
        self.projectiles.append(Projectile(
            y=enemy.y,
            x=enemy.x,
            dy=dy,
            dx=dx,
            speed=self.cfg.enemy_projectile_speed,
            steps_until_move=self.cfg.enemy_projectile_speed,
        ))

    def has_line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Check if there is a clear line of sight between two points."""
        y1, x1 = p1
        y2, x2 = p2
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True
        x_inc, y_inc = dx / steps, dy / steps
        x, y = x1, y1
        for _ in range(steps):
            if self.walls[round(y), round(x)] == 1:
                return False
            x += x_inc
            y += y_inc
        return True

    # ─── Spawning ────────────────────────────────────────────────

    def _generate_map(self):
        """Generate the map layout, including walls, doors, and keys."""
        # Create some random rooms
        for _ in range(5):
            w = self.rng.randint(3, 8)
            h = self.rng.randint(3, 8)
            x = self.rng.randint(1, self.grid_size - w - 1)
            y = self.rng.randint(1, self.grid_size - h - 1)
            self.walls[y:y+h, x] = 1.0
            self.walls[y:y+h, x+w-1] = 1.0
            self.walls[y, x:x+w] = 1.0
            self.walls[y+h-1, x:x+w] = 1.0

        # Place doors
        for _ in range(self.cfg.num_doors):
            while True:
                y = self.rng.randint(1, self.grid_size - 2)
                x = self.rng.randint(1, self.grid_size - 2)
                if self.walls[y, x] == 1 and self.walls[y+1, x] == 0 and self.walls[y-1, x] == 0:
                    self.walls[y, x] = 0
                    self.doors.append((y, x))
                    break
                if self.walls[y, x] == 1 and self.walls[y, x+1] == 0 and self.walls[y, x-1] == 0:
                    self.walls[y, x] = 0
                    self.doors.append((y, x))
                    break
        
        # Place keys
        for _ in range(self.cfg.num_keys):
            self._spawn_resource("key")

        # Spawn other resources
        for _ in range(self.cfg.max_health_packs):
            self._spawn_resource("health")
        for _ in range(self.cfg.max_ammo_packs):
            self._spawn_resource("ammo")
        for _ in range(self.cfg.max_shields):
            self._spawn_resource("shield")
        for _ in range(self.cfg.max_speed_boosts):
            self._spawn_resource("speed_boost")

    def _stochastic_spawns(self):
        """Attempt to spawn enemies and resources based on configured rates."""
        alive_enemies = sum(1 for e in self.enemies if e.alive)
        if alive_enemies < self.cfg.max_enemies and self.rng.random() < self.cfg.enemy_spawn_rate:
            self._spawn_enemy(min_dist_to_agent=3)
            self.total_enemies_spawned += 1

        # Stochastic resource re-spawning
        if len(self.health_packs) < self.cfg.max_health_packs and self.rng.random() < self.cfg.health_pack_spawn_rate:
            self._spawn_resource("health")
        if len(self.ammo_packs) < self.cfg.max_ammo_packs and self.rng.random() < self.cfg.ammo_pack_spawn_rate:
            self._spawn_resource("ammo")
        if len(self.shields) < self.cfg.max_shields and self.rng.random() < self.cfg.shield_spawn_rate:
            self._spawn_resource("shield")
        if len(self.speed_boosts) < self.cfg.max_speed_boosts and self.rng.random() < self.cfg.speed_boost_spawn_rate:
            self._spawn_resource("speed_boost")

    def _spawn_enemy(self, min_dist_to_agent: int = 3):
        """Spawn a single enemy at a random valid position away from the agent."""
        for _ in range(50):  # Max attempts
            y = self.rng.randint(1, self.grid_size - 1)
            x = self.rng.randint(1, self.grid_size - 1)
            if self.walls[y, x] == 0:
                dist = abs(y - self.agent_y) + abs(x - self.agent_x)
                if dist >= min_dist_to_agent:
                    aggression = np.clip(
                        self.cfg.enemy_base_aggression
                        + self.rng.uniform(-0.2, 0.2),
                        0.1, 0.9
                    )
                    self.enemies.append(Enemy(y=y, x=x, aggression=aggression))
                    return

    def _spawn_resource(self, kind: str):
        """Spawn a resource at a random empty interior cell."""
        for _ in range(50):
            y = self.rng.randint(1, self.grid_size - 1)
            x = self.rng.randint(1, self.grid_size - 1)
            if self.walls[y, x] == 0:
                pos = (y, x)
                if pos not in self.health_packs and pos not in self.ammo_packs and pos not in self.shields and pos not in self.speed_boosts and pos not in self.keys:
                    if kind == "health":
                        self.health_packs.append(pos)
                    elif kind == "ammo":
                        self.ammo_packs.append(pos)
                    elif kind == "shield":
                        self.shields.append(pos)
                    elif kind == "speed_boost":
                        self.speed_boosts.append(pos)
                    elif kind == "key":
                        self.keys.append(pos)
                    return

    # ─── Metrics helpers ─────────────────────────────────────────

    def _compute_map_control(self) -> float:
        """
        Map control score ∈ [0, 1].
        Defined as 1 − (alive_enemies / max_enemies), weighted by kill ratio.
        Higher = more dominant.
        """
        alive = sum(1 for e in self.enemies if e.alive)
        control = 1.0 - (alive / max(self.cfg.max_enemies, 1))
        if self.total_enemies_spawned > 0:
            kill_ratio = self.total_enemies_killed / self.total_enemies_spawned
            control = 0.5 * control + 0.5 * kill_ratio
        return float(np.clip(control, 0.0, 1.0))

    def get_enemy_distances(self) -> List[float]:
        """Manhattan distances to all alive enemies (for stress computation)."""
        dists = []
        for e in self.enemies:
            if e.alive:
                d = abs(e.y - self.agent_y) + abs(e.x - self.agent_x)
                dists.append(float(d))
        return dists

    @property
    def alive_enemy_count(self) -> int:
        return sum(1 for e in self.enemies if e.alive)
