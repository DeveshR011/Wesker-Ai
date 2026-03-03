"""
Monte Carlo Tree Search (MCTS) in Latent Space.

Architecture:
    The MCTS operates entirely in the world model's latent space,
    enabling deep lookahead without real environment interaction.

    At each node, the agent:
        1. SELECT    — traverse tree via UCB1 until a leaf
        2. EXPAND    — use policy prior π(a|z) from TacticalAgent
        3. EVALUATE  — imagined rollout value via WorldModel
        4. BACKUP    — propagate value estimates up the tree

    This separates:
        - Fast reflexive policy (TacticalAgent)
        - Slow deliberative planning (MCTS in latent space)

UCB Formula:
    UCB(s, a) = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))

    Q(s,a) = mean value of subtree under action a
    P(s,a) = prior probability from TacticalAgent policy
    N(s)   = visit count at parent node
    N(s,a) = visit count of action a

Dirichlet Noise:
    At the root node, exploration noise is added:
    P̃(s,a) = (1 − ε) · P(s,a) + ε · Dir(α)
    This prevents MCTS from being overly greedy at the tree root.

Action Selection:
    After simulations, sample action proportional to N(s,a)^(1/temp).
    At eval time, temp → 0 → argmax N(s,a).
"""

import math
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict

from .config import Config, MCTSConfig


# ─── MCTS Node ────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    A node in the MCTS search tree.

    Represents a latent state z at some depth in the imagined tree.
    Stores action-level statistics (N, W, Q, P) for all children.
    """

    __slots__ = [
        "z",          # Latent state tensor (1, latent_dim)
        "depth",      # Tree depth
        "N",          # Visit counts: [num_actions]
        "W",          # Total values: [num_actions]
        "P",          # Prior probs from policy: [num_actions]
        "children",   # Dict[action_int → MCTSNode]
        "is_expanded",
        "parent",
        "parent_action",
    ]

    def __init__(
        self,
        z: torch.Tensor,
        depth: int = 0,
        parent: Optional["MCTSNode"] = None,
        parent_action: Optional[int] = None,
    ):
        self.z = z
        self.depth = depth
        self.N: Optional[np.ndarray] = None        # Initialized on expand
        self.W: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded = False
        self.parent = parent
        self.parent_action = parent_action

    @property
    def total_visits(self) -> int:
        return int(self.N.sum()) if self.N is not None else 0

    def Q(self) -> np.ndarray:
        """Mean action values: Q(s,a) = W(s,a) / max(N(s,a), 1)."""
        denom = np.maximum(self.N, 1)
        return self.W / denom

    def ucb_scores(self, c_puct: float) -> np.ndarray:
        """
        UCB1 exploration-exploitation score per action.

        UCB(s,a) = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
        """
        sqrt_total = math.sqrt(max(self.total_visits, 1))
        u = c_puct * self.P * sqrt_total / (1.0 + self.N)
        return self.Q() + u

    def best_action(self, c_puct: float) -> int:
        """Select action with highest UCB score."""
        return int(np.argmax(self.ucb_scores(c_puct)))


# ─── MCTS Search ─────────────────────────────────────────────────────────────

class MCTS:
    """
    Monte Carlo Tree Search in the world model's latent space.

    The MCTS hybrid combines:
        - TacticalAgent policy for fast prior P(a|z)
        - WorldModel for imagined rollouts
        - Value head for leaf evaluation

    Workflow per call to `search()`:
        For k = 1..num_simulations:
            node  = select(root)         # follow UCB to leaf
            value = expand+evaluate(node)
            backup(node, value)
        return action_probs = N^{1/T} / Σ N^{1/T}
    """

    def __init__(self, config: Config):
        self.cfg = config.mcts
        self.num_actions = config.env.num_actions
        self.device = config.device

    @torch.no_grad()
    def search(
        self,
        root_z: torch.Tensor,    # (1, latent_dim)
        agent,                   # TacticalAgent (for policy prior + value)
        world_model,             # WorldModel (for transition + reward)
        obs_ctx: dict,           # Current observation context for agent
    ) -> Tuple[int, np.ndarray]:
        """
        Run MCTS simulations from root_z and return action probabilities.

        Args:
            root_z:     Encoded current latent state (1, latent_dim)
            agent:      TacticalAgent for policy prior and value estimates
            world_model: WorldModel for imagined transitions
            obs_ctx:    Observation window dict for agent forward pass
                        Keys: local_grids, scalars, prev_actions, prev_rewards

        Returns:
            best_action:  int — selected action
            action_probs: (num_actions,) visit count distribution
        """
        root = MCTSNode(z=root_z, depth=0)

        # ── Initialize root with policy prior ──
        prior = self._get_policy_prior(agent, obs_ctx)         # (num_actions,)
        root_value = self._get_value(agent, obs_ctx)

        # Add Dirichlet noise at root for exploration
        eps = self.cfg.dirichlet_epsilon
        alpha = self.cfg.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * self.num_actions)
        noisy_prior = (1 - eps) * prior + eps * noise

        self._expand_node(root, noisy_prior, self.num_actions)
        # Backup root with its own value estimate
        self._backup(root, root_value, parent_action=None)

        # ── Run simulations ──
        for _ in range(self.cfg.num_simulations):
            node, search_path = self._select(root)

            # Evaluate and expand (if not too deep)
            if node.depth < self.cfg.max_depth:
                value = self._expand_and_evaluate(node, agent, world_model)
            else:
                # Max depth: use value head at leaf
                value = self._evaluate_leaf(agent, node.z)

            # Backup
            self._backup_path(search_path, value)

        # ── Compute action probabilities ──
        action_probs = self._action_probs(root, self.cfg.temperature)
        action = int(np.random.choice(self.num_actions, p=action_probs))

        return action, action_probs

    # ── Selection ──

    def _select(
        self, root: MCTSNode
    ) -> Tuple[MCTSNode, List[Tuple[MCTSNode, int]]]:
        """
        Traverse tree following UCB until reaching an unexpanded node.

        Returns:
            leaf:        Terminal node of the selection path
            search_path: List of (node, action_taken) pairs for backup
        """
        node = root
        search_path: List[Tuple[MCTSNode, int]] = []

        while node.is_expanded and node.children:
            action = node.best_action(self.cfg.c_puct)
            search_path.append((node, action))

            if action not in node.children:
                break
            node = node.children[action]

        return node, search_path

    # ── Expansion + Evaluation ──

    def _expand_and_evaluate(
        self,
        node: MCTSNode,
        agent,
        world_model,
    ) -> float:
        """
        Expand a leaf node using the world model + policy prior.

        For each action:
            1. Imagine z_next = transition(z, a)
            2. Create child node with z_next

        Evaluate leaf value via the agent's value head applied to z.
        """
        # Get policy prior from agent's value of this latent (approximated)
        prior = self._prior_from_latent(agent, node.z)  # (num_actions,)

        # Create child nodes for each action
        # OPTIMIZATION: Batch process all actions in one forward pass
        B = node.z.size(0) # Should be 1 typically for MCTS root, but supports B
        actions_t = torch.arange(self.num_actions, dtype=torch.long, device=node.z.device).repeat_interleave(B)
        z_rep = node.z.repeat(self.num_actions, 1) # (num_actions * B, L)
        
        with torch.no_grad():
            z_next_batch = world_model.transition(z_rep, actions_t) # (num_actions * B, L)
            
        for a in range(self.num_actions):
            z_next = z_next_batch[a*B : (a+1)*B]
            child = MCTSNode(z=z_next, depth=node.depth + 1,
                             parent=node, parent_action=a)
            node.children[a] = child

        self._expand_node(node, prior, self.num_actions)

        # Evaluate current latent using world model value signal
        value = self._evaluate_leaf(agent, node.z)
        return value

    def _expand_node(
        self,
        node: MCTSNode,
        prior: np.ndarray,
        num_actions: int,
    ):
        """Initialize node statistics from policy prior."""
        node.N = np.zeros(num_actions, dtype=np.float32)
        node.W = np.zeros(num_actions, dtype=np.float32)
        node.P = prior.copy()
        node.is_expanded = True

    # ── Value Estimation ──

    def _evaluate_leaf(self, agent, z: torch.Tensor) -> float:
        """
        Estimate value at a latent state using the agent's value head.

        We project z through a linear layer to d_model and then run
        the value head. This requires the agent to have a `latent_value`
        convenience method (defined in select_mcts_action wrapper).
        """
        if hasattr(agent, "latent_value_head"):
            with torch.no_grad():
                val = agent.latent_value_head(z)
            return float(val.item())
        # Fallback: return 0 (pure rollout-based value)
        return 0.0

    # ── Backup ──

    def _backup(
        self,
        node: MCTSNode,
        value: float,
        parent_action: Optional[int],
    ):
        """Single-step backup: update parent's stats for parent_action."""
        if node.parent is not None and parent_action is not None:
            node.parent.N[parent_action] += 1
            node.parent.W[parent_action] += value

    def _backup_path(
        self,
        search_path: List[Tuple[MCTSNode, int]],
        value: float,
    ):
        """
        Propagate value backward through the selection path.

        Value is discounted at each step by γ during backup.
        """
        gamma = self.cfg.discount
        discounted_value = value

        for node, action in reversed(search_path):
            if node.N is not None:
                node.N[action] += 1
                node.W[action] += discounted_value
            discounted_value *= gamma

    # ── Policy Prior Helpers ──

    def _get_policy_prior(self, agent, obs_ctx: dict) -> np.ndarray:
        """Get policy prior π(a|h_t) from TacticalAgent for root node."""
        with torch.no_grad():
            output = agent.forward(
                obs_ctx["local_grids"],
                obs_ctx["scalars"],
                obs_ctx["prev_actions"],
                obs_ctx["prev_rewards"],
            )
        logits = output.policy_logits.squeeze(0)     # (num_actions,)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def _get_value(self, agent, obs_ctx: dict) -> float:
        """Get value estimate V(s) from TacticalAgent for root node."""
        with torch.no_grad():
            output = agent.forward(
                obs_ctx["local_grids"],
                obs_ctx["scalars"],
                obs_ctx["prev_actions"],
                obs_ctx["prev_rewards"],
            )
        return float(output.value.item())

    def _prior_from_latent(self, agent, z: torch.Tensor) -> np.ndarray:
        """
        Approximate policy prior from latent z via agent's latent head.

        Falls back to uniform prior if agent doesn't expose latent_policy_head.
        """
        if hasattr(agent, "latent_policy_head"):
            with torch.no_grad():
                logits = agent.latent_policy_head(z).squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        # Uniform fallback
        return np.ones(self.num_actions) / self.num_actions

    # ── Action Probability ──

    def _action_probs(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """
        Convert visit counts to action probabilities via temperature scaling.

        π_MCTS(a) ∝ N(root, a)^{1/T}

        T → 0: deterministic argmax
        T = 1: proportional to visits
        T → ∞: uniform
        """
        N = root.N.astype(np.float64)
        if temperature < 1e-3:
            # Greedy: put all probability on most-visited
            probs = np.zeros_like(N)
            probs[np.argmax(N)] = 1.0
        else:
            N_temp = N ** (1.0 / temperature)
            total = N_temp.sum()
            if total < 1e-10:
                probs = np.ones(self.num_actions) / self.num_actions
            else:
                probs = N_temp / total
        return probs.astype(np.float32)


# ─── Top-Level Convenience Function ──────────────────────────────────────────

def select_mcts_action(
    agent,
    world_model,
    z: torch.Tensor,      # (1, L) current latent
    obs_ctx: dict,
    config: Config,
    temperature: Optional[float] = None,
) -> Tuple[int, np.ndarray]:
    """
    Top-level MCTS action selection.

    Combines MCTS planning in latent space with tactical agent priors.

    Args:
        agent:       TacticalAgent
        world_model: WorldModel
        z:           Current encoded latent state (1, latent_dim)
        obs_ctx:     Observation context window for agent policy
        config:      Master config
        temperature: Override temperature (None = use config)

    Returns:
        action:       Selected action integer
        action_probs: (num_actions,) MCTS visit distribution
    """
    mcts = MCTS(config)
    if temperature is not None:
        mcts.cfg = MCTSConfig(  # Shallow override
            **{**vars(config.mcts), "temperature": temperature}
        )
    action, probs = mcts.search(z, agent, world_model, obs_ctx)
    return action, np.array(probs, dtype=np.float32)
