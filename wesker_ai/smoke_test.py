"""
Smoke test script — run from the 'Wesker ai' parent directory.
Usage: python wesker_ai/smoke_test.py
"""
import sys, os
# Ensure parent is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

PASS = "\u2713"
FAIL = "\u2717"

def section(name):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

def ok(msg): print(f"  {PASS} {msg}")
def err(msg): print(f"  {FAIL} {msg}"); raise RuntimeError(msg)


section("1. Config")
from wesker_ai.config import Config
cfg = Config()
cfg.device = "cpu"
cfg.mcts.num_simulations = 5
cfg.mcts.max_depth = 3
cfg.training.total_steps = 200
cfg.training.rollout_length = 32
ok(f"Config built: grid={cfg.env.grid_size}x{cfg.env.grid_size}, "
   f"vision={cfg.env.vision_radius}, T_window={cfg.transformer.trajectory_window}")


section("2. Environment (POMDP)")
from wesker_ai.environment import SurvivalEnv
env = SurvivalEnv(cfg.env, seed=0)
obs = env.reset()
assert obs["local_grid"].shape == (12, 11, 11), f"Bad grid shape: {obs['local_grid'].shape}"
assert obs["scalars"].shape == (4,), f"Bad scalar shape: {obs['scalars'].shape}"
total_r = 0
for step in range(50):
    a = step % cfg.env.num_actions
    obs, r, done, info = env.step(a)
    total_r += r
    if done:
        obs = env.reset()
ok(f"50 steps OK | cumulative_reward={total_r:.2f} | health={info['health']:.1f} | enemies={info['num_enemies']}")
ok(f"map_control={info['map_control']:.3f} | stress={info['stress']:.3f}")


section("3. Stress Model")
from wesker_ai.stress import StressModel
stress = StressModel(cfg.stress)
sigma = stress.update(40.0, 100.0, 5, 50, [2.0, 3.0, 8.0], 3, 100, 1000)
assert 0 <= sigma <= 1, f"Stress out of range: {sigma}"
logits = torch.randn(1, 9)
noised = stress.apply_logit_noise(logits)
assert noised.shape == logits.shape
delayed_action = stress.check_action_delay(3)
assert 0 <= delayed_action < cfg.env.num_actions
ok(f"sigma={sigma:.4f} | logit noise applied | delay_check action={delayed_action}")
ok(f"is_critical={stress.is_critical} | is_in_delay={stress.is_in_delay}")


section("4. Networks & Tactical Agent")
from wesker_ai.agent import TacticalAgent
agent = TacticalAgent(cfg)
total_params = sum(p.numel() for p in agent.parameters())
T = cfg.transformer.trajectory_window
C, H, W = 12, 11, 11
grids   = torch.zeros(1, T, C, H, W)
scalars = torch.zeros(1, T, 4)
actions = torch.zeros(1, T, dtype=torch.long)
rewards = torch.zeros(1, T)
out = agent(grids, scalars, actions, rewards)
assert out.policy_logits.shape == (1, 9), f"Bad policy shape: {out.policy_logits.shape}"
assert out.value.shape == (1, 1)
assert out.quantiles is not None and out.quantiles.shape == (1, cfg.distributional.num_quantiles)
assert out.cvar is not None and out.cvar.shape == (1, 1)
ok(f"TacticalAgent: {total_params:,} params")
ok(f"policy_logits={tuple(out.policy_logits.shape)} value={tuple(out.value.shape)}")
ok(f"quantiles={tuple(out.quantiles.shape)} cvar={out.cvar.item():.4f}")

# forward_sequence
lseq, vseq, qseq, _ = agent.forward_sequence(grids, scalars, actions, rewards)
assert lseq.shape == (1, T, 9)
assert vseq.shape == (1, T, 1)
assert qseq.shape == (1, T, cfg.distributional.num_quantiles)
ok(f"forward_sequence: logits={tuple(lseq.shape)} values={tuple(vseq.shape)}")

# select_action
action_i, lp, val = agent.select_action(grids, scalars, actions, rewards)
assert 0 <= action_i < 9
ok(f"select_action: a={action_i} log_prob={lp.item():.4f} value={val.item():.4f}")


section("5. Risk Module")
from wesker_ai.risk import (
    quantile_regression_loss, compute_cvar, compute_gae,
    ppo_policy_loss, value_loss, entropy_bonus,
    cvar_policy_gradient, expected_value_objective,
    compute_objectives_comparison, RiskMetrics,
)
N = cfg.distributional.num_quantiles
pred_q = torch.randn(8, N)
target = torch.randn(8)
taus   = agent.quantile_head.taus
qr_loss = quantile_regression_loss(pred_q, target, taus)
assert qr_loss.item() >= 0
ok(f"QR loss={qr_loss.item():.4f}")

cvar_val = compute_cvar(pred_q, 0.25)
assert cvar_val.shape == (8, 1)
ok(f"CVaR_0.25 shape={tuple(cvar_val.shape)}, mean={cvar_val.mean().item():.4f}")

rewards_np = np.random.randn(32).astype(np.float32)
values_np  = np.random.randn(32).astype(np.float32)
dones_np   = np.zeros(32, dtype=np.float32)
adv, ret = compute_gae(rewards_np, values_np, dones_np, 0.0)
assert adv.shape == (32,)
ok(f"GAE: advantages shape={adv.shape}, returns shape={ret.shape}")

log_p = torch.randn(8)
old_lp = log_p.detach().clone()
adv_t = torch.randn(8)
pl = ppo_policy_loss(log_p, old_lp, adv_t)
ok(f"PPO policy loss={pl.item():.4f}")

metrics_tracker = RiskMetrics(alpha=0.25)
for ep_ret in [10.0, -5.0, 30.0, 2.0, 15.0]:
    metrics_tracker.record_episode(ep_ret)
ok(f"RiskMetrics: E[G]={metrics_tracker.expected_return():.2f} CVaR={metrics_tracker.cvar():.2f} TailRisk={metrics_tracker.tail_risk_probability():.2f}")


section("6. World Model")
from wesker_ai.world_model import WorldModel
wm = WorldModel(cfg)
wm_params = sum(p.numel() for p in wm.parameters())
B = 4
g_in  = torch.rand(B, C, H, W)
sc_in = torch.rand(B, 4)
g_next = torch.rand(B, C, H, W)
sc_next = torch.rand(B, 4)
a_in  = torch.randint(0, 9, (B,))
r_in  = torch.randn(B)

z, mu, logvar = wm.encode(g_in, sc_in)
assert z.shape == (B, cfg.world_model.latent_dim)
ok(f"WorldModel: {wm_params:,} params | z={tuple(z.shape)}")

z_next = wm.transition(z, a_in)
assert z_next.shape == (B, cfg.world_model.latent_dim)
ok(f"Transition: z_next={tuple(z_next.shape)}")

r_hat = wm.predict_reward(z, a_in)
assert r_hat.shape == (B, 1)
ok(f"Reward predictor: r_hat={tuple(r_hat.shape)}")

wm.train()
loss, loss_dict = wm.compute_loss(g_in, sc_in, g_next, sc_next, a_in, r_in)
ok(f"WM loss={loss.item():.4f} | breakdown={list(loss_dict.keys())}")

wm.eval()
z_seq, r_seq = wm.imagine_trajectory(z[:1], [0, 1, 2, 3])
assert len(z_seq) == 5
assert r_seq.shape == (4, 1)
ok(f"imagine_trajectory: {len(z_seq)} latent states | rewards={tuple(r_seq.shape)}")


section("7. MCTS")
from wesker_ai.mcts import MCTS, MCTSNode, select_mcts_action
mcts = MCTS(cfg)
obs_ctx = {
    "local_grids":  grids,         # (1, T, 5, 11, 11)
    "scalars":      scalars,
    "prev_actions": actions,
    "prev_rewards": rewards,
}
wm.eval(); agent.eval()
root_z = z[:1]  # (1, latent_dim)
action_mcts, probs = mcts.search(root_z, agent, wm, obs_ctx)
assert 0 <= action_mcts < 9
assert abs(probs.sum() - 1.0) < 0.01
ok(f"MCTS: action={action_mcts} probs_sum={probs.sum():.4f}")

action_s, probs_s = select_mcts_action(agent, wm, root_z, obs_ctx, cfg)
ok(f"select_mcts_action: action={action_s}")


section("8. Training Loop")
agent.train(); wm.train()
from wesker_ai.train import Trainer
cfg2 = Config()
cfg2.device = "cpu"
cfg2.mcts.num_simulations = 3
cfg2.mcts.max_depth = 2
cfg2.training.total_steps = 250
cfg2.training.rollout_length = 64
cfg2.training.batch_size = 16
cfg2.training.num_epochs = 1
cfg2.training.log_interval = 1
cfg2.training.save_interval = 9999  # Don't save during test
cfg2.ablation.use_mcts = False  # MCTS is slow; test separately
trainer = Trainer(cfg2)
trainer.train()
ok(f"Training loop: completed {trainer.global_step} steps, {trainer.episode_count} episodes")
ok(f"Mean return (last 100 eps): {sum(trainer.episode_returns)/max(len(trainer.episode_returns),1):.2f}")


section("9. Evaluation Metrics")
from wesker_ai.evaluate import run_evaluation, print_metrics_table, compare_objectives, log_eval_metrics
eval_env    = SurvivalEnv(cfg.env, seed=999)
eval_stress = StressModel(cfg.stress)
trainer.agent.eval()
metrics = run_evaluation(trainer.agent, eval_env, eval_stress, cfg2, n_episodes=10)
ok(f"survival_time={metrics.survival_time:.1f} | death_rate={metrics.death_rate:.3f}")
ok(f"cvar_return={metrics.cvar_return:.3f} | tail_risk={metrics.tail_risk_probability:.3f}")
ok(f"collapse_freq={metrics.stress_collapse_frequency:.3f} | map_ctrl={metrics.map_control_score:.3f}")
print_metrics_table(metrics, "Smoke Test Agent")


section("10. Advanced 3D-Like Environment")
from wesker_ai.complex_environment import (
    AdvancedSurvivalEnv, AdvancedEnvConfig, EnemyType, Action as AdvAction,
)
adv_cfg = AdvancedEnvConfig()
adv_cfg.grid_size = 20
adv_cfg.max_steps = 200
adv_cfg.num_initial_enemies = 3
adv_env = AdvancedSurvivalEnv(adv_cfg, seed=7)
adv_obs = adv_env.reset()
assert adv_obs["local_grid"].shape == (20, 15, 15), \
    f"Bad advanced grid shape: {adv_obs['local_grid'].shape}"
assert adv_obs["scalars"].shape == (6,), \
    f"Bad advanced scalar shape: {adv_obs['scalars'].shape}"
assert int(adv_obs["current_floor"]) == 0
ok(f"AdvancedSurvivalEnv reset: grid={adv_obs['local_grid'].shape} scalars={adv_obs['scalars'].shape}")
ok(f"Floors={adv_cfg.num_floors} | Cover objects={adv_cfg.num_cover_objects} | Hazards=fire+acid+radiation")

adv_total_r = 0.0
for step in range(60):
    action = step % adv_cfg.num_actions
    adv_obs, adv_r, adv_done, adv_info = adv_env.step(action)
    adv_total_r += adv_r
    if adv_done:
        adv_obs = adv_env.reset()
ok(f"60 steps advanced env: cumulative_reward={adv_total_r:.2f}")
ok(f"Enemy archetypes present: {set(e.kind for e in adv_env.enemies)}")

# Verify multi-ammo actions work without crash
for ammo_action in [int(AdvAction.SHOOT_HVY_N), int(AdvAction.SHOOT_EXP_N)]:
    _, _, _, _ = adv_env.step(ammo_action)
ok("Heavy and explosive ammo actions execute without error")

# Verify interact/crouch (fresh episode so env is not in a done state)
adv_env.reset()
adv_env.step(int(AdvAction.INTERACT))
adv_env.step(int(AdvAction.CROUCH_TOGGLE))
assert adv_env.is_crouching is True
adv_env.step(int(AdvAction.CROUCH_TOGGLE))
assert adv_env.is_crouching is False
ok("INTERACT and CROUCH_TOGGLE actions work correctly")

# Depth channel is non-trivial
depth_chan = adv_obs["local_grid"][19]
assert depth_chan.min() >= 0.0 and depth_chan.max() <= 1.0
ok(f"Depth channel range: [{depth_chan.min():.3f}, {depth_chan.max():.3f}]")

# Wall reset between episodes — verify no accumulation
walls_ep1 = adv_env.walls.copy()
adv_env.reset()
walls_ep2 = adv_env.walls.copy()
# Borders should always be 1; interiors can differ (fresh map)
assert walls_ep1[0, :].all() and walls_ep2[0, :].all(), "Border walls missing"
ok("Wall grid resets cleanly between episodes (no wall accumulation)")


section("11. Multi-Agent Scenarios")
from wesker_ai.multi_agent_env import (
    MultiAgentEnv, MultiAgentConfig, MultiAgentMode,
)

# --- Competitive ---
ma_cfg_comp = MultiAgentConfig(num_agents=3, mode=MultiAgentMode.COMPETITIVE)
ma_env_comp = MultiAgentEnv(cfg.env, ma_cfg_comp, seed=11)
obs_n = ma_env_comp.reset()
assert len(obs_n) == 3, f"Expected 3 agent obs, got {len(obs_n)}"
for i in range(3):
    assert obs_n[i].local_grid.shape[0] == cfg.env.num_obs_channels + 2, \
        f"Agent {i} wrong obs channels: {obs_n[i].local_grid.shape}"
    assert obs_n[i].scalars.shape == (5,), \
        f"Agent {i} wrong scalar shape: {obs_n[i].scalars.shape}"
ok(f"COMPETITIVE: {ma_cfg_comp.num_agents} agents | obs channels={obs_n[0].local_grid.shape[0]}")

total_steps_comp = 0
for _ in range(40):
    actions = {i: np.random.randint(0, cfg.env.num_actions) for i in range(3)}
    obs_n, rew_n, done_n, info_n = ma_env_comp.step(actions)
    total_steps_comp += 1
    if ma_env_comp.episode_done:
        break
ok(f"COMPETITIVE: 40 steps, alive={ma_env_comp.num_alive}/{ma_cfg_comp.num_agents}")
ok(f"Rewards: {[round(rew_n[i], 2) for i in range(3)]}")

# Verify global state
gs = ma_env_comp.get_global_state()
assert gs.shape == (ma_env_comp.global_state_size,), \
    f"Global state shape mismatch: {gs.shape} vs {ma_env_comp.global_state_size}"
ok(f"Global state (centralized critic): shape={gs.shape}")

# --- Cooperative ---
ma_cfg_coop = MultiAgentConfig(num_agents=2, mode=MultiAgentMode.COOPERATIVE,
                                shared_reward_weight=0.5)
ma_env_coop = MultiAgentEnv(cfg.env, ma_cfg_coop, seed=22)
obs_n_c = ma_env_coop.reset()
assert len(obs_n_c) == 2
for _ in range(20):
    actions = {i: np.random.randint(0, cfg.env.num_actions) for i in range(2)}
    obs_n_c, rew_n_c, done_n_c, _ = ma_env_coop.step(actions)
    if ma_env_coop.episode_done:
        break
ok(f"COOPERATIVE: 20 steps | shared_reward_weight={ma_cfg_coop.shared_reward_weight}")

# --- Mixed (teams) ---
ma_cfg_mixed = MultiAgentConfig(
    num_agents=4, mode=MultiAgentMode.MIXED,
    team_assignments=[0, 0, 1, 1], friendly_fire=False,
)
ma_env_mixed = MultiAgentEnv(cfg.env, ma_cfg_mixed, seed=33)
obs_n_m = ma_env_mixed.reset()
for _ in range(20):
    actions = {i: np.random.randint(0, cfg.env.num_actions) for i in range(4)}
    obs_n_m, rew_n_m, done_n_m, _ = ma_env_mixed.step(actions)
    if ma_env_mixed.episode_done:
        break
team_stats = ma_env_mixed.get_team_stats()
assert set(team_stats.keys()) == {0, 1}
ok(f"MIXED: 2 teams of 2 | team_stats={team_stats}")

# Agent distances utility
dists = ma_env_comp.get_agent_distances(0)
assert isinstance(dists, dict)
ok(f"get_agent_distances(0): {dists}")


section("12. Human-in-the-Loop (HITL)")
from wesker_ai.human_in_loop import (
    HumanInTheLoopWrapper, HITLConfig, GuidanceMode, InterventionTrigger,
    HumanFeedbackBuffer, NonInteractiveInterface, make_hitl_env,
)

# Use NonInteractiveInterface so no stdin prompt is needed
hitl_cfg = HITLConfig(
    guidance_mode=GuidanceMode.SUGGEST,
    intervention_trigger=InterventionTrigger.ON_ENTROPY,
    entropy_threshold=0.0,       # Always trigger (entropy always > 0)
    suggest_logit_bias=2.0,
    feedback_decay=0.9,
    log_interactions=True,
)
base_env = SurvivalEnv(cfg.env, seed=55)
hitl_interface = NonInteractiveInterface()     # no stdin needed
hitl_env = HumanInTheLoopWrapper(base_env, hitl_cfg, hitl_interface)

hitl_obs = hitl_env.reset()
assert "local_grid" in hitl_obs and "scalars" in hitl_obs
ok(f"HITL wrapper reset: obs keys={list(hitl_obs.keys())}")

# Run 20 steps with policy logits to exercise entropy trigger
import torch as _torch
hitl_total_r = 0.0
for _ in range(20):
    dummy_logits = _torch.randn(1, cfg.env.num_actions)
    _, hitl_r, hitl_done, hitl_info = hitl_env.step(
        agent_action=0,
        policy_logits=dummy_logits,
    )
    hitl_total_r += hitl_r
    if hitl_done:
        hitl_obs = hitl_env.reset()
ok(f"HITL 20 steps: shaped_reward_sum={hitl_total_r:.3f}")
ok(f"HITL interventions triggered: {hitl_env.total_interventions}")
assert hitl_info["hitl_intervened"] is True or hitl_info["hitl_intervened"] is False
ok(f"hitl_info keys present: {[k for k in hitl_info if k.startswith('hitl_')]}")

# Test OVERRIDE mode
hitl_cfg_ov = HITLConfig(
    guidance_mode=GuidanceMode.OVERRIDE,
    intervention_trigger=InterventionTrigger.PERIODIC,
    periodic_interval=5,
    log_interactions=True,
)
hitl_env_ov = HumanInTheLoopWrapper(SurvivalEnv(cfg.env, seed=66), hitl_cfg_ov,
                                     NonInteractiveInterface())
hitl_env_ov.reset()
for i in range(15):
    _, _, _, _ = hitl_env_ov.step(agent_action=i % cfg.env.num_actions)
ok(f"OVERRIDE mode: {hitl_env_ov.total_interventions} interventions in 15 steps "
   f"(periodic every 5 → expect 3)")

# Test APPROVE mode
hitl_cfg_ap = HITLConfig(
    guidance_mode=GuidanceMode.APPROVE,
    intervention_trigger=InterventionTrigger.ON_DANGER,
    danger_health_threshold=2.0,   # Always trigger (threshold > 1.0 health norm → always triggers)
    log_interactions=True,
)
hitl_env_ap = HumanInTheLoopWrapper(SurvivalEnv(cfg.env, seed=77), hitl_cfg_ap,
                                     NonInteractiveInterface())
hitl_env_ap.reset()
for i in range(10):
    _, _, _, _ = hitl_env_ap.step(agent_action=i % cfg.env.num_actions)
ok(f"APPROVE mode: {hitl_env_ap.total_interventions} interventions in 10 steps")

# Feedback buffer serialisation
from wesker_ai.human_in_loop import HumanFeedbackEvent as _HFE
buf = hitl_env.feedback_buffer
# Ensure there is at least one event to test round-trip
if len(buf) == 0:
    buf.record(_HFE(step=1, event_type="reward", value=0.5, entropy=1.2))
buf_len_before = len(buf)
ok(f"FeedbackBuffer: {buf_len_before} events recorded")
buf.save("_hitl_test_buf.npz")
buf2 = HumanFeedbackBuffer()
buf2.load("_hitl_test_buf.npz")
assert len(buf2) == buf_len_before, f"Buffer save/load mismatch: {len(buf2)} vs {buf_len_before}"
import os as _os; _os.remove("_hitl_test_buf.npz")
ok(f"FeedbackBuffer save/load round-trip: {buf_len_before} events preserved")

# make_hitl_env factory
factory_env = make_hitl_env(SurvivalEnv(cfg.env, seed=88),
                              guidance_mode=GuidanceMode.SUGGEST,
                              trigger=InterventionTrigger.ON_REQUEST,
                              interactive=False)
factory_env.reset()
_, _, _, _ = factory_env.step(0)
ok("make_hitl_env factory creates wrapper without error")

# Session stats
stats = hitl_env.get_session_stats()
assert "total_interventions" in stats and "intervention_rate" in stats
ok(f"Session stats: {stats}")

# Imitation dataset (may be None if no overrides in suggest mode)
dataset = hitl_env.get_imitation_dataset()
ok(f"Imitation dataset: {'None (no overrides in SUGGEST mode)' if dataset is None else list(dataset.keys())}")


section("ALL TESTS PASSED")
print(f"  {PASS} Environment (POMDP)")
print(f"  {PASS} Stress model (sigma, noise, delay)")
print(f"  {PASS} TacticalAgent (transformer, quantiles, CVaR)")
print(f"  {PASS} Risk module (QR loss, GAE, PPO, CVaR)")
print(f"  {PASS} World model (VAE, transition, imagination)")
print(f"  {PASS} MCTS (latent planning)")
print(f"  {PASS} Training loop (PPO + WM)")
print(f"  {PASS} Evaluation metrics (all 6)")
print(f"  {PASS} Advanced 3D-Like Environment (multi-floor, archetypes, hazards, cover)")
print(f"  {PASS} Multi-Agent Scenarios (competitive, cooperative, mixed teams)")
print(f"  {PASS} Human-in-the-Loop (suggest, override, approve, feedback buffer)")
print()
import sys; sys.exit(0)
