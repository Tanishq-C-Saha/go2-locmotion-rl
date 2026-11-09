# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ablation_plot.py — PPO vs PPO+PSO ablation study visualiser.

BEGINNER EXPLANATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An "ablation study" means: take your full system (PPO+PSO),
remove one part (PSO), and compare. This proves whether PSO
actually helped or was just dead weight.

This script:
  1. Loads your saved PPO-only checkpoint.
  2. Loads your saved PPO+PSO checkpoint.
  3. Runs both for one full episode in the Isaac Lab environment.
  4. Records 4 metrics every step: reward, jerk, velocity error, alive.
  5. Plots a 4-panel comparison figure and saves it as a PNG.

HOW TO USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After both training runs are complete:

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/ablation_plot.py \\
      --ppo_checkpoint  logs/rsl_rl/a1_flat/model_2999.pt \\
      --pso_checkpoint  logs/rsl_rl/a1_flat_ppo_pso/model_2999.pt \\
      --task            Isaac-Velocity-Flat-Unitree-A1-v0 \\
      --num_envs        1 \\
      --episode_steps   1000 \\
      --output          ablation_study.png

  Then open ablation_study.png.

METRICS EXPLAINED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cumulative Reward     — total score accumulated over the episode.
                          Higher = better overall performance.

  Joint Jerk            — rate of change of joint acceleration.
                          Lower = smoother, more natural gait.
                          Computed as: ||Δacceleration|| per timestep.

  Velocity Tracking     — |commanded_speed − actual_speed| at each step.
  Error                   Lower = robot follows your speed commands better.

  Survival (alive)      — 1 if robot is upright, 0 if it fell.
                          Shown as a running fraction over the episode.
"""


from __future__ import annotations
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# ARG PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="PPO vs PPO+PSO ablation study plot.")
    parser.add_argument("--ppo_checkpoint",  type=str, required=True,
                        help="Path to saved PPO-only model (.pt file)")
    parser.add_argument("--pso_checkpoint",  type=str, required=True,
                        help="Path to saved PPO+PSO model (.pt file)")
    parser.add_argument("--task",            type=str,
                        default="Isaac-Velocity-Flat-Unitree-A1-v0",
                        help="Isaac Lab task ID to run both agents in")
    parser.add_argument("--num_envs",        type=int, default=1,
                        help="Number of parallel envs (1 for clean ablation)")
    parser.add_argument("--episode_steps",   type=int, default=1000,
                        help="How many steps to record per agent")
    parser.add_argument("--output",          type=str, default="ablation_study.png",
                        help="Output image filename")
    parser.add_argument("--device",          type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--sim_dt",          type=float, default=0.005,
                        help="Simulation timestep (seconds)")
    parser.add_argument("--decimation",      type=int, default=4,
                        help="Policy decimation (physics steps per policy step)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ROLLOUT RECORDER
# ══════════════════════════════════════════════════════════════════════════════

def record_episode(env, policy, num_steps: int, device: str, dt: float) -> dict:
    """
    Run one episode and record all metrics.

    Returns a dict of numpy arrays, one value per timestep:
        rewards         : (num_steps,)  raw per-step reward
        cumulative_rew  : (num_steps,)  running sum of rewards
        jerk            : (num_steps,)  mean joint jerk magnitude
        vel_error       : (num_steps,)  |cmd_vel_x - base_vel_x| + |cmd_vel_y - base_vel_y|
        alive           : (num_steps,)  1 = robot upright, 0 = terminated
    """
    obs, extras = env.get_observations()
    obs = obs.to(device)

    # Buffers for jerk computation (need two prev accelerations)
    prev_joint_vel  = torch.zeros(env.num_envs, 12, device=device)
    prev_joint_acc  = torch.zeros(env.num_envs, 12, device=device)
    policy_dt       = dt * 4  # policy runs at decimation=4

    records = defaultdict(list)
    cumulative = 0.0
    alive_count = 0

    policy.eval()
    with torch.inference_mode():
        for step in range(num_steps):
            # ── Policy action ─────────────────────────────────────────────────
            actions = policy(obs)

            # ── Environment step ──────────────────────────────────────────────
            obs_new, rewards, dones, infos = env.step(actions)
            obs_new = obs_new.to(device)

            reward_val = float(rewards[0].item())
            cumulative += reward_val
            records["rewards"].append(reward_val)
            records["cumulative_rew"].append(cumulative)

            # ── Jerk computation ──────────────────────────────────────────────
            # obs layout (standard Isaac Lab A1):
            #   [0:3]   base lin vel
            #   [3:6]   base ang vel
            #   [6:9]   projected gravity
            #   [9:12]  velocity command
            #   [12:24] joint positions  (12 joints)
            #   [24:36] joint velocities (12 joints)
            #   [36:48] last action      (12 dims)
            joint_vel = obs_new[0, 24:36]                          # (12,)
            joint_acc = (joint_vel - prev_joint_vel[0]) / policy_dt
            jerk      = (joint_acc - prev_joint_acc[0]) / policy_dt
            jerk_mag  = float(jerk.norm().item())
            records["jerk"].append(jerk_mag)
            prev_joint_vel[0] = joint_vel
            prev_joint_acc[0] = joint_acc

            # ── Velocity tracking error ───────────────────────────────────────
            # obs[9:11] = commanded lin vel xy; obs[0:2] = actual base lin vel xy
            cmd_vel    = obs_new[0, 9:11]
            actual_vel = obs_new[0, 0:2]
            vel_err    = float((cmd_vel - actual_vel).abs().sum().item())
            records["vel_error"].append(vel_err)

            # ── Alive ─────────────────────────────────────────────────────────
            alive = 1 if float(dones[0].item()) == 0 else 0
            alive_count += alive
            records["alive"].append(alive_count / (step + 1))  # running fraction

            obs = obs_new

            # Reset on termination (single env) so episode continues
            if dones[0]:
                obs, _ = env.reset()
                obs = obs.to(device)
                prev_joint_vel.zero_()
                prev_joint_acc.zero_()

    return {k: np.array(v) for k, v in records.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_ablation_plot(ppo_data: dict, pso_data: dict,
                        num_steps: int, dt: float, output_path: str):
    """
    4-panel ablation study figure.
    Panel 1 — Cumulative Reward
    Panel 2 — Joint Jerk (smoothed)
    Panel 3 — Velocity Tracking Error (smoothed)
    Panel 4 — Survival Rate (running fraction alive)
    """

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#e6edf3",
        "axes.titlecolor":   "#e6edf3",
        "axes.grid":         True,
        "grid.color":        "#21262d",
        "grid.linewidth":    0.8,
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#e6edf3",
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "font.family":       "monospace",
        "font.size":         10,
    })

    PPO_COLOR = "#58a6ff"   # blue  — PPO only
    PSO_COLOR = "#3fb950"   # green — PPO + PSO
    ALPHA_FILL = 0.12
    LW = 2.0

    steps = np.arange(num_steps)
    time_axis = steps * dt * 4  # convert to seconds (decimation=4)

    def smooth(arr, w=30):
        """Simple moving average smoothing for noisy signals."""
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#0d1117")

    # Super-title
    fig.suptitle(
        "Ablation Study — PPO  vs  PPO + PSO\nUnitree A1 · Flat Terrain · Velocity Tracking",
        fontsize=14,
        fontweight="bold",
        color="#e6edf3",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.08, right=0.97, top=0.91, bottom=0.08)

    # ── Panel 1: Cumulative Reward ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis, ppo_data["cumulative_rew"], color=PPO_COLOR, lw=LW, label="PPO only")
    ax1.plot(time_axis, pso_data["cumulative_rew"], color=PSO_COLOR, lw=LW, label="PPO + PSO")
    ax1.fill_between(time_axis, ppo_data["cumulative_rew"], pso_data["cumulative_rew"],
                     where=pso_data["cumulative_rew"] >= ppo_data["cumulative_rew"],
                     alpha=ALPHA_FILL, color=PSO_COLOR, label="PSO advantage")
    ax1.set_title("① Cumulative Reward  ↑ higher is better", pad=8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total reward")
    ax1.legend(loc="upper left", fontsize=8)
    _annotate_final(ax1, time_axis[-1],
                    ppo_data["cumulative_rew"][-1],
                    pso_data["cumulative_rew"][-1],
                    PPO_COLOR, PSO_COLOR)

    # ── Panel 2: Joint Jerk ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ppo_jerk_s = smooth(ppo_data["jerk"])
    pso_jerk_s = smooth(pso_data["jerk"])
    ax2.plot(time_axis, ppo_jerk_s, color=PPO_COLOR, lw=LW, label="PPO only")
    ax2.plot(time_axis, pso_jerk_s, color=PSO_COLOR, lw=LW, label="PPO + PSO")
    ax2.fill_between(time_axis, ppo_jerk_s, pso_jerk_s,
                     where=pso_jerk_s <= ppo_jerk_s,
                     alpha=ALPHA_FILL, color=PSO_COLOR, label="PSO smoother")
    ax2.set_title("② Joint Jerk  ↓ lower is smoother gait", pad=8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Jerk magnitude (rad/s³)")
    ax2.legend(loc="upper right", fontsize=8)
    _annotate_final(ax2, time_axis[-1],
                    ppo_jerk_s[-1], pso_jerk_s[-1],
                    PPO_COLOR, PSO_COLOR, lower_better=True)

    # ── Panel 3: Velocity Tracking Error ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ppo_vel_s = smooth(ppo_data["vel_error"])
    pso_vel_s = smooth(pso_data["vel_error"])
    ax3.plot(time_axis, ppo_vel_s, color=PPO_COLOR, lw=LW, label="PPO only")
    ax3.plot(time_axis, pso_vel_s, color=PSO_COLOR, lw=LW, label="PPO + PSO")
    ax3.fill_between(time_axis, ppo_vel_s, pso_vel_s,
                     where=pso_vel_s <= ppo_vel_s,
                     alpha=ALPHA_FILL, color=PSO_COLOR, label="PSO more accurate")
    ax3.set_title("③ Velocity Tracking Error  ↓ lower is more accurate", pad=8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("|cmd − actual| (m/s)")
    ax3.legend(loc="upper right", fontsize=8)
    _annotate_final(ax3, time_axis[-1],
                    ppo_vel_s[-1], pso_vel_s[-1],
                    PPO_COLOR, PSO_COLOR, lower_better=True)

    # ── Panel 4: Survival Rate ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_axis, ppo_data["alive"], color=PPO_COLOR, lw=LW, label="PPO only")
    ax4.plot(time_axis, pso_data["alive"], color=PSO_COLOR, lw=LW, label="PPO + PSO")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("④ Survival Rate  ↑ higher = fell less often", pad=8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Fraction of time upright")
    ax4.legend(loc="lower left", fontsize=8)
    _annotate_final(ax4, time_axis[-1],
                    ppo_data["alive"][-1], pso_data["alive"][-1],
                    PPO_COLOR, PSO_COLOR)

    # ── Summary bar at the bottom ──────────────────────────────────────────────
    _add_summary_bar(fig, ppo_data, pso_data, PPO_COLOR, PSO_COLOR)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✅ Ablation plot saved → {os.path.abspath(output_path)}\n")
    plt.close(fig)


def _annotate_final(ax, x_end, ppo_val, pso_val, ppo_color, pso_color,
                    lower_better=False):
    """Annotate the final value of each line at the right edge."""
    winner = pso_val < ppo_val if lower_better else pso_val > ppo_val
    star = " ★" if winner else ""
    ax.annotate(f"{ppo_val:.2f}", xy=(x_end, ppo_val),
                xytext=(4, 0), textcoords="offset points",
                color=ppo_color, fontsize=7, va="center")
    ax.annotate(f"{pso_val:.2f}{star}", xy=(x_end, pso_val),
                xytext=(4, 0), textcoords="offset points",
                color=pso_color, fontsize=7, va="center",
                fontweight="bold" if winner else "normal")


def _add_summary_bar(fig, ppo_data, pso_data, ppo_color, pso_color):
    """
    Text summary row at the very bottom showing who won each metric.
    Beginner note: ★ = PSO won that metric.
    """
    ppo_rew   = ppo_data["cumulative_rew"][-1]
    pso_rew   = pso_data["cumulative_rew"][-1]
    ppo_jerk  = float(np.mean(ppo_data["jerk"]))
    pso_jerk  = float(np.mean(pso_data["jerk"]))
    ppo_vel   = float(np.mean(ppo_data["vel_error"]))
    pso_vel   = float(np.mean(pso_data["vel_error"]))
    ppo_alive = ppo_data["alive"][-1]
    pso_alive = pso_data["alive"][-1]

    def win(pso_v, ppo_v, lower=False):
        return "★ PSO" if (pso_v < ppo_v if lower else pso_v > ppo_v) else "  PPO"

    lines = [
        f"  Cumul. Reward:  PPO={ppo_rew:8.2f}   PSO={pso_rew:8.2f}   winner → {win(pso_rew, ppo_rew)}",
        f"  Mean Jerk:      PPO={ppo_jerk:8.3f}   PSO={pso_jerk:8.3f}   winner → {win(pso_jerk, ppo_jerk, lower=True)}",
        f"  Mean Vel Err:   PPO={ppo_vel:8.3f}   PSO={pso_vel:8.3f}   winner → {win(pso_vel, ppo_vel, lower=True)}",
        f"  Survival Rate:  PPO={ppo_alive:8.3f}   PSO={pso_alive:8.3f}   winner → {win(pso_alive, ppo_alive)}",
    ]
    summary = "\n".join(lines)
    fig.text(0.5, 0.005, summary, ha="center", va="bottom",
             fontsize=8, color="#8b949e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                       edgecolor="#30363d", alpha=0.8))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Isaac Lab imports (deferred so argparse works without Isaac Lab) ───────
    import isaaclab  # noqa: F401 — triggers extension loading
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.envs import ManagerBasedRLEnv

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Build environment (single env, play mode) ─────────────────────────────
    print(f"\n⚙  Building environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, use_fabric=True)
    env_cfg.episode_length_s = (args.episode_steps * args.sim_dt * args.decimation)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim    = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim

    print(f"   obs_dim={obs_dim}  action_dim={action_dim}  device={device}")

    # ── Helper: load a checkpoint into a policy ───────────────────────────────
    def load_policy(checkpoint_path: str):
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner

        # Create runner (same as training)
        runner = RslRlOnPolicyRunner(
            env=env,
            device=device,
        )

        # Load checkpoint (this restores EVERYTHING correctly)
        runner.load(checkpoint_path)

        policy = runner.policy
        policy.eval()

        return policy

    # ── Record both agents ────────────────────────────────────────────────────
    print("\n▶  Recording PPO-only episode...")
    ppo_policy = load_policy(args.ppo_checkpoint)
    ppo_data   = record_episode(env, ppo_policy.act_inference,
                                args.episode_steps, device, args.sim_dt)

    print("\n▶  Recording PPO+PSO episode...")
    pso_policy = load_policy(args.pso_checkpoint)
    pso_data   = record_episode(env, pso_policy.act_inference,
                                args.episode_steps, device, args.sim_dt)

    env.close()

    # ── Print quick console summary ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ABLATION SUMMARY")
    print("═" * 60)
    metrics = [
        ("Cumulative Reward ↑", ppo_data["cumulative_rew"][-1],
                                pso_data["cumulative_rew"][-1], False),
        ("Mean Jerk         ↓", np.mean(ppo_data["jerk"]),
                                np.mean(pso_data["jerk"]),       True),
        ("Mean Vel Error    ↓", np.mean(ppo_data["vel_error"]),
                                np.mean(pso_data["vel_error"]),  True),
        ("Survival Rate     ↑", ppo_data["alive"][-1],
                                pso_data["alive"][-1],           False),
    ]
    for name, ppo_v, pso_v, lower in metrics:
        pso_wins = pso_v < ppo_v if lower else pso_v > ppo_v
        star = "★" if pso_wins else " "
        pct  = (pso_v - ppo_v) / (abs(ppo_v) + 1e-8) * 100
        sign = "+" if pct > 0 else ""
        print(f"  {name}:  PPO={ppo_v:8.3f}  PSO={pso_v:8.3f}  "
              f"Δ={sign}{pct:.1f}%  {star}")
    print("═" * 60)
    print(f"\n  ★ = PSO won that metric\n")

    # ── Build and save plot ───────────────────────────────────────────────────
    build_ablation_plot(ppo_data, pso_data,
                        args.episode_steps, args.sim_dt, args.output)
    simulation_app.close()


if __name__ == "__main__":
    main()