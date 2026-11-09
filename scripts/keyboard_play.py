"""Drive the trained Go2 locomotion policy with the keyboard.

The trained policy outputs joint targets as usual. The keyboard overrides the
velocity command (the same command the policy was trained to track), so you
"steer" the dog by changing what it is asked to do.

Controls (Se2Keyboard):
    Up / Down arrows    (or Numpad 8 / 2) : forward / backward   (v_x)
    Left / Right arrows (or Numpad 4 / 6) : strafe left / right  (v_y)
    Z / X               (or Numpad 7 / 9) : turn left / right    (yaw rate)
    L                                     : zero the command

Usage:
    ipy scripts/keyboard_play.py --task <your-task-id> \
        --checkpoint logs/rsl_rl/<experiment>/<run>/model_<iter>.pt

"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Keyboard-controlled Go2 locomotion policy.")
parser.add_argument("--task", type=str, required=True, help="Registered task ID.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model_*.pt (default: latest run).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots (keyboard drives all of them).")
parser.add_argument("--command_name", type=str, default="base_velocity", help="Velocity command term name in env cfg.")
parser.add_argument("--no_follow", action="store_true", help="Disable the camera that follows the robot.")
parser.add_argument("--real_time", action="store_true", help="Throttle the loop to real-time speed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports that need the Omniverse runtime ───────────────────────────────
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import go2_locomotion_rl.tasks  # noqa: F401  (registers task IDs)


def make_env_cfg():
    """Load the task's env cfg and patch it for interactive driving."""
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    # Long episodes so the dog doesn't time out while you drive (falls still reset).
    env_cfg.episode_length_s = 1e6

    # Keyboard owns the command: no resampling, no heading mode, no standing envs.
    cmd_cfg = getattr(env_cfg.commands, args.command_name)
    cmd_cfg.resampling_time_range = (1e9, 1e9)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.debug_vis = True  # green arrow = commanded velocity, blue = actual

    # Clean demo: no observation noise, no random pushes.
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # Camera follows robot 0.
    if not args.no_follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-2.5, -2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    return env_cfg, cmd_cfg


def main():
    env_cfg, cmd_cfg = make_env_cfg()
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Newer rsl_rl expects cfg["actor"]/["critic"] with class_name; Isaac Lab's
    # train.py/play.py convert the old `policy=` format the same way.
    try:
        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except ImportError:
        print(
            "[WARN] handle_deprecated_rsl_rl_cfg not found. Copy the agent_cfg conversion "
            "lines from scripts/rsl_rl/play.py into this script."
        )

    # ── Checkpoint ──
    if args.checkpoint:
        resume_path = os.path.abspath(args.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # ── Env + policy ──
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    device = unwrapped.device
    cmd_term = unwrapped.command_manager.get_term(args.command_name)
    robot = unwrapped.scene["robot"]

    # Clamp keyboard commands to the ranges the policy was trained on.
    r = cmd_cfg.ranges
    lo = torch.tensor([r.lin_vel_x[0], r.lin_vel_y[0], r.ang_vel_z[0]], device=device)
    hi = torch.tensor([r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]], device=device)

    # Holding a key gives the max trained speed on that axis.
    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=float(hi[0]),
            v_y_sensitivity=float(hi[1]),
            omega_z_sensitivity=float(hi[2]),
        )
    )
    keyboard.reset()
    print(keyboard)
    print(f"[INFO] Command limits  vx {r.lin_vel_x}  vy {r.lin_vel_y}  wz {r.ang_vel_z}")

    obs = env.get_observations()
    if isinstance(obs, tuple):  # older rsl_rl wrapper returns (obs, extras)
        obs = obs[0]

    dt = unwrapped.step_dt
    step = 0

    # ── Loop ──
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            cmd = torch.as_tensor(keyboard.advance(), dtype=torch.float32, device=device)
            cmd = torch.clamp(cmd, lo, hi)
            cmd_term.vel_command_b[:] = cmd  # overwrite the command for all envs

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Print commanded vs actual velocity ~2x per second (good for the demo/report).
            if step % max(1, int(0.5 / dt)) == 0:
                v = robot.data.root_lin_vel_b[0]
                wz = robot.data.root_ang_vel_b[0, 2]
                print(
                    f"cmd    vx={cmd[0].item():+.2f}  vy={cmd[1].item():+.2f}  wz={cmd[2].item():+.2f}   |   "
                    f"actual vx={v[0].item():+.2f}  vy={v[1].item():+.2f}  wz={wz.item():+.2f}"
                )
            step += 1

        if args.real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()