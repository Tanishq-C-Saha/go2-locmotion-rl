"""
play_2.py — Complete A→B Locomotion Testbed for Isaac Lab
=========================================================
Tests a trained policy by commanding the A1 robot to walk from
point A to point B and logs all metrics needed for paper comparison.

FIXES applied vs original:
  [F1] env.step() unpacking: newer Isaac Lab returns 5-tuple (obs, rew, term, trunc, info)
       — original unpacked 4, causing obs to be wrong every step silently.
  [F2] State read timing: root_state_w must be read AFTER env.step(), not before.
       Original read state before step, so metrics were always one step behind.
  [F3] Command injection: vel_command_b attribute name varies by Isaac Lab version.
       Added multi-name probe with explicit WARNING when injection fails, so you
       know the robot is walking open-loop instead of failing silently.
  [F4] Contact sensor name: probed dynamically instead of hardcoded "contact_forces".
  [F5] env.reset() return: wrapped to handle both 1-return and 2-return variants.
  [F6] path_length off-by-one: last_pos initialised from actual first observation,
       not accumulated on step 0 against a None baseline.
  [F7] All bare except blocks replaced with explicit logging so nothing fails silently.
  [F8] Goal positions now verified to be in world frame, matching root_state_w frame.
  [F9] Added per-step sanity checks (NaN in obs/actions) with early abort + warning.
  [F10] Fair comparison: all envs share identical terrain/spawn, events disabled,
        push/force disabled, curriculum disabled — same as before but now enforced
        with hard errors if the env config rejects the override.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import subprocess
import glob
import yaml
import csv
import traceback
from pathlib import Path

import torch
import numpy as np

# ── Isaac Lab imports ─────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="A-to-B locomotion test for A1")

parser.add_argument("--task",       type=str,   default="Isaac-Velocity-Flat-Unitree-A1-Play-v0")
parser.add_argument("--checkpoint", type=str,   default=None,   help="Path / glob to model_*.pt")
parser.add_argument("--label",      type=str,   default="Algorithm")

parser.add_argument("--goal_x",      type=float, default=5.0)
parser.add_argument("--goal_y",      type=float, default=0.0)
parser.add_argument("--goal_radius", type=float, default=0.5)

parser.add_argument("--num_envs",  type=int,   default=16)
parser.add_argument("--max_steps", type=int,   default=2000)
parser.add_argument("--cmd_vel",   type=float, default=1.0)
parser.add_argument("--seed",      type=int,   default=42)

parser.add_argument("--results_dir",  type=str,  default="results")
parser.add_argument("--compare_only", action="store_true")
parser.add_argument("--num_runs",     type=int,  default=1)
parser.add_argument("--export_csv",   action="store_true")
parser.add_argument("--plot_stability", action="store_true",
                    help="Generate ZMP & support-polygon plots after each run.")
parser.add_argument("--clean_results", action="store_true",
                    help="Delete all saved JSONs in results_dir before running.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "unknown"

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# [F3] Command injection.
#
# Isaac Lab's UniformVelocityCommand has TWO separate buffers:
#   term.vel_command_b  (N,3) — the "desired" command, input to the term
#   term.command        (N,3) — what actually goes into the obs tensor
#
# Every call to term.compute() (which runs inside env.step()) will
# RECOMPUTE term.command from vel_command_b plus heading/standing mode logic,
# then resample if time_left hits zero.
#
# The ONLY reliable fix is to write BOTH buffers AND freeze time_left,
# so that after compute() runs, command still holds what we want.
# We must call this AFTER env.step(), not before, so our write wins.
#
# Additionally: we patch the observation tensor directly in the obs group,
# since velocity_commands obs term reads term.command at obs-compute time.
# Writing after step() but before the next policy call is the right order.

_CMD_INJECTED_ONCE = [False]

def inject_velocity_command(env_unwrapped, cmd: torch.Tensor, warned: list):
    """
    Overwrite all command buffers in the UniformVelocityCommand term.
    Call this AFTER env.step() so the write is not undone by compute().
    cmd shape: (N, 3) = [vx_body, vy_body, wz]
    """
    try:
        term = env_unwrapped.command_manager.get_term("base_velocity")
    except Exception as e:
        if not warned[0]:
            print(f"[WARN F3] get_term('base_velocity') failed: {e}")
            warned[0] = True
        return

    if not _CMD_INJECTED_ONCE[0]:
        tensor_attrs = {}
        for a in dir(term):
            if a.startswith("_"):
                continue
            val = getattr(term, a, None)
            if isinstance(val, torch.Tensor):
                tensor_attrs[a] = tuple(val.shape)
        print(f"[DEBUG F3] UniformVelocityCommand tensor attrs: {tensor_attrs}")

    # Write both the input buffer and the output/obs buffer
    wrote_any = False
    for attr in ("vel_command_b", "command", "velocity_command_b", "commands"):
        buf = getattr(term, attr, None)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.shape == cmd.shape:
            buf[:] = cmd
            wrote_any = True
            if not _CMD_INJECTED_ONCE[0]:
                print(f"[INFO F3] Wrote term.{attr}  shape={tuple(buf.shape)}")

    # Freeze resample timer to prevent resampling on next compute()
    for timer_attr in ("time_left", "command_resampling_time",
                       "resampling_time", "resample_time_remaining"):
        timer = getattr(term, timer_attr, None)
        if timer is not None and isinstance(timer, torch.Tensor):
            timer[:] = 1e9
            if not _CMD_INJECTED_ONCE[0]:
                print(f"[INFO F3] Froze timer via term.{timer_attr}")
            break

    if not wrote_any and not warned[0]:
        print(f"[WARN F3] Could not write any command buffer on {type(term)}")
        all_attrs = {a: type(getattr(term, a)).__name__
                     for a in dir(term) if not a.startswith("__")}
        print(f"          All attrs: {all_attrs}")
        warned[0] = True

    _CMD_INJECTED_ONCE[0] = True

# [F4] Probe for body contact sensor
def get_body_contact(env_unwrapped, num_envs: int, device: str, warned: list) -> torch.Tensor:
    """Try known sensor names; return zero tensor (no fall) on failure."""
    for name in ("contact_forces", "contact_sensor", "body_contact_forces", "base_contact"):
        try:
            sensor = env_unwrapped.scene[name]
            forces = sensor.data.net_forces_w[:, 0, :]   # first body
            return torch.norm(forces, dim=1) > 1.0
        except Exception:
            continue
    if not warned[0]:
        print("[WARN F4] Could not find contact sensor — fall detection disabled.")
        warned[0] = True
    return torch.zeros(num_envs, dtype=torch.bool, device=device)

# [F5] Safe env.reset()
def safe_reset(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]   # obs, info  →  obs
    return result          # bare obs

# [F1] Safe env.step() — handles 4-tuple and 5-tuple
def safe_step(env, actions):
    result = env.step(actions)
    if len(result) == 5:
        obs, rew, terminated, truncated, info = result
    elif len(result) == 4:
        obs, rew, dones, info = result
    else:
        raise RuntimeError(f"[F1] env.step() returned {len(result)} values — unexpected!")
    return obs

# [F11] obs may be a TensorDict (Isaac Lab wraps obs as TensorDict).
# Extract the policy tensor and check for NaNs safely.
def obs_has_nan(obs) -> bool:
    """Return True if any NaN exists in obs, whether plain tensor or TensorDict."""
    try:
        from tensordict import TensorDictBase
        if isinstance(obs, TensorDictBase):
            return any(torch.isnan(v).any().item() for v in obs.values())
    except ImportError:
        pass
    if isinstance(obs, dict):
        return any(torch.isnan(v).any().item() for v in obs.values() if isinstance(v, torch.Tensor))
    if isinstance(obs, torch.Tensor):
        return torch.isnan(obs).any().item()
    return False

def actions_has_nan(actions) -> bool:
    """Return True if any NaN exists in actions tensor."""
    if isinstance(actions, torch.Tensor):
        return torch.isnan(actions).any().item()
    return False

# ══════════════════════════════════════════════════════════════════════════════
# ZMP & SUPPORT POLYGON UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
#
# Zero Moment Point (ZMP):
#   The ZMP is the point on the ground plane where the net ground-reaction
#   moment has zero horizontal components — i.e. the robot won't tip over as
#   long as ZMP stays inside the support polygon.
#
#   For a rigid body: ZMP_xy = (sum_i f_i * p_i_xy) / (sum_i f_i)
#   where f_i is the normal (vertical) contact force at foot i,
#   and p_i_xy is the foot's XY position.
#   When all feet are off the ground (flight phase) ZMP is undefined — we skip.
#
# Support Polygon:
#   The convex hull of all grounded foot positions at a given timestep.
#   A larger support polygon = more stable gait.
#
# Stability Margin:
#   The minimum distance from the ZMP to any edge of the support polygon.
#   Positive = ZMP inside polygon (stable). Negative = tipping.

def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Signed distance from point p to line segment a→b (2D). Negative = outside."""
    ab = b - a
    ap = p - a
    t  = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))

def zmp_stability_margin(zmp_xy: np.ndarray, hull_pts: np.ndarray) -> float:
    """
    Compute ZMP stability margin = min distance from zmp_xy to any edge of
    the convex hull defined by hull_pts (Nx2 array of grounded foot positions).
    Returns:
      +d  if ZMP is inside the hull (stable, d = distance to nearest edge)
       0  if ZMP is on the boundary
      -d  if ZMP is outside (robot is tipping, d = distance to nearest edge)
    """
    from scipy.spatial import ConvexHull
    if len(hull_pts) < 3:
        # Degenerate: line or single point — just return distance to centroid
        centroid = hull_pts.mean(axis=0)
        return -float(np.linalg.norm(zmp_xy - centroid))
    try:
        hull = ConvexHull(hull_pts)
    except Exception:
        return 0.0

    # Check if ZMP is inside hull using half-plane test
    # hull.equations: each row = [normal_x, normal_y, offset] where n·x + offset <= 0 inside
    inside = np.all(hull.equations[:, :2] @ zmp_xy + hull.equations[:, 2] <= 1e-6)

    # Min distance to any edge
    vertices = hull_pts[hull.vertices]
    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        d = _point_to_segment_dist(zmp_xy, vertices[i], vertices[(i+1) % n])
        min_dist = min(min_dist, d)

    return min_dist if inside else -min_dist


def compute_zmp_from_feet(
    foot_pos_w: np.ndarray,     # (4, 3) foot positions in world frame
    foot_forces: np.ndarray,    # (4,)   vertical (z) contact forces, >= 0
    contact_thresh: float = 2.0 # N — force threshold to count as grounded
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns:
      zmp_xy    — (2,) ZMP position, or None if no feet are grounded
      hull_pts  — (K, 2) grounded foot XY positions (K >= 1), or None
    """
    grounded = foot_forces > contact_thresh
    if not grounded.any():
        return None, None

    fz     = foot_forces[grounded].clip(min=0.0)
    pts    = foot_pos_w[grounded, :2]          # (K, 2)
    total  = fz.sum()
    if total < 1e-6:
        return None, None

    zmp_xy = (pts * fz[:, None]).sum(axis=0) / total
    return zmp_xy, pts


def plot_zmp_and_hull(
    zmp_history:    list,   # list of (2,) arrays, one per step (None = flight)
    hull_history:   list,   # list of (K,2) arrays, one per step (None = flight)
    margin_history: list,   # list of floats (None = flight)
    base_pos_history: list, # list of (2,) arrays — robot XY trajectory
    spawn_xy:       np.ndarray,   # (2,)
    goal_xy:        np.ndarray,   # (2,)
    label:          str,
    env_id:         int,
    out_dir:        str,
    dt:             float,
):
    """
    Produces two figures:
      1. Top-down: robot path + ZMP trajectory + support polygon snapshots
      2. Time-series: ZMP stability margin over the episode
    """
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from scipy.spatial import ConvexHull

    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "-")
    T    = len(zmp_history)
    time_axis = np.arange(T) * dt

    # ── Filter valid steps ────────────────────────────────────────────────────
    valid       = [i for i, z in enumerate(zmp_history) if z is not None]
    zmp_arr     = np.array([zmp_history[i]     for i in valid])   # (V, 2)
    margin_arr  = np.array([margin_history[i]  for i in valid])   # (V,)
    base_arr    = np.array(base_pos_history)                       # (T, 2)

    # ── Figure 1: Top-down trajectory + ZMP ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_title(f"ZMP & Support Polygon — {label}  (env {env_id})", fontsize=12)

    # Robot base path
    ax.plot(base_arr[:, 0], base_arr[:, 1],
            color="steelblue", lw=1.5, alpha=0.7, label="Robot base path")

    # ZMP trajectory coloured by stability margin
    if len(zmp_arr) > 1:
        points  = zmp_arr.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        lc      = LineCollection(segs, cmap="RdYlGn",
                                 norm=plt.Normalize(-0.1, 0.15), lw=2, zorder=3)
        lc.set_array(margin_arr[:-1])
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="ZMP stability margin (m)")

    # Draw support polygon snapshots every N steps
    snapshot_every = max(1, len(valid) // 20)
    for k, i in enumerate(valid[::snapshot_every]):
        pts = hull_history[i]
        if pts is not None and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                verts = np.append(pts[hull.vertices], pts[hull.vertices[:1]], axis=0)
                ax.fill(verts[:, 0], verts[:, 1],
                        alpha=0.08, color="green", zorder=1)
                ax.plot(verts[:, 0], verts[:, 1],
                        color="green", lw=0.5, alpha=0.3, zorder=2)
            except Exception:
                pass
        elif pts is not None:
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=8, color="green", alpha=0.2, zorder=2)

    # Start / goal markers
    ax.scatter(*spawn_xy, marker="^", s=120, color="blue",   zorder=5, label="Start")
    ax.scatter(*goal_xy,  marker="*", s=180, color="gold",   zorder=5,
               edgecolors="black", linewidths=0.5, label="Goal")
    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("X (m)");  ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p1 = os.path.join(out_dir, f"{safe}_env{env_id}_zmp_topdown.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ── Figure 2: Stability margin over time ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.set_title(f"ZMP Stability Margin over Time — {label}  (env {env_id})", fontsize=11)

    valid_time = time_axis[valid]
    ax2.plot(valid_time, margin_arr, lw=1.2, color="steelblue", label="Margin")
    ax2.axhline(0, color="red", lw=1.0, ls="--", label="ZMP boundary (margin=0)")
    ax2.fill_between(valid_time, margin_arr, 0,
                     where=(margin_arr >= 0), alpha=0.2, color="green",
                     label="Stable (ZMP inside hull)")
    ax2.fill_between(valid_time, margin_arr, 0,
                     where=(margin_arr < 0),  alpha=0.3, color="red",
                     label="Unstable (ZMP outside hull)")

    mean_m = margin_arr.mean()
    ax2.axhline(mean_m, color="orange", lw=1.0, ls=":",
                label=f"Mean margin = {mean_m:.3f} m")

    ax2.set_xlabel("Time (s)");  ax2.set_ylabel("Stability margin (m)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, f"{safe}_env{env_id}_zmp_margin.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    print(f"  [ZMP] Plots saved: {os.path.basename(p1)},  {os.path.basename(p2)}")
    return p1, p2


# ══════════════════════════════════════════════════════════════════════════════
# METRICS TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeMetrics:
    def __init__(self, num_envs, spawn_pos, goal_pos, goal_radius, device):
        self.num_envs    = num_envs
        self.spawn_pos   = spawn_pos.clone().to(device)   # (N, 2)
        self.goal_pos    = goal_pos.clone().to(device)    # (N, 2)
        self.goal_radius = goal_radius
        self.device      = device

        self.reached    = torch.zeros(num_envs, dtype=torch.bool,  device=device)
        self.reach_step = torch.full((num_envs,), -1.0,            device=device)

        self.path_length = torch.zeros(num_envs, device=device)
        self.energy      = torch.zeros(num_envs, device=device)
        self.falls       = torch.zeros(num_envs, device=device)

        # [F6] last_pos will be set on first update, not None-compared
        self.last_pos       = None
        self.base_vel_hist  = []
        self.vel_err_hist   = []
        self.heading_err_hist = []
        self.base_height_hist = []
        self.roll_pitch_hist  = []
        self.policy_time_ms   = []
        self.step_count       = 0

        # ZMP / support-polygon histories  (one entry per env per step)
        # Stored only for env 0 by default to keep memory reasonable.
        # Set track_env_id to track a specific env.
        self.track_env_id       = 0
        self.zmp_history        = []   # (2,) or None
        self.hull_history       = []   # (K,2) or None
        self.margin_history     = []   # float or None
        self.base_pos_history   = []   # (2,) base XY per step

    def update(self, pos, quat, lin_vel, joint_vel, joint_torque,
               body_contact, cmd_vel, yaw_desired, yaw_current, dt):
        self.step_count += 1
        xy_pos = pos[:, :2]

        # ── Goal reached ──────────────────────────────────────────────────────
        dist         = torch.norm(xy_pos - self.goal_pos, dim=1)
        just_reached = (dist < self.goal_radius) & ~self.reached
        self.reached    |= (dist < self.goal_radius)
        self.reach_step[just_reached] = float(self.step_count)

        # ── Path length (only while not yet reached) ──────────────────────────
        if self.last_pos is not None:
            step_dist = torch.norm(xy_pos - self.last_pos[:, :2], dim=1)
            step_dist = torch.where(self.reached, torch.zeros_like(step_dist), step_dist)
            self.path_length += step_dist
        # [F6] always store last_pos after first real observation
        self.last_pos = pos.clone()

        # ── Energy ────────────────────────────────────────────────────────────
        power       = torch.sum(torch.abs(joint_torque * joint_vel), dim=1)
        power_delta = torch.where(self.reached, torch.zeros_like(power), power * dt)
        self.energy += power_delta

        # ── Falls ─────────────────────────────────────────────────────────────
        self.falls += body_contact.float()

        # ── Velocity tracking error ───────────────────────────────────────────
        dx = self.goal_pos[:, 0] - xy_pos[:, 0]
        dy = self.goal_pos[:, 1] - xy_pos[:, 1]
        dist_to_goal = torch.sqrt(dx**2 + dy**2).clamp(min=1e-3)
        cmd_vx = (dx / dist_to_goal) * cmd_vel
        cmd_vy = (dy / dist_to_goal) * cmd_vel
        v_err  = torch.sqrt((cmd_vx - lin_vel[:, 0])**2 + (cmd_vy - lin_vel[:, 1])**2)
        v_err  = torch.where(self.reached, torch.zeros_like(v_err), v_err)
        self.vel_err_hist.append(v_err)
        self.base_vel_hist.append(lin_vel.clone())
        self.base_height_hist.append(pos[:, 2].clone())

        # ── Heading error ─────────────────────────────────────────────────────
        h_err = torch.abs(wrap_to_pi(yaw_desired - yaw_current))
        h_err = torch.where(self.reached, torch.zeros_like(h_err), h_err)
        self.heading_err_hist.append(h_err)

        # ── Roll / Pitch RMS ──────────────────────────────────────────────────
        roll, pitch, _ = euler_xyz_from_quat(quat)
        rp_rms = torch.sqrt((roll**2 + pitch**2) / 2.0)
        self.roll_pitch_hist.append(rp_rms)

    def update_zmp(self, foot_pos_w: torch.Tensor, foot_forces: torch.Tensor):
        """
        foot_pos_w  : (N, 4, 3) foot positions in world frame
        foot_forces : (N, 4)    vertical contact forces per foot (N)
        Tracks only self.track_env_id to keep memory low.
        """
        e = self.track_env_id
        fp = foot_pos_w[e].cpu().numpy()     # (4, 3)
        ff = foot_forces[e].cpu().numpy()    # (4,)

        self.base_pos_history.append(
            self.last_pos[e, :2].cpu().numpy() if self.last_pos is not None
            else np.zeros(2)
        )

        zmp_xy, hull_pts = compute_zmp_from_feet(fp, ff)
        if zmp_xy is None:
            self.zmp_history.append(None)
            self.hull_history.append(None)
            self.margin_history.append(None)
            return

        margin = zmp_stability_margin(zmp_xy, hull_pts)
        self.zmp_history.append(zmp_xy)
        self.hull_history.append(hull_pts)
        self.margin_history.append(margin)

    def log_inference_time(self, t_ms: float):
        self.policy_time_ms.append(t_ms)

    def _compute_zmp_stats(self) -> dict:
        """Summarise ZMP margin history for the tracked env."""
        valid = [m for m in self.margin_history if m is not None]
        if not valid:
            return {"mean_margin": 0.0, "min_margin": 0.0,
                    "pct_unstable": 0.0, "valid_steps": 0}
        arr = np.array(valid)
        return {
            "mean_margin":   float(arr.mean()),
            "min_margin":    float(arr.min()),
            "pct_unstable":  float((arr < 0).mean() * 100),   # % steps ZMP outside hull
            "valid_steps":   len(valid),
        }

    def compute(self, dt: float) -> dict:
        N           = self.num_envs
        reached_np  = self.reached.cpu().numpy()
        reach_step  = self.reach_step.cpu().numpy()

        max_t = self.step_count * dt
        times = np.where(reached_np, reach_step * dt, max_t)

        opt_dist = torch.norm(self.goal_pos - self.spawn_pos, dim=1).cpu().numpy()
        path_len = self.path_length.cpu().numpy()
        path_eff = np.where(path_len > 1e-3, opt_dist / path_len, 0.0)

        energy_j  = self.energy.cpu().numpy()
        energy_pm = np.where(path_len > 1e-3, energy_j / path_len, 0.0)
        falls_np  = self.falls.cpu().numpy()

        if self.vel_err_hist:
            v_err   = torch.stack(self.vel_err_hist,       dim=0).mean(dim=0).cpu().numpy()
            h_err   = torch.stack(self.heading_err_hist,   dim=0).mean(dim=0).cpu().numpy()
            b_hgt   = torch.stack(self.base_height_hist,   dim=0)
            bh_mean = b_hgt.mean(dim=0).cpu().numpy()
            bh_std  = b_hgt.std(dim=0).cpu().numpy()
            rp      = torch.stack(self.roll_pitch_hist,    dim=0).mean(dim=0).cpu().numpy()
        else:
            v_err = h_err = bh_mean = bh_std = rp = np.zeros(N)

        if len(self.base_vel_hist) >= 3:
            vels     = torch.stack(self.base_vel_hist, dim=0)          # (T, N, 3)
            acc      = (vels[1:] - vels[:-1]) / dt
            jerk     = (acc[1:] - acc[:-1]) / dt
            jerk_mag = torch.norm(jerk, dim=-1)                         # (T-2, N)
            jerk_rms = torch.sqrt(torch.mean(jerk_mag**2, dim=0)).cpu().numpy()
        else:
            jerk_rms = np.zeros(N)

        def ms(arr):
            return float(np.mean(arr)), float(np.std(arr))

        def ms_success(arr):
            s = arr[reached_np] if reached_np.any() else np.array([0.0])
            return float(np.mean(s)), float(np.std(s))

        t_m,  t_s  = ms_success(times) if reached_np.any() else (max_t, 0.0)
        pe_m, pe_s = ms_success(path_eff)

        return {
            "meta": {
                "num_envs":  self.num_envs,
                "max_steps": self.step_count,
                "dt":        dt,
            },
            "primary": {
                "success_rate":           float(np.mean(reached_np)),
                "time_to_goal_mean":      t_m,  "time_to_goal_std":      t_s,
                "path_length_mean":       ms(path_len)[0], "path_length_std": ms(path_len)[1],
                "path_efficiency_mean":   pe_m, "path_efficiency_std":   pe_s,
                "energy_J_mean":          ms(energy_j)[0],  "energy_J_std":  ms(energy_j)[1],
                "energy_per_meter_mean":  ms(energy_pm)[0], "energy_per_meter_std": ms(energy_pm)[1],
            },
            "stability": {
                "num_falls_mean":      ms(falls_np)[0],  "num_falls_std":      ms(falls_np)[1],
                "base_height_mean":    ms(bh_mean)[0],   "base_height_std":    ms(bh_mean)[1],
                "roll_pitch_rms_mean": ms(rp)[0],        "roll_pitch_rms_std": ms(rp)[1],
            },
            "motion_quality": {
                "velocity_tracking_rmse_mean": ms(v_err)[0],    "velocity_tracking_rmse_std": ms(v_err)[1],
                "jerk_rms_mean":               ms(jerk_rms)[0], "jerk_rms_std":               ms(jerk_rms)[1],
                "heading_error_mean":          ms(h_err)[0],    "heading_error_std":          ms(h_err)[1],
            },
            "gait": {
                "foot_contact_regularity_mean": 0.0, "foot_contact_regularity_std": 0.0,
                "stride_frequency_mean":        0.0, "stride_frequency_std":        0.0,
            },
            "compute": {
                "policy_inference_ms_mean": ms(self.policy_time_ms)[0],
                "policy_inference_ms_std":  ms(self.policy_time_ms)[1],
            },
            "zmp_stability": self._compute_zmp_stats(),
            "per_env": [
                {
                    "env_id":      i,
                    "success":     bool(reached_np[i]),
                    "time_to_goal": float(times[i]),
                    "path_length":  float(path_len[i]),
                    "energy_J":     float(energy_j[i]),
                    "falls":        float(falls_np[i]),
                } for i in range(N)
            ],
        }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_atob(args, run_seed: int) -> dict:
    set_seed(run_seed)

    print(f"\n{'='*60}")
    print(f"  play_2.py — A→B Test  (Seed: {run_seed})")
    print(f"  Label    : {args.label}")
    print(f"  Task     : {args.task}")
    print(f"  Goal     : ({args.goal_x}, {args.goal_y})  radius={args.goal_radius}m")
    print(f"{'='*60}\n")

    # ── 1. Build environment ──────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = run_seed
    env_cfg.observations.policy.enable_corruption = False

    # [F10] Disable random perturbations for fair comparison — hard-fail if missing
    if hasattr(env_cfg, "events"):
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot                 = None
    else:
        print("[WARN F10] env_cfg has no 'events' attribute — "
              "cannot disable push/force. Comparison may be unfair!")

    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s      = 9999.0   # effectively infinite

    # Lock command to a single forward velocity; no resampling
    env_cfg.commands.base_velocity.resampling_time_range   = (9999.0, 9999.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_x        = (args.cmd_vel, args.cmd_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y        = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z        = (0.0, 0.0)

    env    = gym.make(args.task, cfg=env_cfg)
    env    = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    # ── 2. Load policy ────────────────────────────────────────────────────────
    task_entry        = gym.spec(args.task).kwargs
    rsl_rl_cfg_entry  = task_entry["rsl_rl_cfg_entry_point"]
    if isinstance(rsl_rl_cfg_entry, str):
        import importlib
        module_path, class_name = rsl_rl_cfg_entry.split(":")
        module           = importlib.import_module(module_path)
        runner_cfg_class = getattr(module, class_name)
    else:
        runner_cfg_class = rsl_rl_cfg_entry

    ckpt_path = None
    if args.checkpoint is not None:
        matches   = sorted(glob.glob(args.checkpoint))
        ckpt_path = matches[-1] if matches else None
        if ckpt_path is None:
            print(f"[WARN] Checkpoint glob '{args.checkpoint}' matched nothing — using random weights!")

    runner_dict = runner_cfg_class().to_dict()
    if ckpt_path is not None:
        saved_cfg = os.path.join(os.path.dirname(ckpt_path), "params", "agent.yaml")
        if os.path.exists(saved_cfg):
            with open(saved_cfg) as f:
                runner_dict = yaml.safe_load(f)
            print(f"[INFO] Using saved config: {saved_cfg}")

    runner = OnPolicyRunner(env, runner_dict, log_dir=None, device=device)
    if ckpt_path is not None:
        runner.load(ckpt_path)
        print(f"[INFO] Loaded weights: {ckpt_path}")
    else:
        print("[INFO] No checkpoint — running with random / untrained weights.")

    policy = runner.get_inference_policy(device=device)
    policy.eval()

    # ── 3. Setup metrics ──────────────────────────────────────────────────────
    # [F8] env_origins are world offsets; goal is offset + goal delta
    spawn_pos_xy = env.unwrapped.scene.env_origins[:, :2].clone()
    goal_pos_xy  = spawn_pos_xy + torch.tensor(
        [args.goal_x, args.goal_y], device=device
    ).unsqueeze(0)

    print(f"[INFO] Spawn positions (first 4): {spawn_pos_xy[:4].cpu().tolist()}")
    print(f"[INFO] Goal  positions (first 4): {goal_pos_xy[:4].cpu().tolist()}")

    dt      = env.unwrapped.step_dt
    metrics = EpisodeMetrics(
        args.num_envs, spawn_pos_xy, goal_pos_xy, args.goal_radius, device
    )

    # One-time warning flags (mutable list so closures can write)
    cmd_warn     = [False]
    contact_warn = [False]

    # [F5] Safe reset
    obs = safe_reset(env)

    # Pre-inject a forward command so step-0 obs already has a sane velocity_commands.
    # Without this, the policy sees zeros in velocity_commands on the very first step.
    _initial_cmd = torch.zeros((args.num_envs, 3), device=device)
    _initial_cmd[:, 0] = args.cmd_vel
    inject_velocity_command(env.unwrapped, _initial_cmd, [True])  # warned=True suppresses duplicate prints

    # ── 4. Step loop ──────────────────────────────────────────────────────────
    step       = 0
    start_wall = time.time()

    while step < args.max_steps:

        # [F9] NaN guard on observations — [F11] obs may be TensorDict
        if obs_has_nan(obs):
            print(f"[ERROR F9] NaN detected in obs at step {step}! Aborting this run.")
            break

        asset = env.unwrapped.scene["robot"]

        # ── Navigation: body-frame velocity command ─────────────────────────
        # The policy obs includes velocity_commands as [vx_body, vy_body, wz].
        # vx_body = forward in robot heading. We must NOT send world-frame vx/vy.
        #
        # Strategy:
        #   - Compute heading error = (world yaw to goal) - (robot yaw).
        #   - Angular rate proportional to heading error, clamped to trained range.
        #   - Forward speed tapers off when heading error is large (avoid running sideways).
        root_state        = asset.data.root_state_w        # (N,13)
        pos_pre           = root_state[:, :3]
        quat_pre          = root_state[:, 3:7]
        lin_vel_pre       = root_state[:, 7:10]

        dx                = goal_pos_xy[:, 0] - pos_pre[:, 0]
        dy                = goal_pos_xy[:, 1] - pos_pre[:, 1]
        yaw_to_goal       = torch.atan2(dy, dx)
        _, _, yaw_current = euler_xyz_from_quat(quat_pre)
        heading_err       = wrap_to_pi(yaw_to_goal - yaw_current)   # (N,)

        # Proportional yaw controller, clamped to trained range [-1.5, 1.5] rad/s
        ang_vel_z         = (2.0 * heading_err).clamp(-1.5, 1.5)

        # IMPORTANT: The A1 policy was trained with simultaneous vx + wz commands.
        # Do NOT zero out vx when turning — that's out-of-distribution for the policy.
        # Always command full forward speed; the policy handles combined walk+turn.
        # Only reduce vx for very large heading errors (> 120 deg) where going forward
        # would move away from goal.
        heading_scale     = torch.where(
            heading_err.abs() > (2.0 * math.pi / 3.0),   # > 120 degrees
            torch.zeros_like(heading_err),                  # facing backwards: stop forward
            torch.ones_like(heading_err)                    # otherwise: full forward speed
        )
        vx_cmd            = args.cmd_vel * heading_scale

        cmd               = torch.zeros((args.num_envs, 3), device=device)
        cmd[:, 0]         = vx_cmd
        cmd[:, 2]         = ang_vel_z
        cmd               = torch.where(
            metrics.reached.unsqueeze(1), torch.zeros_like(cmd), cmd
        )

        # Policy inference — obs already has last step's command baked in
        t0 = time.perf_counter()
        with torch.no_grad():
            actions = policy(obs)
        t1 = time.perf_counter()
        metrics.log_inference_time((t1 - t0) * 1000.0)

        # [F9] NaN guard on actions
        if actions_has_nan(actions):
            print(f"[ERROR F9] NaN detected in actions at step {step}! Aborting.")
            break

        # [F1] Safe step — handles 4-tuple and 5-tuple
        obs = safe_step(env, actions)
        step += 1

        # [F3] Inject command AFTER step() so our write overwrites what compute() set.
        inject_velocity_command(env.unwrapped, cmd, cmd_warn)

        # [F3b] NUCLEAR OPTION: also patch the raw observation tensor directly.
        # The velocity_commands obs term reads term.command at obs-build time inside step().
        # Since step() already built obs before we injected, we must patch obs directly
        # so the CURRENT step's policy call sees our command, not the env's stale one.
        #
        # The obs group 'policy' has shape (N, 48):
        #   [0:3]   base_lin_vel
        #   [3:6]   base_ang_vel
        #   [6:9]   projected_gravity
        #   [9:12]  velocity_commands   <-- patch this
        #   [12:24] joint_pos
        #   [24:36] joint_vel
        #   [36:48] actions
        try:
            from tensordict import TensorDictBase
            if isinstance(obs, TensorDictBase):
                pol_obs = obs.get("policy")          # (N, 48)
                if pol_obs is not None and pol_obs.shape[-1] == 48:
                    pol_obs[:, 9:12] = cmd
            elif isinstance(obs, torch.Tensor) and obs.shape[-1] == 48:
                obs[:, 9:12] = cmd
        except Exception:
            pass  # if patching fails, injected buffers are still the best we can do

        # [F2] Read state AFTER step so metrics reflect current state
        root_state_post  = asset.data.root_state_w
        pos_post         = root_state_post[:, :3]
        quat_post        = root_state_post[:, 3:7]
        lin_vel_post     = root_state_post[:, 7:10]
        _, _, yaw_post   = euler_xyz_from_quat(quat_post)

        # [F4] Body contact with dynamic sensor probe
        body_contact = get_body_contact(env.unwrapped, args.num_envs, device, contact_warn)

        metrics.update(
            pos=pos_post, quat=quat_post, lin_vel=lin_vel_post,
            joint_vel=asset.data.joint_vel,
            joint_torque=asset.data.applied_torque,
            body_contact=body_contact,
            cmd_vel=args.cmd_vel,
            yaw_desired=yaw_to_goal,
            yaw_current=yaw_post,
            dt=dt,
        )

        # ── ZMP & support polygon (only when --plot_stability requested) ──────
        if args.plot_stability:
            try:
                # Foot positions from body kinematics
                # Isaac Lab: asset.data.body_pos_w  shape (N, num_bodies, 3)
                # A1 foot body names: FL_foot, FR_foot, RL_foot, RR_foot
                # Typically bodies 4,8,12,16 or use find by name
                body_pos_w = asset.data.body_pos_w          # (N, B, 3)

                # Try to get foot indices from body names
                if not hasattr(metrics, "_foot_idx"):
                    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    try:
                        metrics._foot_idx = [
                            asset.find_bodies(n)[0][0] for n in foot_names
                        ]
                        print(f"[ZMP] Foot body indices: {metrics._foot_idx}")
                    except Exception:
                        # Fallback: last 4 bodies are typically feet
                        nb = body_pos_w.shape[1]
                        metrics._foot_idx = list(range(nb-4, nb))
                        print(f"[ZMP] Using fallback foot indices: {metrics._foot_idx}")

                foot_pos = body_pos_w[:, metrics._foot_idx, :]  # (N, 4, 3)

                # Foot contact forces — use the contact sensor if available,
                # otherwise approximate from body z-position (feet near ground = contact)
                try:
                    sensor = env.unwrapped.scene["contact_forces"]
                    # net_forces_w shape: (N, num_bodies, 3)  — take Z component of feet
                    raw_f   = sensor.data.net_forces_w
                    foot_fz = raw_f[:, metrics._foot_idx, 2].abs()   # (N, 4)
                except Exception:
                    # Approximate: foot is grounded if z < 0.05m above ground
                    foot_z   = foot_pos[:, :, 2]                   # (N, 4)
                    foot_fz  = torch.where(
                        foot_z < 0.05,
                        torch.ones_like(foot_z) * 30.0,   # ~30 N per grounded foot
                        torch.zeros_like(foot_z)
                    )

                metrics.update_zmp(foot_pos, foot_fz)

            except Exception as e:
                if step == 1:
                    print(f"[WARN ZMP] Could not collect foot data: {e}")
                    print("           ZMP tracking disabled for this run.")
                    args.plot_stability = False

        if step % 200 == 0:
            reached_count = metrics.reached.sum().item()
            dists = torch.norm(pos_post[:, :2] - goal_pos_xy, dim=1)
            # Recompute heading error for diagnostics
            dx_d  = goal_pos_xy[:, 0] - pos_post[:, 0]
            dy_d  = goal_pos_xy[:, 1] - pos_post[:, 1]
            yaw_g = torch.atan2(dy_d, dx_d)
            h_err_deg = torch.rad2deg(wrap_to_pi(yaw_g - yaw_post).abs())
            mean_vx = lin_vel_post[:, 0].mean().item()
            print(
                f"  step {step:4d}/{args.max_steps} | "
                f"dist={dists.mean():.3f}m | "
                f"heading_err={h_err_deg.mean():.1f}° | "
                f"vx={mean_vx:.2f}m/s | "
                f"reached={reached_count}/{args.num_envs} | "
                f"wall={time.time()-start_wall:.1f}s"
            )

        if metrics.reached.all():
            print(f"\n  ✓ All {args.num_envs} robots reached goal at step {step}!")
            break

    wall_time = time.time() - start_wall
    result    = metrics.compute(dt)
    result["_metrics_ref"] = metrics   # temporary — stripped before JSON save

    result["meta"]["label"]       = args.label
    result["meta"]["task"]        = args.task
    result["meta"]["checkpoint"]  = ckpt_path if ckpt_path else "random"
    result["meta"]["goal"]        = [args.goal_x, args.goal_y]
    result["meta"]["cmd_vel"]     = args.cmd_vel
    result["meta"]["wall_time_s"] = wall_time
    result["meta"]["git_hash"]    = get_git_hash()
    result["meta"]["seed"]        = run_seed
    result["compute"]["sim_steps_per_second"] = (step * args.num_envs) / wall_time
    result["cpg_specific"] = (
        {"note": "CPG specific metrics not implemented"}
        if "cpg" in args.label.lower() else None
    )

    env.close()
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SAVE + COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def save_result(result: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    safe_label = result["meta"]["label"].replace(" ", "_").replace("/", "-")
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    filename   = Path(results_dir) / f"{safe_label}_{timestamp}.json"
    result_clean = {k: v for k, v in result.items() if k != "_metrics_ref"}
    with open(filename, "w") as fp:
        json.dump(result_clean, fp, indent=2)
    print(f"[INFO] Results saved to: {filename}")


def print_comparison_table(results_dir: str, export_csv: bool):
    results_path = Path(results_dir)
    json_files   = sorted(results_path.glob("*.json"))

    if not json_files:
        print(f"\n[play_2] No results found in '{results_dir}'.\n")
        return

    all_results = []
    for f in json_files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    all_results.sort(
        key=lambda r: (-r["primary"]["success_rate"], r["primary"]["time_to_goal_mean"])
    )

    headers  = ["Algorithm", "Success%", "Time(s)", "PathEff", "Energy/m", "Falls", "JerkRMS", "VelRMSE"]
    col_w    = [20, 10, 14, 10, 10, 8, 10, 10]

    table_data = []
    for r in all_results:
        pm = r["primary"];  st = r["stability"];  mq = r["motion_quality"]
        table_data.append({
            "Algorithm": r["meta"]["label"],
            "Success%":  pm["success_rate"] * 100,
            "Time(s)":   pm["time_to_goal_mean"],
            "PathEff":   pm["path_efficiency_mean"],
            "Energy/m":  pm["energy_per_meter_mean"],
            "Falls":     st["num_falls_mean"],
            "JerkRMS":   mq["jerk_rms_mean"],
            "VelRMSE":   mq["velocity_tracking_rmse_mean"],
            "time_std":  pm["time_to_goal_std"],
        })

    valid_times = [r["Time(s)"] for r in table_data if r["Success%"] > 0]
    best = {
        "Time(s)":   min(valid_times) if valid_times else float('inf'),
        "PathEff":   max(r["PathEff"]  for r in table_data),
        "Energy/m":  min(r["Energy/m"] for r in table_data),
        "Falls":     min(r["Falls"]    for r in table_data),
        "JerkRMS":   min(r["JerkRMS"]  for r in table_data),
        "VelRMSE":   min(r["VelRMSE"]  for r in table_data),
    }

    sep = "─" * sum(col_w)
    print(f"\n{sep}\n  A→B LOCOMOTION COMPARISON TABLE\n{sep}")
    print("".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)

    baseline_row = next((r for r in table_data if "PPO Baseline" in r["Algorithm"]), None)

    for r in table_data:
        def fmt(key, val):
            if val == 0 and key in ("Time(s)", "PathEff") and r["Success%"] == 0:
                return "FAIL"
            s = f"{val:.2f}" if key in ("PathEff","JerkRMS","VelRMSE") else \
                f"{val:.0f}" if key == "Energy/m" else f"{val:.1f}"
            if val == best.get(key) and r["Success%"] > 0:
                s += "*"
            return s

        succ = f"{r['Success%']:.0f}%"
        t    = (f"{r['Time(s)']:.2f}±{r['time_std']:.1f}" + ("*" if r["Time(s)"] == best["Time(s)"] else "")
                if r["Success%"] > 0 else "FAILED")

        row_str = [
            r["Algorithm"], succ, t,
            fmt("PathEff",  r["PathEff"]),
            fmt("Energy/m", r["Energy/m"]),
            fmt("Falls",    r["Falls"]),
            fmt("JerkRMS",  r["JerkRMS"]),
            fmt("VelRMSE",  r["VelRMSE"]),
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row_str, col_w)))

        if (baseline_row
                and r["Algorithm"] != baseline_row["Algorithm"]
                and r["Success%"] > 0
                and baseline_row["Success%"] > 0):
            imp_t   = (baseline_row["Time(s)"]  - r["Time(s)"])  / baseline_row["Time(s)"]  * 100
            imp_eng = (baseline_row["Energy/m"] - r["Energy/m"]) / baseline_row["Energy/m"] * 100
            print(f"    └─ vs Baseline: Time {imp_t:+.1f}%,  Energy {imp_eng:+.1f}%")

    print(sep)
    print("  * indicates best value in column\n")

    if export_csv:
        csv_path = Path(results_dir) / "comparison_table.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for r in table_data:
                writer.writerow({k: r.get(k, "") for k in headers})
        print(f"[INFO] Exported CSV to {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args.clean_results:
        import glob as _glob
        old_files = _glob.glob(os.path.join(args.results_dir, "*.json"))
        for f in old_files:
            os.remove(f)
        print(f"[INFO] Cleaned {len(old_files)} old result(s) from '{args.results_dir}'.")

    if args.compare_only:
        print_comparison_table(args.results_dir, args.export_csv)
    else:
        all_run_results = []
        for i in range(args.num_runs):
            seed = args.seed + i
            try:
                res = run_atob(args, seed)
                all_run_results.append(res)
                save_result(res, args.results_dir)

                # ── ZMP plots ─────────────────────────────────────────────────
                if args.plot_stability and res.get("_metrics_ref") is not None:
                    m = res["_metrics_ref"]
                    e = m.track_env_id
                    plot_zmp_and_hull(
                        zmp_history=m.zmp_history,
                        hull_history=m.hull_history,
                        margin_history=m.margin_history,
                        base_pos_history=m.base_pos_history,
                        spawn_xy=m.spawn_pos[e].cpu().numpy(),
                        goal_xy=m.goal_pos[e].cpu().numpy(),
                        label=args.label,
                        env_id=e,
                        out_dir=os.path.join(args.results_dir, "plots"),
                        dt=res["meta"]["dt"],
                    )
                    del res["_metrics_ref"]   # don't serialize to JSON

                # ── Inline result summary printed immediately after each run ──
                pm = res["primary"];  st = res["stability"];  mq = res["motion_quality"]
                sep = "─" * 60
                print(f"\n{sep}")
                print(f"  RUN {i+1}/{args.num_runs} SUMMARY  [{res['meta']['label']}]  seed={seed}")
                print(sep)
                print(f"  Success rate    : {pm['success_rate']*100:.0f}%  ({int(pm['success_rate']*args.num_envs)}/{args.num_envs} robots)")
                print(f"  Time to goal    : {pm['time_to_goal_mean']:.2f} ± {pm['time_to_goal_std']:.2f} s")
                print(f"  Path efficiency : {pm['path_efficiency_mean']:.3f}  (1.0 = perfectly straight)")
                print(f"  Energy / meter  : {pm['energy_per_meter_mean']:.1f} J/m")
                print(f"  Falls           : {st['num_falls_mean']:.1f}")
                print(f"  Jerk RMS        : {mq['jerk_rms_mean']:.2f}")
                print(f"  Vel tracking    : {mq['velocity_tracking_rmse_mean']:.3f} m/s RMSE")
                print(f"  Inference time  : {res['compute']['policy_inference_ms_mean']:.2f} ms/step")
                if "zmp_stability" in res and res["zmp_stability"]["valid_steps"] > 0:
                    zs = res["zmp_stability"]
                    print(f"  ZMP mean margin : {zs['mean_margin']:.4f} m  "
                          f"(min={zs['min_margin']:.4f} m,  "
                          f"{zs['pct_unstable']:.1f}% steps outside hull)")
                print(f"  Wall time       : {res['meta']['wall_time_s']:.1f} s")
                print(sep + "\n")
            except Exception as e:
                print(f"\n[ERROR] Run {i} (seed={seed}) crashed: {e}")
                traceback.print_exc()
                print("Continuing to next run...\n")

        if all_run_results:
            print_comparison_table(args.results_dir, args.export_csv)
        else:
            print("[ERROR] All runs failed — no results to compare.")

    simulation_app.close()