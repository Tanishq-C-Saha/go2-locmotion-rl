"""Turn an rsl_rl TensorBoard run into clean PNG plots for the README / report.

Usage:
    ipy scripts/plot_training.py logs/rsl_rl/unitree_go2_rough/<run_dir> --out docs/media
    ipy scripts/plot_training.py logs/rsl_rl/unitree_go2_rough/<run_dir> --list   # show all logged tags

Produces:
    reward_and_length.png        mean reward + mean episode length
    tracking_error.png           velocity tracking error (xy, yaw)
    reward_terms.png             every Episode_Reward/* term
    curriculum_terminations.png  terrain level + termination ratios
    ppo_diagnostics.png          value loss, surrogate loss, entropy, action std
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def smooth(values, weight):
    """Exponential moving average (same idea as TensorBoard's smoothing slider)."""
    out, last = [], values[0] if values else 0.0
    for v in values:
        last = weight * last + (1 - weight) * v
        out.append(last)
    return out


def find_tag(tags, candidates):
    """Return the first tag that exists (tag names differ slightly across rsl_rl versions)."""
    for c in candidates:
        if c in tags:
            return c
    return None


def plot_panels(ea, tags, panels, title, path, smoothing):
    """panels: list of (subplot title, [candidate tag names], y label)."""
    found = [(t, find_tag(tags, cands), y) for t, cands, y in panels]
    found = [f for f in found if f[1] is not None]
    if not found:
        print(f"[skip] {os.path.basename(path)}: none of its tags were logged")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(5 * len(found), 3.6), squeeze=False)
    for ax, (sub_title, tag, ylabel) in zip(axes[0], found):
        events = ea.Scalars(tag)
        x = [e.step for e in events]
        y = [e.value for e in events]
        ax.plot(x, y, alpha=0.25, linewidth=1, color="tab:blue")
        ax.plot(x, smooth(y, smoothing), linewidth=2, color="tab:blue")
        ax.set_title(sub_title)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok]   {path}")


def plot_reward_terms(ea, tags, path, smoothing):
    terms = sorted(t for t in tags if t.startswith("Episode_Reward/"))
    if not terms:
        print("[skip] reward_terms.png: no Episode_Reward/* tags")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag in terms:
        events = ea.Scalars(tag)
        ax.plot([e.step for e in events], smooth([e.value for e in events], smoothing),
                linewidth=1.8, label=tag.split("/", 1)[1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Reward terms per episode (weighted)", fontweight="bold")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Contribution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok]   {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="rsl_rl run folder containing events.out.tfevents.*")
    p.add_argument("--out", default="docs/media", help="Output folder for PNGs")
    p.add_argument("--smoothing", type=float, default=0.9, help="EMA weight, 0 = raw")
    p.add_argument("--list", action="store_true", help="Only print the logged scalar tags")
    a = p.parse_args()

    ea = EventAccumulator(a.run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    if a.list:
        for t in sorted(tags):
            print(t)
        return

    os.makedirs(a.out, exist_ok=True)
    out = lambda name: os.path.join(a.out, name)  # noqa: E731
    s = a.smoothing

    plot_panels(ea, tags, [
        ("Mean reward", ["Train/mean_reward"], "Reward"),
        ("Mean episode length", ["Train/mean_episode_length"], "Steps"),
    ], "Training progress", out("reward_and_length.png"), s)

    plot_panels(ea, tags, [
        ("Linear velocity error (xy)", ["Metrics/base_velocity/error_vel_xy"], "m/s"),
        ("Yaw rate error", ["Metrics/base_velocity/error_vel_yaw"], "rad/s"),
    ], "Command tracking error", out("tracking_error.png"), s)

    plot_reward_terms(ea, tags, out("reward_terms.png"), s)

    plot_panels(ea, tags, [
        ("Terrain level", ["Curriculum/terrain_levels"], "Level"),
        ("Falls (base contact)", ["Episode_Termination/base_contact"], "Fraction"),
        ("Time-outs", ["Episode_Termination/time_out"], "Fraction"),
    ], "Curriculum and terminations", out("curriculum_terminations.png"), s)

    plot_panels(ea, tags, [
        ("Value loss", ["Loss/value_function", "Loss/value"], "Loss"),
        ("Surrogate loss", ["Loss/surrogate"], "Loss"),
        ("Entropy", ["Loss/entropy", "Policy/entropy"], "Entropy"),
        ("Action std", ["Policy/mean_noise_std", "Policy/mean_std", "Policy/std"], "Std"),
    ], "PPO diagnostics", out("ppo_diagnostics.png"), s)


if __name__ == "__main__":
    main()