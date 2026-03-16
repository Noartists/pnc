"""
Dataset and model prediction visualization tool.

Usage:
  # Check dataset quality (randomly sample N trajectories)
  python -m learning.visualize_dataset --h5 learning/datasets/pilot.h5

  # Also load model to compare prediction vs ground truth
  python -m learning.visualize_dataset --h5 learning/datasets/pilot.h5 \
      --checkpoint learning/checkpoints/best.pt

  # Inspect N trajectories
  python -m learning.visualize_dataset --h5 learning/datasets/pilot.h5 --n-trajs 5
"""

import os
import sys
import argparse
import numpy as np
import h5py

from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _make_output_dir(h5_path: str) -> str:
    """Build output dir: learning/figures/<dataset_name>/<YYYYMMDD_HHMMSS>/"""
    dataset_name = os.path.splitext(os.path.basename(h5_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_DIR, "learning", "figures", dataset_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _apply_style():
    """Apply Times New Roman font globally."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 120,
    })


# ============================================================
#                   HDF5 dataset summary
# ============================================================

def print_dataset_summary(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        cfg = f["config"]
        n = cfg.attrs["n_trajectories"]
        dt = cfg.attrs["dt"]

        print(f"\n{'='*60}")
        print(f"Dataset: {os.path.basename(h5_path)}")
        print(f"  Trajectories: {n}")
        print(f"  Time step:    {dt} s")

        traj_grp = f["trajectories"]
        keys = sorted(traj_grp.keys())

        lengths, init_alts, reasons = [], [], []
        for k in keys:
            g = traj_grp[k]
            lengths.append(g.attrs["valid_steps"])
            reasons.append(g.attrs["reason"])
            init_alts.append(float(g["states"][0, 2]))

        lengths = np.array(lengths)
        init_alts = np.array(init_alts)

        from collections import Counter
        reason_counts = Counter(reasons)

        print(f"\n  Length: min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.0f} steps")
        print(f"  Duration: {lengths.min()*dt:.1f}s ~ {lengths.max()*dt:.1f}s")
        print(f"  Init altitude: {init_alts.min():.0f}m ~ {init_alts.max():.0f}m")
        print(f"\n  Termination reasons:")
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count} ({100*count/n:.1f}%)")
        print(f"{'='*60}\n")

    return lengths, init_alts


# ============================================================
#              Single trajectory plot
# ============================================================

def plot_trajectory(ax_dict, states, actions, winds, traj_key, dt):
    T = len(states)
    t = np.arange(T) * dt

    pos = states[:, 0:3]
    euler = np.degrees(states[:, 3:6])
    vel_canopy = states[:, 8:11]

    # 3D trajectory
    ax = ax_dict['3d']
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1.0, alpha=0.8)
    ax.scatter(*pos[0], color='g', s=40, zorder=5, label='Start')
    ax.scatter(*pos[-1], color='r', s=40, zorder=5, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory [{traj_key}]')
    ax.legend()

    # Altitude
    ax = ax_dict['alt']
    ax.plot(t, pos[:, 2], color='steelblue')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude'); ax.grid(True, alpha=0.3)

    # Velocity
    ax = ax_dict['vel']
    ax.plot(t, vel_canopy[:, 0], label='u (forward)')
    ax.plot(t, vel_canopy[:, 1], label='v (lateral)')
    ax.plot(t, vel_canopy[:, 2], label='w (vertical)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Canopy Velocity (body frame)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Euler angles
    ax = ax_dict['euler']
    ax.plot(t, euler[:, 0], label=r'$\phi$ (roll)')
    ax.plot(t, euler[:, 1], label=r'$\theta$ (pitch)')
    ax.plot(t, euler[:, 2], label=r'$\psi$ (heading)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)')
    ax.set_title('Euler Angles')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Control inputs
    ax = ax_dict['action']
    ta = np.arange(len(actions)) * dt
    ax.plot(ta, actions[:, 0], label='Left (m)', color='tab:blue')
    ax.plot(ta, actions[:, 1], label='Right (m)', color='tab:orange')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Deflection (m)')
    ax.set_title('Control Input')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.45)

    # Wind
    ax = ax_dict['wind']
    tw = np.arange(len(winds)) * dt
    ax.plot(tw, winds[:, 0], label='Wx')
    ax.plot(tw, winds[:, 1], label='Wy')
    ax.plot(tw, winds[:, 2], label='Wz')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Wind speed (m/s)')
    ax.set_title('Wind Field')
    ax.legend(); ax.grid(True, alpha=0.3)


def visualize_trajectories(h5_path: str, out_dir: str, n_trajs: int = 4, seed: int = 0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _apply_style()

    with h5py.File(h5_path, "r") as f:
        dt = float(f["config"].attrs["dt"])
        keys = sorted(f["trajectories"].keys())

    rng = np.random.default_rng(seed)
    selected = rng.choice(keys, size=min(n_trajs, len(keys)), replace=False)

    for traj_key in selected:
        with h5py.File(h5_path, "r") as f:
            g = f["trajectories"][traj_key]
            valid = g.attrs["valid_steps"]
            states = g["states"][:valid]
            actions = g["actions"][:valid]
            winds = g["winds"][:valid]
            reason = g.attrs["reason"]

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"{traj_key}  |  termination: {reason}  |  "
            f"{valid} steps ({valid*dt:.1f}s)",
            fontsize=11
        )

        ax3d    = fig.add_subplot(2, 3, 1, projection='3d')
        ax_alt  = fig.add_subplot(2, 3, 2)
        ax_vel  = fig.add_subplot(2, 3, 3)
        ax_euler  = fig.add_subplot(2, 3, 4)
        ax_action = fig.add_subplot(2, 3, 5)
        ax_wind   = fig.add_subplot(2, 3, 6)

        plot_trajectory(
            {'3d': ax3d, 'alt': ax_alt, 'vel': ax_vel,
             'euler': ax_euler, 'action': ax_action, 'wind': ax_wind},
            states, actions, winds, traj_key, dt
        )

        fig.tight_layout()
        save_path = os.path.join(out_dir, f"traj_{traj_key}.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
#         Dataset-wide distribution plots
# ============================================================

def plot_dataset_distributions(h5_path: str, out_dir: str, max_trajs: int = 200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _apply_style()

    init_alts, final_alts = [], []
    all_u, all_v, all_w = [], [], []
    all_left, all_right = [], []
    all_psi = []
    lengths = []

    with h5py.File(h5_path, "r") as f:
        dt = float(f["config"].attrs["dt"])
        keys = sorted(f["trajectories"].keys())[:max_trajs]
        for k in keys:
            g = f["trajectories"][k]
            valid = g.attrs["valid_steps"]
            states = g["states"][:valid]
            actions = g["actions"][:valid]
            lengths.append(valid)
            init_alts.append(states[0, 2])
            final_alts.append(states[-1, 2])
            idx = np.linspace(0, valid - 1, min(50, valid), dtype=int)
            all_u.extend(states[idx, 8].tolist())
            all_v.extend(states[idx, 9].tolist())
            all_w.extend(states[idx, 10].tolist())
            all_left.extend(actions[idx, 0].tolist())
            all_right.extend(actions[idx, 1].tolist())
            all_psi.extend(np.degrees(states[idx, 5]).tolist())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"Dataset Distribution Overview  |  {len(keys)} trajectories  |  "
        f"{os.path.basename(h5_path)}",
        fontsize=11
    )

    def hist(ax, data, title, xlabel, color='steelblue', bins=40):
        ax.hist(data, bins=bins, color=color, edgecolor='white', linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.axvline(np.mean(data), color='red', linestyle='--',
                   linewidth=1.2, label=f'mean={np.mean(data):.2f}')
        ax.legend()

    hist(axes[0, 0], lengths,    'Trajectory Length',   f'Steps (x{dt}s)', )
    hist(axes[0, 1], init_alts,  'Initial Altitude',    'Altitude (m)',    color='teal')
    hist(axes[0, 2], final_alts, 'Final Altitude',      'Altitude (m)',    color='coral')
    hist(axes[0, 3], all_psi,    'Heading Distribution', r'$\psi$ (deg)',   color='mediumpurple')
    hist(axes[1, 0], all_u,      'Forward Velocity u',  'm/s',             color='dodgerblue')
    hist(axes[1, 1], all_v,      'Lateral Velocity v',  'm/s',             color='darkorange')
    hist(axes[1, 2], all_left,   'Left Deflection',     'm',               color='seagreen')
    hist(axes[1, 3], all_right,  'Right Deflection',    'm',               color='firebrick')

    fig.tight_layout()
    save_path = os.path.join(out_dir, "distributions.png")
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"\nDistribution plot saved: {save_path}")


# ============================================================
#         Prediction vs ground truth (requires checkpoint)
# ============================================================

def plot_prediction_vs_truth(h5_path: str, checkpoint_path: str, out_dir: str,
                             n_samples: int = 4, context_length: int = 50,
                             horizon: int = 20):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _apply_style()

    from learning.model import InContextDynamicsTransformer, ModelConfig
    from learning.dataset import (
        NormalizationStats, compute_normalization_stats,
        encode_state_token, encode_state_delta,
        TOKEN_DIM, TARGET_DIM,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    tcfg = ckpt.get("config", {})
    model_cfg = ModelConfig(
        token_dim=TOKEN_DIM, target_dim=TARGET_DIM,
        d_model=tcfg.get("d_model", 128),
        n_heads=tcfg.get("n_heads", 4),
        n_layers=tcfg.get("n_layers", 4),
        d_ff=tcfg.get("d_ff", 512),
        dropout=0.0,
        max_context_length=context_length + 10,
        prediction_horizon=horizon,
        proj_hidden=tcfg.get("d_model", 128),
    )
    model = InContextDynamicsTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: epoch={ckpt.get('epoch', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.6f}")

    norm_path = tcfg.get("norm_stats_path", "")
    if norm_path and not os.path.isabs(norm_path):
        norm_path = os.path.join(ROOT_DIR, norm_path)
    if norm_path and os.path.exists(norm_path):
        norm = NormalizationStats.load(norm_path)
    else:
        print("Norm stats not found, recomputing...")
        norm = compute_normalization_stats(h5_path)

    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "r") as f:
        dt = float(f["config"].attrs["dt"])
        keys = sorted(f["trajectories"].keys())

    min_len = context_length + horizon + 1
    valid_keys = []
    with h5py.File(h5_path, "r") as f:
        for k in keys:
            if f["trajectories"][k].attrs["valid_steps"] >= min_len:
                valid_keys.append(k)

    selected = rng.choice(valid_keys, size=min(n_samples, len(valid_keys)), replace=False)

    channel_names = [
        r'$d\phi$', r'$d\theta$', r'$d\psi$',
        r'$d\theta_r$', r'$d\psi_r$',
        'du', 'dv', 'dw',
        'dp', 'dq', 'dr',
        r'$du_p$', r'$dv_p$', r'$dw_p$',
        r'$dp_p$', r'$dq_p$', r'$dr_p$',
    ]

    for i, traj_key in enumerate(selected):
        with h5py.File(h5_path, "r") as f:
            g = f["trajectories"][traj_key]
            valid = g.attrs["valid_steps"]
            states = g["states"][:valid]
            actions = g["actions"][:valid]

        t_start = rng.integers(0, max(1, valid - min_len))
        t_end = t_start + context_length + horizon + 1

        seg_states = states[t_start:t_end]
        seg_actions = actions[t_start:min(t_end, len(actions))]
        if len(seg_actions) < context_length + horizon:
            pad = np.zeros((context_length + horizon - len(seg_actions), 2))
            seg_actions = np.vstack([seg_actions, pad])

        ctx = np.zeros((context_length, TOKEN_DIM), dtype=np.float32)
        for j in range(context_length):
            ctx[j] = encode_state_token(seg_states[j], seg_actions[j], norm)

        true_deltas = np.zeros((horizon, TARGET_DIM), dtype=np.float32)
        for j in range(horizon):
            si = context_length + j
            true_deltas[j] = encode_state_delta(seg_states[si], seg_states[si + 1], norm)

        with torch.no_grad():
            ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
            pred = model(ctx_t)[0].cpu().numpy()

        show_channels = [0, 1, 2, 5, 6, 7, 8, 9, 10]
        n_ch = len(show_channels)
        n_cols = 3
        n_rows = (n_ch + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        axes = axes.flatten()
        h_steps = np.arange(1, horizon + 1)

        for idx, ch in enumerate(show_channels):
            ax = axes[idx]
            ax.plot(h_steps, true_deltas[:, ch], 'o-', color='steelblue',
                    linewidth=1.5, markersize=4, label='Ground truth')
            ax.plot(h_steps, pred[:, ch], 's--', color='tomato',
                    linewidth=1.5, markersize=4, label='Prediction')
            rmse = float(np.sqrt(np.mean((pred[:, ch] - true_deltas[:, ch]) ** 2)))
            ax.set_title(f'{channel_names[ch]}  RMSE={rmse:.4f}')
            ax.set_xlabel('Horizon step H')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        for idx in range(n_ch, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Prediction vs Ground Truth  [{traj_key}]  t={t_start*dt:.1f}s",
            fontsize=11
        )
        fig.tight_layout()
        save_path = os.path.join(out_dir, f"pred_vs_truth_{i:02d}_{traj_key}.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
#                      CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dataset and model prediction visualization")
    parser.add_argument("--h5", type=str, required=True,
                        help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--n-trajs", type=int, default=4,
                        help="Number of trajectories to plot (default: 4)")
    parser.add_argument("--context-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-dist", action="store_true",
                        help="Skip distribution overview plot")
    args = parser.parse_args()

    h5_path = args.h5
    if not os.path.isabs(h5_path):
        h5_path = os.path.join(ROOT_DIR, h5_path)

    if not os.path.exists(h5_path):
        print(f"Error: file not found: {h5_path}")
        return

    out_dir = _make_output_dir(h5_path)
    print(f"Output directory: {out_dir}")

    print_dataset_summary(h5_path)

    if not args.skip_dist:
        print("Plotting dataset distributions...")
        plot_dataset_distributions(h5_path, out_dir)

    if args.n_trajs > 0:
        print(f"Plotting {args.n_trajs} trajectory details...")
        visualize_trajectories(h5_path, out_dir, n_trajs=args.n_trajs, seed=args.seed)

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
        print("\nPlotting prediction vs ground truth...")
        plot_prediction_vs_truth(
            h5_path, ckpt_path, out_dir,
            n_samples=args.n_trajs,
            context_length=args.context_length,
            horizon=args.horizon,
        )

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
