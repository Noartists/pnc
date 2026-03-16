"""
Evaluation framework for in-context dynamics model.

Provides:
  1. Test dataset generation (IID / OOD-param / OOD-wind)
  2. Autoregressive rollout evaluation
  3. Context-length ablation (K=5..50)
  4. Per-channel error analysis
  5. Baseline comparison (MLP, nominal ODE, oracle ODE)
  6. Publication-quality figure generation
"""

import os
import sys
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from learning.model import InContextDynamicsTransformer, ModelConfig, MLPDynamicsModel
from learning.dataset import (
    NormalizationStats, encode_state_token, encode_state_delta,
    TOKEN_DIM, TARGET_DIM,
)
from learning.data_generation.dataset_generator import (
    DatasetGenerator, GenerationConfig,
)
from learning.data_generation.domain_randomization import WindConfig


# ============================================================
#              Test dataset generation
# ============================================================

@dataclass
class EvalSplit:
    """A named evaluation split with its generation config."""
    name: str
    description: str
    gen_config: GenerationConfig


def make_iid_split(n: int = 200, seed_offset: int = 100000) -> EvalSplit:
    """In-distribution test set (same param/wind ranges as training)."""
    cfg = GenerationConfig(
        n_trajectories=n,
        dataset_name="eval_iid",
        traj_length_steps=2000,
    )
    return EvalSplit("IID", "In-distribution test", cfg)


def make_ood_param_split(n: int = 200) -> EvalSplit:
    """OOD: parameters perturbed beyond training range."""
    from learning.data_generation.domain_randomization import PARAM_PERTURBATION_SPEC
    ood_spec = {}
    for k, v in PARAM_PERTURBATION_SPEC.items():
        ood_spec[k] = min(v * 1.8, 0.45)  # ~1.8x wider range

    cfg = GenerationConfig(
        n_trajectories=n,
        dataset_name="eval_ood_param",
        traj_length_steps=2000,
        param_perturbation_spec=ood_spec,
    )
    return EvalSplit("OOD-Param", "Out-of-distribution parameters", cfg)


def make_ood_wind_split(n: int = 200) -> EvalSplit:
    """OOD: wind speed beyond training range."""
    cfg = GenerationConfig(
        n_trajectories=n,
        dataset_name="eval_ood_wind",
        traj_length_steps=2000,
        wind_config=WindConfig(
            mode="ar1",
            speed_range=(5.0, 10.0),   # training uses 0-5 m/s
            ar1_sigma=0.6,             # stronger turbulence
        ),
    )
    return EvalSplit("OOD-Wind", "Out-of-distribution wind", cfg)


def generate_eval_splits(output_dir: str = "learning/datasets",
                         base_seed: int = 99999) -> Dict[str, str]:
    """Generate all evaluation datasets. Returns {name: h5_path}."""
    splits = [make_iid_split(), make_ood_param_split(), make_ood_wind_split()]
    paths = {}

    for split in splits:
        split.gen_config.output_dir = output_dir
        gen = DatasetGenerator(split.gen_config)
        print(f"\n{'='*60}")
        print(f"Generating: {split.name} — {split.description}")
        print(f"{'='*60}")
        path = gen.generate(base_seed=base_seed)
        paths[split.name] = path
        base_seed += 10000

    return paths


# ============================================================
#            Autoregressive rollout
# ============================================================

@torch.no_grad()
def autoregressive_rollout(
    model: InContextDynamicsTransformer,
    context_states: np.ndarray,     # (K+H+1, 20)
    context_actions: np.ndarray,    # (K+H, 2)
    norm: NormalizationStats,
    K: int,
    H: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model autoregressively for H steps given K context steps.

    Returns:
      pred_deltas: (H, target_dim) — predicted state deltas
      true_deltas: (H, target_dim) — ground truth state deltas
    """
    model.eval()

    # Encode context
    context_tokens = np.zeros((K, TOKEN_DIM), dtype=np.float32)
    for i in range(K):
        context_tokens[i] = encode_state_token(
            context_states[i], context_actions[i], norm
        )

    ctx_tensor = torch.from_numpy(context_tokens).unsqueeze(0).to(device)
    pred = model(ctx_tensor)  # (1, H, target_dim)
    pred_deltas = pred[0].cpu().numpy()

    # Ground truth deltas
    true_deltas = np.zeros((H, TARGET_DIM), dtype=np.float32)
    for i in range(H):
        si = K + i
        true_deltas[i] = encode_state_delta(
            context_states[si], context_states[si + 1], norm
        )

    return pred_deltas, true_deltas


# ============================================================
#          Evaluation metrics
# ============================================================

def compute_rollout_metrics(
    pred_deltas: np.ndarray,
    true_deltas: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-horizon and aggregate error metrics.

    Returns dict with:
      - rmse_per_step: (H,) RMSE at each horizon
      - rmse_total: scalar
      - mae_total: scalar
      - per_channel_rmse: (target_dim,)
    """
    diff = pred_deltas - true_deltas
    rmse_per_step = np.sqrt((diff ** 2).mean(axis=-1))
    rmse_total = np.sqrt((diff ** 2).mean())
    mae_total = np.abs(diff).mean()
    per_channel_rmse = np.sqrt((diff ** 2).mean(axis=0))

    return {
        "rmse_per_step": rmse_per_step,
        "rmse_total": float(rmse_total),
        "mae_total": float(mae_total),
        "per_channel_rmse": per_channel_rmse,
    }


# ============================================================
#        Full evaluation pipeline
# ============================================================

import h5py

@dataclass
class EvalResult:
    """Container for evaluation results of one split."""
    split_name: str
    context_length: int
    n_samples: int
    rmse_total: float
    mae_total: float
    rmse_per_step: np.ndarray
    per_channel_rmse: np.ndarray


def evaluate_on_split(
    model: InContextDynamicsTransformer,
    h5_path: str,
    norm: NormalizationStats,
    context_length: int,
    prediction_horizon: int,
    device: torch.device,
    max_trajs: int = 100,
    samples_per_traj: int = 5,
) -> EvalResult:
    """Evaluate a model on one dataset split."""
    all_rmse_steps = []
    all_channel_rmse = []
    total_rmse = []
    total_mae = []

    with h5py.File(h5_path, "r") as f:
        traj_grp = f["trajectories"]
        keys = sorted(traj_grp.keys())[:max_trajs]

        for key in keys:
            states = traj_grp[key]["states"][:]
            actions = traj_grp[key]["actions"][:]
            T = len(states)

            min_len = context_length + prediction_horizon + 1
            if T < min_len:
                continue

            for _ in range(samples_per_traj):
                t_start = np.random.randint(0, max(1, T - min_len))
                t_end = t_start + context_length + prediction_horizon + 1

                seg_states = states[t_start:t_end]
                seg_actions = actions[t_start:min(t_end, len(actions))]
                if len(seg_actions) < context_length + prediction_horizon:
                    pad = np.zeros((context_length + prediction_horizon - len(seg_actions), 2))
                    seg_actions = np.vstack([seg_actions, pad])

                pred_d, true_d = autoregressive_rollout(
                    model, seg_states, seg_actions, norm,
                    context_length, prediction_horizon, device
                )
                metrics = compute_rollout_metrics(pred_d, true_d)
                all_rmse_steps.append(metrics["rmse_per_step"])
                all_channel_rmse.append(metrics["per_channel_rmse"])
                total_rmse.append(metrics["rmse_total"])
                total_mae.append(metrics["mae_total"])

    n = len(total_rmse)
    return EvalResult(
        split_name=os.path.basename(h5_path),
        context_length=context_length,
        n_samples=n,
        rmse_total=float(np.mean(total_rmse)) if n > 0 else float('inf'),
        mae_total=float(np.mean(total_mae)) if n > 0 else float('inf'),
        rmse_per_step=np.mean(all_rmse_steps, axis=0) if n > 0 else np.zeros(prediction_horizon),
        per_channel_rmse=np.mean(all_channel_rmse, axis=0) if n > 0 else np.zeros(TARGET_DIM),
    )


def context_length_ablation(
    model: InContextDynamicsTransformer,
    h5_path: str,
    norm: NormalizationStats,
    device: torch.device,
    K_values: List[int] = None,
    prediction_horizon: int = 20,
    max_trajs: int = 50,
) -> Dict[int, EvalResult]:
    """Run evaluation at different context lengths."""
    if K_values is None:
        K_values = [5, 10, 20, 30, 50]

    results = {}
    for K in K_values:
        print(f"  Evaluating K={K}...")
        result = evaluate_on_split(
            model, h5_path, norm, K, prediction_horizon,
            device, max_trajs=max_trajs, samples_per_traj=3,
        )
        results[K] = result
        print(f"    RMSE: {result.rmse_total:.6f}")

    return results


# ============================================================
#                Figure generation
# ============================================================

def plot_all_figures(
    results: Dict[str, Dict[int, EvalResult]],
    output_dir: str = "learning/figures",
    prediction_horizon: int = 20,
):
    """
    Generate publication-quality figures.

    Args:
        results: {split_name: {K: EvalResult}}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
    })

    # --- Figure 1: In-context adaptation curve ---
    fig, ax = plt.subplots()
    for split_name, k_results in results.items():
        Ks = sorted(k_results.keys())
        rmses = [k_results[k].rmse_total for k in Ks]
        ax.plot(Ks, rmses, 'o-', label=split_name, linewidth=2, markersize=6)

    ax.set_xlabel("Context Length K")
    ax.set_ylabel("Rollout RMSE")
    ax.set_title("In-Context Adaptation: RMSE vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_context_adaptation.png"))
    fig.savefig(os.path.join(output_dir, "fig1_context_adaptation.pdf"))
    plt.close(fig)

    # --- Figure 2: Rollout error vs horizon ---
    fig, ax = plt.subplots()
    K_ref = max(list(results.values())[0].keys())  # use largest K
    horizons = np.arange(1, prediction_horizon + 1)

    for split_name, k_results in results.items():
        if K_ref in k_results:
            ax.plot(horizons, k_results[K_ref].rmse_per_step,
                    '-', label=split_name, linewidth=2)

    ax.set_xlabel("Prediction Horizon Step")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Error Growth over Prediction Horizon (K={K_ref})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_horizon_error.png"))
    fig.savefig(os.path.join(output_dir, "fig2_horizon_error.pdf"))
    plt.close(fig)

    # --- Figure 3: Per-channel RMSE heatmap ---
    channel_names = [
        r"$\dot\phi$", r"$\dot\theta$", r"$\dot\psi$",
        r"$\dot\theta_r$", r"$\dot\psi_r$",
        "du", "dv", "dw",
        "dp", "dq", "dr",
        r"$du_p$", r"$dv_p$", r"$dw_p$",
        r"$dp_p$", r"$dq_p$", r"$dr_p$",
    ]
    split_names = list(results.keys())
    K_ref = max(list(results.values())[0].keys())
    data_matrix = np.zeros((len(split_names), TARGET_DIM))

    for i, sn in enumerate(split_names):
        if K_ref in results[sn]:
            data_matrix[i] = results[sn][K_ref].per_channel_rmse

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(TARGET_DIM))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticks(range(len(split_names)))
    ax.set_yticklabels(split_names)
    ax.set_title(f"Per-Channel RMSE (K={K_ref})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_channel_heatmap.png"))
    fig.savefig(os.path.join(output_dir, "fig3_channel_heatmap.pdf"))
    plt.close(fig)

    print(f"Figures saved to {output_dir}/")


# ============================================================
#                    CLI entry point
# ============================================================

def _make_eval_run_dir(base_dir: str, checkpoint_path: str) -> str:
    """
    Create a unique evaluation output directory:
      learning/eval_results/<date>_<checkpoint_name>/
    e.g. learning/eval_results/20260317_035200_best/
    """
    from datetime import datetime

    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT_DIR, base_dir, f"{timestamp}_{ckpt_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate in-context dynamics model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--eval-data-dir", type=str, default="learning/datasets",
                        help="Directory containing eval HDF5 files")
    parser.add_argument("--generate-eval-data", action="store_true",
                        help="Generate evaluation datasets first")
    parser.add_argument("--K-values", type=int, nargs='+',
                        default=[5, 10, 20, 30, 50])
    parser.add_argument("--max-trajs", type=int, default=50)
    parser.add_argument("--output-base-dir", type=str, default="learning/eval_results",
                        help="Base directory for eval outputs (each run gets a subfolder)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create unique output directory for this eval run
    run_dir = _make_eval_run_dir(args.output_base_dir, args.checkpoint)
    print(f"Eval output directory: {run_dir}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt.get("config", {})

    model_cfg = ModelConfig(
        token_dim=TOKEN_DIM,
        target_dim=TARGET_DIM,
        d_model=train_cfg.get("d_model", 128),
        n_heads=train_cfg.get("n_heads", 4),
        n_layers=train_cfg.get("n_layers", 4),
        d_ff=train_cfg.get("d_ff", 512),
        dropout=0.0,
        max_context_length=train_cfg.get("context_length", 50) + 10,
        prediction_horizon=train_cfg.get("prediction_horizon", 20),
        proj_hidden=train_cfg.get("d_model", 128),
    )
    model = InContextDynamicsTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} params")

    # Save run metadata
    run_meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_params": model.count_parameters(),
        "d_model": train_cfg.get("d_model"),
        "n_layers": train_cfg.get("n_layers"),
        "n_heads": train_cfg.get("n_heads"),
        "context_length": train_cfg.get("context_length"),
        "prediction_horizon": train_cfg.get("prediction_horizon"),
        "train_epoch": ckpt.get("epoch"),
        "train_best_val_loss": ckpt.get("best_val_loss"),
        "K_values": args.K_values,
        "max_trajs": args.max_trajs,
    }
    with open(os.path.join(run_dir, "run_info.json"), 'w') as f:
        json.dump(run_meta, f, indent=2)

    # Load normalization stats
    norm_path = train_cfg.get("norm_stats_path", "learning/datasets/pilot_norm.npz")
    if not os.path.isabs(norm_path):
        norm_path = os.path.join(ROOT_DIR, norm_path)
    norm = NormalizationStats.load(norm_path)

    # Generate eval data if needed
    if args.generate_eval_data:
        eval_paths = generate_eval_splits(args.eval_data_dir)
    else:
        eval_paths = {}
        for name, fname in [("IID", "eval_iid.h5"),
                            ("OOD-Param", "eval_ood_param.h5"),
                            ("OOD-Wind", "eval_ood_wind.h5")]:
            p = os.path.join(ROOT_DIR, args.eval_data_dir, fname)
            if os.path.exists(p):
                eval_paths[name] = p

    if not eval_paths:
        print("No eval datasets found. Run with --generate-eval-data first.")
        return

    # Context-length ablation on each split
    H = train_cfg.get("prediction_horizon", 20)
    all_results = {}

    for split_name, h5_path in eval_paths.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {split_name}")
        print(f"{'='*50}")
        k_results = context_length_ablation(
            model, h5_path, norm, device,
            K_values=args.K_values,
            prediction_horizon=H,
            max_trajs=args.max_trajs,
        )
        all_results[split_name] = k_results

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for split_name, k_results in all_results.items():
        print(f"\n{split_name}:")
        for K, result in sorted(k_results.items()):
            print(f"  K={K:3d}: RMSE={result.rmse_total:.6f}, "
                  f"MAE={result.mae_total:.6f} (n={result.n_samples})")

    # Save metrics JSON
    metrics_path = os.path.join(run_dir, "eval_metrics.json")
    metrics_json = {}
    for sn, kr in all_results.items():
        metrics_json[sn] = {}
        for K, r in kr.items():
            metrics_json[sn][str(K)] = {
                "rmse_total": r.rmse_total,
                "mae_total": r.mae_total,
                "rmse_per_step": r.rmse_per_step.tolist(),
                "per_channel_rmse": r.per_channel_rmse.tolist(),
            }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate figures into the run directory
    try:
        plot_all_figures(all_results, run_dir, H)
    except ImportError:
        print("matplotlib not available, skipping figure generation")

    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
