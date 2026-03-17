"""
Training loop for the in-context dynamics transformer.

Features:
  - Multi-step prediction loss with exponential decay weights
  - Per-channel loss weighting (angular velocity > velocity > angles)
  - Curriculum learning on prediction horizon
  - Periodic autoregressive validation
  - Checkpoint saving / resumption
  - TensorBoard logging (optional)
  - wandb logging (optional)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from learning.model import InContextDynamicsTransformer, ModelConfig
from learning.dataset import (
    NormalizationStats, compute_normalization_stats, create_dataloaders,
    BatchEncoder, TOKEN_DIM, TARGET_DIM,
    WIND_LABEL_DIM, PARAM_LABEL_DIM,
)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    h5_path: str = "learning/datasets/pilot.h5"
    norm_stats_path: str = "learning/datasets/pilot_norm.npz"

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    context_length: int = 50
    prediction_horizon: int = 20

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    warmup_steps: int = 500
    grad_clip: float = 1.0
    use_amp: bool = True

    # Loss
    horizon_decay: float = 0.95    # lambda_i = decay^i
    loss_type: str = "huber"       # "huber" or "mse"
    huber_delta: float = 1.0

    # Channel weights for loss (17D target)
    # euler(3), rel_angles(2), vel_canopy(3), omega_canopy(3),
    # vel_payload(3), omega_payload(3)
    channel_weights: Optional[List[float]] = None

    # Rollout loss (autoregressive)
    rollout_loss_enabled: bool = False
    rollout_weight: float = 0.5
    rollout_warmup_epochs: int = 10
    rollout_steps: int = 0

    # Auxiliary task-identification loss
    aux_loss_enabled: bool = False
    aux_loss_weight: float = 0.1
    aux_predict_wind: bool = True
    aux_predict_params: bool = False

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_start_H: int = 1
    curriculum_epoch_per_step: int = 5  # increase H every N epochs

    # Validation
    val_every_n_epochs: int = 1
    samples_per_traj: int = 10
    train_ratio: float = 0.9
    num_workers: int = 4
    preload_dataset: bool = True  # load split into RAM; faster, uses more memory

    # Checkpointing
    checkpoint_dir: str = "learning/checkpoints"
    save_every_n_epochs: int = 10
    resume_from: Optional[str] = None

    # Logging
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "learning/runs"
    use_wandb: bool = False
    wandb_project: str = "parafoil-dynamics"
    wandb_run_name: Optional[str] = None


class MultiStepLoss(nn.Module):
    """
    Multi-step prediction loss with horizon decay and channel weighting.
    """

    def __init__(self, horizon: int, decay: float = 0.95,
                 loss_type: str = "huber", huber_delta: float = 1.0,
                 channel_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.horizon = horizon
        self.decay = decay
        self.channel_weights = channel_weights

        if loss_type == "huber":
            self.base_loss = nn.HuberLoss(reduction='none', delta=huber_delta)
        else:
            self.base_loss = nn.MSELoss(reduction='none')

        # Pre-compute horizon weights
        weights = torch.tensor([decay ** i for i in range(horizon)])
        weights = weights / weights.sum()
        self.register_buffer('horizon_weights', weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                active_horizon: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, target_dim)
            target: (B, H, target_dim)
            active_horizon: if set, only use first active_horizon steps

        Returns:
            scalar loss
        """
        H = active_horizon or self.horizon
        pred = pred[:, :H, :]
        target = target[:, :H, :]

        # Per-element loss: (B, H, target_dim)
        loss = self.base_loss(pred, target)

        # Channel weighting
        if self.channel_weights is not None:
            cw = self.channel_weights.to(loss.device)
            loss = loss * cw.unsqueeze(0).unsqueeze(0)

        # Average over channels, then apply horizon weights
        loss = loss.mean(dim=-1)  # (B, H)
        hw = self.horizon_weights[:H].to(loss.device)
        hw = hw / hw.sum()
        loss = (loss * hw.unsqueeze(0)).sum(dim=-1)  # (B,)

        return loss.mean()


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Main training orchestrator."""

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve paths
        self.h5_path = os.path.join(ROOT_DIR, config.h5_path)
        self.norm_path = os.path.join(ROOT_DIR, config.norm_stats_path)
        self.ckpt_dir = os.path.join(ROOT_DIR, config.checkpoint_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Normalization stats
        if os.path.exists(self.norm_path):
            self.norm_stats = NormalizationStats.load(self.norm_path)
            print(f"Loaded normalization stats from {self.norm_path}")
        else:
            print("Computing normalization statistics...")
            self.norm_stats = compute_normalization_stats(self.h5_path)
            self.norm_stats.save(self.norm_path)
            print(f"Saved normalization stats to {self.norm_path}")

        # Auxiliary loss dimensions
        self.aux_dim = 0
        if config.aux_loss_enabled:
            if config.aux_predict_wind:
                self.aux_dim += WIND_LABEL_DIM
            if config.aux_predict_params:
                self.aux_dim += PARAM_LABEL_DIM

        # Data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.h5_path, self.norm_stats,
            context_length=config.context_length,
            prediction_horizon=config.prediction_horizon,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            num_workers=config.num_workers,
            samples_per_traj=config.samples_per_traj,
            load_aux_labels=config.aux_loss_enabled,
            preload=config.preload_dataset,
        )
        tr_ds, va_ds = self.train_loader.dataset, self.val_loader.dataset
        print(f"Train samples: {len(tr_ds)}, Val samples: {len(va_ds)}")
        if config.preload_dataset and getattr(tr_ds, "_states_list", None):
            def _ram_mb(ds):
                if not ds._states_list:
                    return 0.0
                s = sum(a.nbytes for a in ds._states_list)
                a = sum(a.nbytes for a in ds._actions_list)
                w = sum(a.nbytes for a in ds._winds_list) if ds._winds_list else 0
                return (s + a + w) / (1024 ** 2)
            print(f"  Dataset in RAM ≈ train {_ram_mb(tr_ds):.0f} MB + val {_ram_mb(va_ds):.0f} MB")

        # Model
        model_cfg = ModelConfig(
            token_dim=TOKEN_DIM,
            target_dim=TARGET_DIM,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_context_length=config.context_length + 10,
            prediction_horizon=config.prediction_horizon,
            proj_hidden=config.d_model,
        )
        self.model = InContextDynamicsTransformer(
            model_cfg, aux_dim=self.aux_dim
        ).to(self.device)
        n_params = self.model.count_parameters()
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        if config.aux_loss_enabled:
            print(f"  Aux head: dim={self.aux_dim} "
                  f"(wind={config.aux_predict_wind}, "
                  f"params={config.aux_predict_params})")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        total_steps = config.max_epochs * len(self.train_loader)
        self.scheduler = get_lr_scheduler(
            self.optimizer, config.warmup_steps, total_steps
        )

        # Loss
        channel_weights = None
        if config.channel_weights:
            channel_weights = torch.tensor(config.channel_weights, dtype=torch.float32)
        else:
            # Default weights by control importance:
            # dφ(1.5) dθ(1.5) dψ(3.0) | dθr(1.0) dψr(1.0) |
            # du(2.5) dv(2.5) dw(2.5) | dp(2.0) dq(2.0) dr(2.0) |
            # du_p(1.5) dv_p(1.5) dw_p(1.5) | dp_p(1.0) dq_p(1.0) dr_p(1.0)
            channel_weights = torch.tensor(
                [1.5, 1.5, 3.0,  1.0, 1.0,  2.5, 2.5, 2.5,
                 2.0, 2.0, 2.0,  1.5, 1.5, 1.5,  1.0, 1.0, 1.0],
                dtype=torch.float32
            )
        self.loss_fn = MultiStepLoss(
            config.prediction_horizon,
            decay=config.horizon_decay,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            channel_weights=channel_weights,
        ).to(self.device)

        # AMP
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            print("Mixed precision (AMP) enabled")

        # GPU batch encoder (replaces per-sample NumPy loops)
        self.encoder = BatchEncoder(self.norm_stats, self.device)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        # TensorBoard
        self.tb_writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(ROOT_DIR, config.tensorboard_log_dir)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard 日志目录: {log_dir}")
                print(f"  运行: tensorboard --logdir={log_dir}")
            except ImportError:
                print("tensorboard 未安装，跳过。运行: pip install tensorboard")

        # wandb
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=asdict(config),
                )
            except ImportError:
                print("wandb not installed, skipping logging")

    def train(self):
        """Run the full training loop."""
        cfg = self.cfg
        print(f"\nStarting training on {self.device}")
        print(f"  Epochs: {cfg.max_epochs}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Context K={cfg.context_length}, Horizon H={cfg.prediction_horizon}")
        if cfg.curriculum_enabled:
            print(f"  Curriculum: H starts at {cfg.curriculum_start_H}, "
                  f"+1 every {cfg.curriculum_epoch_per_step} epochs")
        if cfg.rollout_loss_enabled:
            print(f"  Rollout loss: weight={cfg.rollout_weight}, "
                  f"warmup={cfg.rollout_warmup_epochs} epochs")
        if cfg.aux_loss_enabled:
            print(f"  Aux loss: weight={cfg.aux_loss_weight}")
        print()

        epoch_pbar = tqdm(range(self.epoch, cfg.max_epochs),
                          desc="Training", unit="epoch",
                          initial=self.epoch, total=cfg.max_epochs)
        for epoch in epoch_pbar:
            self.epoch = epoch

            # Curriculum: determine active horizon
            if cfg.curriculum_enabled:
                active_H = min(
                    cfg.curriculum_start_H + epoch // cfg.curriculum_epoch_per_step,
                    cfg.prediction_horizon
                )
            else:
                active_H = cfg.prediction_horizon

            # Train
            train_stats = self._train_epoch(active_H)
            train_loss = train_stats["total"]

            # Validate
            val_loss = None
            curriculum_complete = (active_H >= cfg.prediction_horizon)
            if (epoch + 1) % cfg.val_every_n_epochs == 0:
                val_loss = self._validate(active_H)

                if curriculum_complete and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
                    tqdm.write(f"  ** New best val loss: {val_loss:.6f}")

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            postfix = {"H": active_H, "train": f"{train_loss:.5f}", "lr": f"{lr:.2e}"}
            if val_loss is not None:
                postfix["val"] = f"{val_loss:.5f}"
            epoch_pbar.set_postfix(postfix)

            if self.tb_writer:
                ep = epoch + 1
                gs = self.global_step
                self.tb_writer.add_scalar("train/loss", train_loss, gs)
                self.tb_writer.add_scalar("train/lr", lr, gs)
                self.tb_writer.add_scalar("train/active_H", active_H, gs)
                if val_loss is not None:
                    self.tb_writer.add_scalar("val/loss", val_loss, gs)
                # 分项（step 轴，每 epoch 一点）
                self.tb_writer.add_scalar("train/loss_tf", train_stats["tf"], gs)
                self.tb_writer.add_scalar(
                    "train/loss_rollout", train_stats["rollout"], gs
                )
                self.tb_writer.add_scalar("train/loss_aux", train_stats["aux"], gs)
                self.tb_writer.add_scalar(
                    "train/loss_aux_weighted", train_stats["aux_weighted"], gs
                )
                # 按 epoch 横轴 1,2,3…
                self.tb_writer.add_scalar("epoch/loss_total", train_loss, ep)
                self.tb_writer.add_scalar("epoch/loss_tf", train_stats["tf"], ep)
                self.tb_writer.add_scalar(
                    "epoch/loss_rollout", train_stats["rollout"], ep
                )
                self.tb_writer.add_scalar("epoch/loss_aux", train_stats["aux"], ep)
                self.tb_writer.add_scalar(
                    "epoch/loss_aux_weighted", train_stats["aux_weighted"], ep
                )
                self.tb_writer.add_scalar("epoch/lr", lr, ep)
                self.tb_writer.add_scalar("epoch/active_H", float(active_H), ep)
                if val_loss is not None:
                    self.tb_writer.add_scalar("epoch/val_loss", val_loss, ep)

            if self.wandb_run:
                log = {
                    "train/loss": train_loss,
                    "train/loss_tf": train_stats["tf"],
                    "train/loss_rollout": train_stats["rollout"],
                    "train/loss_aux": train_stats["aux"],
                    "train/loss_aux_weighted": train_stats["aux_weighted"],
                    "train/lr": lr,
                    "train/active_H": active_H,
                    "epoch": epoch,
                }
                if val_loss is not None:
                    log["val/loss"] = val_loss
                self.wandb_run.log(log, step=self.global_step)

            # Checkpoint
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1:04d}.pt")

        # Final save
        self._save_checkpoint("final.pt")
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")

    def _build_aux_label(self, batch: dict) -> torch.Tensor:
        """Concatenate wind and/or param labels into a single target vector."""
        parts = []
        if self.cfg.aux_predict_wind and "wind_label" in batch:
            parts.append(batch["wind_label"])
        if self.cfg.aux_predict_params and "param_label" in batch:
            parts.append(batch["param_label"])
        return torch.cat(parts, dim=-1).to(self.device) if parts else None

    def _train_epoch(self, active_H: int) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        sum_tf = 0.0
        sum_rollout = 0.0
        sum_aux = 0.0
        n_batches = 0
        cfg = self.cfg

        use_rollout = (
            cfg.rollout_loss_enabled
            and self.epoch >= cfg.rollout_warmup_epochs
            and active_H >= cfg.prediction_horizon
        )
        use_aux = cfg.aux_loss_enabled and self.aux_dim > 0

        K = cfg.context_length
        batch_pbar = tqdm(self.train_loader, desc=f"  Train E{self.epoch+1}",
                          leave=False, unit="batch")
        for batch in batch_pbar:
            raw_s = batch["raw_states"].to(self.device, non_blocking=True)
            raw_a = batch["raw_actions"].to(self.device, non_blocking=True)

            context = self.encoder.encode_tokens(raw_s[:, :K], raw_a[:, :K])
            target = self.encoder.encode_deltas(
                raw_s[:, K:K+active_H], raw_s[:, K+1:K+active_H+1]
            )

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if use_aux:
                    pred, aux_pred = self.model(context, return_aux=True)
                    aux_label = self._build_aux_label(batch)
                    aux_loss = nn.functional.mse_loss(aux_pred, aux_label)
                    sum_aux += float(aux_loss.detach())
                else:
                    pred = self.model(context)
                    aux_loss = torch.zeros((), device=self.device)

                tf_loss = self.loss_fn(pred, target, active_horizon=active_H)

                if use_rollout:
                    r_loss = self._compute_rollout_loss(
                        raw_s, raw_a, context, target, active_H
                    )
                    sum_rollout += float(r_loss.detach())
                    alpha = cfg.rollout_weight
                    loss = (1 - alpha) * tf_loss + alpha * r_loss
                else:
                    loss = tf_loss

                loss = loss + cfg.aux_loss_weight * aux_loss

            sum_tf += float(tf_loss.detach())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.5f}")

        n = max(1, n_batches)
        avg_aux = sum_aux / n
        return {
            "total": total_loss / n,
            "tf": sum_tf / n,
            "rollout": sum_rollout / n if use_rollout else 0.0,
            "aux": avg_aux if use_aux else 0.0,
            "aux_weighted": (cfg.aux_loss_weight * avg_aux) if use_aux else 0.0,
        }

    @torch.no_grad()
    def _validate(self, active_H: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        K = self.cfg.context_length

        val_pbar = tqdm(self.val_loader, desc=f"  Val   E{self.epoch+1}",
                        leave=False, unit="batch")
        for batch in val_pbar:
            raw_s = batch["raw_states"].to(self.device, non_blocking=True)
            raw_a = batch["raw_actions"].to(self.device, non_blocking=True)

            context = self.encoder.encode_tokens(raw_s[:, :K], raw_a[:, :K])
            target = self.encoder.encode_deltas(
                raw_s[:, K:K+active_H], raw_s[:, K+1:K+active_H+1]
            )

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(context)
                loss = self.loss_fn(pred, target, active_horizon=active_H)

            total_loss += loss.item()
            n_batches += 1
            val_pbar.set_postfix(loss=f"{total_loss/n_batches:.5f}")

        return total_loss / max(1, n_batches)

    def _compute_rollout_loss(self, raw_states: torch.Tensor,
                              raw_actions: torch.Tensor,
                              context: torch.Tensor,
                              target: torch.Tensor,
                              active_H: int) -> torch.Tensor:
        """
        Autoregressive rollout loss — fully on GPU, no Python loops over batch.
        """
        B, K, _ = context.shape
        rollout_H = min(active_H, self.cfg.rollout_steps or active_H)

        pred_deltas = []
        current_context = context.clone()
        current_state = raw_states[:, K, :].clone()  # (B, 20)

        for h in range(rollout_H):
            pred_delta_norm = self.model.predict_single_step(current_context)
            pred_deltas.append(pred_delta_norm)

            pred_delta_raw = pred_delta_norm * self.encoder.target_std
            next_state = current_state.clone()
            next_state[:, 3:20] = next_state[:, 3:20] + pred_delta_raw

            action_h = raw_actions[:, K + h, :]  # (B, 2)
            new_token = self.encoder.encode_tokens(next_state, action_h)  # (B, TOKEN_DIM)

            current_context = torch.cat([
                current_context[:, 1:, :],
                new_token.unsqueeze(1),
            ], dim=1)

            current_state = next_state

        pred_stack = torch.stack(pred_deltas, dim=1)
        target_slice = target[:, :rollout_H, :]

        return self.loss_fn(pred_stack, target_slice, active_horizon=rollout_H)

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg) if hasattr(self.cfg, '__dataclass_fields__') else vars(self.cfg),
        }, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Resumed from {path}, epoch={self.epoch}, "
              f"step={self.global_step}")


# ============================================================
#                      CLI entry point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train in-context dynamics model")
    parser.add_argument("--h5-path", type=str, default="learning/datasets/pilot.h5")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=50)
    parser.add_argument("--prediction-horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    cfg = TrainConfig(
        h5_path=args.h5_path,
        norm_stats_path=args.h5_path.replace(".h5", "_norm.npz"),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        dropout=args.dropout,
        curriculum_enabled=args.curriculum,
        resume_from=args.resume,
        use_wandb=args.wandb,
        wandb_run_name=args.run_name,
    )

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
