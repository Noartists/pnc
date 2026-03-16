"""
Training loop for the in-context dynamics transformer.

Features:
  - Multi-step prediction loss with exponential decay weights
  - Per-channel loss weighting (angular velocity > velocity > angles)
  - Curriculum learning on prediction horizon
  - Periodic autoregressive validation
  - Checkpoint saving / resumption
  - wandb logging (optional)
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from learning.model import InContextDynamicsTransformer, ModelConfig
from learning.dataset import (
    ParafoilDynamicsDataset, NormalizationStats,
    compute_normalization_stats, create_dataloaders,
    TOKEN_DIM, TARGET_DIM,
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

    # Loss
    horizon_decay: float = 0.95    # lambda_i = decay^i
    loss_type: str = "huber"       # "huber" or "mse"
    huber_delta: float = 1.0

    # Channel weights for loss (17D target)
    # euler(3), rel_angles(2), vel_canopy(3), omega_canopy(3),
    # vel_payload(3), omega_payload(3)
    channel_weights: Optional[List[float]] = None

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_start_H: int = 1
    curriculum_epoch_per_step: int = 5  # increase H every N epochs

    # Validation
    val_every_n_epochs: int = 1
    samples_per_traj: int = 10
    train_ratio: float = 0.9
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "learning/checkpoints"
    save_every_n_epochs: int = 10
    resume_from: Optional[str] = None

    # Logging
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

        # Data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.h5_path, self.norm_stats,
            context_length=config.context_length,
            prediction_horizon=config.prediction_horizon,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            num_workers=config.num_workers,
            samples_per_traj=config.samples_per_traj,
        )
        print(f"Train samples: {len(self.train_loader.dataset)}, "
              f"Val samples: {len(self.val_loader.dataset)}")

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
        self.model = InContextDynamicsTransformer(model_cfg).to(self.device)
        n_params = self.model.count_parameters()
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

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
            # Default: upweight angular velocities
            # euler(3), rel(2), v_c(3), w_c(3), v_p(3), w_p(3) = 17
            channel_weights = torch.tensor(
                [1.0]*3 + [1.0]*2 + [1.0]*3 + [2.0]*3 + [1.0]*3 + [2.0]*3,
                dtype=torch.float32
            )
        self.loss_fn = MultiStepLoss(
            config.prediction_horizon,
            decay=config.horizon_decay,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            channel_weights=channel_weights,
        ).to(self.device)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

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
        print()

        for epoch in range(self.epoch, cfg.max_epochs):
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
            train_loss = self._train_epoch(active_H)

            # Validate
            val_loss = None
            if (epoch + 1) % cfg.val_every_n_epochs == 0:
                val_loss = self._validate(active_H)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch+1:3d}/{cfg.max_epochs} | "
                   f"H={active_H:2d} | "
                   f"train_loss={train_loss:.6f} | "
                   f"lr={lr:.2e}")
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.6f}"
            print(msg)

            if self.wandb_run:
                log = {"train/loss": train_loss, "train/lr": lr,
                       "train/active_H": active_H, "epoch": epoch}
                if val_loss is not None:
                    log["val/loss"] = val_loss
                self.wandb_run.log(log, step=self.global_step)

            # Checkpoint
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1:04d}.pt")

        # Final save
        self._save_checkpoint("final.pt")
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")

    def _train_epoch(self, active_H: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)

            pred = self.model(context)
            loss = self.loss_fn(pred, target, active_horizon=active_H)

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _validate(self, active_H: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)

            pred = self.model(context)
            loss = self.loss_fn(pred, target, active_horizon=active_H)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

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
