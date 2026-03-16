"""
Unified training entry point.

Usage:
  python -m learning.train --config learning/configs/pilot.yaml
  python -m learning.train --config learning/configs/full.yaml
  python -m learning.train --config learning/configs/full.yaml --resume learning/checkpoints/best.pt
"""

import os
import sys
import argparse
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from learning.trainer import Trainer, TrainConfig


def load_config_from_yaml(yaml_path: str, overrides: dict) -> TrainConfig:
    """Load TrainConfig from YAML with CLI overrides."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    loss = cfg.get("loss", {})

    tc = TrainConfig(
        h5_path=data.get("h5_path", "learning/datasets/pilot.h5"),
        norm_stats_path=data.get("h5_path", "learning/datasets/pilot.h5").replace(".h5", "_norm.npz"),
        d_model=model.get("d_model", 128),
        n_heads=model.get("n_heads", 4),
        n_layers=model.get("n_layers", 4),
        d_ff=model.get("d_ff", 512),
        dropout=model.get("dropout", 0.1),
        context_length=model.get("context_length", 50),
        prediction_horizon=model.get("prediction_horizon", 20),
        batch_size=training.get("batch_size", 64),
        learning_rate=training.get("learning_rate", 3e-4),
        weight_decay=training.get("weight_decay", 1e-4),
        max_epochs=training.get("max_epochs", 100),
        warmup_steps=training.get("warmup_steps", 500),
        grad_clip=training.get("grad_clip", 1.0),
        curriculum_enabled=training.get("curriculum_enabled", True),
        curriculum_start_H=training.get("curriculum_start_H", 1),
        curriculum_epoch_per_step=training.get("curriculum_epoch_per_step", 5),
        horizon_decay=loss.get("horizon_decay", 0.95),
        loss_type=loss.get("loss_type", "huber"),
        huber_delta=loss.get("huber_delta", 1.0),
    )

    # CLI overrides
    if overrides.get("resume"):
        tc.resume_from = overrides["resume"]
    if overrides.get("wandb"):
        tc.use_wandb = True
    if overrides.get("run_name"):
        tc.wandb_run_name = overrides["run_name"]
    if overrides.get("epochs"):
        tc.max_epochs = overrides["epochs"]

    return tc


def main():
    parser = argparse.ArgumentParser(description="Train in-context dynamics model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()

    overrides = {
        "resume": args.resume,
        "wandb": args.wandb,
        "run_name": args.run_name,
        "epochs": args.epochs,
    }

    cfg = load_config_from_yaml(args.config, overrides)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
