"""
PyTorch Dataset for in-context dynamics learning.

Loads HDF5 trajectories and provides (context, target) pairs with:
  - State normalization (per-channel)
  - Sin/cos encoding for Euler angles
  - Relative position (delta) instead of absolute
  - Configurable context length K and prediction horizon H
"""

import os
import json
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NormalizationStats:
    """Per-channel mean and std for state normalization."""
    state_mean: np.ndarray   # (20,)
    state_std: np.ndarray    # (20,)
    action_mean: np.ndarray  # (2,)
    action_std: np.ndarray   # (2,)

    def save(self, path: str):
        np.savez(path,
                 state_mean=self.state_mean, state_std=self.state_std,
                 action_mean=self.action_mean, action_std=self.action_std)

    @classmethod
    def load(cls, path: str) -> 'NormalizationStats':
        d = np.load(path)
        return cls(d['state_mean'], d['state_std'],
                   d['action_mean'], d['action_std'])


def compute_normalization_stats(h5_path: str,
                                max_trajs: int = 500) -> NormalizationStats:
    """Compute running mean/std from an HDF5 dataset."""
    state_sum = np.zeros(20, dtype=np.float64)
    state_sq_sum = np.zeros(20, dtype=np.float64)
    action_sum = np.zeros(2, dtype=np.float64)
    action_sq_sum = np.zeros(2, dtype=np.float64)
    total = 0

    with h5py.File(h5_path, "r") as f:
        traj_grp = f["trajectories"]
        keys = sorted(traj_grp.keys())[:max_trajs]
        for k in keys:
            states = traj_grp[k]["states"][:]
            actions = traj_grp[k]["actions"][:]
            n = len(states)
            state_sum += states.sum(axis=0)
            state_sq_sum += (states ** 2).sum(axis=0)
            action_sum += actions.sum(axis=0)
            action_sq_sum += (actions ** 2).sum(axis=0)
            total += n

    state_mean = state_sum / total
    state_std = np.sqrt(state_sq_sum / total - state_mean ** 2)
    state_std = np.maximum(state_std, 1e-6)

    action_mean = action_sum / total
    action_std = np.sqrt(action_sq_sum / total - action_mean ** 2)
    action_std = np.maximum(action_std, 1e-6)

    return NormalizationStats(
        state_mean.astype(np.float32),
        state_std.astype(np.float32),
        action_mean.astype(np.float32),
        action_std.astype(np.float32),
    )


def encode_state_token(state: np.ndarray,
                       action: np.ndarray,
                       norm: Optional[NormalizationStats] = None,
                       include_altitude: bool = True) -> np.ndarray:
    """
    Encode a single timestep into a token vector.

    Input state (20D) -> token (22-23D):
      sin(phi), cos(phi), sin(theta), cos(theta), sin(psi), cos(psi),  # 6
      theta_r, psi_r,                                                    # 2
      u, v, w,                                                           # 3
      p, q, r,                                                           # 3
      u_p, v_p, w_p,                                                     # 3
      p_p, q_p, r_p,                                                     # 3
      delta_left, delta_right,                                           # 2
      [altitude]                                                         # 0 or 1
    Total: 22 or 23
    """
    phi, theta, psi = state[3], state[4], state[5]
    theta_r, psi_r = state[6], state[7]
    vel_canopy = state[8:11]
    omega_canopy = state[11:14]
    vel_payload = state[14:17]
    omega_payload = state[17:20]

    # Normalize velocities and angular velocities
    if norm is not None:
        vel_canopy = (vel_canopy - norm.state_mean[8:11]) / norm.state_std[8:11]
        omega_canopy = (omega_canopy - norm.state_mean[11:14]) / norm.state_std[11:14]
        vel_payload = (vel_payload - norm.state_mean[14:17]) / norm.state_std[14:17]
        omega_payload = (omega_payload - norm.state_mean[17:20]) / norm.state_std[17:20]
        theta_r = (theta_r - norm.state_mean[6]) / norm.state_std[6]
        psi_r = (psi_r - norm.state_mean[7]) / norm.state_std[7]
        action = (action - norm.action_mean) / norm.action_std

    parts = [
        np.array([np.sin(phi), np.cos(phi),
                  np.sin(theta), np.cos(theta),
                  np.sin(psi), np.cos(psi)]),
        np.array([theta_r, psi_r]),
        vel_canopy,
        omega_canopy,
        vel_payload,
        omega_payload,
        action,
    ]
    if include_altitude:
        alt_norm = state[2] / 1000.0  # scale to ~O(1)
        parts.append(np.array([alt_norm]))

    return np.concatenate(parts).astype(np.float32)


def encode_state_delta(state_curr: np.ndarray,
                       state_next: np.ndarray,
                       norm: Optional[NormalizationStats] = None) -> np.ndarray:
    """
    Compute prediction target: state delta for dynamics-relevant channels.

    Returns (17D):
      d_euler (3), d_rel_angles (2), d_vel_canopy (3), d_omega_canopy (3),
      d_vel_payload (3), d_omega_payload (3)

    Position delta is NOT included (it's an integral, not dynamics).
    """
    delta = state_next - state_curr

    # Wrap angle deltas
    for i in [3, 4, 5, 6, 7]:
        delta[i] = _wrap_angle_delta(state_curr[i], state_next[i])

    target = np.concatenate([
        delta[3:20]  # euler(3) + rel_angles(2) + velocities(12)
    ])

    if norm is not None:
        target_std = np.concatenate([
            norm.state_std[3:20]
        ])
        target = target / target_std

    return target.astype(np.float32)


def _wrap_angle_delta(a, b):
    d = b - a
    return (d + np.pi) % (2 * np.pi) - np.pi


# ============================================================
#                  Token dimensions
# ============================================================

TOKEN_DIM = 23      # with altitude
TARGET_DIM = 17     # euler(3) + rel_angles(2) + velocities(12)


# ============================================================
#                  PyTorch Dataset
# ============================================================

class ParafoilDynamicsDataset(Dataset):
    """
    Dataset that yields (context, target) pairs from HDF5 trajectories.

    context: (K, token_dim) — past K steps of encoded (state, action)
    target:  (H, target_dim) — future H steps of state deltas
    """

    def __init__(self, h5_path: str,
                 context_length: int = 50,
                 prediction_horizon: int = 20,
                 norm_stats: Optional[NormalizationStats] = None,
                 max_trajs: Optional[int] = None,
                 samples_per_traj: int = 10):
        self.h5_path = h5_path
        self.K = context_length
        self.H = prediction_horizon
        self.norm = norm_stats
        self.samples_per_traj = samples_per_traj

        # Load trajectory metadata
        self._traj_keys = []
        self._traj_lengths = []
        with h5py.File(h5_path, "r") as f:
            traj_grp = f["trajectories"]
            keys = sorted(traj_grp.keys())
            if max_trajs is not None:
                keys = keys[:max_trajs]
            for k in keys:
                length = traj_grp[k].attrs["valid_steps"]
                min_len = self.K + self.H + 1
                if length >= min_len:
                    self._traj_keys.append(k)
                    self._traj_lengths.append(length)

        self._n_trajs = len(self._traj_keys)
        self._total_samples = self._n_trajs * samples_per_traj

        # Pre-compute valid start ranges
        self._max_starts = [l - self.K - self.H for l in self._traj_lengths]

        # HDF5 file handle (opened lazily per worker)
        self._h5 = None

    def __len__(self):
        return self._total_samples

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()

        traj_idx = idx // self.samples_per_traj
        key = self._traj_keys[traj_idx]
        grp = self._h5["trajectories"][key]

        # Random start within valid range
        max_start = self._max_starts[traj_idx]
        t_start = np.random.randint(0, max(1, max_start))

        # Load slice
        t_end_ctx = t_start + self.K
        t_end_pred = t_end_ctx + self.H

        states = grp["states"][t_start:t_end_pred + 1]
        actions = grp["actions"][t_start:t_end_pred]

        # Encode context tokens
        context = np.zeros((self.K, TOKEN_DIM), dtype=np.float32)
        for i in range(self.K):
            context[i] = encode_state_token(
                states[i], actions[i], self.norm
            )

        # Encode prediction targets (state deltas)
        target = np.zeros((self.H, TARGET_DIM), dtype=np.float32)
        for i in range(self.H):
            si = self.K + i
            target[i] = encode_state_delta(
                states[si], states[si + 1], self.norm
            )

        # Also provide the "current state" at prediction start for autoregressive rollout
        current_state = encode_state_token(
            states[self.K], actions[self.K] if self.K < len(actions) else actions[-1],
            self.norm
        )

        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "current_state": torch.from_numpy(current_state),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def create_dataloaders(
    h5_path: str,
    norm_stats: NormalizationStats,
    context_length: int = 50,
    prediction_horizon: int = 20,
    batch_size: int = 64,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    samples_per_traj: int = 10,
):
    """Create train/val DataLoaders from a single HDF5 file."""
    # Count total trajectories
    with h5py.File(h5_path, "r") as f:
        n_total = len(f["trajectories"])

    n_train = int(n_total * train_ratio)

    train_ds = ParafoilDynamicsDataset(
        h5_path, context_length, prediction_horizon,
        norm_stats, max_trajs=n_train,
        samples_per_traj=samples_per_traj,
    )
    val_ds = ParafoilDynamicsDataset(
        h5_path, context_length, prediction_horizon,
        norm_stats, max_trajs=n_total,
        samples_per_traj=max(1, samples_per_traj // 2),
    )
    # Offset val to use trajectories after train split
    val_ds._traj_keys = val_ds._traj_keys[n_train:]
    val_ds._traj_lengths = val_ds._traj_lengths[n_train:]
    val_ds._max_starts = val_ds._max_starts[n_train:]
    val_ds._n_trajs = len(val_ds._traj_keys)
    val_ds._total_samples = val_ds._n_trajs * val_ds.samples_per_traj

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
