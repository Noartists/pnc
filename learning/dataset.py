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
from typing import Optional, Tuple, Dict, List, Sequence
from dataclasses import dataclass
from tqdm import tqdm


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
        for k in tqdm(keys, desc="Computing norm stats", leave=False, unit="traj"):
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
#       GPU-batched encoding (replaces per-sample NumPy loops)
# ============================================================

class BatchEncoder:
    """Vectorized state/delta encoding on GPU. Zero Python loops."""

    def __init__(self, norm: NormalizationStats, device: torch.device):
        self.device = device
        self.s_mean = torch.from_numpy(norm.state_mean).float().to(device)
        self.s_std = torch.from_numpy(norm.state_std).float().to(device)
        self.a_mean = torch.from_numpy(norm.action_mean).float().to(device)
        self.a_std = torch.from_numpy(norm.action_std).float().to(device)
        self.target_std = self.s_std[3:20]

    def encode_tokens(self, states: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states:  (..., 20)   on device
            actions: (..., 2)    on device
        Returns:
            tokens:  (..., 23)
        """
        phi   = states[..., 3]
        theta = states[..., 4]
        psi   = states[..., 5]

        theta_r = (states[..., 6:7] - self.s_mean[6]) / self.s_std[6]
        psi_r   = (states[..., 7:8] - self.s_mean[7]) / self.s_std[7]

        vel_c  = (states[..., 8:11]  - self.s_mean[8:11])  / self.s_std[8:11]
        om_c   = (states[..., 11:14] - self.s_mean[11:14]) / self.s_std[11:14]
        vel_p  = (states[..., 14:17] - self.s_mean[14:17]) / self.s_std[14:17]
        om_p   = (states[..., 17:20] - self.s_mean[17:20]) / self.s_std[17:20]

        act_n  = (actions - self.a_mean) / self.a_std
        alt    = states[..., 2:3] / 1000.0

        return torch.cat([
            phi.sin().unsqueeze(-1),   phi.cos().unsqueeze(-1),
            theta.sin().unsqueeze(-1), theta.cos().unsqueeze(-1),
            psi.sin().unsqueeze(-1),   psi.cos().unsqueeze(-1),
            theta_r, psi_r,
            vel_c, om_c, vel_p, om_p,
            act_n, alt,
        ], dim=-1)

    def encode_deltas(self, states_curr: torch.Tensor,
                      states_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states_curr: (..., 20)
            states_next: (..., 20)
        Returns:
            deltas: (..., 17) normalized
        """
        delta = states_next[..., 3:20] - states_curr[..., 3:20]
        for i in range(5):
            raw = states_next[..., 3 + i] - states_curr[..., 3 + i]
            delta[..., i] = (raw + torch.pi) % (2 * torch.pi) - torch.pi
        return delta / self.target_std


WIND_LABEL_DIM = 3  # mean wind vector (wx, wy, wz)
WIND_LABEL_SCALE = 10.0

PARAM_NAMES_ORDER: List[str] = [
    "mc", "mp", "b", "Ac", "As", "Ap",
    "k_psi", "k_r", "k_f", "c_f",
    "CD0", "CDa2", "CDds", "CDp",
    "CL0", "CLa", "CLds", "CYbeta",
    "Cm0", "Cma", "Cmq",
    "Clp", "Clbeta", "Clr", "Clda",
    "Cnbeta", "Cnp", "Cnr", "Cnda",
]
PARAM_LABEL_DIM = len(PARAM_NAMES_ORDER)  # 29


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
                 samples_per_traj: int = 10,
                 load_aux_labels: bool = False,
                 preload: bool = False,
                 traj_keys: Optional[Sequence[str]] = None,
                 preload_desc: str = "Preload RAM"):
        self.h5_path = h5_path
        self.K = context_length
        self.H = prediction_horizon
        self.norm = norm_stats
        self.samples_per_traj = samples_per_traj
        self.load_aux_labels = load_aux_labels
        self._preload = preload

        self._traj_keys = []
        self._traj_lengths = []
        self._param_multipliers: List[np.ndarray] = []
        self._has_winds = False
        min_len = self.K + self.H + 1

        with h5py.File(h5_path, "r") as f:
            traj_grp = f["trajectories"]
            if traj_keys is not None:
                key_iter = list(traj_keys)
            else:
                keys = sorted(traj_grp.keys())
                if max_trajs is not None:
                    keys = keys[:max_trajs]
                key_iter = []
                for k in keys:
                    if int(traj_grp[k].attrs["valid_steps"]) >= min_len:
                        key_iter.append(k)

            for k in key_iter:
                length = int(traj_grp[k].attrs["valid_steps"])
                if length < min_len:
                    continue
                self._traj_keys.append(k)
                self._traj_lengths.append(length)
                if load_aux_labels:
                    meta_str = traj_grp[k].attrs.get("meta", "{}")
                    meta = json.loads(meta_str)
                    pm = meta.get("param_multipliers", {})
                    vec = np.array(
                        [pm.get(n, 1.0) for n in PARAM_NAMES_ORDER],
                        dtype=np.float32,
                    )
                    self._param_multipliers.append(vec)

            first_key = self._traj_keys[0] if self._traj_keys else None
            if first_key and "winds" in traj_grp[first_key]:
                self._has_winds = True

        self._n_trajs = len(self._traj_keys)
        self._total_samples = self._n_trajs * samples_per_traj
        self._max_starts = [l - self.K - self.H for l in self._traj_lengths]
        self._h5 = None

        self._states_list: List[np.ndarray] = []
        self._actions_list: List[np.ndarray] = []
        self._winds_list: List[np.ndarray] = []

        self._preload_desc = preload_desc
        if preload and self._n_trajs > 0:
            self._load_into_ram()

    def _load_into_ram(self):
        """Load all trajectories in this split into RAM (no disk I/O in __getitem__)."""
        with h5py.File(self.h5_path, "r") as f:
            traj_grp = f["trajectories"]
            desc = getattr(self, "_preload_desc", "Preload RAM")
            for k in tqdm(self._traj_keys, desc=desc, leave=False, unit="traj"):
                g = traj_grp[k]
                valid = int(g.attrs["valid_steps"])
                self._states_list.append(
                    np.asarray(g["states"][:valid], dtype=np.float32))
                self._actions_list.append(
                    np.asarray(g["actions"][:valid], dtype=np.float32))
                if self.load_aux_labels and self._has_winds:
                    self._winds_list.append(
                        np.asarray(g["winds"][:valid], dtype=np.float32))

    def __len__(self):
        return self._total_samples

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        traj_idx = idx // self.samples_per_traj

        max_start = self._max_starts[traj_idx]
        t_start = np.random.randint(0, max(1, max_start))

        t_end_ctx = t_start + self.K
        t_end_pred = t_end_ctx + self.H

        if self._preload:
            states = self._states_list[traj_idx][t_start:t_end_pred + 1]
            actions = self._actions_list[traj_idx][t_start:t_end_pred]
        else:
            self._ensure_open()
            key = self._traj_keys[traj_idx]
            grp = self._h5["trajectories"][key]
            states = grp["states"][t_start:t_end_pred + 1].astype(np.float32)
            actions = grp["actions"][t_start:t_end_pred].astype(np.float32)

        result = {
            "raw_states": torch.from_numpy(np.asarray(states, dtype=np.float32)),
            "raw_actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        }

        if self.load_aux_labels:
            if self._has_winds:
                if self._preload:
                    winds = self._winds_list[traj_idx][t_start:t_end_ctx]
                else:
                    winds = grp["winds"][t_start:t_end_ctx]
                wind_mean = np.asarray(winds, dtype=np.float32).mean(axis=0)
                result["wind_label"] = torch.from_numpy(
                    wind_mean / WIND_LABEL_SCALE
                )
            else:
                result["wind_label"] = torch.zeros(WIND_LABEL_DIM)

            if self._param_multipliers:
                pm = self._param_multipliers[traj_idx]
                result["param_label"] = torch.from_numpy(pm - 1.0)
            else:
                result["param_label"] = torch.zeros(PARAM_LABEL_DIM)

        return result

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def split_train_val_keys(
    h5_path: str,
    context_length: int,
    prediction_horizon: int,
    train_ratio: float,
) -> Tuple[List[str], List[str]]:
    """Same split as legacy create_dataloaders: train = valid keys in first file-prefix; val = rest."""
    min_len = context_length + prediction_horizon + 1
    with h5py.File(h5_path, "r") as f:
        traj_grp = f["trajectories"]
        all_sorted = sorted(traj_grp.keys())
        n_cut = max(1, int(len(all_sorted) * train_ratio))
        train_keys = [
            k for k in all_sorted[:n_cut]
            if int(traj_grp[k].attrs["valid_steps"]) >= min_len
        ]
        all_valid = [
            k for k in all_sorted
            if int(traj_grp[k].attrs["valid_steps"]) >= min_len
        ]
        val_keys = all_valid[n_cut:]
    return train_keys, val_keys


def create_dataloaders(
    h5_path: str,
    norm_stats: NormalizationStats,
    context_length: int = 50,
    prediction_horizon: int = 20,
    batch_size: int = 64,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    samples_per_traj: int = 10,
    load_aux_labels: bool = False,
    preload: bool = True,
):
    """Create train/val DataLoaders from a single HDF5 file."""
    train_keys, val_keys = split_train_val_keys(
        h5_path, context_length, prediction_horizon, train_ratio
    )

    train_ds = ParafoilDynamicsDataset(
        h5_path, context_length, prediction_horizon,
        norm_stats, max_trajs=None,
        samples_per_traj=samples_per_traj,
        load_aux_labels=load_aux_labels,
        preload=preload,
        traj_keys=train_keys,
        preload_desc="Preload train → RAM",
    )
    val_ds = ParafoilDynamicsDataset(
        h5_path, context_length, prediction_horizon,
        norm_stats, max_trajs=None,
        samples_per_traj=max(1, samples_per_traj // 2),
        load_aux_labels=load_aux_labels,
        preload=preload,
        traj_keys=val_keys,
        preload_desc="Preload val → RAM",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
