"""
Domain randomization for parafoil dynamics data generation.

Provides randomization of:
  - Aerodynamic / structural parameters
  - Wind field (constant + AR(1) turbulence)
  - Actuator dynamics (first-order lag + transport delay)
  - Sensor noise (Gaussian + low-frequency drift)
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# ============================================================
#            Parameter perturbation specification
# ============================================================

# (param_name, nominal_value, relative_range)
# relative_range is the *fractional* half-width, e.g. 0.2 means ±20%
# Sensitive params get tighter ranges to avoid ODE divergence.
PARAM_PERTURBATION_SPEC: Dict[str, float] = {
    # --- mass ---
    "mc":       0.25,
    "mp":       0.25,
    # --- geometry ---
    "b":        0.18,
    "Ac":       0.18,
    "As":       0.18,
    "Ap":       0.25,
    # --- hinge (sensitive) ---
    "k_psi":    0.35,
    "k_r":      0.35,
    "k_f":      0.35,
    "c_f":      0.25,
    # --- aerodynamics ---
    "CD0":      0.35,
    "CDa2":     0.35,
    "CDds":     0.35,
    "CDp":      0.35,
    "CL0":      0.18,
    "CLa":      0.18,
    "CLds":     0.35,
    "CYbeta":   0.35,
    "Cm0":      0.35,
    "Cma":      0.35,
    "Cmq":      0.35,
    "Clp":      0.35,
    "Clbeta":   0.35,
    "Clr":      0.45,
    "Clda":     0.35,
    "Cnbeta":   0.45,
    "Cnp":      0.45,
    "Cnr":      0.35,
    "Cnda":     0.35,
}


def randomize_params(para, rng: np.random.Generator,
                     spec: Optional[Dict[str, float]] = None):
    """
    Apply multiplicative perturbation to a ParafoilParams object.

    Returns a (deep-copied, perturbed) params object and a dict of
    the actual multipliers applied (for logging).
    """
    if spec is None:
        spec = PARAM_PERTURBATION_SPEC

    para_new = copy.deepcopy(para)
    multipliers = {}

    for name, half_range in spec.items():
        if not hasattr(para_new, name):
            continue
        nominal = getattr(para, name)
        mult = rng.uniform(1.0 - half_range, 1.0 + half_range)
        setattr(para_new, name, nominal * mult)
        multipliers[name] = mult

    # Recompute derived quantities that depend on perturbed values
    para_new.c = para_new.Ac / para_new.b
    para_new.r = para_new.b / para_new.ca
    para_new.t = (para_new.c * para.t / (para.Ac / para.b))  # keep thickness_ratio
    para_new.sloc = 2 * para_new.r * np.sin(para_new.ca / 2) / para_new.ca

    line_length_nom = para.rcOc[2, 0] + para.r - para.sloc
    para_new.rcOc = np.array([[0, 0, line_length_nom - para_new.r + para_new.sloc]]).T

    return para_new, multipliers


# ============================================================
#                     Wind field models
# ============================================================

@dataclass
class WindConfig:
    """Wind randomization configuration."""
    mode: str = "ar1"            # "constant" or "ar1"
    speed_range: Tuple[float, float] = (0.0, 12.0)  # m/s
    ar1_alpha: float = 0.98      # AR(1) correlation coefficient
    ar1_sigma: float = 0.5       # innovation std (m/s per axis)
    vertical_speed_range: Tuple[float, float] = (-1.0, 1.0)
    regime_change_prob: float = 0.3
    regime_change_speed_delta: float = 5.0


class WindField:
    """Generates time-varying wind vectors with optional regime changes."""

    def __init__(self, config: WindConfig, rng: np.random.Generator,
                 total_steps: int = 0):
        self.cfg = config
        self.rng = rng

        speed = rng.uniform(*config.speed_range)
        direction = rng.uniform(0, 2 * np.pi)
        wz = rng.uniform(*config.vertical_speed_range)

        self.base_wind = np.array([
            speed * np.cos(direction),
            speed * np.sin(direction),
            wz
        ])
        self._current = self.base_wind.copy()
        self._step_count = 0

        self._regime_change_step = -1
        self._new_base_wind = None
        if (total_steps > 0
                and config.regime_change_prob > 0
                and rng.random() < config.regime_change_prob):
            change_frac = rng.uniform(0.3, 0.7)
            self._regime_change_step = int(total_steps * change_frac)
            delta = config.regime_change_speed_delta
            new_speed = max(0.0, speed + rng.uniform(-delta, delta))
            new_dir = direction + rng.uniform(-np.pi / 2, np.pi / 2)
            new_wz = rng.uniform(*config.vertical_speed_range)
            self._new_base_wind = np.array([
                new_speed * np.cos(new_dir),
                new_speed * np.sin(new_dir),
                new_wz,
            ])

    def reset(self):
        self._current = self.base_wind.copy()
        self._step_count = 0

    def step(self, dt: float) -> np.ndarray:
        """Advance one timestep. Returns (3,) wind vector."""
        self._step_count += 1

        if (self._regime_change_step > 0
                and self._step_count == self._regime_change_step
                and self._new_base_wind is not None):
            self.base_wind = self._new_base_wind.copy()

        if self.cfg.mode == "constant":
            return self.base_wind.copy()

        alpha = self.cfg.ar1_alpha
        eps = self.rng.normal(0, self.cfg.ar1_sigma, size=3)
        self._current = alpha * self._current + (1 - alpha) * self.base_wind + eps
        return self._current.copy()

    def get_current(self) -> np.ndarray:
        return self._current.copy()

    def get_meta(self) -> dict:
        meta = {
            "mode": self.cfg.mode,
            "base_wind": self.base_wind.tolist(),
            "ar1_alpha": self.cfg.ar1_alpha,
            "ar1_sigma": self.cfg.ar1_sigma,
        }
        if self._regime_change_step > 0:
            meta["regime_change_step"] = self._regime_change_step
            if self._new_base_wind is not None:
                meta["new_base_wind"] = self._new_base_wind.tolist()
        return meta


# ============================================================
#                   Actuator dynamics
# ============================================================

@dataclass
class ActuatorConfig:
    """First-order lag + transport delay model."""
    tau_range: Tuple[float, float] = (0.05, 0.30)   # time constant (s)
    delay_steps_range: Tuple[int, int] = (0, 3)      # discrete delay (steps)


class ActuatorModel:
    """
    Models actuator dynamics as: first-order lag + pure delay.

    u_actual(t) = lag_filter( u_command(t - delay) )
    """

    def __init__(self, config: ActuatorConfig, dt: float,
                 rng: np.random.Generator):
        self.tau = rng.uniform(*config.tau_range)
        self.delay_steps = rng.integers(config.delay_steps_range[0],
                                         config.delay_steps_range[1] + 1)
        self.dt = dt
        self.alpha = dt / (self.tau + dt)

        # Delay buffer for 2 channels (left, right)
        buf_len = max(1, self.delay_steps + 1)
        self._buffer = np.zeros((buf_len, 2))
        self._idx = 0

        # Lag state
        self._state = np.zeros(2)

    def reset(self):
        self._buffer[:] = 0.0
        self._idx = 0
        self._state[:] = 0.0

    def step(self, u_cmd: np.ndarray) -> np.ndarray:
        """
        Process a command [left, right] through delay + lag.
        Returns the actual actuator output [left, right].
        """
        u_cmd = np.asarray(u_cmd, dtype=np.float64)

        # Write into circular buffer
        self._buffer[self._idx % len(self._buffer)] = u_cmd
        self._idx += 1

        # Read delayed command
        read_idx = (self._idx - 1 - self.delay_steps) % len(self._buffer)
        u_delayed = self._buffer[read_idx]

        # First-order lag
        self._state = self._state + self.alpha * (u_delayed - self._state)
        return self._state.copy()

    def get_meta(self) -> dict:
        return {
            "tau": float(self.tau),
            "delay_steps": int(self.delay_steps),
        }


# ============================================================
#                   Sensor noise model
# ============================================================

@dataclass
class SensorNoiseConfig:
    """Gaussian noise + low-frequency drift on state observations."""
    enabled: bool = True
    # Per-group noise std (applied after normalization in training,
    # but here we apply raw noise to the state vector).
    euler_noise_std: float = 0.005       # rad
    rel_angle_noise_std: float = 0.002   # rad
    velocity_noise_std: float = 0.05     # m/s
    angular_vel_noise_std: float = 0.005 # rad/s
    position_noise_std: float = 0.5      # m
    # Low-frequency drift
    drift_rate: float = 0.001            # drift per second


class SensorNoiseModel:
    """Adds measurement noise + slow drift to state observations."""

    # Indices of state groups in 20D state vector
    _GROUPS = {
        "position":    (slice(0, 3),   "position_noise_std"),
        "euler":       (slice(3, 6),   "euler_noise_std"),
        "rel_angle":   (slice(6, 8),   "rel_angle_noise_std"),
        "vel_canopy":  (slice(8, 11),  "velocity_noise_std"),
        "omega_canopy":(slice(11, 14), "angular_vel_noise_std"),
        "vel_payload": (slice(14, 17), "velocity_noise_std"),
        "omega_payload":(slice(17, 20),"angular_vel_noise_std"),
    }

    def __init__(self, config: SensorNoiseConfig, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self._drift = np.zeros(20)

    def reset(self):
        self._drift = np.zeros(20)

    def apply(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Return a noisy observation of the true state."""
        if not self.cfg.enabled:
            return state.copy()

        noisy = state.copy()
        for _, (slc, attr) in self._GROUPS.items():
            std = getattr(self.cfg, attr)
            n = slc.stop - slc.start if slc.stop is not None else 3
            noisy[slc] += self.rng.normal(0, std, size=n)

        # Low-frequency drift
        self._drift += self.rng.normal(0, self.cfg.drift_rate * dt, size=20)
        noisy += self._drift

        return noisy
