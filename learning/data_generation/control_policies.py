"""
Control input sampling strategies for data generation.

Three types:
  1. Closed-loop controller (ADRC) + optional noise
  2. Random piecewise-constant actions
  3. Open-loop excitation signals (chirp, step, doublet)
"""

import os
import sys
import numpy as np
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from control.adrc_controller import ADRC
    _ADRC_AVAILABLE = True
except ImportError:
    _ADRC_AVAILABLE = False


MAX_DEFLECTION = 0.4  # meters — physical limit of control ropes


class ControlPolicy:
    """Base class for control policies."""

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        """Return [left, right] control in meters."""
        raise NotImplementedError

    def reset(self):
        pass


class RandomPiecewiseConstant(ControlPolicy):
    """
    Piecewise-constant random actions with rate limiting.
    Each segment holds a constant action for a random duration.
    """

    def __init__(self, rng: np.random.Generator, dt: float,
                 segment_duration_range=(1.0, 5.0),
                 rate_limit=0.1):
        self.rng = rng
        self.dt = dt
        self.seg_range = segment_duration_range
        self.rate_limit = rate_limit  # max change per second

        self._target = np.zeros(2)
        self._current = np.zeros(2)
        self._next_switch = 0.0

    def reset(self):
        self._target = np.zeros(2)
        self._current = np.zeros(2)
        self._next_switch = 0.0

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        if t >= self._next_switch:
            self._target = self.rng.uniform(0, MAX_DEFLECTION, size=2)
            dur = self.rng.uniform(*self.seg_range)
            self._next_switch = t + dur

        # Rate-limited slew toward target
        max_delta = self.rate_limit * self.dt
        diff = self._target - self._current
        diff = np.clip(diff, -max_delta, max_delta)
        self._current = np.clip(self._current + diff, 0, MAX_DEFLECTION)
        return self._current.copy()


class ExcitationSignal(ControlPolicy):
    """
    Open-loop excitation: chirp / step / doublet signals.
    Useful for covering frequency-response space for system identification.
    """

    def __init__(self, rng: np.random.Generator, dt: float,
                 signal_type: Optional[str] = None):
        self.rng = rng
        self.dt = dt
        if signal_type is None:
            signal_type = rng.choice(["chirp", "step", "doublet"])
        self.signal_type = signal_type

        # Chirp params
        self.f0 = rng.uniform(0.05, 0.2)   # start freq Hz
        self.f1 = rng.uniform(0.5, 2.0)    # end freq Hz
        self.chirp_duration = rng.uniform(10.0, 30.0)
        self.chirp_amplitude = rng.uniform(0.05, 0.2)

        # Step params
        self.step_times = np.sort(rng.uniform(1.0, 20.0, size=rng.integers(2, 6)))
        self.step_levels = rng.uniform(0, MAX_DEFLECTION,
                                        size=(len(self.step_times) + 1, 2))

        # Doublet params
        self.doublet_start = rng.uniform(1.0, 5.0)
        self.doublet_width = rng.uniform(0.5, 2.0)
        self.doublet_amp = rng.uniform(0.05, 0.25)

        # Which channel gets the signal (0=symmetric, 1=asymmetric)
        self.channel_mode = rng.choice(["symmetric", "asymmetric", "both"])

    def reset(self):
        pass

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        if self.signal_type == "chirp":
            return self._chirp(t)
        elif self.signal_type == "step":
            return self._step(t)
        else:
            return self._doublet(t)

    def _chirp(self, t: float) -> np.ndarray:
        phase_t = min(t / self.chirp_duration, 1.0)
        freq = self.f0 + (self.f1 - self.f0) * phase_t
        val = self.chirp_amplitude * np.sin(2 * np.pi * freq * t)
        base = MAX_DEFLECTION * 0.3
        return self._apply_to_channels(base + val)

    def _step(self, t: float) -> np.ndarray:
        idx = np.searchsorted(self.step_times, t)
        return np.clip(self.step_levels[idx], 0, MAX_DEFLECTION)

    def _doublet(self, t: float) -> np.ndarray:
        base = MAX_DEFLECTION * 0.2
        ds = self.doublet_start
        dw = self.doublet_width
        if ds <= t < ds + dw:
            val = base + self.doublet_amp
        elif ds + dw <= t < ds + 2 * dw:
            val = base - self.doublet_amp
        else:
            val = base
        return self._apply_to_channels(val)

    def _apply_to_channels(self, val: float) -> np.ndarray:
        u = np.ones(2) * MAX_DEFLECTION * 0.15
        if self.channel_mode == "symmetric":
            u[:] = val
        elif self.channel_mode == "asymmetric":
            u[0] = val
            u[1] = MAX_DEFLECTION * 0.3 - (val - MAX_DEFLECTION * 0.15)
        else:
            u[0] = val
            u[1] = val * 0.8
        return np.clip(u, 0, MAX_DEFLECTION)


class NoisyControllerPolicy(ControlPolicy):
    """
    Uses the real ADRC heading controller with additive exploration noise.
    Periodically changes the target heading to produce diverse closed-loop data.
    Falls back to small constant deflection + noise if ADRC is unavailable.
    """

    # How often (seconds) to pick a new random target heading
    _HEADING_CHANGE_INTERVAL = (10.0, 30.0)

    def __init__(self, controller=None, noise_std: float = 0.03,
                 rng: Optional[np.random.Generator] = None,
                 dt: float = 0.01):
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()
        self.dt = dt

        self._target_heading: Optional[float] = None
        self._next_heading_change: float = 0.0

        if _ADRC_AVAILABLE:
            self._adrc = ADRC(
                td_r=30.0,
                td_h=dt,
                eso_omega=20.0,
                eso_order=2,
                use_linear_eso=True,
                kp=2.0,
                kd=0.5,
                use_linear_sef=True,
                b0=0.5,
                u_min=-MAX_DEFLECTION,
                u_max=MAX_DEFLECTION,
            )
        else:
            self._adrc = None

    def reset(self):
        self._target_heading = None
        self._next_heading_change = 0.0
        if self._adrc is not None:
            self._adrc.reset()

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        noise = self.rng.normal(0, self.noise_std, size=2)

        if self._adrc is None:
            # Fallback: small symmetric deflection + noise
            base = np.array([0.05, 0.05])
            return np.clip(base + noise, 0, MAX_DEFLECTION)

        # Periodically pick a new random target heading
        if self._target_heading is None or t >= self._next_heading_change:
            self._target_heading = self.rng.uniform(-np.pi, np.pi)
            interval = self.rng.uniform(*self._HEADING_CHANGE_INTERVAL)
            self._next_heading_change = t + interval

        current_heading = float(state[5])  # psi from 20D state vector

        # ADRC heading control → differential deflection [-MAX, MAX]
        delta_diff = self._adrc.update(self._target_heading, current_heading, self.dt)

        # Convert to [left, right] deflection (same logic as ParafoilADRCController)
        if delta_diff > 0:
            delta_left, delta_right = 0.0, delta_diff
        elif delta_diff < 0:
            delta_left, delta_right = -delta_diff, 0.0
        else:
            delta_left, delta_right = 0.0, 0.0

        u = np.array([delta_left, delta_right]) + noise
        return np.clip(u, 0, MAX_DEFLECTION)


def make_control_policy(policy_type: str, rng: np.random.Generator,
                        dt: float) -> ControlPolicy:
    """Factory for control policies."""
    if policy_type == "random":
        return RandomPiecewiseConstant(rng, dt)
    elif policy_type == "excitation":
        return ExcitationSignal(rng, dt)
    elif policy_type == "noisy_controller":
        return NoisyControllerPolicy(rng=rng, dt=dt)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def sample_policy_type(rng: np.random.Generator,
                       weights=(0.5, 0.3, 0.2)) -> str:
    """
    Sample a policy type with given weights.
    Default: 50% random piecewise, 30% excitation, 20% noisy controller.
    """
    types = ["random", "excitation", "noisy_controller"]
    return rng.choice(types, p=weights)
