"""
Parallel dataset generator for in-context dynamics learning.

Generates diverse rollouts from the 8-DOF parafoil ODE with:
  - Randomized parameters
  - Time-varying wind
  - Actuator dynamics (lag + delay)
  - Mixed control policies
  - Optional sensor noise

Outputs HDF5 dataset ready for training.
"""

import os
import sys
import time
import json
import numpy as np
import h5py
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict
from multiprocessing import cpu_count, freeze_support
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.parafoil_model import (
    ParafoilParams, parafoil_dynamics, euler_to_dcm
)
from learning.data_generation.domain_randomization import (
    randomize_params, WindConfig, WindField,
    ActuatorConfig, ActuatorModel,
    SensorNoiseConfig, SensorNoiseModel,
    PARAM_PERTURBATION_SPEC,
)
from learning.data_generation.control_policies import (
    make_control_policy, sample_policy_type, MAX_DEFLECTION
)


# ============================================================
#                   Generation config
# ============================================================

@dataclass
class GenerationConfig:
    """Full configuration for dataset generation."""
    # Paths
    model_config_path: str = "cfg/config.yaml"
    output_dir: str = "learning/datasets"
    dataset_name: str = "pilot"

    # Rollout settings
    n_trajectories: int = 1000
    traj_length_steps: int = 2000
    dt: float = 0.01                     # ODE integration step (s)
    control_dt: float = 0.01             # control update interval (s)

    # Initial conditions
    altitude_range: Tuple[float, float] = (200.0, 1500.0)
    velocity_range: Tuple[float, float] = (8.0, 14.0)
    heading_range: Tuple[float, float] = (0.0, 6.2832)
    pitch_range: Tuple[float, float] = (0.05, 0.2)    # rad, ~3-11 deg

    # Domain randomization
    randomize_params: bool = True
    param_perturbation_spec: Optional[Dict[str, float]] = None

    # Wind
    wind_config: WindConfig = field(default_factory=WindConfig)

    # Actuator
    actuator_config: ActuatorConfig = field(default_factory=ActuatorConfig)

    # Sensor noise
    sensor_noise_config: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)

    # Control policy weights: [random, excitation, noisy_controller]
    policy_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # Parallelism
    n_workers: int = 0   # 0 = auto (cpu_count - 1)

    # Save settings
    save_subsample_step: int = 1  # downsample when saving (1=no change, 10=dt_eff=0.1s)

    # Stability
    max_derivative_threshold: float = 1e6
    max_state_threshold: float = 1e5


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================
#              Single trajectory generation
# ============================================================

def generate_single_trajectory(
    traj_idx: int,
    seed: int,
    cfg: GenerationConfig,
    nominal_para: ParafoilParams,
) -> Optional[dict]:
    """
    Generate a single rollout. Returns dict with arrays or None if failed.
    """
    rng = np.random.default_rng(seed)

    # --- Randomize parameters ---
    if cfg.randomize_params:
        para, multipliers = randomize_params(
            nominal_para, rng, cfg.param_perturbation_spec
        )
    else:
        para = nominal_para
        multipliers = {}

    # --- Wind ---
    wind = WindField(cfg.wind_config, rng, total_steps=cfg.traj_length_steps)

    # --- Actuator ---
    actuator = ActuatorModel(cfg.actuator_config, cfg.control_dt, rng)

    # --- Sensor noise ---
    sensor = SensorNoiseModel(cfg.sensor_noise_config, rng)

    # --- Control policy ---
    policy_type = sample_policy_type(rng, weights=cfg.policy_weights)
    policy = make_control_policy(policy_type, rng, cfg.control_dt)

    # --- Initial state ---
    alt = rng.uniform(*cfg.altitude_range)
    vel = rng.uniform(*cfg.velocity_range)
    heading = rng.uniform(*cfg.heading_range)
    pitch = rng.uniform(*cfg.pitch_range)

    y0 = np.zeros(20)
    y0[0] = 0.0
    y0[1] = 0.0
    y0[2] = alt
    y0[3] = 0.0         # phi
    y0[4] = pitch        # theta
    y0[5] = heading      # psi
    y0[8] = vel          # u (forward)
    y0[9] = 0.0          # v (lateral)
    y0[10] = vel * np.tan(pitch)  # w (approx downward component)
    # Payload velocities ≈ canopy velocities initially
    y0[14:17] = y0[8:11]

    T = cfg.traj_length_steps
    n_substeps = max(1, int(cfg.control_dt / cfg.dt))
    actual_dt = cfg.control_dt / n_substeps

    # Storage
    states = np.zeros((T, 20), dtype=np.float32)
    observed_states = np.zeros((T, 20), dtype=np.float32)
    actions = np.zeros((T, 2), dtype=np.float32)
    winds = np.zeros((T, 3), dtype=np.float32)

    state = y0.copy()
    states[0] = state
    observed_states[0] = sensor.apply(state, cfg.control_dt)

    for step in range(T - 1):
        t = step * cfg.control_dt

        # Update wind
        w_vec = wind.step(cfg.control_dt)
        para.vw = w_vec.reshape(3, 1)
        winds[step] = w_vec

        # Update air density based on altitude
        para.update_density(max(state[2], 0.0))

        # Get control command from policy
        obs = observed_states[step]
        u_cmd = policy.step(obs, t)
        u_cmd = np.clip(u_cmd, 0, MAX_DEFLECTION)

        # Actuator filter
        u_actual = actuator.step(u_cmd)
        u_actual = np.clip(u_actual, 0, MAX_DEFLECTION)
        actions[step] = u_actual

        # Apply control
        para.left = u_actual[0]
        para.right = u_actual[1]

        # Euler integration with sub-stepping
        next_state = state.copy()
        diverged = False
        for _ in range(n_substeps):
            try:
                dydt = parafoil_dynamics(next_state, t, para)
            except Exception:
                diverged = True
                break

            if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                diverged = True
                break

            if np.max(np.abs(dydt)) > cfg.max_derivative_threshold:
                diverged = True
                break

            next_state = next_state + dydt * actual_dt

            # Normalize heading
            next_state[5] = _wrap_angle(next_state[5])

            # Clamp pitch/roll to avoid Euler singularity
            next_state[3] = np.clip(next_state[3], -np.radians(85), np.radians(85))
            next_state[4] = np.clip(next_state[4], -np.radians(85), np.radians(85))

        if diverged:
            return _make_result(traj_idx, seed, states, observed_states,
                                actions, winds, step, multipliers, wind,
                                actuator, policy_type, alt, heading,
                                success=False, reason="diverged")

        if np.max(np.abs(next_state)) > cfg.max_state_threshold:
            return _make_result(traj_idx, seed, states, observed_states,
                                actions, winds, step, multipliers, wind,
                                actuator, policy_type, alt, heading,
                                success=False, reason="state_explosion")

        state = next_state
        states[step + 1] = state
        observed_states[step + 1] = sensor.apply(state, cfg.control_dt)

        # Stop if altitude drops below ground
        if state[2] < 0:
            return _make_result(traj_idx, seed, states, observed_states,
                                actions, winds, step + 1, multipliers, wind,
                                actuator, policy_type, alt, heading,
                                success=True, reason="landed")

    # Fill last action/wind
    actions[-1] = actions[-2] if T > 1 else np.zeros(2)
    winds[-1] = winds[-2] if T > 1 else np.zeros(3)

    return _make_result(traj_idx, seed, states, observed_states,
                        actions, winds, T, multipliers, wind,
                        actuator, policy_type, alt, heading,
                        success=True, reason="complete")


def _make_result(traj_idx, seed, states, observed_states, actions, winds,
                 valid_steps, multipliers, wind, actuator,
                 policy_type, init_alt, init_heading,
                 success, reason):
    return {
        "traj_idx": traj_idx,
        "seed": seed,
        "states": states[:valid_steps].copy(),
        "observed_states": observed_states[:valid_steps].copy(),
        "actions": actions[:valid_steps].copy(),
        "winds": winds[:valid_steps].copy(),
        "valid_steps": valid_steps,
        "success": success,
        "reason": reason,
        "meta": {
            "param_multipliers": multipliers,
            "wind": wind.get_meta(),
            "actuator": actuator.get_meta(),
            "policy_type": policy_type,
            "init_altitude": float(init_alt),
            "init_heading": float(init_heading),
        }
    }


# ============================================================
#         Worker function for multiprocessing
# ============================================================

def _worker(args):
    """Top-level worker for Pool.map (must be picklable)."""
    traj_idx, seed, cfg_dict, yaml_path = args
    cfg = _dict_to_config(cfg_dict)
    para = ParafoilParams.from_yaml(yaml_path)
    return generate_single_trajectory(traj_idx, seed, cfg, para)


# ============================================================
#                    Main generator
# ============================================================

class DatasetGenerator:
    """Orchestrates parallel data generation and HDF5 output."""

    def __init__(self, config: GenerationConfig):
        self.cfg = config
        yaml_path = os.path.join(ROOT_DIR, config.model_config_path)
        self.nominal_para = ParafoilParams.from_yaml(yaml_path)
        self.yaml_path = yaml_path

    def generate(self, base_seed: int = 42) -> str:
        """
        Generate the full dataset. Returns path to HDF5 file.
        """
        cfg = self.cfg
        n = cfg.n_trajectories

        if cfg.n_workers == 1:
            n_workers = 1
        else:
            n_workers = cfg.n_workers or max(1, min(cpu_count() // 3, 6))

        ss = np.random.SeedSequence(base_seed)
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n)]

        cfg_dict = _config_to_dict(cfg)

        print(f"Generating {n} trajectories with {n_workers} workers...")
        print(f"  Steps per trajectory: {cfg.traj_length_steps}")
        print(f"  dt={cfg.dt}s, control_dt={cfg.control_dt}s")
        if cfg.save_subsample_step > 1:
            eff_dt = cfg.control_dt * cfg.save_subsample_step
            eff_steps = cfg.traj_length_steps // cfg.save_subsample_step
            print(f"  Subsample: every {cfg.save_subsample_step} steps "
                  f"-> effective dt={eff_dt}s, ~{eff_steps} saved steps/traj")
        print(f"  Domain randomization: {cfg.randomize_params}")

        args_list = [
            (i, seeds[i], cfg_dict, self.yaml_path) for i in range(n)
        ]

        t0 = time.time()

        if n_workers <= 1:
            results = []
            pbar = tqdm(args_list, desc="Generating trajectories",
                        unit="traj")
            for args in pbar:
                r = _worker(args)
                results.append(r)
                status = "OK" if (r is not None and r["success"]) else "FAIL"
                pbar.set_postfix(status=status)
        else:
            results = [None] * n
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {}
                for idx, args in enumerate(args_list):
                    f = executor.submit(_worker, args)
                    future_to_idx[f] = idx

                pbar = tqdm(as_completed(future_to_idx),
                            total=n, desc="Generating trajectories",
                            unit="traj")
                for future in pbar:
                    idx = future_to_idx[future]
                    try:
                        r = future.result()
                    except Exception as e:
                        r = None
                        tqdm.write(f"Worker {idx} error: {e}")
                    results[idx] = r
                    status = "OK" if (r is not None and r["success"]) else "FAIL"
                    pbar.set_postfix(status=status)

        elapsed = time.time() - t0
        print(f"\nGeneration done in {elapsed:.1f}s")

        # Statistics
        successes = sum(1 for r in results if r is not None and r["success"])
        failures = sum(1 for r in results if r is None or not r["success"])
        print(f"  Successes: {successes}/{n} ({100*successes/n:.1f}%)")
        print(f"  Failures:  {failures}/{n} ({100*failures/n:.1f}%)")

        if failures > 0:
            reasons = {}
            for r in results:
                if r is not None and not r["success"]:
                    reason = r.get("reason", "unknown")
                    reasons[reason] = reasons.get(reason, 0) + 1
            print(f"  Failure reasons: {reasons}")

        # Save to HDF5
        output_path = self._save_hdf5(results)
        print(f"\nDataset saved to: {output_path}")
        return output_path

    def _save_hdf5(self, results: List[Optional[dict]]) -> str:
        os.makedirs(os.path.join(ROOT_DIR, self.cfg.output_dir), exist_ok=True)
        path = os.path.join(ROOT_DIR, self.cfg.output_dir,
                            f"{self.cfg.dataset_name}.h5")

        valid_results = [r for r in results if r is not None and r["success"]]
        ss = self.cfg.save_subsample_step
        effective_dt = self.cfg.control_dt * ss

        if ss > 1:
            print(f"  Subsampling: every {ss} steps → effective dt={effective_dt}s")

        with h5py.File(path, "w") as f:
            config_grp = f.create_group("config")
            config_grp.attrs["n_trajectories"] = len(valid_results)
            config_grp.attrs["dt"] = effective_dt
            config_grp.attrs["control_dt"] = effective_dt
            config_grp.attrs["ode_dt"] = self.cfg.dt
            config_grp.attrs["save_subsample_step"] = ss
            config_grp.attrs["state_dim"] = 20
            config_grp.attrs["action_dim"] = 2

            traj_grp = f.create_group("trajectories")
            for i, r in tqdm(enumerate(valid_results),
                             total=len(valid_results),
                             desc="Saving to HDF5", unit="traj"):
                states = r["states"][::ss]
                observed = r["observed_states"][::ss]
                actions = r["actions"][::ss]
                winds_arr = r["winds"][::ss]

                g = traj_grp.create_group(f"traj_{i:06d}")
                g.create_dataset("states", data=states,
                                 compression="gzip", compression_opts=4)
                g.create_dataset("observed_states", data=observed,
                                 compression="gzip", compression_opts=4)
                g.create_dataset("actions", data=actions,
                                 compression="gzip", compression_opts=4)
                g.create_dataset("winds", data=winds_arr,
                                 compression="gzip", compression_opts=4)
                g.attrs["valid_steps"] = len(states)
                g.attrs["reason"] = r["reason"]
                g.attrs["seed"] = r["seed"]
                g.attrs["meta"] = json.dumps(r["meta"], default=_json_default)

        return path


def _config_to_dict(cfg: GenerationConfig) -> dict:
    """Convert config to a plain dict for pickling across processes."""
    d = {}
    for k, v in cfg.__dict__.items():
        if hasattr(v, '__dataclass_fields__'):
            d[k] = asdict(v)
        else:
            d[k] = v
    return d


def _dict_to_config(d: dict) -> GenerationConfig:
    """Reconstruct GenerationConfig from a plain dict (reverses _config_to_dict)."""
    nested_fields = {
        'wind_config': WindConfig,
        'actuator_config': ActuatorConfig,
        'sensor_noise_config': SensorNoiseConfig,
    }
    kwargs = {}
    for k, v in d.items():
        if k in nested_fields and isinstance(v, dict):
            kwargs[k] = nested_fields[k](**v)
        else:
            kwargs[k] = v
    return GenerationConfig(**kwargs)


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ============================================================
#                      CLI entry point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate parafoil dynamics dataset for in-context learning"
    )
    parser.add_argument("--n-trajectories", type=int, default=1000,
                        help="Number of trajectories (default: 1000)")
    parser.add_argument("--traj-length", type=int, default=2000,
                        help="Steps per trajectory (default: 2000)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Integration timestep (default: 0.01)")
    parser.add_argument("--dataset-name", type=str, default="pilot",
                        help="Dataset name (default: pilot)")
    parser.add_argument("--output-dir", type=str, default="learning/datasets",
                        help="Output directory")
    parser.add_argument("--n-workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Disable parameter randomization")
    parser.add_argument("--wind-mode", type=str, default="ar1",
                        choices=["constant", "ar1"])
    parser.add_argument("--no-sensor-noise", action="store_true",
                        help="Disable sensor noise")
    parser.add_argument("--subsample-step", type=int, default=1,
                        help="Downsample factor when saving (10 -> dt_eff=0.1s)")
    parser.add_argument("--no-regime-change", action="store_true",
                        help="Disable mid-trajectory wind regime changes")
    parser.add_argument("--wind-speed-max", type=float, default=None,
                        help="Override max wind speed (m/s)")
    parser.add_argument("--param-scale", type=float, default=1.0,
                        help="Scale factor for param perturbation ranges")

    args = parser.parse_args()

    wind_cfg = WindConfig(mode=args.wind_mode)
    if args.no_regime_change:
        wind_cfg.regime_change_prob = 0.0
    if args.wind_speed_max is not None:
        wind_cfg.speed_range = (0.0, args.wind_speed_max)

    param_spec = None
    if args.param_scale != 1.0:
        from learning.data_generation.domain_randomization import PARAM_PERTURBATION_SPEC
        param_spec = {k: v * args.param_scale for k, v in PARAM_PERTURBATION_SPEC.items()}

    cfg = GenerationConfig(
        n_trajectories=args.n_trajectories,
        traj_length_steps=args.traj_length,
        dt=args.dt,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        randomize_params=not args.no_randomize,
        wind_config=wind_cfg,
        param_perturbation_spec=param_spec,
        sensor_noise_config=SensorNoiseConfig(enabled=not args.no_sensor_noise),
        save_subsample_step=args.subsample_step,
    )

    generator = DatasetGenerator(cfg)
    generator.generate(base_seed=args.seed)


if __name__ == "__main__":
    freeze_support()
    main()
