"""
Fixed-trajectory controller comparison utility.

For each seed, this script plans once, samples one initial condition once,
then replays the same reference trajectory and initial state with ADRC and PID.
This isolates controller behavior from planning randomness as much as possible.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from planning.trajectory_postprocess import validate_trajectory
from simulation.closed_loop_sim import ClosedLoopSimulator


def parse_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in spec:
        return [int(item.strip()) for item in spec.split(",") if item.strip()]
    return [int(spec)]


def load_json(path: str) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_diagnostics(sim: ClosedLoopSimulator) -> Dict[str, float]:
    trajectory = sim.trajectory
    constraints = sim.map_manager.constraints
    validation = validate_trajectory(
        trajectory,
        min_glide_ratio=constraints.min_glide_ratio,
        max_glide_ratio=constraints.glide_ratio,
        min_turn_radius=constraints.min_turn_radius,
    )

    curvatures = np.abs(trajectory.get_curvatures())
    finite_radii = np.divide(
        1.0,
        curvatures,
        out=np.full_like(curvatures, np.inf),
        where=curvatures > 1e-6,
    )
    finite_radii = finite_radii[np.isfinite(finite_radii)]
    robust_radii = finite_radii[finite_radii > 5.0]

    curvature_samples = curvatures[curvatures > 1e-6]
    curvature_p95 = float(np.percentile(curvature_samples, 95)) if curvature_samples.size else 0.0
    turn_rate_p95 = float(np.degrees(sim.reference_speed * curvature_p95))

    glide_stats = validation.get("statistics", {}).get("glide_ratio", {})

    return {
        "trajectory_valid": bool(validation.get("valid", False)),
        "raw_min_turn_radius": float(np.min(finite_radii)) if finite_radii.size else float("inf"),
        "robust_turn_radius_p05": float(np.percentile(robust_radii, 5)) if robust_radii.size else float("inf"),
        "curvature_p95": curvature_p95,
        "turn_rate_p95_deg_s": turn_rate_p95,
        "glide_ratio_mean": float(glide_stats.get("mean", np.nan)),
        "glide_ratio_max": float(glide_stats.get("max", np.nan)),
        "min_turn_radius_constraint": float(constraints.min_turn_radius),
    }


def run_controller(
    seed: int,
    controller_type: str,
    trajectory,
    init_state: np.ndarray,
    planning_time: float,
    controller_kwargs: Dict,
) -> Dict:
    sim = ClosedLoopSimulator(
        map_config_path="cfg/map_config.yaml",
        model_config_path="cfg/config.yaml",
        seed=seed,
        controller_type=controller_type,
        controller_kwargs=controller_kwargs,
        quiet=True,
    )
    sim.trajectory = trajectory
    sim.controller.set_trajectory(trajectory)
    sim.planning_time = planning_time
    sim.init_state(
        position=init_state[0:3].copy(),
        heading=float(init_state[5]),
        velocity=float(init_state[8]),
        use_rng=False,
    )
    sim.run(
        max_time=trajectory.duration + 60.0,
        enable_failure_detection=True,
        verbose=False,
    )
    metrics = sim.compute_metrics()
    final_pos = sim._last_final_state[0:3]
    target_pos = sim.map_manager.target.position
    return {
        "controller": controller_type,
        "termination_reason": sim._last_termination_reason.value if sim._last_termination_reason else "unknown",
        "ADE": float(metrics.get("ADE", np.nan)),
        "FDE_horizontal": float(np.linalg.norm(final_pos[:2] - target_pos[:2])),
        "mean_cross_track_error": float(metrics.get("mean_cross_track_error", np.nan)),
        "max_cross_track_error": float(metrics.get("max_cross_track_error", np.nan)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare ADRC and PID on fixed planned trajectories.")
    parser.add_argument("--seeds", default="6-10", help="Seed range, comma list, or single seed.")
    parser.add_argument("--output-dir", default="benchmark/outputs", help="Directory for CSV/JSON summaries.")
    parser.add_argument("--common-config", default="", help="Path to JSON file with controller kwargs for both controllers.")
    parser.add_argument("--adrc-config", default="", help="Path to JSON file with ADRC-only controller kwargs.")
    parser.add_argument("--pid-config", default="", help="Path to JSON file with PID-only controller kwargs.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    common_overrides = load_json(args.common_config)
    adrc_overrides = dict(common_overrides)
    adrc_overrides.update(load_json(args.adrc_config))
    pid_overrides = dict(common_overrides)
    pid_overrides.update(load_json(args.pid_config))

    rows = []
    for seed in seeds:
        base = ClosedLoopSimulator(
            map_config_path="cfg/map_config.yaml",
            model_config_path="cfg/config.yaml",
            seed=seed,
            controller_type="adrc",
            quiet=True,
        )
        planned = base.plan(max_time=30.0)
        if not planned:
            rows.append({
                "seed": seed,
                "planning_success": False,
                "controller": "none",
            })
            continue

        init_state = base.init_state(use_rng=True)
        diag = trajectory_diagnostics(base)
        for controller_type, overrides in (("adrc", adrc_overrides), ("pid", pid_overrides)):
            result = run_controller(
                seed=seed,
                controller_type=controller_type,
                trajectory=base.trajectory,
                init_state=init_state,
                planning_time=base.planning_time,
                controller_kwargs=overrides,
            )
            row = {
                "seed": seed,
                "planning_success": True,
                **diag,
                **result,
            }
            rows.append(row)
            print(
                f"seed {seed:03d} {controller_type:>4} "
                f"FDEh={row['FDE_horizontal']:.2f}m "
                f"ADE={row['ADE']:.2f}m "
                f"XTE={row['mean_cross_track_error']:.2f}m "
                f"term={row['termination_reason']}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"controller_compare_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"controller_compare_{timestamp}.json")

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")


if __name__ == "__main__":
    main()
