"""
Controller test script - avoid Chinese output
"""
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress print from map_manager
import io
from contextlib import redirect_stdout

from planning.map_manager import MapManager
from planning.global_planner import RRTStarPlanner
from planning.path_smoother import PathSmoother
from control.adrc_controller import ParafoilADRCController
from models.parafoil_model import ParafoilParams, parafoil_dynamics


def test_controller():
    print("=" * 60)
    print("Controller Test")
    print("=" * 60)

    # Load config silently
    f = io.StringIO()
    with redirect_stdout(f):
        map_manager = MapManager.from_yaml('cfg/map_config.yaml')
        para = ParafoilParams.from_yaml('cfg/config.yaml')

    # Create planner and smoother
    planner = RRTStarPlanner(map_manager)
    smoother = PathSmoother(
        turn_radius=map_manager.constraints.min_turn_radius,
        reference_speed=12.0,
        control_frequency=100.0
    )

    # Create controller - conservative params for parafoil (turns via roll)
    # With descent rate control via symmetric deflection
    controller = ParafoilADRCController(
        heading_kp=0.8,
        heading_kd=0.4,
        heading_eso_omega=3.0,
        heading_td_r=10.0,
        lateral_kp=0.002,
        glide_ratio_natural=11.0,   # Natural glide ratio (no symmetric deflection)
        glide_ratio_min=5.0,        # Minimum glide ratio (max symmetric deflection)
        descent_kp=0.5,             # Descent rate control gain
        descent_margin=1.2,         # Glide ratio margin factor
        reference_speed=12.0,
        lookahead_distance=100.0,
        max_deflection=0.5,         # Increased for descent control
        dt=0.01
    )
    controller.set_debug(False)

    # Plan
    print("\n[1] Planning...")
    f = io.StringIO()
    with redirect_stdout(f):
        path, info = planner.plan(max_time=20.0)

    if path is None:
        print("Planning failed!")
        return

    print(f"    Path found: {len(path)} waypoints, length={info['path_length']:.1f}m")

    # Smooth
    print("\n[2] Smoothing...")
    end_heading = map_manager.target.approach_heading if map_manager.target else None
    trajectory = smoother.smooth(path, end_heading=end_heading, waypoint_density=15)
    print(f"    Trajectory: {len(trajectory)} points, duration={trajectory.duration:.1f}s")

    # Set trajectory
    controller.set_trajectory(trajectory)

    # Init state
    print("\n[3] Initializing state...")
    init_point = trajectory[0]
    state = np.zeros(20)
    state[0:3] = init_point.position + np.array([10, -15, 0])  # Position noise
    state[3] = 0.0                           # phi (roll)
    state[4] = np.radians(8)                 # theta (pitch)
    state[5] = init_point.heading + 0.1      # psi (heading) with noise
    state[8] = 10.0                          # u (forward speed)
    state[9] = 0.0                           # v (side speed)
    state[10] = 5.0                          # w (descent speed)

    print(f"    Init pos: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})")
    print(f"    Init heading: {np.degrees(state[5]):.1f} deg")

    # Run simulation
    print("\n[4] Running simulation (500 steps = 5s)...")
    print("-" * 60)

    control_dt = 0.01
    dynamics_dt = 0.002
    n_substeps = int(control_dt / dynamics_dt)
    MAX_DEFLECTION_METERS = 0.4

    t = 0.0
    roll_exceeded = 0
    for step in range(500):
        # Extract state
        position = state[0:3]
        euler = state[3:6]
        velocity_body = state[8:11]
        heading = euler[2]

        # Normalize heading
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi

        # Body to inertial velocity
        psi = heading
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        vx = velocity_body[0] * cos_psi - velocity_body[1] * sin_psi
        vy = velocity_body[0] * sin_psi + velocity_body[1] * cos_psi
        vz = -velocity_body[2]
        velocity = np.array([vx, vy, vz])

        # Controller update
        ctrl = controller.update(
            current_pos=position,
            current_vel=velocity,
            current_heading=heading,
            t=t
        )

        # Set control input
        para.left = ctrl.delta_left * MAX_DEFLECTION_METERS
        para.right = ctrl.delta_right * MAX_DEFLECTION_METERS
        para.update_density(position[2])

        # Print progress
        if step % 100 == 0:
            roll = np.degrees(euler[0])
            heading_deg = np.degrees(heading)
            heading_error_deg = np.degrees(ctrl.heading_error)
            print(f"  t={t:5.2f}s | pos=({position[0]:7.1f}, {position[1]:7.1f}, {position[2]:6.1f})")
            print(f"          | heading={heading_deg:6.1f}deg, roll={roll:5.1f}deg")
            print(f"          | ctrl: L={ctrl.delta_left:.3f}, R={ctrl.delta_right:.3f}, d_s={ctrl.delta_symmetric:.3f}")
            print(f"          | glide: required={ctrl.glide_ratio_required:.1f}, current={ctrl.glide_ratio_current:.1f}")
            print(f"          | error: heading={heading_error_deg:5.1f}deg, cross_track={ctrl.cross_track_error:6.1f}m")
            print()

        # Dynamics integration
        next_state = state.copy()
        actual_dt = control_dt / n_substeps

        for i in range(n_substeps):
            try:
                dydt = parafoil_dynamics(next_state, t, para)
            except Exception as e:
                print(f"Dynamics error at t={t:.3f}s: {e}")
                return

            if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                print(f"NaN/Inf in dynamics at t={t:.3f}s")
                return

            next_state = next_state + dydt * actual_dt

            # Normalize heading
            while next_state[5] > np.pi:
                next_state[5] -= 2 * np.pi
            while next_state[5] < -np.pi:
                next_state[5] += 2 * np.pi

            # Limit roll/pitch
            theta_max = np.radians(85)
            if abs(next_state[4]) > theta_max:
                next_state[4] = np.sign(next_state[4]) * theta_max
            if abs(next_state[3]) > theta_max:
                next_state[3] = np.sign(next_state[3]) * theta_max
                roll_exceeded += 1

        state = next_state
        t += control_dt

        # Check for ground
        if state[2] < 0:
            print(f"Ground contact at t={t:.1f}s")
            break

    print("-" * 60)
    print("\n[5] Final state:")
    print(f"    Position: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})")
    print(f"    Heading: {np.degrees(state[5]):.1f} deg")
    print(f"    Roll: {np.degrees(state[3]):.1f} deg")
    print(f"    Progress: {controller.get_progress()*100:.1f}%")
    print(f"    Roll limit exceeded: {roll_exceeded} times")
    print()

    # Calculate final error
    final_target = trajectory[-1].position
    final_error = np.linalg.norm(state[0:3] - final_target)
    print(f"    Final position error: {final_error:.1f}m")

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == "__main__":
    test_controller()
