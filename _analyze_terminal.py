# -*- coding: utf-8 -*-
"""Analyze terminal phase of latest benchmark run"""
import json, numpy as np, os, glob

# Find latest experiment
dirs = sorted(glob.glob('benchmark/outputs/exp_*/'))
latest = dirs[-1]
cf = os.path.join(latest, 'default/seed_001/case.json')
print(f"Analyzing: {cf}")

with open(cf, 'r') as f:
    data = json.load(f)

ref = data['reference_trajectory']
act = data['actual_trajectory']
target = np.array(data['target_position'])
start = np.array(data['start_position'])

print(f"Target: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
print(f"Start:  ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})")
print(f"Ref trajectory: {len(ref)} points, t=[{ref[0]['t']:.1f}, {ref[-1]['t']:.1f}]")
print(f"Ref end: ({ref[-1]['x']:.1f}, {ref[-1]['y']:.1f}, {ref[-1]['z']:.1f})")
print(f"Act trajectory: {len(act)} points, t=[{act[0]['t']:.1f}, {act[-1]['t']:.1f}]")

# Find when the parafoil is closest to target (2D)
min_dist = float('inf')
closest_t = 0
closest_pos = None
closest_z = 0
for ap in act:
    d2d = np.sqrt((ap['x']-target[0])**2 + (ap['y']-target[1])**2)
    if d2d < min_dist:
        min_dist = d2d
        closest_t = ap['t']
        closest_pos = (ap['x'], ap['y'], ap['z'])
        closest_z = ap['z']

print(f"\nClosest approach to target (2D):")
print(f"  t={closest_t:.1f}s, dist_2d={min_dist:.1f}m, altitude={closest_z:.1f}m")
print(f"  pos=({closest_pos[0]:.1f}, {closest_pos[1]:.1f}, {closest_pos[2]:.1f})")

# Analyze last 50s of flight
print(f"\n=== Last 50 seconds ===")
last_entries = [ap for ap in act if ap['t'] > act[-1]['t'] - 50]
for ap in last_entries[::500]:  # Every 5 seconds
    d2d = np.sqrt((ap['x']-target[0])**2 + (ap['y']-target[1])**2)
    d2d_ref = np.sqrt((ap['x']-ref[-1]['x'])**2 + (ap['y']-ref[-1]['y'])**2)
    heading = np.degrees(ap.get('yaw', 0))
    angle_to_target = np.degrees(np.arctan2(target[1]-ap['y'], target[0]-ap['x']))
    print(f"  t={ap['t']:6.1f}s: pos=({ap['x']:7.1f},{ap['y']:7.1f},{ap['z']:5.1f}) "
          f"dist2d_target={d2d:6.1f}m dist2d_refend={d2d_ref:6.1f}m "
          f"yaw={heading:6.1f} angle_to_target={angle_to_target:6.1f}")

# Check: does ref trajectory end at target?
ref_end = np.array([ref[-1]['x'], ref[-1]['y'], ref[-1]['z']])
print(f"\nRef trajectory endpoint: ({ref_end[0]:.1f}, {ref_end[1]:.1f}, {ref_end[2]:.1f})")
print(f"Target:                 ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
print(f"Distance ref_end to target: {np.linalg.norm(ref_end - target):.1f}m")
