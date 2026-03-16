# -*- coding: utf-8 -*-
import os, json, glob
dirs = sorted(glob.glob('benchmark/outputs/exp_*/'))
latest = dirs[-1]
mf = os.path.join(latest, 'default/seed_001/metrics.json')
with open(mf) as f:
    m = json.load(f)

print(f"Success: {m['success']}")
print(f"Termination: {m['termination_reason']}")
print(f"Flight time: {m['timing']['flight_time']:.1f}s")
print()
q = m['quality']
print(f"ADE:              {q['ADE']:.1f}m")
print(f"FDE:              {q['FDE']:.1f}m")
print(f"FDE_horizontal:   {q['FDE_horizontal']:.1f}m")
print(f"FDE_vertical:     {q['FDE_vertical']:.4f}m")
print(f"mean_cross_track: {q['mean_cross_track_error']:.1f}m")
print(f"max_cross_track:  {q['max_cross_track_error']:.1f}m")
print(f"saturation_ratio: {q['saturation_ratio']:.3f}")
print(f"mean_ctrl_effort: {q['mean_control_effort']:.3f}")
