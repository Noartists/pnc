"""Analyze case.json cross-track errors"""
import json
import numpy as np

with open('benchmark/outputs/exp_20260208_182653/default/seed_001/case.json', 'r') as f:
    data = json.load(f)

ref_traj = data['reference_trajectory']
act_traj = data['actual_trajectory']
print(f'Reference trajectory points: {len(ref_traj)}')
print(f'Actual trajectory points: {len(act_traj)}')

# Print structure of first entries
if ref_traj:
    print(f'Ref traj first entry keys: {list(ref_traj[0].keys())}')
if act_traj:
    print(f'Act traj first entry keys: {list(act_traj[0].keys())}')

# Print first/last entries
if ref_traj:
    r0 = ref_traj[0]
    rn = ref_traj[-1]
    print(f'\nRef start: t={r0.get("t","?")}, pos=({r0.get("x","?")}, {r0.get("y","?")}, {r0.get("z","?")})')
    print(f'Ref end:   t={rn.get("t","?")}, pos=({rn.get("x","?")}, {rn.get("y","?")}, {rn.get("z","?")})')
if act_traj:
    a0 = act_traj[0]
    an = act_traj[-1]
    print(f'\nAct start: t={a0.get("t","?")}, pos=({a0.get("x","?")}, {a0.get("y","?")}, {a0.get("z","?")})')
    print(f'Act end:   t={an.get("t","?")}, pos=({an.get("x","?")}, {an.get("y","?")}, {an.get("z","?")})')

# Compute cross-track errors ourselves
if ref_traj and act_traj:
    ref_pts = np.array([[p['x'], p['y'], p['z']] for p in ref_traj])
    ref_times = np.array([p['t'] for p in ref_traj])
    
    ct_errors = []
    z_errors = []
    total_errors = []
    times = []
    for ap in act_traj[::100]:  # Sample every 100th point (every 1 second)
        t = ap['t']
        ax, ay, az = ap['x'], ap['y'], ap['z']
        # Find closest ref point by time
        idx = np.searchsorted(ref_times, t)
        idx = min(idx, len(ref_pts)-1)
        rx, ry, rz = ref_pts[idx]
        ct = np.sqrt((ax-rx)**2 + (ay-ry)**2)
        z_err = abs(az - rz)
        total = np.sqrt((ax-rx)**2 + (ay-ry)**2 + (az-rz)**2)
        ct_errors.append(ct)
        z_errors.append(z_err)
        total_errors.append(total)
        times.append(t)
    
    ct_errors = np.array(ct_errors)
    z_errors = np.array(z_errors)
    total_errors = np.array(total_errors)
    
    print(f'\n=== Cross-track error analysis ===')
    print(f'  Mean lateral error: {np.mean(ct_errors):.1f} m')
    print(f'  Max lateral error: {np.max(ct_errors):.1f} m')
    print(f'  Mean altitude error: {np.mean(z_errors):.1f} m')
    print(f'  Max altitude error: {np.max(z_errors):.1f} m')
    print(f'  Mean total error: {np.mean(total_errors):.1f} m')
    
    # Print samples
    print(f'\n=== Time series (sampled) ===')
    step = max(1, len(ct_errors) // 25)
    for i in range(0, len(ct_errors), step):
        print(f'  t={times[i]:6.1f}s: lat_err={ct_errors[i]:6.1f}m, z_err={z_errors[i]:6.1f}m, total={total_errors[i]:6.1f}m')
    # Always print last
    print(f'  t={times[-1]:6.1f}s: lat_err={ct_errors[-1]:6.1f}m, z_err={z_errors[-1]:6.1f}m, total={total_errors[-1]:6.1f}m')
    
    # When does cross_track error stay above 50m?
    print(f'\n=== Periods with lateral error > 50m ===')
    in_violation = False
    start_t = 0
    for i, (t, e) in enumerate(zip(times, ct_errors)):
        if e > 50 and not in_violation:
            in_violation = True
            start_t = t
        elif e <= 50 and in_violation:
            in_violation = False
            print(f'  t=[{start_t:.1f}, {t:.1f}], duration={t-start_t:.1f}s')
    if in_violation:
        print(f'  t=[{start_t:.1f}, {times[-1]:.1f}], duration={times[-1]-start_t:.1f}s (ongoing at sim end)')
    
    # Events
    print(f'\n=== Events ===')
    for ev in data.get('events', []):
        print(f'  {ev}')
