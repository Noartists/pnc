# -*- coding: utf-8 -*-
"""Patch the controller to add terminal homing - match on ASCII parts only"""
with open('control/adrc_controller.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the target region by ASCII content
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'target_pos = output.ref_position_closest' in line and start_idx is None:
        # Go back to find "# 2."
        for j in range(i, max(i-5, 0), -1):
            if '# 2.' in lines[j]:
                start_idx = j
                break
    if start_idx is not None and 'output.delta_symmetric = delta_s' in line:
        end_idx = i + 3  # Include the glide_ratio lines + blank line
        break

if start_idx is not None and end_idx is not None:
    print(f"Found target region: lines {start_idx+1} to {end_idx+1}")
    for i in range(start_idx, end_idx):
        print(f"  {i+1}: {lines[i].rstrip()[:60]}")
    
    # Build the replacement
    new_lines = [
        '        # 2. 计算下降率控制 (对称偏转)\n',
        '        if getattr(self, \'_terminal_homing\', False):\n',
        '            # 末端归航模式：使用大对称偏转快速下降\n',
        '            # 70% 预算给下降，留 30% 给航向修正\n',
        '            delta_s = self.max_deflection * 0.7\n',
        '            glide_required = 2.47\n',
        '            v_horiz = np.linalg.norm(current_vel[:2])\n',
        '            v_vert = abs(current_vel[2]) if len(current_vel) > 2 else 1.0\n',
        '            glide_current = v_horiz / max(v_vert, 0.1)\n',
        '        else:\n',
        '            # 正常模式：高度控制使用最近轨迹点\n',
        '            target_pos = output.ref_position_closest\n',
        '            delta_s, glide_required, glide_current = self.compute_symmetric_deflection(\n',
        '                current_pos=current_pos,\n',
        '                current_vel=current_vel,\n',
        '                target_pos=target_pos\n',
        '            )\n',
        '\n',
    ]
    
    # Replace
    result = lines[:start_idx] + new_lines + lines[end_idx:]
    
    with open('control/adrc_controller.py', 'w', encoding='utf-8') as f:
        f.writelines(result)
    print("SUCCESS: Patch applied")
else:
    print(f"ERROR: Could not find target region. start_idx={start_idx}, end_idx={end_idx}")
