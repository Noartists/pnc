# -*- coding: utf-8 -*-
"""Patch terminal homing: descent-only, no heading override"""
with open('control/adrc_controller.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the terminal homing block in compute_control
# Strategy: replace the homing block to NOT override heading
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# ========== 0.' in line and 'terminal' in lines[i].lower() or 'terminal' in lines[i+1].lower() if i+1 < len(lines) else False:
        start_idx = i
    if start_idx and '# ========== 1.' in line:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    print(f"Found terminal homing block: lines {start_idx+1} to {end_idx+1}")
    
    # Replace with simpler version: only set flag, no heading override
    new_block = [
        '        # ========== 0. 末端快速下降检测 ==========\n',
        '        # 当轨迹跟踪接近完成且翼伞仍有高度时，\n',
        '        # 增加对称偏转加速下降（不改变航向引导）\n',
        '        progress = self.get_progress()\n',
        '        target_final = self.trajectory[-1].position\n',
        '        altitude_above_target = current_pos[2] - target_final[2]\n',
        '        \n',
        '        # 条件：进度>90% 且 高度>10m → 加速下降\n',
        '        self._terminal_descent = (\n',
        '            progress > 0.90\n',
        '            and altitude_above_target > 10.0\n',
        '        )\n',
        '        \n',
    ]
    
    lines = lines[:start_idx] + new_block + lines[end_idx:]
    print("Replaced terminal homing with terminal descent (heading unchanged)")
else:
    print(f"ERROR: Could not find block. start={start_idx}, end={end_idx}")
    # Debug
    for i, line in enumerate(lines):
        if '========== 0' in line or 'terminal' in line.lower():
            print(f"  Line {i+1}: {line.rstrip()[:80]}")
    import sys; sys.exit(1)

# Now find and update the if-else in compute_control that handles heading
# Remove the heading override block (self._terminal_homing check for heading)
found = False
for i, line in enumerate(lines):
    if 'if self._terminal_homing:' in line:
        # Find the end of this if-else block
        j = i + 1
        indent = len(line) - len(line.lstrip())
        while j < len(lines):
            stripped = lines[j].lstrip()
            if stripped.startswith('else:') and (len(lines[j]) - len(stripped)) == indent:
                # Found the else clause
                # Now find the end of the else block
                k = j + 1
                while k < len(lines):
                    if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= indent:
                        break
                    k += 1
                # Replace the entire if-else with just the else body (normal tracking)
                else_body_start = j + 1
                else_body_end = k
                # Get the else body lines and de-indent by 4 spaces
                normal_lines = []
                for m in range(else_body_start, else_body_end):
                    old_line = lines[m]
                    # Remove one level of indentation (4 spaces)
                    if old_line.startswith(' ' * (indent + 8)):
                        normal_lines.append(' ' * (indent + 4) + old_line[indent + 8:])
                    else:
                        normal_lines.append(old_line)
                
                lines = lines[:i] + normal_lines + lines[else_body_end:]
                found = True
                print(f"Removed heading override block at lines {i+1}-{else_body_end+1}")
                break
            j += 1
        break

if not found:
    print("WARNING: Could not find _terminal_homing heading block (might already be removed)")

# Now update the symmetric deflection check to use _terminal_descent
for i, line in enumerate(lines):
    if "_terminal_homing" in line:
        lines[i] = line.replace("_terminal_homing", "_terminal_descent")
        print(f"  Updated reference at line {i+1}")

with open('control/adrc_controller.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("SUCCESS: Patch applied")
