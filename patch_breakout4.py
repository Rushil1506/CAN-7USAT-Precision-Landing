import math
import re

with open("guidance/breakout_guidance.py", "r") as f:
    content = f.read()

# I will replace the inside of `calculate_breakout_geometry` again
# We want to fast path zero wind, and limit search to 0..180

replacement_func = """
def _find_best_turn_angle(position, heading, target, forward_speed, wind_x, wind_y, turn_radius, initial_h_err):
    if abs(wind_x) < 1e-6 and abs(wind_y) < 1e-6:
        return abs(initial_h_err), 1.0 if initial_h_err >= 0.0 else -1.0
        
    best_ang = None
    best_dir = None
    min_mismatch = 999
    
    primary_dir = 1.0 if initial_h_err >= 0.0 else -1.0
    
    for turn_dir in [primary_dir, -primary_dir]:
        prev_err = None
        prev_ang = None
        
        # Scan 0 to 180 in 5 deg steps
        for step in range(37):
            ang = step * 5.0
            res = _evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target)
            if res is None: continue
            err = res[0]
            
            if abs(err) < abs(min_mismatch):
                min_mismatch = err
                best_ang = ang
                best_dir = turn_dir
                
            if prev_err is not None:
                if prev_err * err <= 0 and abs(prev_err - err) < 180:
                    a, b = prev_ang, ang
                    for _ in range(15):
                        mid = (a + b) / 2
                        m_res = _evaluate_turn_angle(mid, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target)
                        if m_res is None: break
                        m_err = m_res[0]
                        if m_err * prev_err <= 0:
                            b = mid
                            err = m_err
                        else:
                            a = mid
                            prev_err = m_err
                    if abs(m_err) < abs(min_mismatch):
                        min_mismatch = m_err
                        best_ang = mid
                        best_dir = turn_dir
            prev_err = err
            prev_ang = ang
            
        if abs(min_mismatch) < 0.5:
            break
            
    return best_ang, best_dir
"""

pattern = r"def _find_best_turn_angle\(.*?\):.*?return best_ang, best_dir"
content = re.sub(pattern, replacement_func.strip("\n"), content, flags=re.DOTALL)

with open("guidance/breakout_guidance.py", "w") as f:
    f.write(content)
print("done")
