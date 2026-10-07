import math
from environment.wind import PiecewiseWindField
from guidance.glide_range import _integrate_glide_range

def test_convergence():
    wind = PiecewiseWindField([(0.0, (10.0, 0.0)), (1000.0, (0.0, 0.0))])
    altitude = 1000.0
    descent_rate = 2.5
    horizontal_speed = 8.0
    course = 90.0 # East (Tailwind increasing to headwind? Wait, wind is (10,0) at 0 altitude, meaning it blows East. So tailwind at 0, zero wind at 1000)

    for dh in [10.0, 1.0, 0.1]:
        # I'll monkeypatch dh inside _integrate_glide_range for this test
        # Actually I can't monkeypatch local variable. I'll just copy the function here to test convergence.
        pass

def integrate_test(dh: float):
    wind = PiecewiseWindField([(0.0, (10.0, 0.0)), (1000.0, (0.0, 0.0))])
    altitude = 1000.0
    descent_rate = 2.5
    horizontal_speed = 8.0
    desired_course_deg = 90.0

    if altitude < 1e-3:
        return 0.0, True
        
    num_steps = int(math.ceil(altitude / dh))
    actual_dh = altitude / num_steps
    
    integrated_range = 0.0
    
    # Evaluate at h_0
    from guidance.wind_compensation import calculate_wind_corrected_heading
    h_prev = altitude
    wx_prev, wy_prev = wind.wind_at_altitude(h_prev)
    comp_prev = calculate_wind_corrected_heading(desired_course_deg, horizontal_speed, wx_prev, wy_prev)
    
    v_prev = comp_prev.ground_speed_along_track
    
    for i in range(1, num_steps + 1):
        h_curr = altitude - i * actual_dh
        if h_curr < 0:
            h_curr = 0.0
            
        wx_curr, wy_curr = wind.wind_at_altitude(h_curr)
        comp_curr = calculate_wind_corrected_heading(desired_course_deg, horizontal_speed, wx_curr, wy_curr)
        
        v_curr = comp_curr.ground_speed_along_track
        
        dt = actual_dh / descent_rate
        integrated_range += 0.5 * (v_prev + v_curr) * dt
        v_prev = v_curr
        
    return integrated_range

if __name__ == "__main__":
    for dh in [100.0, 10.0, 1.0, 0.1, 0.01]:
        res = integrate_test(dh)
        print(f"dh={dh:6.2f}m : range = {res:.6f}m")
