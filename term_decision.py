import csv
from flight.state_machine import FlightState
from validate_wind_baseline import run_mission_with_wind
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from guidance.wind_compensation import calculate_wind_corrected_heading

def analyze_terminal_decision():
    scenarios = [
        ("Baseline", 0.0, 0.0),
        ("North (Crosswind)", 0.0, 3.0),
        ("South (Crosswind)", 0.0, -3.0),
        ("East (Tailwind)", 3.0, 0.0),
        ("West (Headwind)", -3.0, 0.0),
        ("Diagonal (NE)", 2.12, 2.12),
        ("Strong (90% Airspeed N)", 0.0, 7.2),
        ("Strong (90% Airspeed S)", 0.0, -7.2),
        ("Strong (90% Airspeed E)", 7.2, 0.0),
        ("Strong (90% Airspeed W)", -7.2, 0.0),
    ]
    
    target = LocalPoint(1200.0, 0.0)
    
    for name, wx, wy in scenarios:
        sim, history = run_mission_with_wind(wx, wy)
        terminal_entry_samples = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
        if terminal_entry_samples:
            entry = terminal_entry_samples[0]
            desired_course = bearing(LocalPoint(entry.x, entry.y), target)
            comp = calculate_wind_corrected_heading(desired_course, 8.0, wx, wy)
            print(f"[{name}] ENTERED TERMINAL GUIDANCE")
            print(f"  Alt: {entry.altitude:.1f}m, Dist: {sim.terminal_entry_distance:.1f}m")
            print(f"  Course: {desired_course:.1f}, W_cross: {comp.crosswind_component:.1f}, W_along: {comp.along_track_wind:.1f}")
            print(f"  Proj GS: {comp.ground_speed_along_track:.1f}, Rg: {sim.terminal_entry_glide_range:.1f}")
            print(f"  Margin: {sim.terminal_margin}, Failsafe: {sim.state_machine.get_state() == FlightState.FAILSAFE}")
        else:
            # Never entered
            # Where did it end up?
            final = history[-1]
            desired_course = bearing(LocalPoint(final.x, final.y), target)
            comp = calculate_wind_corrected_heading(desired_course, 8.0, wx, wy)
            print(f"[{name}] NEVER ENTERED TERMINAL GUIDANCE")
            print(f"  Final Alt: {final.altitude:.1f}m, Final Dist: {sim.terminal_entry_distance if hasattr(sim, 'terminal_entry_distance') else 'N/A'}")
            print(f"  Course: {desired_course:.1f}, W_cross: {comp.crosswind_component:.1f}, W_along: {comp.along_track_wind:.1f}")
            print(f"  Proj GS: {comp.ground_speed_along_track:.1f}")
            print(f"  Failsafe: {sim.state_machine.get_state() == FlightState.FAILSAFE}")

analyze_terminal_decision()
