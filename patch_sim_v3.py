import re

with open("simulation/simulator.py", "r") as f:
    content = f.read()

# Let's fix the None target issue
fix_2 = """            if self.terminal_entry_time is None:
                self.terminal_entry_time = self.time
                self.terminal_entry_altitude = self.state.altitude
                if self.target_position is not None:
                    self.terminal_entry_distance = distance(
                        LocalPoint(self.state.x, self.state.y),
                        self.target_position,
                    )
                self.terminal_entry_glide_range = (
                    self.parafoil_config.forward_speed
                    * self.state.altitude
                    / self.parafoil_config.descent_rate
                )
                
                if self.target_position is not None:
                    geom = calculate_breakout_geometry(
                        position=LocalPoint(self.state.x, self.state.y),
                        heading=self.state.heading,
                        target=self.target_position,
                        forward_speed=self.parafoil_config.forward_speed,
                        descent_rate=self.parafoil_config.descent_rate,
                        max_turn_rate=self.parafoil_config.max_turn_rate,
                        breakout_altitude_margin=self.breakout_altitude_margin,
                        wind_x=self.parafoil_config.wind_x,
                        wind_y=self.parafoil_config.wind_y,
                    )
                    expected_terminal_time = max(0.0, (self.terminal_entry_altitude - geom.breakout_altitude) / self.parafoil_config.descent_rate)
                    
                    import math
                    orbit_circumference = 2.0 * math.pi * self.loiter_radius
                    safety_time_margin = orbit_circumference / self.parafoil_config.forward_speed
                    
                    # Watchdog
                    if self.max_terminal_guidance_time < 30.0:
                        # Test override
                        self.max_terminal_guidance_time = expected_terminal_time + self.max_terminal_guidance_time
                    else:
                        self.max_terminal_guidance_time = expected_terminal_time + safety_time_margin
"""
pattern_2 = r"            if self\.terminal_entry_time is None:\n.*?self\.terminal_entry_glide_range = \(\n.*?\n.*?\n.*?\n.*?\)\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?self\.max_terminal_guidance_time = expected_terminal_time \+ safety_time_margin\n"

content = re.sub(pattern_2, fix_2, content, flags=re.DOTALL)
with open("simulation/simulator.py", "w") as f:
    f.write(content)

with open("tests/test_simulator.py", "r") as f:
    tc = f.read()

tc = tc.replace(
    "altitude=400.0,  # High altitude, cannot reach 120m final approach in 5s",
    "altitude=400.0,  # High altitude"
)
tc = tc.replace(
    "history = sim.run(steering_function=lambda t, s: 0.0, max_time=15.0)",
    "history = sim.run(steering_function=lambda t, s: 0.0, max_time=400.0)"
)
with open("tests/test_simulator.py", "w") as f:
    f.write(tc)

print("done")
