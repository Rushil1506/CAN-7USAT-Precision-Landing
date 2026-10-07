import re

with open("simulation/simulator.py", "r") as f:
    content = f.read()

# 1. Remove the available_terminal_time check from transition
replace_1 = """                    available_terminal_time = (self.state.altitude - geom.breakout_altitude) / self.parafoil_config.descent_rate
                    
                    if available_terminal_time <= self.max_terminal_guidance_time:
                        self.state_machine.transition_to(
                            FlightState.TERMINAL_GUIDANCE
                        )
                    self.terminal_entry_time = self.time
                    self.terminal_entry_altitude = self.state.altitude
                    self.terminal_entry_distance = target_distance"""

with_1 = """                    self.state_machine.transition_to(
                        FlightState.TERMINAL_GUIDANCE
                    )"""

content = content.replace(replace_1, with_1)

# 2. Add dynamic watchdog assignment
pattern_2 = r"            if self\.terminal_entry_time is None:\n.*?self\.terminal_entry_glide_range = \(\n.*?\n.*?\n.*?\n.*?\)"

with_2 = """            if self.terminal_entry_time is None:
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
                
                # Calculate expected terminal duration at entry
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
                
                # Dynamic watchdog
                self.max_terminal_guidance_time = expected_terminal_time + safety_time_margin
"""
content = re.sub(pattern_2, with_2, content, flags=re.DOTALL)

with open("simulation/simulator.py", "w") as f:
    f.write(content)
print("done")
