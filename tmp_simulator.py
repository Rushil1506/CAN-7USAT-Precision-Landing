                        target=self.target_position,
                        current_altitude=self.state.altitude,
                        wind_x=self.parafoil_config.wind_x,
                        wind_y=self.parafoil_config.wind_y,
                        forward_speed=self.parafoil_config.forward_speed,
                        descent_rate=self.parafoil_config.descent_rate,
                        orbit_radius=self.loiter_radius,
                        breakout_altitude_margin=self.breakout_altitude_margin,
                    )
                    
                    if self.loiter_guidance is not None:
                        self.loiter_guidance.center = self.terminal_orbit_plan.orbit_center
                        self.loiter_guidance.direction = self.terminal_orbit_plan.orbit_direction
                        
                    expected_terminal_time = max(0.0, (self.terminal_entry_altitude - self.terminal_orbit_plan.breakout_altitude) / self.parafoil_config.descent_rate)
                    
                    import math
                    orbit_circumference = 2.0 * math.pi * self.loiter_radius
                    safety_time_margin = orbit_circumference / self.parafoil_config.forward_speed
                    
                    # Watchdog
                    if self.max_terminal_guidance_time < 30.0:
                        # Test override
                        self.max_terminal_guidance_time = expected_terminal_time + self.max_terminal_guidance_time
                    else:
                        self.max_terminal_guidance_time = expected_terminal_time + safety_time_margin


            if self.target_position is not None:
                target_distance = distance(
                    LocalPoint(self.state.x, self.state.y),
                    self.target_position,
                )

                # Safety guard: maximum time in terminal guidance exceeded
                if (
                    self.terminal_entry_time is not None
                    and (self.time - self.terminal_entry_time) >= self.max_terminal_guidance_time
                ):
                    self.state_machine.transition_to(
                        FlightState.FAILSAFE
                    )
                else:
                    # Calculate dynamic energy-aligned breakout geometry
                    pos = LocalPoint(self.state.x, self.state.y)
                    geom = calculate_breakout_geometry(
                        position=pos,
                        heading=self.state.heading,
                        target=self.target_position,
                        forward_speed=self.parafoil_config.forward_speed,
                        descent_rate=self.parafoil_config.descent_rate,
                        max_turn_rate=self.parafoil_config.max_turn_rate,
                        breakout_altitude_margin=self.breakout_altitude_margin,
                        wind_x=self.parafoil_config.wind_x,
                        wind_y=self.parafoil_config.wind_y,
                    )
                    self.last_breakout_geometry = geom

                    # Explicit low-altitude feasibility guard:
                    # If current altitude is too low to safely execute the required breakout turn,
                    # do not initiate an aggressive maneuver; enter FAILSAFE.
                    min_turn_altitude = geom.turn_arc_length / geom.glide_ratio
                    if self.state.altitude < min_turn_altitude and geom.turn_angle > 0.0:
                        self.state_machine.transition_to(
                            FlightState.FAILSAFE
                        )

                    # Dynamic energy-aligned breakout condition:
                    # Trigger transition to FINAL_APPROACH when current altitude has depleted
                    # to the calculated breakout altitude.
                    elif self.state.altitude <= geom.breakout_altitude:
                        self.state_machine.transition_to(
                            FlightState.FINAL_APPROACH
                        )
                        self.final_approach_entry_time = self.time
                        self.final_approach_entry_altitude = self.state.altitude
                        self.final_approach_entry_distance = target_distance
                        self.breakout_time = self.time
                        self.breakout_altitude = self.state.altitude
                        self.breakout_distance = target_distance
                        self.breakout_heading = self.state.heading
            else:
                if (
                    self.terminal_entry_time is not None
                    and (self.time - self.terminal_entry_time) >= self.max_terminal_guidance_time
                ):
                    self.state_machine.transition_to(
                        FlightState.FAILSAFE
                    )

        # --------------------------------------------------
        # PRECISION GUIDANCE / TERMINAL GUIDANCE / FINAL APPROACH → LANDING
        # --------------------------------------------------

        current_state = self.state_machine.get_state()

        if self.state.altitude <= 0.0:

            if current_state in {
                FlightState.PRECISION_GUIDANCE,
                FlightState.TERMINAL_GUIDANCE,
                FlightState.FINAL_APPROACH,
            }:
                self.state_machine.transition_to(FlightState.LANDING)

    # ======================================================
    # One simulation step
    # ======================================================

    def step(self, steering: float = 0.0) -> None:
        """
        Advance the simulation by one timestep.
        """

        current_state = self.state_machine.get_state()

        # --------------------------------------------------
        # DROGUE DESCENT
        # --------------------------------------------------

        if current_state == FlightState.DROGUE_DESCENT:

            # No precision steering during drogue descent.
            self.model.update_drogue(dt=self.dt)

            steering = 0.0

        # --------------------------------------------------
        # PARAFOIL DEPLOYMENT
        # --------------------------------------------------

        elif current_state == FlightState.PARAFOIL_DEPLOYMENT:

            # Steering remains disabled while the parafoil
            # is deploying/inflating.
            self.model.update_parafoil(steering=0.0, dt=self.dt)

            steering = 0.0

        # --------------------------------------------------
        # PRECISION GUIDANCE
        # --------------------------------------------------

        elif current_state == FlightState.PRECISION_GUIDANCE:

            self.model.update_parafoil(steering=steering, dt=self.dt)

        # --------------------------------------------------
        # TERMINAL GUIDANCE
        # --------------------------------------------------

        elif current_state == FlightState.TERMINAL_GUIDANCE:

            self.model.update_parafoil(steering=steering, dt=self.dt)

        # --------------------------------------------------
        # FINAL APPROACH
        # --------------------------------------------------

        elif current_state == FlightState.FINAL_APPROACH:

            self.model.update_parafoil(steering=steering, dt=self.dt)

        # --------------------------------------------------
        # LANDING
        # --------------------------------------------------

        elif current_state == FlightState.LANDING:

            steering = 0.0

        # --------------------------------------------------
        # FAILSAFE
        # --------------------------------------------------

        elif current_state == FlightState.FAILSAFE:

            # No precision steering in failsafe.
            steering = 0.0

        # --------------------------------------------------
        # Advance simulation time
        # --------------------------------------------------

        self.time += self.dt

        # --------------------------------------------------
        # Check for state transitions
        # --------------------------------------------------

        self.update_mission_state()

        # --------------------------------------------------
        # Record updated state
        # --------------------------------------------------

        self.record_state(steering=steering)

    # ======================================================
    # Run complete simulation
    # ======================================================

    def run(
        self,
        steering_function: Callable,
        max_time: float,
    ) -> List[SimulationSample]:
        """
        Run the simulation until:

            - maximum simulation time is reached
            - CanSat reaches the ground
            - LANDING state is reached
            - FAILSAFE state is reached

        steering_function is only allowed to influence
        the CanSat during precision guidance and final
        approach.
        """

        if max_time <= 0.0:

            raise ValueError("max_time must be greater than zero")

        # --------------------------------------------------
        # Record initial condition
        # --------------------------------------------------

        self.record_state(steering=0.0)

        # --------------------------------------------------
        # Main simulation loop
        # --------------------------------------------------

        while (
            self.time < max_time
            and self.state.altitude > 0.0
            and self.state_machine.get_state() != FlightState.LANDING
            and self.state_machine.get_state() != FlightState.FAILSAFE
        ):

            current_state = self.state_machine.get_state()

            # --------------------------------------------------
            # Precision guidance / final approach
            # --------------------------------------------------

            if current_state in {
                FlightState.PRECISION_GUIDANCE,
                FlightState.FINAL_APPROACH,
            }:

                steering = steering_function(self.time, self.state)

            # --------------------------------------------------
            # Terminal guidance (autonomous loiter holding orbit)
            # --------------------------------------------------

            elif current_state == FlightState.TERMINAL_GUIDANCE:

                if self.loiter_guidance is not None:
                    loiter_out = self.loiter_guidance.update(
                        position=LocalPoint(self.state.x, self.state.y),
                        current_heading=self.state.heading,
                    )
                    self.last_loiter_output = loiter_out
                    self.loiter_history.append((self.time, loiter_out))
                    steering = loiter_out.steering_command
                    if self.loiter_capture_time is None and loiter_out.phase == LoiterPhase.ORBIT_TRACKING:
                        self.loiter_capture_time = self.time

                    # Track orbit angular progression
                    if self.target_position is not None:
                        current_pos = LocalPoint(self.state.x, self.state.y)
                        current_bearing = bearing(self.target_position, current_pos)
                        if self._last_loiter_bearing is not None:
                            delta_bearing = (
                                (current_bearing - self._last_loiter_bearing + 180.0) % 360.0
                                - 180.0
                            )
                            self.cumulative_loiter_angle_deg += abs(delta_bearing)
                        self._last_loiter_bearing = current_bearing
                else:
                    steering = 0.0

            # --------------------------------------------------
            # All other phases
            # --------------------------------------------------

            else:

                steering = 0.0

            # --------------------------------------------------
            # Advance simulation
            # --------------------------------------------------

            self.step(steering=steering)

        return self.history
