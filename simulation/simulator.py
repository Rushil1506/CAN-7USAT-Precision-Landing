from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.distance import distance

from guidance.terminal_guidance import should_enter_terminal_guidance
from guidance.loiter_guidance import (
    LoiterDirection,
    LoiterGuidance,
    LoiterGuidanceOutput,
    LoiterPhase,
)

from flight.parafoil import (
    ParafoilState,
    ParafoilConfig,
    DrogueConfig,
    ParafoilModel,
)

from flight.state_machine import (
    FlightState,
    FlightStateMachine,
)


@dataclass
class SimulationSample:
    """
    One recorded point in the simulated flight.
    """

    time: float

    x: float
    y: float

    altitude: float

    vx: float
    vy: float
    vz: float

    heading: float
    heading_rate: float

    steering: float

    state: FlightState

    # Loiter holding orbit telemetry (populated when in TERMINAL_GUIDANCE)
    loiter_phase: Optional[LoiterPhase] = None
    loiter_radial_distance: Optional[float] = None
    loiter_radial_error: Optional[float] = None
    loiter_desired_heading: Optional[float] = None


class ParafoilSimulator:
    """
    Two-stage CAN-7USAT descent simulator.

    Mission:

        DROGUE_DESCENT
              ↓
        600 m trigger
              ↓
        PARAFOIL_DEPLOYMENT
              ↓
        deployment complete
              ↓
        PRECISION_GUIDANCE
              ↓
        FINAL_APPROACH
              ↓
        LANDING

    Any unsafe condition can eventually lead to FAILSAFE.
    """

    def __init__(
        self,
        state: ParafoilState,
        drogue_config: DrogueConfig,
        parafoil_config: ParafoilConfig,
        dt: float = 0.1,
        parafoil_deployment_altitude: float = 600.0,
        deployment_duration: float = 2.0,
        target_position: Optional[LocalPoint] = None,
        final_approach_altitude: float = 120.0,
        final_approach_radius: float = 100.0,
        terminal_margin: float = 100.0,
        terminal_capture_radius: float = 200.0,
        loiter_radius: float = 30.0,
        loiter_capture_tolerance: float = 5.0,
        loiter_direction: LoiterDirection = LoiterDirection.CLOCKWISE,
        max_terminal_guidance_time: float = 120.0,
        loiter_guidance: Optional[LoiterGuidance] = None,
    ):

        # --------------------------------------------------
        # Validate configuration
        # --------------------------------------------------

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        if parafoil_deployment_altitude < 0.0:
            raise ValueError("Deployment altitude cannot be negative")

        if deployment_duration < 0.0:
            raise ValueError("Deployment duration cannot be negative")

        if final_approach_altitude < 0.0:
            raise ValueError("Final approach altitude cannot be negative")

        if final_approach_radius <= 0.0:
            raise ValueError("Final approach radius must be greater than zero")

        if terminal_margin < 0.0:
            raise ValueError("Terminal margin cannot be negative")

        if terminal_capture_radius <= 0.0:
            raise ValueError("Terminal capture radius must be greater than zero")

        if loiter_radius <= 0.0:
            raise ValueError("Loiter radius must be greater than zero")

        if loiter_capture_tolerance <= 0.0:
            raise ValueError("Loiter capture tolerance must be greater than zero")

        if max_terminal_guidance_time <= 0.0:
            raise ValueError("Max terminal guidance time must be greater than zero")

        # --------------------------------------------------
        # Store flight model
        # --------------------------------------------------

        self.state = state
        self.parafoil_config = parafoil_config

        self.model = ParafoilModel(
            state=state,
            drogue_config=drogue_config,
            parafoil_config=parafoil_config,
        )

        # --------------------------------------------------
        # Simulation parameters
        # --------------------------------------------------

        self.dt = dt

        self.parafoil_deployment_altitude = parafoil_deployment_altitude

        self.deployment_duration = deployment_duration

        # Final-approach trigger parameters.
        self.target_position = target_position
        self.final_approach_altitude = final_approach_altitude
        self.final_approach_radius = final_approach_radius
        self.terminal_margin = terminal_margin
        self.terminal_capture_radius = terminal_capture_radius

        # Loiter holding orbit parameters (simulation/prototyping values)
        self.loiter_radius = loiter_radius
        self.loiter_capture_tolerance = loiter_capture_tolerance
        self.loiter_direction = loiter_direction
        self.max_terminal_guidance_time = max_terminal_guidance_time

        if loiter_guidance is not None:
            self.loiter_guidance = loiter_guidance
        elif self.target_position is not None:
            self.loiter_guidance = LoiterGuidance(
                center=self.target_position,
                loiter_radius=self.loiter_radius,
                direction=self.loiter_direction,
                capture_tolerance=self.loiter_capture_tolerance,
                forward_speed=self.parafoil_config.forward_speed,
                max_turn_rate=self.parafoil_config.max_turn_rate,
            )
        else:
            self.loiter_guidance = None

        self.last_loiter_output: Optional[LoiterGuidanceOutput] = None
        self.loiter_history: List[Tuple[float, LoiterGuidanceOutput]] = []

        # Tracking metrics for terminal and approach phases
        self.terminal_entry_time: Optional[float] = None
        self.terminal_entry_altitude: Optional[float] = None
        self.terminal_entry_distance: Optional[float] = None
        self.terminal_entry_glide_range: Optional[float] = None
        self.final_approach_entry_time: Optional[float] = None
        self.final_approach_entry_altitude: Optional[float] = None
        self.final_approach_entry_distance: Optional[float] = None

        self.cumulative_loiter_angle_deg: float = 0.0
        self._last_loiter_bearing: Optional[float] = None

        # --------------------------------------------------
        # Simulation time
        # --------------------------------------------------

        self.time = 0.0

        # None until the 600 m deployment trigger occurs
        self.deployment_start_time: Optional[float] = None

        # --------------------------------------------------
        # State machine
        # --------------------------------------------------

        self.state_machine = FlightStateMachine()

        # --------------------------------------------------
        # Simulation history
        # --------------------------------------------------

        self.history: List[SimulationSample] = []

        # --------------------------------------------------
        # Start mission
        # --------------------------------------------------

        self.state_machine.transition_to(FlightState.WAITING)

        self.state_machine.transition_to(FlightState.DROGUE_DESCENT)

    @property
    def loiter_orbits_completed(self) -> float:
        """Total number of full or partial loiter revolutions completed."""
        return self.cumulative_loiter_angle_deg / 360.0

    @property
    def max_radial_error_during_loiter(self) -> float:
        """Maximum absolute radial error observed during terminal loiter guidance."""
        errs = [
            abs(s.loiter_radial_error)
            for s in self.history
            if s.state == FlightState.TERMINAL_GUIDANCE and s.loiter_radial_error is not None
        ]
        return max(errs) if errs else 0.0

    @property
    def mean_radial_error_during_loiter(self) -> float:
        """Mean absolute radial error observed during terminal loiter guidance."""
        errs = [
            abs(s.loiter_radial_error)
            for s in self.history
            if s.state == FlightState.TERMINAL_GUIDANCE and s.loiter_radial_error is not None
        ]
        return (sum(errs) / len(errs)) if errs else 0.0

    # ======================================================
    # Record simulation state
    # ======================================================

    def record_state(self, steering: float) -> None:
        """
        Store the current state of the simulation.
        """
        current_state = self.state_machine.get_state()
        loiter_phase = None
        loiter_r_dist = None
        loiter_r_err = None
        loiter_cmd_hdg = None

        if current_state == FlightState.TERMINAL_GUIDANCE and self.last_loiter_output is not None:
            loiter_phase = self.last_loiter_output.phase
            loiter_r_dist = self.last_loiter_output.radial_distance
            loiter_r_err = self.last_loiter_output.radial_error
            loiter_cmd_hdg = self.last_loiter_output.desired_heading

        self.history.append(
            SimulationSample(
                time=self.time,
                x=self.state.x,
                y=self.state.y,
                altitude=self.state.altitude,
                vx=self.state.vx,
                vy=self.state.vy,
                vz=self.state.vz,
                heading=self.state.heading,
                heading_rate=self.state.heading_rate,
                steering=steering,
                state=current_state,
                loiter_phase=loiter_phase,
                loiter_radial_distance=loiter_r_dist,
                loiter_radial_error=loiter_r_err,
                loiter_desired_heading=loiter_cmd_hdg,
            )
        )

    # ======================================================
    # Update mission state
    # ======================================================

    def update_mission_state(self) -> None:
        """
        Check whether the CanSat should transition
        to another flight phase.
        """

        current_state = self.state_machine.get_state()

        # --------------------------------------------------
        # DROGUE → PARAFOIL DEPLOYMENT
        # --------------------------------------------------

        if current_state == FlightState.DROGUE_DESCENT:

            if self.state.altitude <= self.parafoil_deployment_altitude:

                self.state_machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)

                # Store the exact simulation time at which
                # the parafoil deployment sequence begins.
                self.deployment_start_time = self.time

        # --------------------------------------------------
        # PARAFOIL DEPLOYMENT → PRECISION GUIDANCE
        # --------------------------------------------------

        elif current_state == FlightState.PARAFOIL_DEPLOYMENT:

            # This should never be None while in the
            # deployment state, but explicitly checking it
            # keeps the code safe and type-checker friendly.
            if self.deployment_start_time is None:

                raise RuntimeError(
                    "Parafoil deployment start time " "was not initialized."
                )

            elapsed = self.time - self.deployment_start_time

            if elapsed >= self.deployment_duration:

                self.state_machine.transition_to(FlightState.PRECISION_GUIDANCE)

        # --------------------------------------------------
        # PRECISION GUIDANCE → FINAL APPROACH / TERMINAL GUIDANCE
        # --------------------------------------------------

        current_state = self.state_machine.get_state()

        if current_state == FlightState.PRECISION_GUIDANCE:

            if self.target_position is not None:
                target_distance = distance(
                    LocalPoint(self.state.x, self.state.y),
                    self.target_position,
                )

                if (
                    self.state.altitude <= self.final_approach_altitude
                    and target_distance <= self.final_approach_radius
                ):
                    self.state_machine.transition_to(
                        FlightState.FINAL_APPROACH
                    )
                    self.final_approach_entry_time = self.time
                    self.final_approach_entry_altitude = self.state.altitude
                    self.final_approach_entry_distance = target_distance

                elif should_enter_terminal_guidance(
                    altitude=self.state.altitude,
                    horizontal_speed=self.parafoil_config.forward_speed,
                    descent_rate=self.parafoil_config.descent_rate,
                    target_distance=target_distance,
                    terminal_margin=self.terminal_margin,
                    terminal_capture_radius=self.terminal_capture_radius,
                ):
                    self.state_machine.transition_to(
                        FlightState.TERMINAL_GUIDANCE
                    )
                    self.terminal_entry_time = self.time
                    self.terminal_entry_altitude = self.state.altitude
                    self.terminal_entry_distance = target_distance
                    self.terminal_entry_glide_range = (
                        self.parafoil_config.forward_speed
                        * self.state.altitude
                        / self.parafoil_config.descent_rate
                    )

        # --------------------------------------------------
        # TERMINAL GUIDANCE → FINAL APPROACH / FAILSAFE
        # --------------------------------------------------

        elif current_state == FlightState.TERMINAL_GUIDANCE:

            if self.terminal_entry_time is None:
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
                target_distance = distance(
                    LocalPoint(self.state.x, self.state.y),
                    self.target_position,
                )

                # FINAL_APPROACH has priority over other terminal logic
                if (
                    self.state.altitude <= self.final_approach_altitude
                    and target_distance <= self.final_approach_radius
                ):
                    self.state_machine.transition_to(
                        FlightState.FINAL_APPROACH
                    )
                    self.final_approach_entry_time = self.time
                    self.final_approach_entry_altitude = self.state.altitude
                    self.final_approach_entry_distance = target_distance

                # Safety guard: maximum time in terminal guidance exceeded
                elif (
                    self.terminal_entry_time is not None
                    and (self.time - self.terminal_entry_time) >= self.max_terminal_guidance_time
                ):
                    self.state_machine.transition_to(
                        FlightState.FAILSAFE
                    )
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
