from dataclasses import dataclass
from typing import Callable, List, Optional

from navigation.coordinates import LocalPoint
from navigation.distance import distance

from guidance.terminal_guidance import should_enter_terminal_guidance

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

    # ======================================================
    # Record simulation state
    # ======================================================

    def record_state(self, steering: float) -> None:
        """
        Store the current state of the simulation.
        """

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
                state=self.state_machine.get_state(),
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

        # --------------------------------------------------
        # TERMINAL GUIDANCE → FINAL APPROACH
        # --------------------------------------------------

        elif current_state == FlightState.TERMINAL_GUIDANCE:

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

            # Safe placeholder behavior: kinematic parafoil update without premature maneuver
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
        ):

            current_state = self.state_machine.get_state()

            # --------------------------------------------------
            # Precision guidance / final approach
            # --------------------------------------------------

            if current_state in {
                FlightState.PRECISION_GUIDANCE,
                FlightState.TERMINAL_GUIDANCE,
                FlightState.FINAL_APPROACH,
            }:

                steering = steering_function(self.time, self.state)

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
