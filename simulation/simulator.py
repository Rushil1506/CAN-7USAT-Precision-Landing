import math
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
from guidance.breakout_guidance import (
    BreakoutGeometry,
    calculate_breakout_geometry,
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

from environment.wind import WindField, ConstantWindField
from guidance.wind_estimator import WindEstimator
from control.actuator import ActuatorModel, ActuatorConfig
from guidance.loiter_center_planner import LoiterCenterPlanner, LoiterCenterPlannerConfig
from sensors.sensor_model import SensorSuite, SensorNoiseConfig, SensorMeasurement
from estimation.ekf import ExtendedKalmanFilter, EKFConfig, EKFStateEstimate


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

    # Telemetry
    wind_x: float
    wind_y: float
    ground_speed: float
    course_over_ground: float
    crab_angle: float

    # Loiter holding orbit telemetry (populated when in TERMINAL_GUIDANCE)
    loiter_phase: Optional[LoiterPhase] = None
    loiter_radial_distance: Optional[float] = None
    loiter_radial_error: Optional[float] = None
    loiter_desired_heading: Optional[float] = None

    # Loiter dynamic breakout telemetry (populated when in TERMINAL_GUIDANCE)
    breakout_required_altitude: Optional[float] = None
    breakout_altitude_threshold: Optional[float] = None

    # Actuator telemetry
    commanded_steering: Optional[float] = None

    # Loiter planned center telemetry
    loiter_planned_center_x: Optional[float] = None
    loiter_planned_center_y: Optional[float] = None
    loiter_center_offset_x: Optional[float] = None
    loiter_center_offset_y: Optional[float] = None

    # State estimation telemetry (Phase 5.1 EKF & sensor noise)
    estimated_x: Optional[float] = None
    estimated_y: Optional[float] = None
    estimated_altitude: Optional[float] = None
    estimated_heading: Optional[float] = None
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    sigma_heading: Optional[float] = None


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
        breakout_altitude_margin: float = 2.0,
        wind_x: float = 0.0,
        wind_y: float = 0.0,
        wind_field: Optional[WindField] = None,
        wind_estimator: Optional[WindEstimator] = None,
        actuator_model: Optional[ActuatorModel] = None,
        loiter_center_planner: Optional[LoiterCenterPlanner] = None,
        sensor_suite: Optional[SensorSuite] = None,
        ekf: Optional[ExtendedKalmanFilter] = None,
        state_estimation_mode: str = "ideal",
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

        if breakout_altitude_margin < 0.0:
            raise ValueError("Breakout altitude margin cannot be negative")

        # --------------------------------------------------
        # Store flight model
        # --------------------------------------------------

        self.state = state
        self.drogue_config = drogue_config
        self.parafoil_config = parafoil_config

        if wind_field is not None:
            self.wind_field = wind_field
        else:
            self.wind_field = ConstantWindField(wind_x, wind_y)

        # Override config wind fields for convenience as required by single source of truth rule
        # Use initial altitude wind
        initial_wx, initial_wy = self.wind_field.wind_at(self.state.altitude, 0.0)
        self.parafoil_config.wind_x = initial_wx
        self.parafoil_config.wind_y = initial_wy

        self.wind_estimator = wind_estimator or WindEstimator()
        self.wind_estimator.update(initial_wx, initial_wy)

        self.model = ParafoilModel(
            state=state,
            drogue_config=drogue_config,
            parafoil_config=parafoil_config,
        )

        self.actuator_model = actuator_model
        if self.actuator_model is not None:
            self.actuator_model.reset(0.0)

        self.loiter_center_planner = loiter_center_planner
        if self.loiter_center_planner is not None:
            self.loiter_center_planner.reset()

        # State estimation and sensor noise configuration (Phase 5.1)
        valid_modes = {"ideal", "noisy_direct", "noisy_ekf"}
        if state_estimation_mode not in valid_modes:
            raise ValueError(
                f"Invalid state_estimation_mode '{state_estimation_mode}'. "
                f"Must be one of {valid_modes}"
            )
        self.state_estimation_mode = state_estimation_mode
        self.sensor_suite = sensor_suite
        self.ekf = ekf

        if self.state_estimation_mode != "ideal" and self.sensor_suite is None:
            self.sensor_suite = SensorSuite(SensorNoiseConfig(enabled=True))

        if self.state_estimation_mode == "noisy_ekf" and self.ekf is None:
            self.ekf = ExtendedKalmanFilter(
                EKFConfig(
                    enabled=True,
                    forward_speed=self.parafoil_config.forward_speed,
                    descent_rate=self.parafoil_config.descent_rate,
                    max_turn_rate_deg_s=self.parafoil_config.max_turn_rate,
                )
            )

        if self.sensor_suite is not None:
            self.sensor_suite.reset()
            self.last_sensor_meas = self.sensor_suite.update(
                time=0.0,
                true_x=self.state.x,
                true_y=self.state.y,
                true_altitude=self.state.altitude,
                true_vx=self.state.vx,
                true_vy=self.state.vy,
                true_heading_deg=self.state.heading,
            )
        else:
            self.last_sensor_meas = None

        if self.ekf is not None:
            self.ekf.reset(
                init_x=self.state.x,
                init_y=self.state.y,
                init_alt=self.state.altitude,
                init_heading_deg=self.state.heading,
                init_time=0.0,
                forward_speed=0.0,
                descent_rate=self.drogue_config.descent_rate,
            )
            self.last_ekf_estimate = self.ekf.get_estimate()
        else:
            self.last_ekf_estimate = None

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

        # Energy-aligned dynamic breakout parameters (simulation/prototyping values)
        self.breakout_altitude_margin = breakout_altitude_margin
        self.last_breakout_geometry: Optional[BreakoutGeometry] = None
        self.breakout_time: Optional[float] = None
        self.breakout_altitude: Optional[float] = None
        self.breakout_distance: Optional[float] = None
        self.breakout_heading: Optional[float] = None
        self.loiter_capture_time: Optional[float] = None

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

    @property
    def loiter_tracking_duration(self) -> float:
        """Total duration spent in steady ORBIT_TRACKING in seconds."""
        if self.loiter_capture_time is None:
            return 0.0
        end_time = self.breakout_time if self.breakout_time is not None else self.time
        return max(0.0, end_time - self.loiter_capture_time)

    @property
    def final_approach_duration(self) -> float:
        """Total duration spent in FINAL_APPROACH in seconds."""
        if self.final_approach_entry_time is None:
            return 0.0
        return max(0.0, self.time - self.final_approach_entry_time)

    def get_navigation_state(self) -> Tuple[LocalPoint, float, float]:
        """
        Return (position, heading_deg, altitude) available to flight guidance.

        Behavior depends strictly on self.state_estimation_mode:
        - "ideal": Returns simulator ground truth (state.x, state.y, heading, altitude).
        - "noisy_direct": Returns latest raw noisy sensor measurements.
        - "noisy_ekf": Returns EKF state estimates.
        """
        if self.state_estimation_mode == "ideal":
            return (
                LocalPoint(self.state.x, self.state.y),
                self.state.heading,
                self.state.altitude,
            )
        elif self.state_estimation_mode == "noisy_direct":
            if self.sensor_suite is not None:
                return (
                    LocalPoint(self.sensor_suite.latest_x, self.sensor_suite.latest_y),
                    self.sensor_suite.latest_heading_deg,
                    self.sensor_suite.latest_altitude,
                )
            return (
                LocalPoint(self.state.x, self.state.y),
                self.state.heading,
                self.state.altitude,
            )
        elif self.state_estimation_mode == "noisy_ekf":
            if self.last_ekf_estimate is not None:
                return (
                    LocalPoint(self.last_ekf_estimate.x, self.last_ekf_estimate.y),
                    self.last_ekf_estimate.heading_deg,
                    self.last_ekf_estimate.altitude,
                )
            return (
                LocalPoint(self.state.x, self.state.y),
                self.state.heading,
                self.state.altitude,
            )
        return (
            LocalPoint(self.state.x, self.state.y),
            self.state.heading,
            self.state.altitude,
        )

    # ======================================================
    # Record simulation state
    # ======================================================

    def record_state(self, steering: float, commanded_steering: Optional[float] = None) -> None:
        """
        Store the current state of the simulation.
        """
        if commanded_steering is None:
            commanded_steering = steering

        current_state = self.state_machine.get_state()
        loiter_phase = None
        loiter_r_dist = None
        loiter_r_err = None
        loiter_cmd_hdg = None
        breakout_req_alt = None
        breakout_thresh = None

        loiter_center_x = None
        loiter_center_y = None
        loiter_offset_x = None
        loiter_offset_y = None

        if current_state == FlightState.TERMINAL_GUIDANCE:
            if self.last_loiter_output is not None:
                loiter_phase = self.last_loiter_output.phase
                loiter_r_dist = self.last_loiter_output.radial_distance
                loiter_r_err = self.last_loiter_output.radial_error
                loiter_cmd_hdg = self.last_loiter_output.desired_heading
            if self.last_breakout_geometry is not None:
                breakout_req_alt = self.last_breakout_geometry.minimum_required_altitude
                breakout_thresh = self.last_breakout_geometry.breakout_altitude
            if self.loiter_guidance is not None:
                loiter_center_x = self.loiter_guidance.center.x
                loiter_center_y = self.loiter_guidance.center.y
                if self.target_position is not None:
                    loiter_offset_x = loiter_center_x - self.target_position.x
                    loiter_offset_y = loiter_center_y - self.target_position.y

        # Telemetry calculations
        gs = math.hypot(self.state.vx, self.state.vy)

        if gs > 1e-6:
            cog = math.degrees(math.atan2(self.state.vx, self.state.vy)) % 360.0
        else:
            cog = self.state.heading

        crab = (cog - self.state.heading + 180.0) % 360.0 - 180.0

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
                wind_x=self.parafoil_config.wind_x,
                wind_y=self.parafoil_config.wind_y,
                ground_speed=gs,
                course_over_ground=cog,
                crab_angle=crab,
                loiter_phase=loiter_phase,
                loiter_radial_distance=loiter_r_dist,
                loiter_radial_error=loiter_r_err,
                loiter_desired_heading=loiter_cmd_hdg,
                breakout_required_altitude=breakout_req_alt,
                breakout_altitude_threshold=breakout_thresh,
                commanded_steering=commanded_steering,
                loiter_planned_center_x=loiter_center_x,
                loiter_planned_center_y=loiter_center_y,
                loiter_center_offset_x=loiter_offset_x,
                loiter_center_offset_y=loiter_offset_y,
                estimated_x=self.last_ekf_estimate.x if self.last_ekf_estimate is not None else None,
                estimated_y=self.last_ekf_estimate.y if self.last_ekf_estimate is not None else None,
                estimated_altitude=self.last_ekf_estimate.altitude if self.last_ekf_estimate is not None else None,
                estimated_heading=self.last_ekf_estimate.heading_deg if self.last_ekf_estimate is not None else None,
                sigma_x=self.last_ekf_estimate.sigma_x if self.last_ekf_estimate is not None else None,
                sigma_y=self.last_ekf_estimate.sigma_y if self.last_ekf_estimate is not None else None,
                sigma_heading=self.last_ekf_estimate.sigma_heading_deg if self.last_ekf_estimate is not None else None,
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
        prev_state = self.state_machine.get_state()
        nav_pos, nav_heading, nav_alt = self.get_navigation_state()
        current_state = self.state_machine.get_state()

        # --------------------------------------------------
        # DROGUE → PARAFOIL DEPLOYMENT
        # --------------------------------------------------

        if current_state == FlightState.DROGUE_DESCENT:

            if nav_alt <= self.parafoil_deployment_altitude:

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
                    nav_pos,
                    self.target_position,
                )

                if (
                    nav_alt <= self.final_approach_altitude
                    and target_distance <= self.final_approach_radius
                ):
                    self.state_machine.transition_to(
                        FlightState.FINAL_APPROACH
                    )
                    self.final_approach_entry_time = self.time
                    self.final_approach_entry_altitude = nav_alt
                    self.final_approach_entry_distance = target_distance

                elif should_enter_terminal_guidance(
                    altitude=nav_alt,
                    horizontal_speed=self.parafoil_config.forward_speed,
                    descent_rate=self.parafoil_config.descent_rate,
                    target_distance=target_distance,
                    terminal_margin=self.terminal_margin,
                    terminal_capture_radius=self.terminal_capture_radius,
                    wind_x=self.parafoil_config.wind_x,
                    wind_y=self.parafoil_config.wind_y,
                    desired_course_deg=bearing(
                        nav_pos,
                        self.target_position
                    ),
                    wind_field=self.wind_field,
                ):
                    pos = nav_pos

                    from guidance.shear_orbit_planner import solve_shear_terminal_plan

                    plan = solve_shear_terminal_plan(
                        position=pos,
                        target=self.target_position,
                        current_altitude=nav_alt,
                        wind_field=self.wind_field,
                        forward_speed=self.parafoil_config.forward_speed,
                        descent_rate=self.parafoil_config.descent_rate,
                        orbit_radius=self.loiter_radius,
                        breakout_altitude_margin=self.breakout_altitude_margin,
                    )

                    if plan is not None and plan.feasible:
                        self.terminal_orbit_plan = plan
                        self.state_machine.transition_to(
                            FlightState.TERMINAL_GUIDANCE
                        )

                        if self.loiter_guidance is not None:
                            self.loiter_guidance.center = self.terminal_orbit_plan.orbit_center
                            self.loiter_guidance.direction = self.terminal_orbit_plan.orbit_direction

                        if (
                            self.loiter_center_planner is not None
                            and self.loiter_center_planner.config.enabled
                        ):
                            planned_center, _ = self.loiter_center_planner.plan_center(
                                target=self.target_position,
                                wind_x=self.wind_estimator.wind_x,
                                wind_y=self.wind_estimator.wind_y,
                            )
                            if self.loiter_guidance is not None:
                                self.loiter_guidance.center = planned_center

                        from guidance.glide_range import calculate_glide_range
                        self.terminal_entry_glide_range = calculate_glide_range(
                            altitude=nav_alt,
                            descent_rate=self.parafoil_config.descent_rate,
                            horizontal_speed=self.parafoil_config.forward_speed,
                            wind_field=self.wind_field,
                            desired_course_deg=bearing(
                                nav_pos,
                                self.target_position
                            )
                        )


        # --------------------------------------------------
        # TERMINAL GUIDANCE → FINAL APPROACH / FAILSAFE
        # --------------------------------------------------

        elif current_state == FlightState.TERMINAL_GUIDANCE:

            if self.terminal_entry_time is None:
                self.terminal_entry_time = self.time
                self.terminal_entry_altitude = nav_alt
                if self.target_position is not None:
                    self.terminal_entry_distance = distance(
                        nav_pos,
                        self.target_position,
                    )
                if self.target_position is not None and hasattr(self, 'terminal_orbit_plan'):

                    if self.loiter_guidance is not None:
                        self.loiter_guidance.center = self.terminal_orbit_plan.orbit_center
                        self.loiter_guidance.direction = self.terminal_orbit_plan.orbit_direction

                    if (
                        self.loiter_center_planner is not None
                        and self.loiter_center_planner.config.enabled
                    ):
                        planned_center, _ = self.loiter_center_planner.plan_center(
                            target=self.target_position,
                            wind_x=self.wind_estimator.wind_x,
                            wind_y=self.wind_estimator.wind_y,
                        )
                        if self.loiter_guidance is not None:
                            self.loiter_guidance.center = planned_center

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
                    nav_pos,
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
                    pos = nav_pos
                    geom = calculate_breakout_geometry(
                        position=pos,
                        heading=nav_heading,
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
                    if nav_alt < min_turn_altitude and geom.turn_angle > 0.0:
                        self.state_machine.transition_to(
                            FlightState.FAILSAFE
                        )

                    # Dynamic energy-aligned breakout condition:
                    # Trigger transition to FINAL_APPROACH when current altitude has depleted
                    # to the calculated breakout altitude.
                    elif nav_alt <= geom.breakout_altitude:
                        self.state_machine.transition_to(
                            FlightState.FINAL_APPROACH
                        )
                        self.final_approach_entry_time = self.time
                        self.final_approach_entry_altitude = nav_alt
                        self.final_approach_entry_distance = target_distance
                        self.breakout_time = self.time
                        self.breakout_altitude = nav_alt
                        self.breakout_distance = target_distance
                        self.breakout_heading = nav_heading
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

        new_state = self.state_machine.get_state()
        if new_state != prev_state and self.sensor_suite is not None and getattr(self.sensor_suite, "fault_config", None) is not None:
            self.sensor_suite.fault_config.notify_state_transition(new_state, self.time)

    # ======================================================
    # One simulation step
    # ======================================================

    def step(self, steering: float = 0.0) -> None:
        """
        Advance the simulation by one timestep.
        """

        # Update current environment wind based on altitude and time
        current_wx, current_wy = self.wind_field.wind_at(self.state.altitude, getattr(self, "time", 0.0))
        self.parafoil_config.wind_x = current_wx
        self.parafoil_config.wind_y = current_wy

        # Determine current flight phase kinematics
        current_state = self.state_machine.get_state()
        if current_state == FlightState.DROGUE_DESCENT:
            current_va = 0.0
            current_vz = self.drogue_config.descent_rate
            step_steering = 0.0
        elif current_state == FlightState.PARAFOIL_DEPLOYMENT:
            current_va = self.parafoil_config.forward_speed
            current_vz = self.parafoil_config.descent_rate
            step_steering = 0.0
        else:
            current_va = self.parafoil_config.forward_speed
            current_vz = self.parafoil_config.descent_rate
            step_steering = steering

        # Update synthetic sensor suite from true physical state
        if self.sensor_suite is not None:
            self.last_sensor_meas = self.sensor_suite.update(
                time=self.time,
                true_x=self.state.x,
                true_y=self.state.y,
                true_altitude=self.state.altitude,
                true_vx=self.state.vx,
                true_vy=self.state.vy,
                true_heading_deg=self.state.heading,
            )

        # Update shared wind estimator (representing flight software)
        # Flight software calculates wind as measured GNSS ground velocity minus airspeed vector.
        # Under drogue descent, airspeed is zero and horizontal ground velocity is pure wind drift.
        if self.state_estimation_mode in {"noisy_ekf", "noisy_direct"} and self.sensor_suite is not None:
            if current_state == FlightState.DROGUE_DESCENT:
                meas_wx = self.sensor_suite.latest_vx
                meas_wy = self.sensor_suite.latest_vy
            else:
                if self.state_estimation_mode == "noisy_ekf" and self.last_ekf_estimate is not None:
                    psi_deg = self.last_ekf_estimate.heading_deg
                else:
                    psi_deg = self.sensor_suite.latest_heading_deg
                psi_rad = math.radians(psi_deg)
                air_vx = self.parafoil_config.forward_speed * math.sin(psi_rad)
                air_vy = self.parafoil_config.forward_speed * math.cos(psi_rad)
                meas_wx = self.sensor_suite.latest_vx - air_vx
                meas_wy = self.sensor_suite.latest_vy - air_vy
            self.wind_estimator.update(meas_wx, meas_wy)
        else:
            # Ideal baseline: pass true wind to wind_estimator
            self.wind_estimator.update(current_wx, current_wy)

        # Update EKF if enabled
        if self.ekf is not None and self.last_sensor_meas is not None:
            # Inject flight software wind estimate and flight phase kinematics into EKF
            self.ekf.set_kinematics(current_va, current_vz)
            self.ekf.set_wind_estimate(self.wind_estimator.wind_x, self.wind_estimator.wind_y)
            # EKF prediction step using commanded/actual steering
            self.ekf.predict(self.dt, step_steering)
            # EKF measurement correction steps
            if self.last_sensor_meas.gnss_valid:
                self.ekf.correct_gps(
                    meas_x=self.last_sensor_meas.gnss_x,
                    meas_y=self.last_sensor_meas.gnss_y,
                    meas_vx=self.last_sensor_meas.gnss_vx,
                    meas_vy=self.last_sensor_meas.gnss_vy,
                    meas_time=self.last_sensor_meas.gnss_timestamp if self.last_sensor_meas.gnss_timestamp is not None else self.last_sensor_meas.time,
                )
            if self.last_sensor_meas.baro_valid:
                self.ekf.correct_baro(
                    meas_alt=self.last_sensor_meas.baro_altitude,
                    meas_time=self.last_sensor_meas.baro_timestamp if self.last_sensor_meas.baro_timestamp is not None else self.last_sensor_meas.time,
                )
            if self.last_sensor_meas.heading_valid:
                self.ekf.correct_heading(
                    meas_heading_deg=self.last_sensor_meas.heading_deg,
                    meas_time=self.last_sensor_meas.heading_timestamp if self.last_sensor_meas.heading_timestamp is not None else self.last_sensor_meas.time,
                )

            self.last_ekf_estimate = self.ekf.get_estimate()

        current_state = self.state_machine.get_state()

        commanded_steering = steering
        actual_steering = steering

        # --------------------------------------------------
        # DROGUE DESCENT
        # --------------------------------------------------

        if current_state == FlightState.DROGUE_DESCENT:

            # No precision steering during drogue descent.
            self.model.update_drogue(dt=self.dt)

            commanded_steering = 0.0
            actual_steering = 0.0
            if self.actuator_model is not None:
                self.actuator_model.reset(0.0)

        # --------------------------------------------------
        # PARAFOIL DEPLOYMENT
        # --------------------------------------------------

        elif current_state == FlightState.PARAFOIL_DEPLOYMENT:

            # Steering remains disabled while the parafoil
            # is deploying/inflating.
            self.model.update_parafoil(steering=0.0, dt=self.dt)

            commanded_steering = 0.0
            actual_steering = 0.0
            if self.actuator_model is not None:
                self.actuator_model.reset(0.0)

        # --------------------------------------------------
        # PRECISION GUIDANCE
        # --------------------------------------------------

        elif current_state == FlightState.PRECISION_GUIDANCE:

            if self.actuator_model is not None:
                actual_steering = self.actuator_model.update(commanded_steering, self.dt)
            else:
                actual_steering = commanded_steering

            self.model.update_parafoil(steering=actual_steering, dt=self.dt)

        # --------------------------------------------------
        # TERMINAL GUIDANCE
        # --------------------------------------------------

        elif current_state == FlightState.TERMINAL_GUIDANCE:

            if self.actuator_model is not None:
                actual_steering = self.actuator_model.update(commanded_steering, self.dt)
            else:
                actual_steering = commanded_steering

            self.model.update_parafoil(steering=actual_steering, dt=self.dt)

        # --------------------------------------------------
        # FINAL APPROACH
        # --------------------------------------------------

        elif current_state == FlightState.FINAL_APPROACH:

            if self.actuator_model is not None:
                actual_steering = self.actuator_model.update(commanded_steering, self.dt)
            else:
                actual_steering = commanded_steering

            self.model.update_parafoil(steering=actual_steering, dt=self.dt)

        # --------------------------------------------------
        # LANDING
        # --------------------------------------------------

        elif current_state == FlightState.LANDING:

            commanded_steering = 0.0
            actual_steering = 0.0
            if self.actuator_model is not None:
                self.actuator_model.reset(0.0)

        # --------------------------------------------------
        # FAILSAFE
        # --------------------------------------------------

        elif current_state == FlightState.FAILSAFE:

            # No precision steering in failsafe.
            commanded_steering = 0.0
            actual_steering = 0.0
            if self.actuator_model is not None:
                self.actuator_model.reset(0.0)

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

        self.record_state(steering=actual_steering, commanded_steering=commanded_steering)

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
                    # Provide the estimator's filtered wind to loiter guidance
                    self.loiter_guidance.set_wind(self.wind_estimator.wind_x, self.wind_estimator.wind_y)
                    nav_pos, nav_heading, _ = self.get_navigation_state()
                    loiter_out = self.loiter_guidance.update(
                        position=nav_pos,
                        current_heading=nav_heading,
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
