import math
from dataclasses import dataclass


@dataclass
class ParafoilState:
    """
    Current state of the simulated CAN-7USAT.

    Coordinate system:
        +X = East
        +Y = North

    Heading:
        0°   = North
        90°  = East
        180° = South
        270° = West
    """

    # Position
    x: float
    y: float

    # Altitude
    altitude: float

    # Ground velocity
    vx: float
    vy: float

    # Vertical velocity
    vz: float

    # Heading
    heading: float

    # Heading rate
    heading_rate: float


@dataclass
class DrogueConfig:
    """
    Drogue parachute flight characteristics.

    These values are configurable simulation parameters.
    """

    # Nominal vertical descent rate [m/s]
    descent_rate: float = 17.0

    # Horizontal velocity multiplier.
    #
    # For V1 we assume the drogue has negligible
    # intentional horizontal flight capability.
    horizontal_speed: float = 0.0


@dataclass
class ParafoilConfig:
    """
    Steerable ram-air parafoil parameters.

    These are currently simulation parameters and will
    later be replaced by values obtained from the actual
    parafoil design/testing.
    """

    # Forward airspeed [m/s]
    forward_speed: float = 8.0

    # Vertical descent rate [m/s]
    descent_rate: float = 2.5

    # Maximum heading rate [deg/s]
    max_turn_rate: float = 30.0

    # Wind velocity [m/s]
    wind_x: float = 0.0
    wind_y: float = 0.0


class ParafoilModel:
    """
    Simplified kinematic model of the CAN-7USAT descent system.

    The model supports two flight phases:

        1. DROGUE
        2. PARAFOIL

    Drogue:
        - No steering
        - High vertical descent rate
        - Horizontal motion comes from wind

    Parafoil:
        - Steering enabled
        - Forward flight
        - Wind affects ground velocity
        - Controlled heading change
    """

    def __init__(
        self,
        state: ParafoilState,
        drogue_config: DrogueConfig,
        parafoil_config: ParafoilConfig,
    ):

        self.state = state

        self.drogue_config = drogue_config

        self.parafoil_config = parafoil_config

    # --------------------------------------------------
    # Drogue update
    # --------------------------------------------------

    def update_drogue(self, dt: float) -> None:
        """
        Advance the simulation during drogue descent.

        No precision steering is applied.
        """

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        # --------------------------------------------------
        # Horizontal velocity
        #
        # At this stage we assume no intentional forward
        # flight. Horizontal motion is entirely driven by wind.
        # --------------------------------------------------

        self.state.vx = self.parafoil_config.wind_x
        self.state.vy = self.parafoil_config.wind_y

        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt

        # --------------------------------------------------
        # No intentional heading control
        # --------------------------------------------------

        self.state.heading_rate = 0.0

        # --------------------------------------------------
        # Vertical velocity
        # --------------------------------------------------

        self.state.vz = -self.drogue_config.descent_rate

        # --------------------------------------------------
        # Update altitude
        # --------------------------------------------------

        self.state.altitude += self.state.vz * dt

        self.state.altitude = max(0.0, self.state.altitude)

    # --------------------------------------------------
    # Parafoil update
    # --------------------------------------------------

    def update_parafoil(self, steering: float, dt: float) -> None:
        """
        Advance the simulation during precision flight.
        """

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        # --------------------------------------------------
        # Limit steering
        # --------------------------------------------------

        steering = max(-1.0, min(1.0, steering))

        # --------------------------------------------------
        # Heading rate
        # --------------------------------------------------

        self.state.heading_rate = steering * self.parafoil_config.max_turn_rate

        # --------------------------------------------------
        # Heading
        # --------------------------------------------------

        self.state.heading += self.state.heading_rate * dt

        self.state.heading %= 360.0

        # --------------------------------------------------
        # Convert heading to radians
        # --------------------------------------------------

        heading_rad = math.radians(self.state.heading)

        # --------------------------------------------------
        # Air-relative velocity
        # --------------------------------------------------

        air_vx = self.parafoil_config.forward_speed * math.sin(heading_rad)

        air_vy = self.parafoil_config.forward_speed * math.cos(heading_rad)

        # --------------------------------------------------
        # Wind
        # --------------------------------------------------

        self.state.vx = air_vx + self.parafoil_config.wind_x

        self.state.vy = air_vy + self.parafoil_config.wind_y

        # --------------------------------------------------
        # Horizontal position
        # --------------------------------------------------

        self.state.x += self.state.vx * dt

        self.state.y += self.state.vy * dt

        # --------------------------------------------------
        # Vertical velocity
        # --------------------------------------------------

        self.state.vz = -self.parafoil_config.descent_rate

        # --------------------------------------------------
        # Altitude
        # --------------------------------------------------

        self.state.altitude += self.state.vz * dt

        self.state.altitude = max(0.0, self.state.altitude)
