from enum import Enum


class FlightState(Enum):
    BOOT = 0
    WAITING = 1
    DROGUE_DESCENT = 2
    PARAFOIL_DEPLOYMENT = 3
    PRECISION_GUIDANCE = 4
    TERMINAL_GUIDANCE = 5
    FINAL_APPROACH = 6
    LANDING = 7
    FAILSAFE = 8


# --------------------------------------------------
# Valid state transitions
# --------------------------------------------------

VALID_TRANSITIONS = {
    FlightState.BOOT: {FlightState.WAITING, FlightState.FAILSAFE},
    FlightState.WAITING: {FlightState.DROGUE_DESCENT, FlightState.FAILSAFE},
    FlightState.DROGUE_DESCENT: {FlightState.PARAFOIL_DEPLOYMENT, FlightState.FAILSAFE},
    FlightState.PARAFOIL_DEPLOYMENT: {
        FlightState.PRECISION_GUIDANCE,
        FlightState.FAILSAFE,
    },
    FlightState.PRECISION_GUIDANCE: {
        FlightState.TERMINAL_GUIDANCE,
        FlightState.FINAL_APPROACH,
        FlightState.LANDING,
        FlightState.FAILSAFE,
    },
    FlightState.TERMINAL_GUIDANCE: {
        FlightState.FINAL_APPROACH,
        FlightState.LANDING,
        FlightState.FAILSAFE,
    },
    FlightState.FINAL_APPROACH: {FlightState.LANDING, FlightState.FAILSAFE},
    FlightState.LANDING: {FlightState.LANDING},
    FlightState.FAILSAFE: {FlightState.FAILSAFE},
}


# --------------------------------------------------
# Flight State Machine
# --------------------------------------------------


class FlightStateMachine:

    def __init__(self):
        self.state = FlightState.BOOT

    def transition_to(self, new_state: FlightState):

        allowed_states = VALID_TRANSITIONS[self.state]

        if new_state not in allowed_states:

            raise ValueError(
                f"Invalid transition: " f"{self.state.name} → {new_state.name}"
            )

        self.state = new_state

    def get_state(self) -> FlightState:
        return self.state
