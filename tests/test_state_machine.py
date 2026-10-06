import pytest

from flight.state_machine import FlightState, FlightStateMachine


def test_initial_state():

    machine = FlightStateMachine()

    assert machine.get_state() == FlightState.BOOT


def test_boot_to_waiting():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)

    assert machine.get_state() == FlightState.WAITING


def test_waiting_to_drogue():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)

    machine.transition_to(FlightState.DROGUE_DESCENT)

    assert machine.get_state() == FlightState.DROGUE_DESCENT


def test_drogue_to_parafoil_deployment():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)

    machine.transition_to(FlightState.DROGUE_DESCENT)

    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)

    assert machine.get_state() == FlightState.PARAFOIL_DEPLOYMENT


def test_parafoil_deployment_to_guidance():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)

    machine.transition_to(FlightState.DROGUE_DESCENT)

    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)

    machine.transition_to(FlightState.PRECISION_GUIDANCE)

    assert machine.get_state() == FlightState.PRECISION_GUIDANCE


def test_complete_normal_sequence():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)
    machine.transition_to(FlightState.FINAL_APPROACH)
    machine.transition_to(FlightState.LANDING)

    assert machine.get_state() == FlightState.LANDING


def test_invalid_transition():

    machine = FlightStateMachine()

    with pytest.raises(ValueError):

        machine.transition_to(FlightState.LANDING)


def test_failsafe():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.FAILSAFE)

    assert machine.get_state() == FlightState.FAILSAFE


def test_failsafe_is_terminal():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.FAILSAFE)

    with pytest.raises(ValueError):

        machine.transition_to(FlightState.WAITING)


def test_landing_is_terminal():

    machine = FlightStateMachine()

    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)
    machine.transition_to(FlightState.FINAL_APPROACH)
    machine.transition_to(FlightState.LANDING)

    machine.transition_to(FlightState.LANDING)

    assert machine.get_state() == FlightState.LANDING


def test_precision_guidance_to_terminal_guidance():
    machine = FlightStateMachine()
    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)

    machine.transition_to(FlightState.TERMINAL_GUIDANCE)
    assert machine.get_state() == FlightState.TERMINAL_GUIDANCE


def test_terminal_guidance_to_final_approach():
    machine = FlightStateMachine()
    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)
    machine.transition_to(FlightState.TERMINAL_GUIDANCE)

    machine.transition_to(FlightState.FINAL_APPROACH)
    assert machine.get_state() == FlightState.FINAL_APPROACH


def test_terminal_guidance_to_landing():
    machine = FlightStateMachine()
    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)
    machine.transition_to(FlightState.TERMINAL_GUIDANCE)

    machine.transition_to(FlightState.LANDING)
    assert machine.get_state() == FlightState.LANDING


def test_terminal_guidance_to_failsafe():
    machine = FlightStateMachine()
    machine.transition_to(FlightState.WAITING)
    machine.transition_to(FlightState.DROGUE_DESCENT)
    machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    machine.transition_to(FlightState.PRECISION_GUIDANCE)
    machine.transition_to(FlightState.TERMINAL_GUIDANCE)

    machine.transition_to(FlightState.FAILSAFE)
    assert machine.get_state() == FlightState.FAILSAFE


@pytest.mark.parametrize(
    "source_state",
    [
        FlightState.BOOT,
        FlightState.WAITING,
        FlightState.DROGUE_DESCENT,
        FlightState.PARAFOIL_DEPLOYMENT,
        FlightState.FINAL_APPROACH,
        FlightState.LANDING,
        FlightState.FAILSAFE,
    ],
)
def test_invalid_transitions_to_terminal_guidance(source_state):
    machine = FlightStateMachine()
    machine.state = source_state
    with pytest.raises(ValueError):
        machine.transition_to(FlightState.TERMINAL_GUIDANCE)
