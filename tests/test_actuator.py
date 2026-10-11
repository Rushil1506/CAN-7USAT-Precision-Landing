"""
Tests for Actuator Dynamics Model (Phase 4.3).

Validates:
- Command saturation
- Transport delay buffer FIFO
- Slew rate limiting (positive and negative)
- Asymmetric deadband thresholding and linear rescaling
- State reset
- Configuration parameter validation
- Pass-through identity
- Integration with ParafoilSimulator
"""

import math
import pytest

from control.actuator import ActuatorConfig, ActuatorModel
from navigation.coordinates import LocalPoint
from flight.parafoil import ParafoilConfig, DrogueConfig, ParafoilState
from flight.state_machine import FlightState
from simulation.simulator import ParafoilSimulator


# ======================================================================
# Unit Tests: Configuration & Validation
# ======================================================================


def test_actuator_config_defaults():
    config = ActuatorConfig()
    assert config.steering_min == -1.0
    assert config.steering_max == 1.0
    assert config.max_slew_rate is None
    assert config.transport_delay == 0.0
    assert config.deadband_positive == 0.0
    assert config.deadband_negative == 0.0
    assert config.is_passthrough is True


def test_actuator_config_validation_invalid_limits():
    with pytest.raises(ValueError, match="steering_min must be less than steering_max"):
        ActuatorConfig(steering_min=1.0, steering_max=-1.0)

    with pytest.raises(ValueError, match="steering_min must be less than steering_max"):
        ActuatorConfig(steering_min=0.5, steering_max=0.5)


def test_actuator_config_validation_invalid_slew():
    with pytest.raises(ValueError, match="max_slew_rate must be positive"):
        ActuatorConfig(max_slew_rate=0.0)

    with pytest.raises(ValueError, match="max_slew_rate must be positive"):
        ActuatorConfig(max_slew_rate=-2.0)


def test_actuator_config_validation_invalid_delay():
    with pytest.raises(ValueError, match="transport_delay cannot be negative"):
        ActuatorConfig(transport_delay=-0.05)


def test_actuator_config_validation_invalid_deadband():
    with pytest.raises(ValueError, match="deadband_positive cannot be negative"):
        ActuatorConfig(deadband_positive=-0.01)

    with pytest.raises(ValueError, match="deadband_negative cannot be negative"):
        ActuatorConfig(deadband_negative=-0.01)

    with pytest.raises(ValueError, match="deadband_positive must be strictly less than steering_max"):
        ActuatorConfig(deadband_positive=1.0)

    with pytest.raises(ValueError, match="deadband_negative must be strictly less than abs"):
        ActuatorConfig(deadband_negative=1.0)


# ======================================================================
# Unit Tests: ActuatorModel Behavior
# ======================================================================


def test_actuator_passthrough():
    model = ActuatorModel(ActuatorConfig())
    assert model.is_passthrough is True
    assert model.update(0.5, dt=0.1) == pytest.approx(0.5)
    assert model.update(-0.8, dt=0.1) == pytest.approx(-0.8)
    assert model.update(0.0, dt=0.1) == pytest.approx(0.0)


def test_actuator_command_saturation():
    config = ActuatorConfig(steering_min=-0.8, steering_max=0.6)
    model = ActuatorModel(config)
    assert model.update(1.0, dt=0.1) == pytest.approx(0.6)
    assert model.update(-1.0, dt=0.1) == pytest.approx(-0.8)
    assert model.update(0.2, dt=0.1) == pytest.approx(0.2)


def test_actuator_slew_rate_limiting():
    # Slew rate 2.0/s: in dt=0.1s, max change is 0.2
    config = ActuatorConfig(max_slew_rate=2.0)
    model = ActuatorModel(config)
    
    # Step from 0 to 1.0
    out1 = model.update(1.0, dt=0.1)
    assert out1 == pytest.approx(0.2)
    out2 = model.update(1.0, dt=0.1)
    assert out2 == pytest.approx(0.4)
    out3 = model.update(1.0, dt=0.1)
    assert out3 == pytest.approx(0.6)

    # Step down to -1.0 from 0.6
    out4 = model.update(-1.0, dt=0.1)
    assert out4 == pytest.approx(0.4)
    out5 = model.update(-1.0, dt=0.1)
    assert out5 == pytest.approx(0.2)
    out6 = model.update(-1.0, dt=0.1)
    assert out6 == pytest.approx(0.0)
    out7 = model.update(-1.0, dt=0.1)
    assert out7 == pytest.approx(-0.2)


def test_actuator_slew_rate_small_step_not_limited():
    config = ActuatorConfig(max_slew_rate=2.0)
    model = ActuatorModel(config)
    
    # Small step within rate capability (0.1 < 0.2)
    out = model.update(0.1, dt=0.1)
    assert out == pytest.approx(0.1)


def test_actuator_transport_delay():
    # Delay 0.2s with dt=0.1s -> 2 step delay
    config = ActuatorConfig(transport_delay=0.2)
    model = ActuatorModel(config)
    
    # Initial command is 1.0
    out1 = model.update(1.0, dt=0.1)
    assert out1 == pytest.approx(0.0)  # Delayed: seeing initial buffer value (0.0)
    
    out2 = model.update(1.0, dt=0.1)
    assert out2 == pytest.approx(0.0)  # Delayed: seeing initial buffer value (0.0)
    
    out3 = model.update(1.0, dt=0.1)
    assert out3 == pytest.approx(1.0)  # 2 steps elapsed, command arrives


def test_actuator_transport_delay_with_varying_commands():
    config = ActuatorConfig(transport_delay=0.3)
    model = ActuatorModel(config)
    dt = 0.1  # 3 delay steps
    
    cmds = [0.2, 0.4, 0.6, 0.8, 1.0]
    # For steps 0, 1, 2, outputs should be 0.0
    assert model.update(cmds[0], dt=dt) == pytest.approx(0.0)
    assert model.update(cmds[1], dt=dt) == pytest.approx(0.0)
    assert model.update(cmds[2], dt=dt) == pytest.approx(0.0)
    # Step 3 receives cmds[0]
    assert model.update(cmds[3], dt=dt) == pytest.approx(0.2)
    # Step 4 receives cmds[1]
    assert model.update(cmds[4], dt=dt) == pytest.approx(0.4)


def test_actuator_deadband_symmetric():
    config = ActuatorConfig(deadband_positive=0.05, deadband_negative=0.05)
    model = ActuatorModel(config)
    dt = 0.1
    
    # Commands within deadband produce 0.0
    assert model.update(0.02, dt=dt) == pytest.approx(0.0)
    assert model.update(-0.03, dt=dt) == pytest.approx(0.0)
    assert model.update(0.05, dt=dt) == pytest.approx(0.0)
    assert model.update(-0.05, dt=dt) == pytest.approx(0.0)
    
    # Command beyond deadband is rescaled
    # For cmd = 0.10: (0.10 - 0.05) / (1.0 - 0.05) = 0.05 / 0.95 ≈ 0.0526315
    assert model.update(0.10, dt=dt) == pytest.approx(0.05 / 0.95)
    
    # At full scale cmd = 1.0: (1.0 - 0.05) / (1.0 - 0.05) = 1.0
    assert model.update(1.0, dt=dt) == pytest.approx(1.0)
    assert model.update(-1.0, dt=dt) == pytest.approx(-1.0)


def test_actuator_deadband_asymmetric():
    config = ActuatorConfig(deadband_positive=0.10, deadband_negative=0.04)
    model = ActuatorModel(config)
    dt = 0.1
    
    # Within respective deadbands
    assert model.update(0.08, dt=dt) == pytest.approx(0.0)
    assert model.update(-0.03, dt=dt) == pytest.approx(0.0)
    
    # Beyond negative deadband: (-0.10 + 0.04) / (1.0 - 0.04) = -0.06 / 0.96 = -0.0625
    assert model.update(-0.10, dt=dt) == pytest.approx(-0.0625)
    
    # Beyond positive deadband: (0.20 - 0.10) / (1.0 - 0.10) = 0.10 / 0.90
    assert model.update(0.20, dt=dt) == pytest.approx(0.10 / 0.90)


def test_actuator_combined_dynamics():
    # Delay: 0.1s (1 step)
    # Slew rate: 2.0/s (max step 0.2 at dt=0.1)
    # Deadband: 0.05
    config = ActuatorConfig(
        transport_delay=0.1,
        max_slew_rate=2.0,
        deadband_positive=0.05,
        deadband_negative=0.05,
    )
    model = ActuatorModel(config)
    dt = 0.1
    
    # Step 1: Cmd 0.8. In delay buffer -> output is 0.0
    out1 = model.update(0.8, dt=dt)
    assert out1 == pytest.approx(0.0)
    
    # Step 2: Cmd 0.8. Out of buffer comes 0.8 from step 1.
    # Deadband rescales 0.8 -> (0.8 - 0.05) / 0.95 = 0.75 / 0.95 ≈ 0.78947
    # Slew rate limits delta from 0.0 to 0.2 max!
    out2 = model.update(0.8, dt=dt)
    assert out2 == pytest.approx(0.2)
    
    # Step 3: Delta again limited to +0.2 -> 0.4
    out3 = model.update(0.8, dt=dt)
    assert out3 == pytest.approx(0.4)


def test_actuator_reset():
    config = ActuatorConfig(transport_delay=0.2, max_slew_rate=1.0)
    model = ActuatorModel(config)
    dt = 0.1
    
    model.update(1.0, dt=dt)
    model.update(1.0, dt=dt)
    model.update(1.0, dt=dt)
    assert model.current_output > 0.0
    
    # Reset to 0.0
    model.reset(0.0)
    assert model.current_output == 0.0
    
    # Buffer should be reset: step 1 output should be 0.0
    assert model.update(1.0, dt=dt) == pytest.approx(0.0)


def test_actuator_invalid_dt():
    model = ActuatorModel(ActuatorConfig())
    with pytest.raises(ValueError, match="dt must be positive"):
        model.update(0.5, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        model.update(0.5, dt=-0.1)


# ======================================================================
# Integration Tests: Simulator with ActuatorModel
# ======================================================================


def _make_test_simulator(actuator_model=None):
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=500.0,
        vx=0.0,
        vy=8.0,
        vz=-2.5,
        heading=0.0,
        heading_rate=0.0,
    )
    drogue_config = DrogueConfig(descent_rate=15.0)
    parafoil_config = ParafoilConfig(
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
    )
    return ParafoilSimulator(
        state=state,
        drogue_config=drogue_config,
        parafoil_config=parafoil_config,
        dt=0.1,
        parafoil_deployment_altitude=600.0,  # Below initial altitude -> starts in PRECISION_GUIDANCE directly after deployment logic
        target_position=LocalPoint(0.0, 500.0),
        actuator_model=actuator_model,
    )


def test_simulator_without_actuator_model_records_identical_steering():
    sim = _make_test_simulator(actuator_model=None)
    # Transition to PRECISION_GUIDANCE via PARAFOIL_DEPLOYMENT
    sim.state_machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    sim.state_machine.transition_to(FlightState.PRECISION_GUIDANCE)
    
    sim.step(0.75)
    sample = sim.history[-1]
    assert sample.steering == pytest.approx(0.75)
    assert sample.commanded_steering == pytest.approx(0.75)


def test_simulator_with_actuator_model_records_dynamics():
    actuator = ActuatorModel(ActuatorConfig(max_slew_rate=2.0))
    sim = _make_test_simulator(actuator_model=actuator)
    sim.state_machine.transition_to(FlightState.PARAFOIL_DEPLOYMENT)
    sim.state_machine.transition_to(FlightState.PRECISION_GUIDANCE)
    
    # Command 1.0, slew rate limits to 0.2 at dt=0.1
    sim.step(1.0)
    sample = sim.history[-1]
    assert sample.commanded_steering == pytest.approx(1.0)
    assert sample.steering == pytest.approx(0.2)
    assert sim.model.state.heading_rate == pytest.approx(30.0 * 0.2)
