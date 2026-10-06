import math
import pytest

from control.pid import PIDController


def test_zero_error():

    pid = PIDController(
        kp=0.1,
        ki=0.01,
        kd=0.05
    )

    output = pid.update(
        error=0.0,
        dt=0.1
    )

    assert math.isclose(output, 0.0)


def test_positive_error():

    pid = PIDController(
        kp=0.1,
        ki=0.0,
        kd=0.0
    )

    output = pid.update(
        error=5.0,
        dt=0.1
    )

    assert math.isclose(output, 0.5)


def test_negative_error():

    pid = PIDController(
        kp=0.1,
        ki=0.0,
        kd=0.0
    )

    output = pid.update(
        error=-5.0,
        dt=0.1
    )

    assert math.isclose(output, -0.5)


def test_output_saturation():

    pid = PIDController(
        kp=1.0,
        ki=0.0,
        kd=0.0
    )

    output = pid.update(
        error=100.0,
        dt=0.1
    )

    assert output == 1.0


def test_negative_output_saturation():

    pid = PIDController(
        kp=1.0,
        ki=0.0,
        kd=0.0
    )

    output = pid.update(
        error=-100.0,
        dt=0.1
    )

    assert output == -1.0


def test_integral_limit():

    pid = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        integral_min=-1.0,
        integral_max=1.0
    )

    for _ in range(100):

        pid.update(
            error=10.0,
            dt=1.0
        )

    assert pid.integral == 1.0


def test_reset():

    pid = PIDController(
        kp=0.1,
        ki=0.1,
        kd=0.1
    )

    pid.update(
        error=10.0,
        dt=0.1
    )

    pid.reset()

    assert pid.integral == 0.0
    assert pid.previous_error == 0.0
    assert pid.initialized is False


def test_invalid_dt():

    pid = PIDController(
        kp=0.1,
        ki=0.1,
        kd=0.1
    )

    with pytest.raises(ValueError):

        pid.update(
            error=10.0,
            dt=0.0
        )