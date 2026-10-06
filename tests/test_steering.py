import math

from control.steering import (
    BrakeCommand,
    steering_to_brakes
)

from control.servo_mapping import (
    ServoConfig,
    brake_to_servo
)


def test_max_left():

    command = steering_to_brakes(-1.0)

    assert command.left == 1.0
    assert command.right == 0.0


def test_max_right():

    command = steering_to_brakes(1.0)

    assert command.left == 0.0
    assert command.right == 1.0


def test_neutral():

    command = steering_to_brakes(0.0)

    assert command.left == 0.0
    assert command.right == 0.0


def test_partial_left():

    command = steering_to_brakes(-0.4)

    assert math.isclose(command.left, 0.4)
    assert math.isclose(command.right, 0.0)


def test_partial_right():

    command = steering_to_brakes(0.65)

    assert math.isclose(command.left, 0.0)
    assert math.isclose(command.right, 0.65)


def test_steering_clamped():

    command = steering_to_brakes(2.0)

    assert command.right == 1.0

    command = steering_to_brakes(-2.0)

    assert command.left == 1.0


def test_servo_mapping_neutral():

    brake = BrakeCommand(
        left=0.0,
        right=0.0
    )

    left = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=140.0
    )

    right = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=40.0
    )

    result = brake_to_servo(
        brake,
        left,
        right
    )

    assert result.left_angle == 90.0
    assert result.right_angle == 90.0


def test_servo_mapping_max_brake():

    brake = BrakeCommand(
        left=1.0,
        right=1.0
    )

    left = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=140.0
    )

    right = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=40.0
    )

    result = brake_to_servo(
        brake,
        left,
        right
    )

    assert result.left_angle == 140.0
    assert result.right_angle == 40.0


def test_servo_mapping_half_brake():

    brake = BrakeCommand(
        left=0.5,
        right=0.5
    )

    left = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=140.0
    )

    right = ServoConfig(
        neutral_angle=90.0,
        max_brake_angle=40.0
    )

    result = brake_to_servo(
        brake,
        left,
        right
    )

    assert result.left_angle == 115.0
    assert result.right_angle == 65.0