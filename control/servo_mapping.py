from dataclasses import dataclass

from .steering import BrakeCommand


@dataclass
class ServoConfig:
    neutral_angle: float
    max_brake_angle: float


@dataclass
class ServoCommand:
    left_angle: float
    right_angle: float


def brake_to_servo(
    brake: BrakeCommand,
    left_config: ServoConfig,
    right_config: ServoConfig
) -> ServoCommand:
    """
    Convert normalized brake commands to servo angles.

    brake values:
        0.0 = neutral
        1.0 = maximum brake

    Servo angle is linearly interpolated between
    neutral and maximum-brake positions.
    """

    left_angle = (
        left_config.neutral_angle
        + brake.left
        * (
            left_config.max_brake_angle
            - left_config.neutral_angle
        )
    )

    right_angle = (
        right_config.neutral_angle
        + brake.right
        * (
            right_config.max_brake_angle
            - right_config.neutral_angle
        )
    )

    return ServoCommand(
        left_angle=left_angle,
        right_angle=right_angle
    )