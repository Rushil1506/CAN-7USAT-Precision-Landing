from dataclasses import dataclass


@dataclass
class BrakeCommand:
    """
    Normalized brake commands.

    0.0 = no brake
    1.0 = maximum brake
    """

    left: float
    right: float


def steering_to_brakes(
    steering: float
) -> BrakeCommand:
    """
    Convert normalized steering command into
    differential brake commands.

    Convention:

        steering = -1
            → maximum LEFT brake

        steering = 0
            → both brakes neutral

        steering = +1
            → maximum RIGHT brake

    Returns:
        BrakeCommand with values in [0, 1].
    """

    # Protect against values outside the expected range
    steering = max(-1.0, min(1.0, steering))

    if steering < 0.0:

        return BrakeCommand(
            left=abs(steering),
            right=0.0
        )

    elif steering > 0.0:

        return BrakeCommand(
            left=0.0,
            right=steering
        )

    else:

        return BrakeCommand(
            left=0.0,
            right=0.0
        )