def heading_error(
    current_heading: float,
    desired_bearing: float
) -> float:
    """
    Calculate the shortest signed angular error.

    Heading convention:
        0°   = North
        90°  = East
        180° = South
        270° = West

    Returns:
        Error in degrees, in the range [-180, 180).
    """

    error = desired_bearing - current_heading

    # Wrap to [-180, 180)
    error = (error + 180.0) % 360.0 - 180.0

    return error