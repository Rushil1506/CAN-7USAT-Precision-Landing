from dataclasses import dataclass


@dataclass
class PIDController:
    """
    PID controller with output saturation and anti-windup.

    Input:
        Error in degrees.

    Output:
        Normalized steering command in [-1, +1].

        -1.0 = maximum left
         0.0 = neutral
        +1.0 = maximum right
    """

    kp: float
    ki: float
    kd: float

    output_min: float = -1.0
    output_max: float = 1.0

    integral_min: float = -1.0
    integral_max: float = 1.0

    integral: float = 0.0
    previous_error: float = 0.0

    initialized: bool = False

    def reset(self):
        """Reset controller state."""

        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False

    def update(
        self,
        error: float,
        dt: float
    ) -> float:
        """
        Calculate PID output.

        Parameters
        ----------
        error:
            Current control error.

        dt:
            Time since previous controller update in seconds.

        Returns
        -------
        float:
            Saturated control output.
        """

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        # -----------------------------------------
        # Proportional term
        # -----------------------------------------

        proportional = self.kp * error

        # -----------------------------------------
        # Integral term
        # -----------------------------------------

        self.integral += error * dt

        # Anti-windup:
        # limit the accumulated integral
        self.integral = max(
            self.integral_min,
            min(self.integral, self.integral_max)
        )

        integral_term = self.ki * self.integral

        # -----------------------------------------
        # Derivative term
        # -----------------------------------------

        if not self.initialized:

            derivative = 0.0
            self.initialized = True

        else:

            derivative = (
                error - self.previous_error
            ) / dt

        derivative_term = self.kd * derivative

        # -----------------------------------------
        # PID output
        # -----------------------------------------

        output = (
            proportional
            + integral_term
            + derivative_term
        )

        # -----------------------------------------
        # Output saturation
        # -----------------------------------------

        output = max(
            self.output_min,
            min(output, self.output_max)
        )

        # Store error for next iteration
        self.previous_error = error

        return output