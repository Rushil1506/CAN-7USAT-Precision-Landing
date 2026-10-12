#include "cansat/steering.hpp"

#include <algorithm>
#include <cmath>

namespace cansat {

BrakeCommand steering_to_brakes(double steering) noexcept {
    // Handle non-finite inputs defensively
    if (!std::isfinite(steering)) {
        return BrakeCommand{0.0, 0.0};
    }

    // Clamp input to expected range [-1.0, +1.0]
    double clamped_steering = steering;
    if (clamped_steering < -1.0) clamped_steering = -1.0;
    if (clamped_steering > 1.0) clamped_steering = 1.0;

    if (clamped_steering < 0.0) {
        return BrakeCommand{std::abs(clamped_steering), 0.0};
    } else if (clamped_steering > 0.0) {
        return BrakeCommand{0.0, clamped_steering};
    } else {
        return BrakeCommand{0.0, 0.0};
    }
}

} // namespace cansat
