#ifndef CANSAT_ANGLES_HPP_
#define CANSAT_ANGLES_HPP_

#include <cmath>

namespace cansat {

// Exact mathematical constants matching IEEE-754 double precision
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

/**
 * @brief Python-compatible floor modulo for floating-point values.
 *
 * Guaranteed to return result in [0.0, n) for positive n, exactly matching
 * Python's `a % n` semantics across negative dividends.
 */
inline double floor_mod(double a, double n) noexcept {
    double m = std::fmod(a, n);
    if (m < 0.0) {
        m += n;
    }
    // Handle floating-point negative zero or boundary cases
    if (m == 0.0) {
        m = 0.0;
    }
    return m;
}

/**
 * @brief Normalize angle in degrees to range [0.0, 360.0).
 *
 * Heading convention:
 *   0°   = North (+Y)
 *   90°  = East  (+X)
 *   180° = South (-Y)
 *   270° = West  (-X)
 */
inline double wrap_360(double deg) noexcept {
    return floor_mod(deg, 360.0);
}

/**
 * @brief Calculate the shortest signed angular error between current heading
 *        and desired bearing in degrees.
 *
 * Sign convention:
 *   Positive error (> 0): Target is clockwise / to the right -> steer right.
 *   Negative error (< 0): Target is counter-clockwise / to the left -> steer left.
 *
 * @return Signed heading error in degrees, strictly in [-180.0, +180.0).
 */
inline double heading_error(double current_heading_deg, double desired_bearing_deg) noexcept {
    double raw_diff = desired_bearing_deg - current_heading_deg;
    return floor_mod(raw_diff + 180.0, 360.0) - 180.0;
}

/**
 * @brief Calculate navigation bearing from one local EN coordinate to another.
 *
 * Convention:
 *   +X = East, +Y = North
 *   0° = North, 90° = East
 *
 * @return Bearing in degrees, strictly in [0.0, 360.0).
 */
inline double bearing(double from_x, double from_y, double to_x, double to_y) noexcept {
    double dx = to_x - from_x;
    double dy = to_y - from_y;
    double angle_rad = std::atan2(dx, dy);
    double angle_deg = angle_rad * kRadToDeg;
    return floor_mod(angle_deg, 360.0);
}

/**
 * @brief Convert degrees to radians.
 */
inline double deg_to_rad(double deg) noexcept {
    return deg * kDegToRad;
}

/**
 * @brief Convert radians to degrees.
 */
inline double rad_to_deg(double rad) noexcept {
    return rad * kRadToDeg;
}

} // namespace cansat

#endif // CANSAT_ANGLES_HPP_
