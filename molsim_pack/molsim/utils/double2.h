#pragma once

/**
 * \brief Represents a 2D vector with double precision.
 *
 * The double2 struct encapsulates two double-precision floating-point values,
 * commonly used to represent 2D coordinates or vectors in mathematical and
 * simulation calculations.
 */
struct double2
{
  double x;  ///< The x-coordinate or first component of the vector.
  double y;  ///< The y-coordinate or second component of the vector.

  /**
   * \brief Constructs a double2 object with specified x and y values.
   *
   * \param x Initial value for the x-coordinate (default is 0.0).
   * \param y Initial value for the y-coordinate (default is 0.0).
   */
  double2(double x = 0.0, double y = 0.0) : x(x), y(y) {};
};

static inline double2 operator+(const double2& a, const double2& b) { return double2(a.x + b.x, a.y + b.y); };
static inline double2 operator-(const double2& a, const double2& b) { return double2(a.x - b.x, a.y - b.y); };
static inline double dot(const double2& a, const double2& b) { return a.x * b.x + a.y * b.y; };

/**
 * @brief Wraps a vector into the simulation box using periodic boundary
 * conditions.
 *
 * Adjusts each component of the vector by subtracting the box size times the
 * nearest integer of the component divided by the box size.
 *
 * @param v The vector to be wrapped.
 * @param boxSize The size of the simulation box.
 * @return The wrapped vector within the simulation box.
 */
static double2 wrap(const double2& v, const double& boxSize)
{
  // Apply periodic boundary conditions using round
  return double2(v.x - (boxSize * std::round(v.x / boxSize)), v.y - (boxSize * std::round(v.y / boxSize)));
}