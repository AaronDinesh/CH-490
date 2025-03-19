#pragma once

#include <algorithm>
#include <cmath>
#include <string>

/**
 * \brief Represents a 3D vector with double precision.
 *
 * The double3 union encapsulates three double-precision floating-point values
 * (x, y, z) and provides common vector operations for mathematical computations.
 */
union double3
{
  double v[3];  ///< Array representation for accessing vector components.
  struct
  {
    double x, y, z;  ///< Components of the vector.
  };

  /**
   * \brief Default constructor initializing all components to 0.0.
   */
  double3() : x(0.0), y(0.0), z(0.0) {}

  /**
   * \brief Constructs a double3 object with specified values for x, y, and z.
   *
   * \param x Value for the x-component.
   * \param y Value for the y-component.
   * \param z Value for the z-component.
   */
  double3(double x, double y, double z) : x(x), y(y), z(z) {}

  bool operator==(double3 const &rhs) const { return (x == rhs.x) && (y == rhs.y) && (z == rhs.z); }

  double3 operator-() const { return double3(-this->x, -this->y, -this->z); }
  double3 &operator+=(const double3 &b)
  {
    this->x += b.x, this->y += b.y, this->z += b.z;
    return *this;
  }
  double3 &operator-=(const double3 &b)
  {
    this->x -= b.x, this->y -= b.y, this->z -= b.z;
    return *this;
  }
  double3 &operator/=(const double &s)
  {
    this->x /= s, this->y /= s, this->z /= s;
    return *this;
  }
  double3 &operator*=(const double &s)
  {
    this->x *= s, this->y *= s, this->z *= s;
    return *this;
  }

  /**
   * \brief Computes the absolute value of each component of the vector.
   *
   * \param v1 The input vector.
   * \return A new double3 vector with absolute values of the components.
   */
  inline static double3 abs(double3 v1) { return double3(std::abs(v1.x), std::abs(v1.y), std::abs(v1.z)); }

  /**
   * \brief Computes the dot product of two vectors.
   *
   * \param v1 First input vector.
   * \param v2 Second input vector.
   * \return The dot product as a double.
   */
  inline static double dot(const double3 &v1, const double3 &v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

  /**
   * \brief Computes the component-wise maximum of two vectors.
   *
   * \param v1 First input vector.
   * \param v2 Second input vector.
   * \return A double3 object with the maximum of each component.
   */
  inline static double3 max(const double3 &v1, const double3 &v2)
  {
    return double3(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
  }

  /**
   * \brief Computes the component-wise minimum of two vectors.
   *
   * \param v1 First input vector.
   * \param v2 Second input vector.
   * \return A double3 object with the minimum of each component.
   */
  inline static double3 min(const double3 &v1, const double3 &v2)
  {
    return double3(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
  }

  /**
   * \brief Computes the cross product of two vectors.
   *
   * \param v1 First input vector.
   * \param v2 Second input vector.
   * \return A double3 object representing the cross product.
   */
  inline static double3 cross(const double3 &v1, const double3 &v2)
  {
    return double3(v1.y * v2.z - v2.y * v1.z, v1.z * v2.x - v2.z * v1.x, v1.x * v2.y - v2.x * v1.y);
  }

  /**
   * \brief Normalizes the vector to have a magnitude of 1.
   *
   * \param v The input vector.
   * \return A normalized double3 vector.
   */
  inline static double3 normalize(const double3 &v)
  {
    double f = 1.0 / sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
    return double3(f * v.x, f * v.y, f * v.z);
  }

  /**
   * \brief Converts the vector to a string representation.
   *
   * \return A string in the format "(x, y, z)".
   */
  std::string to_string()
  {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
  }
};

inline double3 operator*(const double3 &a, const double3 &b) { return double3(a.x * b.x, a.y * b.y, a.z * b.z); }

inline double3 operator-(const double3 &a, const double3 &b) { return double3(a.x - b.x, a.y - b.y, a.z - b.z); }

inline double3 operator+(const double3 &a, const double3 &b) { return double3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline double3 operator/(const double3 &a, const double3 &b) { return double3(a.x / b.x, a.y / b.y, a.z / b.z); }

inline double3 operator+(const double3 &a, double b) { return double3(a.x + b, a.y + b, a.z + b); }

inline double3 operator-(const double3 &a, double b) { return double3(a.x - b, a.y - b, a.z - b); }

inline double3 operator*(const double3 &a, double b) { return double3(a.x * b, a.y * b, a.z * b); }

inline double3 operator/(const double3 &a, double b) { return double3(a.x / b, a.y / b, a.z / b); }

inline double3 operator*(const double &a, const double3 &b) { return double3(a * b.x, a * b.y, a * b.z); }

inline double3 sqrt(const double3 &a) { return double3(std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z)); }

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
static double3 wrap(const double3 &v, const double &boxSize)
{
  // Apply periodic boundary conditions using round
  return double3(v.x - (boxSize * std::round(v.x / boxSize)), v.y - (boxSize * std::round(v.y / boxSize)),
                 v.z - (boxSize * std::round(v.z / boxSize)));
}

/**
 * @brief Wraps a vector into the simulation box using floor-based periodic
 * boundary conditions.
 *
 * Adjusts each component of the vector by subtracting the box size times the
 * floor of the component divided by the box size.
 *
 * @param v The vector to be wrapped.
 * @param boxSize The size of the simulation box.
 * @return The wrapped vector within the simulation box.
 */
static double3 wrapFloor(const double3 &v, const double &boxSize)
{
  // Apply periodic boundary conditions using floor
  return double3(v.x - (boxSize * std::floor(v.x / boxSize)), v.y - (boxSize * std::floor(v.y / boxSize)),
                 v.z - (boxSize * std::floor(v.z / boxSize)));
}
