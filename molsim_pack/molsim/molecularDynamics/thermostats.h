#include <random>
#include <vector>

#include "double3.h"

/**
 * \brief Implements velocity scaling for temperature control in simulations.
 *
 * The VelocityScaling struct provides a simple method to adjust particle velocities
 * to maintain a desired temperature. It scales the velocities based on the current
 * kinetic energy and the target temperature, ensuring that the system remains at
 * the specified thermal state.
 */
struct VelocityScaling
{
  double temperature;
  int degreesOfFreedom;

  /**
   * \brief Constructs a VelocityScaling object with specified temperature and degrees of freedom.
   *
   * Initializes the velocity scaling mechanism with the desired temperature and the number
   * of degrees of freedom, which are essential for calculating the scaling factor.
   *
   * \param temperature        The target temperature to maintain.
   * \param degreesOfFreedom   The number of degrees of freedom in the system.
   */
  VelocityScaling(double temperature, int degreesOfFreedom);

  /**
   * \brief Scales the velocities of particles to achieve the target temperature.
   *
   * This function adjusts the velocities of all particles in the system based on the current
   * kinetic energy. It ensures that the system's temperature matches the desired target by
   * applying a uniform scaling factor to all velocities.
   *
   * \param velocities      A vector of double3 structures representing the velocities of particles.
   * \param kineticEnergy   The current total kinetic energy of the system.
   */
  void scale(std::vector<double3>& velocities, double& kineticEnergy);
};

/**
 * \brief Implements the Nose-Hoover thermostat for NVT ensemble simulations.
 *
 * The NoseHooverNVT struct provides a more sophisticated method for temperature control
 * in molecular dynamics simulations by implementing the Nose-Hoover thermostat. This
 * approach maintains the desired temperature by introducing additional degrees of freedom
 * that act as a heat bath, allowing for more accurate sampling of the canonical ensemble.
 */
struct NoseHooverNVT
{
  double temperature;
  double degreesOfFreedom;
  double timescaleParameter;
  double timeStep;
  double thermostatMass;
  double thermostatVelocity;
  double thermostatForce;
  double thermostatPosition;

  std::mt19937 mt;
  std::normal_distribution<double> normal_dist;

  /**
   * \brief Constructs a NoseHooverNVT object with specified parameters.
   *
   * Initializes the Nose-Hoover thermostat with the target temperature, degrees of freedom,
   * timescale parameter, and integration time step. An optional seed can be provided for
   * the random number generator to ensure reproducibility.
   *
   * \param temperature          The target temperature to maintain.
   * \param degreesOfFreedom     The number of degrees of freedom in the system.
   * \param timescaleParameter   The timescale parameter for the thermostat's dynamics.
   * \param timeStep             The integration time step for the simulation.
   * \param seed                 (Optional) Seed for the random number generator. Defaults to 12.
   */
  NoseHooverNVT(double temperature, int degreesOfFreedom, double timescaleParameter, double timeStep, int seed = 12);

  /**
   * \brief Scales the velocities of particles using the Nose-Hoover thermostat.
   *
   * This function adjusts the particle velocities based on the Nose-Hoover thermostat algorithm,
   * which introduces additional degrees of freedom to control the system's temperature. It
   * ensures that the kinetic energy of the system is regulated to match the desired temperature.
   *
   * \param velocities      A vector of double3 structures representing the velocities of particles.
   * \param kineticEnergy   The current total kinetic energy of the system.
   */
  void scale(std::vector<double3>& velocities, double& kineticEnergy);

  /**
   * \brief Retrieves the current energy associated with the Nose-Hoover thermostat.
   *
   * This function returns the energy of the thermostat's additional degree of freedom, which
   * is part of the Nose-Hoover conserved Hamiltonian.
   *
   * \return The current energy of the Nose-Hoover thermostat.
   */
  double getEnergy();
};