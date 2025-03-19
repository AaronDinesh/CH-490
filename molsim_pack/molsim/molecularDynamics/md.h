#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "double3.h"
#include "sample.h"
#include "thermostats.h"
#include "utils.h"

/**
 * \brief The main MolecularDynamics object for running MD simulations.
 *
 * The MolecularDynamics class represents the main simulation object used to run Molecular Dynamics (MD) simulations.
 * It is defined in `src/molecularDynamics/md.h` and can be configured through its constructor to set various
 * parameters for the simulation. The thermostats available (e.g. Nose-Hoover, VelocityScaling) can be found in
 * `src/molecularDynamics/thermostats.h`, and additional sampling tools (e.g. RDF and MSD samplers) in
 * `src/molecularDynamics/samplers.h`.
 *
 * The constructor `MolecularDynamics::MolecularDynamics` initializes the simulation:
 * - It sets up the system based on input parameters such as number of particles, temperature, box size, and number of
 * steps.
 * - It precomputes system properties like volume and density.
 * - It initializes particle velocities according to the Maxwell-Boltzmann distribution (mass = 1, hence momentum =
 * velocity).
 * - It calls `MolecularDynamics::latticeInitialization` to place particles on a cubic lattice to avoid overlaps.
 * - It then refines the initial configuration using `MolecularDynamics::gradientDescent` to move towards a local energy
 * minimum.
 *
 * After initialization, the simulation uses the velocity Verlet integrator to propagate the system:
 * - In the velocity Verlet scheme, velocities are updated in half-steps and positions in full steps.
 * - Forces are calculated by `MolecularDynamics::calculateForce`, which implements pairwise Lennard-Jones interactions.
 * - The Lennard-Jones potential models both repulsive ($r^{-12}$) and attractive ($r^{-6}$) interactions, and is
 * combined with a cutoff.
 * - The potential energy, pressure (via the virial), and forces are computed during the force calculation.
 *
 * The equations of motion implemented are for an NVE ensemble:
 * \f[
 * \mathcal{H}(p,q) = \sum_i^N \frac{p_i^2}{2} + \mathcal{U}(q)
 * \f]
 * with
 * \f[
 * \dot{q} = \frac{\partial \mathcal{H}}{\partial p}, \quad \dot{p} = -\frac{\partial \mathcal{H}}{\partial q}.
 * \f]
 * Using the velocity Verlet integrator:
 * \f[
 * v(t+\frac{\Delta t}{2}) = v(t) + F(t) \frac{\Delta t}{2}
 * \]
 * \f[
 * q(t+\Delta t) = q(t) + v(t)\Delta t + F(t)\frac{(\Delta t)^2}{2}
 * \]
 * \f[
 * v(t+\Delta t) = v(t+\frac{\Delta t}{2}) + F(t+\Delta t) \frac{\Delta t}{2}
 * \]
 * where \f$F(t) = -\frac{\partial \mathcal{U}}{\partial q}\f$.
 *
 * For equilibration, a `VelocityScaling` thermostat may be used to maintain the input temperature. Although the final
 * runs may be NVE (no thermostat), during initial equilibration the thermostat helps adjust velocities to match the
 * given temperature. Other thermostats, such as Nose-Hoover, may also be used (defined in `thermostats.h`).
 *
 * In addition, the class provides sampling tools to measure properties like RDF and MSD from `samplers.h`, and logging
 * capabilities. Overall, this class provides the main interface to run MD simulations, from initialization, through
 * equilibration and production, to analysis of thermodynamic and structural properties.
 */
struct MolecularDynamics
{
  int numberOfParticles;           ///< Number of particles in the simulation.
  double temperature;              ///< Current temperature of the system.
  double dt;                       ///< Integration time step.
  double boxSize;                  ///< Length of the cubic simulation box.
  int numberOfEquilibrationSteps;  ///< Number of equilibration steps before production runs.
  int numberOfProductionSteps;     ///< Number of production steps for data collection.
  int sampleFrequency;             ///< Frequency (in steps) at which samples are taken.
  bool outputPDB;                  ///< Flag indicating whether PDB output should be generated.
  bool useNoseHoover;              ///< Flag indicating whether the Nose-Hoover thermostat is used.

  int step;               ///< Current simulation step index.
  double cutOff = 3.0;    ///< Lennard-Jones potential cutoff radius.
  double cutOffEnergy;    ///< Shifted potential energy value at the cutoff radius.
  double volume;          ///< Volume of the simulation box.
  double density;         ///< Particle density (number of particles per volume).
  int degreesOfFreedom;   ///< Degrees of freedom for the system (3N - constraints).
  double3 totalMomentum;  ///< Total linear momentum of all particles combined.
  double gridSize;        ///< Grid spacing used during lattice initialization.

  VelocityScaling velocityScaling;  ///< Velocity scaling instance for simple thermostatting.
  NoseHooverNVT noseHoover;         ///< Nose-Hoover thermostat instance for NVT ensemble simulation.

  std::vector<double3> positions;  ///< Positions of all particles in the system.
  std::vector<double3> momenta;    ///< Momenta of all particles.
  std::vector<double3> forces;     ///< Forces acting on each particle.

  std::mt19937 mt;                                      ///< Mersenne Twister random number generator.
  std::uniform_real_distribution<double> uniform_dist;  ///< Uniform distribution for random sampling.
  std::normal_distribution<double> normal_dist;         ///< Normal (Gaussian) distribution for random sampling.

  int numberOfSamples;              ///< Number of samples recorded during production.
  double pressure{0.0};             ///< Instantaneous pressure of the system.
  double kineticEnergy{0.0};        ///< Instantaneous kinetic energy.
  double potentialEnergy{0.0};      ///< Instantaneous potential energy.
  double conservedEnergy{0.0};      ///< Instantaneous conserved energy (kinetic + potential [+ thermostat]).
  double observedTemperature{0.0};  ///< Instantaneous observed temperature.

  std::vector<double> time;                  ///< Times at which samples are taken.
  std::vector<double> pressures;             ///< Pressure samples over simulation time.
  std::vector<double> kineticEnergies;       ///< Kinetic energy samples over simulation time.
  std::vector<double> potentialEnergies;     ///< Potential energy samples over simulation time.
  std::vector<double> conservedEnergies;     ///< Conserved energy samples over simulation time.
  std::vector<double> observedTemperatures;  ///< Temperature samples over simulation time.

  double baselineEnergy = 0.0;  ///< Baseline energy for assessing energy drift.
  double driftEnergy = 0.0;     ///< Computed drift in conserved energy.

  SampleRDF rdfSampler;  ///< Radial distribution function (RDF) sampler.
  SampleMSD msdSampler;  ///< Mean-square displacement (MSD) sampler.

  Logger logger;        ///< Logger instance for output messages.
  int frameNumber = 1;  ///< Frame counter for outputting snapshots.

  /**
   * @brief Constructs the MolecularDynamics simulation object.
   *
   * Initializes key parameters including the number of particles, temperature,
   * integration time step, simulation box size, equilibration and production steps,
   * sample frequency, logging level, random number generation seed, and thermostat usage.
   *
   * @param numberOfParticles Number of particles in the simulation.
   * @param temperature Initial temperature of the system.
   * @param dt Integration time step size.
   * @param boxSize Length of the cubic simulation box.
   * @param numberOfEquilibrationSteps Number of equilibration steps.
   * @param numberOfProductionSteps Number of production steps.
   * @param outputPDB Whether to output PDB files.
   * @param sampleFrequency Frequency at which samples are taken.
   * @param logLevel Logging verbosity level.
   * @param seed Random number generator seed.
   * @param useNoseHoover Whether to use the Nose-Hoover thermostat.
   */
  MolecularDynamics(int numberOfParticles, double temperature, double dt, double boxSize,
                    int numberOfEquilibrationSteps, int numberOfProductionSteps, bool outputPDB = true,
                    int sampleFrequency = 100, int logLevel = 0, int seed = 12, bool useNoseHoover = false,
                    int noseHooverTimeScaleParameter = 500);

  double uniform() { return uniform_dist(mt); }
  double normal() { return normal_dist(mt); }

  /**
   * @brief Initializes particle positions on a cubic lattice with slight random perturbations.
   *
   * Distributes particles evenly on a grid within the simulation box.
   * Optionally adds small random displacements to avoid particle overlap.
   */
  void latticeInitialization();

  /**
   * @brief Minimizes the potential energy using a steepest descent method.
   *
   * Iteratively adjusts particle positions in the direction opposite to the gradient of the energy.
   * Steps are accepted or rejected based on energy reduction. Step sizes are adjusted adaptively.
   */
  void gradientDescent();

  /**
   * @brief Calculates forces, potential energy, and pressure using pairwise Lennard-Jones interactions.
   *
   * Only considers interactions within the cutoff radius.
   * Accumulates forces on all particles and computes the total potential energy and pressure.
   */
  void calculateForceFast();

  /**
   * @brief Calculates forces, potential energy, and pressure using pairwise Lennard-Jones interactions.
   *
   * Only considers interactions within the cutoff radius.
   * Accumulates forces on all particles and computes the total potential energy and pressure.
   */
  void calculateForceSlow();

  /**
   * @brief Applies the appropriate thermostat, if any, to control temperature.
   *
   * Uses Nose-Hoover thermostat if enabled or simple velocity scaling during equilibration otherwise.
   */
  void thermostat();

  /**
   * @brief Integrates equations of motion by performing one step of the Verlet integrator.
   *
   * Updates particle positions and momenta, applies thermostats, and recalculates energies and pressures.
   */
  void integrate();

  /**
   * @brief Runs the molecular dynamics simulation for the specified number of equilibration and production steps.
   *
   * Performs force calculations and integration each step. Logs and samples
   * system properties at specified intervals.
   */
  void run();

  /**
   * @brief Logs time-averaged thermodynamic properties (temperature, pressure, energies) at the end of the simulation.
   *
   * Uses block averaging to estimate statistical uncertainties.
   */
  void logThermodynamicalAverages();

  /**
   * @brief Returns a string representation of the current state of the simulation.
   *
   * Includes configuration, run parameters, and current thermodynamic properties.
   *
   * @return A formatted string containing simulation details.
   */
  std::string repr();
};