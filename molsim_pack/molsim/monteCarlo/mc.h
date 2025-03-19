#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "double3.h"
#include "energyVirial.h"
#include "utils.h"

/**
 * \brief Monte Carlo simulation class for particle systems.
 *
 * The MonteCarlo class encapsulates the Monte Carlo simulation methods and properties
 * for simulating particle interactions in a defined system. It includes system settings,
 * particle positions, energy calculations, and simulation control functions.
 */
struct MonteCarlo
{
  int numberOfParticles;
  double temperature;
  double boxSize;
  int numberOfInitCycles;
  int numberOfProdCycles;
  int sampleFrequency;
  double maxDisplacement;
  double translationProbability;
  double pressure;
  double volumeProbability;
  double maxVolumeChange;
  double swapProbability;
  bool optimizeMCMoves;
  double chemicalPotential;
  bool outputPDB;

  double cutOff{3.0};
  EnergyVirial cutOffPrefactor;
  double volume;
  double density;
  double beta;
  double fugacity;

  int cycle{0};
  double translationAttempted{0.0};
  double translationAccepted{0.0};
  double translationAcceptance{0.0};
  double volumeAttempted{0.0};
  double volumeAccepted{0.0};
  double volumeAcceptance{0.0};

  double insertionAttempted{0.0};
  double insertionAccepted{0.0};
  double insertionAcceptance{0.0};
  double deletionAttempted{0.0};
  double deletionAccepted{0.0};
  double deletionAcceptance{0.0};

  std::vector<double3> positions;

  std::mt19937 mt;
  std::uniform_real_distribution<double> uniform_dist;

  EnergyVirial runningEnergyVirial;
  EnergyVirial totalEnergyVirial;

  EnergyVirial drift;
  double virialPressure;
  std::vector<double> driftEnergies;
  std::vector<double> pressures;
  std::vector<double> widomWeights;
  std::vector<double> energies;
  std::vector<double> volumes;
  std::vector<double> particleCounts;
  std::vector<double> densities;
  std::vector<double> chemicalPotentials;

  Logger logger;
  int frameNumber = 1;

  /**
   * \brief Constructs a MonteCarlo simulation object.
   *
   * Initializes the Monte Carlo simulation with the specified parameters.
   *
   * \param numberOfParticles Number of particles in the system.
   * \param numberOfInitCycles Number of initialization cycles.
   * \param numberOfProdCycles Number of production cycles.
   * \param temperature Temperature of the system.
   * \param boxSize Size of the simulation box.
   * \param maxDisplacement Maximum displacement in a move.
   * \param translationProbability
   * \param pressure Pressure to couple to for NPT.
   * \param volumeProbability Probability of performing a volume move between (0, 1)
   * \param maxVolumeChange Maximum volume change of a volume move
   * \param chemicalPotential
   * \param swapProbability probability to add or delete a particle.
   * \param sampleFrequency Frequency of sampling the system.
   * \param logLevel Logging level.
   * \param outputPDB Flag whether to write to a pdb file.
   * \param seed Seed for random number generator.
   */
  MonteCarlo(int numberOfParticles, int numberOfInitCycles, int numberOfProdCycles, double temperature, double boxSize,
             double maxDisplacement, double translationProbability = 1.0, bool optimizeMCMoves = true,
             double pressure = 0.0, double volumeProbability = 0.0, double maxVolumeChange = 1.0,
             double chemicalPotential = 0.0, double swapProbability = 0.0, int sampleFrequency = 100, int logLevel = 0,
             bool outputPDB = true, int seed = 12);

  /**
   * \brief Generates a uniform random number between 0 and 1.
   *
   * \return A random double between 0 and 1.
   */
  double uniform() { return uniform_dist(mt); }

  /**
   * \brief Performs a translation move on a particle.
   *
   * Attempts to move a particle to a new position and accepts or rejects based on energy change.
   */
  void translationMove();

  /**
   * \brief Optimizes the maximum displacement based on acceptance rate.
   *
   * Adjusts the maximum displacement to achieve an optimal acceptance rate.
   */
  void optimizeMaxDisplacement();

  /**
   * \brief Performs a volume move on the box.
   *
   * Attempts to change the boxSize and accepts or rejects based on energy change.
   */
  void volumeMove();

  /**
   * \brief Optimizes the maximum volume displacement based on acceptance rate.
   *
   * Adjusts the maxvolume change to achieve an optimal acceptance rate.
   */
  void optimizeVolumeChange();

  /**
   * \brief Performs an insertion or deletion move on the box.
   *
   * Attempts to change the paritcle number and accepts or rejects based on energy change.
   */
  void swapMove();

  /**
   * \brief Computes the pressure of the system.
   *
   * Calculates the current pressure and records it.
   */
  void computePressure();

  /**
   * \brief Computes the chemical potential of the system.
   *
   * Estimates the chemical potential and records it.
   */
  void computeChemicalPotential();

  /**
   * \brief Runs the Monte Carlo simulation.
   *
   * Executes the simulation for the specified number of cycles.
   */
  void run();

  void logThermodynamicalAverages();

  /**
   * \brief Returns a string representation of the Monte Carlo simulation.
   *
   * Provides detailed information about the current state of the simulation.
   *
   * \return A string containing the representation.
   */
  std::string repr();
};