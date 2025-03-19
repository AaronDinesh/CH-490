#include "mc.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "writePDB.h"

MonteCarlo::MonteCarlo(int numberOfParticles, int numberOfInitCycles, int numberOfProdCycles, double temperature,
                       double boxSize, double maxDisplacement, double translationProbability, bool optimizeMCMoves,
                       double pressure, double volumeProbability, double maxVolumeChange, double chemicalPotential,
                       double swapProbability, int sampleFrequency, int logLevel, bool outputPDB, int seed)
    : numberOfParticles(numberOfParticles),
      numberOfInitCycles(numberOfInitCycles),
      numberOfProdCycles(numberOfProdCycles),
      temperature(temperature),
      boxSize(boxSize),
      maxDisplacement(maxDisplacement),
      translationProbability(translationProbability),
      optimizeMCMoves(optimizeMCMoves),
      pressure(pressure),
      volumeProbability(volumeProbability),
      maxVolumeChange(maxVolumeChange),
      sampleFrequency(sampleFrequency),
      chemicalPotential(chemicalPotential),
      swapProbability(swapProbability),
      positions(numberOfParticles),
      mt(seed),
      uniform_dist(0.0, 1.0),
      logger(logLevel),
      outputPDB(outputPDB)

{
  // Calculate cutoff energy, and system properties
  volume = boxSize * boxSize * boxSize;
  density = numberOfParticles / volume;
  beta = 1 / temperature;

  double invCutoff = 1.0 / cutOff;
  double r3i = invCutoff * invCutoff * invCutoff;
  cutOffPrefactor = EnergyVirial((8.0 / 3.0) * M_PI * ((1.0 / 3.0) * r3i * r3i * r3i - r3i),
                                 (16.0 / 3.0) * M_PI * ((2.0 / 3.0) * r3i * r3i * r3i - r3i));

  fugacity = std::exp(beta * chemicalPotential);

  // Empty movie.pdb file
  std::ofstream file("movie.pdb");
  file.close();

  // Initialize MD simulation
  logger.info("Class MC created.");
  logger.debug(repr());

  // Calculate number of grid points and grid spacing based on number of particles
  int numGrids = static_cast<int>(std::round(std::pow(numberOfParticles, 1.0 / 3.0) + 0.5));
  double gridSize = boxSize / (static_cast<double>(numGrids) + 2.0);
  logger.debug("numGrids " + std::to_string(numGrids) + " gridSize " + std::to_string(gridSize));

  // Initialize particle positions on a grid with slight random perturbations
  int counter = 0;
  for (int i = 0; i < numGrids; ++i)
  {
    for (int j = 0; j < numGrids; ++j)
    {
      for (int k = 0; k < numGrids; ++k)
      {
        if (counter < numberOfParticles)
        {
          positions[counter] =
              double3((i + 0.01 * (uniform() - 0.5)) * gridSize, (j + 0.01 * (uniform() - 0.5)) * gridSize,
                      (k + 0.01 * (uniform() - 0.5)) * gridSize);
          ++counter;
        }
      }
    }
  }

  // normalize and accumulate move probablities
  double totalProbability = translationProbability + volumeProbability + swapProbability;
  volumeProbability = volumeProbability / totalProbability;
  swapProbability = volumeProbability + swapProbability / totalProbability;

  if (outputPDB)
  {
    writePDB("movie.pdb", positions, boxSize, frameNumber);
    ++frameNumber;
  }

  totalEnergyVirial = systemEnergyVirial(positions, boxSize, cutOff, cutOffPrefactor);
  runningEnergyVirial = totalEnergyVirial;

  logger.info("(Init) completed. Total energy: " + std::to_string(totalEnergyVirial.energy) +
              ", Total virial: " + std::to_string(totalEnergyVirial.virial));
  logger.info(repr());
};

void MonteCarlo::computePressure()
{
  // Compute pressure using virial theorem and record it
  virialPressure = (density / beta) + (runningEnergyVirial.virial / (3.0 * volume));
  virialPressure += (cutOffPrefactor.virial * density * density);
  pressures.push_back(virialPressure);
}

void MonteCarlo::computeChemicalPotential()
{
  // Estimate chemical potential by inserting random particles
  double widomWeight = 0.0;
  for (int k = 0; k < 10; k++)
  {
    double3 randomPosition = boxSize * double3(uniform(), uniform(), uniform());
    EnergyVirial insertionEnergy = particleEnergyVirial(positions, randomPosition, -1, boxSize, cutOff);
    insertionEnergy += (cutOffEnergyVirial(numberOfParticles + 1, boxSize, cutOffPrefactor) -
                        cutOffEnergyVirial(numberOfParticles, boxSize, cutOffPrefactor));

    widomWeight += std::exp(-beta * insertionEnergy.energy);
  }
  widomWeight /= 10;
  widomWeights.push_back(widomWeight / density);
}

void MonteCarlo::run()
{
  // Main simulation loop
  for (cycle = 0; cycle < numberOfInitCycles + numberOfProdCycles; ++cycle)
  {
    // Attempt translation moves
    for (int i = 0; i < numberOfParticles; ++i)
    {
      double rand = uniform();
      if (rand < volumeProbability)
      {
        volumeMove();
      }
      else if (rand < swapProbability)
      {
        swapMove();
      }
      else
      {
        translationMove();
      }
    }

    if (cycle > numberOfInitCycles)
    {
      computeChemicalPotential();
    }

    // Every sampleFrequency cycles, compute and log properties
    if (cycle % sampleFrequency == 0)
    {
      totalEnergyVirial = systemEnergyVirial(positions, boxSize, cutOff, cutOffPrefactor);

      translationAcceptance = translationAccepted / translationAttempted;
      volumeAcceptance = volumeAccepted / volumeAttempted;
      insertionAcceptance = insertionAccepted / insertionAttempted;
      deletionAcceptance = deletionAccepted / deletionAttempted;

      logger.debug(repr());
      if (outputPDB)
      {
        writePDB("movie.pdb", positions, boxSize, frameNumber);
        ++frameNumber;
      }

      if (optimizeMCMoves)
      {
        optimizeMaxDisplacement();
        optimizeVolumeChange();
      }
      // If in production phase, compute observables
      if (cycle > numberOfInitCycles)
      {
        computePressure();
        driftEnergies.push_back(runningEnergyVirial.energy - totalEnergyVirial.energy);
        energies.push_back(runningEnergyVirial.energy);
        volumes.push_back(volume);
        particleCounts.push_back(numberOfParticles);
        densities.push_back(numberOfParticles / volume);
      }
    }
  }

  computePressure();
  computeChemicalPotential();
  insertionAcceptance = insertionAccepted / insertionAttempted;
  deletionAcceptance = deletionAccepted / deletionAttempted;

  totalEnergyVirial = systemEnergyVirial(positions, boxSize, cutOff, cutOffPrefactor);
  logger.info(repr());

  if (outputPDB)
  {
    writePDB("movie.pdb", positions, boxSize, frameNumber);
    ++frameNumber;
  }

  logThermodynamicalAverages();
}

void MonteCarlo::logThermodynamicalAverages()
{
  std::string s;
  std::pair<double, double> avePressure = blockAverage(pressures);
  std::pair<double, double> avePotentialEnergy = blockAverage(energies);
  std::pair<double, double> aveDriftEnergy = blockAverage(driftEnergies);
  std::pair<double, double> aveVolumes = blockAverage(volumes);
  std::pair<double, double> aveParticleCounts = blockAverage(particleCounts);
  std::pair<double, double> aveDensities = blockAverage(densities);

  s += "Thermodynamical averages\n";
  s += "----------------------------\n";
  s +=
      "Pressure             : " + std::to_string(avePressure.first) + " ± " + std::to_string(avePressure.second) + "\n";
  s += "Potential energy     : " + std::to_string(avePotentialEnergy.first) + " ± " +
       std::to_string(avePotentialEnergy.second) + "\n";
  s += "Drift energy         : " + std::to_string(aveDriftEnergy.first) + " ± " +
       std::to_string(aveDriftEnergy.second) + "\n";
  s += "Volume               : " + std::to_string(aveVolumes.first) + " ± " + std::to_string(aveVolumes.second) + "\n";
  s += "Number of particles  : " + std::to_string(aveParticleCounts.first) + " ± " +
       std::to_string(aveParticleCounts.second) + "\n";
  s += "Density              : " + std::to_string(aveDensities.first) + " ± " + std::to_string(aveDensities.second) +
       "\n";

  // block averaging of the chemical potential
  std::vector<double> widomSums(5);
  std::vector<double> counts(5);
  for (int i = 0; i < widomWeights.size(); ++i)
  {
    int bin = std::floor(i * 5 / widomWeights.size());
    widomSums[bin] += widomWeights[i];
    ++counts[bin];
  }

  chemicalPotentials.resize(5);
  std::vector<double> fugacities(5);
  for (int i = 0; i < 5; ++i)
  {
    double widomAverage = widomSums[i] / counts[i];
    chemicalPotentials[i] = -(1.0 / beta) * std::log(widomAverage);
    fugacities[i] = 1.0 / widomAverage;
  }

  double chemPotConf95 = 2.776 * std::sqrt(variance(chemicalPotentials) / 5);
  double fugConf95 = 2.776 * std::sqrt(variance(fugacities) / 5);

  s += "Chemical potential   : " + std::to_string(average(chemicalPotentials)) + " ± " + std::to_string(chemPotConf95) +
       "\n";
  s += "Fugacity             : " + std::to_string(average(fugacities)) + " ± " + std::to_string(fugConf95) + "\n";

  logger.info(s);
}

std::string MonteCarlo::repr()
{
  std::string s;
  EnergyVirial drift = (runningEnergyVirial - totalEnergyVirial);

  s += "Monte Carlo program\n";
  s += "----------------------------\n";
  s += "Number of particles  : " + std::to_string(numberOfParticles) + "\n";
  s += "Temperature          : " + std::to_string(temperature) + "\n";
  if (volumeProbability != 0.0)
  {
    s += "Pressure             : " + std::to_string(pressure) + "\n";
  }
  s += "Box length           : " + std::to_string(boxSize) + "\n";
  s += "Volume               : " + std::to_string(volume) + "\n";
  s += "Density              : " + std::to_string(density) + "\n";
  s += "CutOff radius        : " + std::to_string(cutOff) + "\n";
  s += "CutOff energy        : " + std::to_string(cutOffPrefactor.energy * density) + "\n";
  s += "Steps run            : " + std::to_string(cycle) + "\n";
  s += "Max displacement     : " + std::to_string(maxDisplacement) + "\n";
  s += "Translation acc.     : " + std::to_string(translationAcceptance) + "\n";
  if (volumeProbability != 0.0)
  {
    s += "Max volume change    : " + std::to_string(maxVolumeChange) + "\n";
    s += "Volume acc.          : " + std::to_string(volumeAcceptance) + "\n";
  }
  s += "Running energy       : " + std::to_string(runningEnergyVirial.energy) + "\n";
  s += "Total energy         : " + std::to_string(totalEnergyVirial.energy) + "\n";
  s += "Drift energy         : " + std::to_string(drift.energy / totalEnergyVirial.energy) + "\n";
  s += "Running virial       : " + std::to_string(runningEnergyVirial.virial) + "\n";
  s += "Total virial         : " + std::to_string(totalEnergyVirial.virial) + "\n";
  s += "Drift virial         : " + std::to_string(drift.virial / totalEnergyVirial.virial) + "\n";
  s += "\n";
  return s;
}
