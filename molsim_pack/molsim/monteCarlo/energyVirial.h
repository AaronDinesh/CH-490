#pragma once

#include "utils.h"

/**
 * \brief Represents the energy and virial of a system or particle.
 *
 * The EnergyVirial struct encapsulates the total energy and virial contributions
 * which are used in the calculations of system properties in Monte Carlo simulations.
 */
struct EnergyVirial
{
  double energy;
  double virial;

  /**
   * \brief Constructs an EnergyVirial object with specified energy and virial.
   *
   * \param energy Initial energy value.
   * \param virial Initial virial value.
   */
  EnergyVirial(double energy = 0.0, double virial = 0.0) : energy(energy), virial(virial) {};

  EnergyVirial& operator+=(const EnergyVirial& other)
  {
    this->energy += other.energy;
    this->virial += other.virial;
    return *this;
  };

  EnergyVirial& operator-=(const EnergyVirial& other)
  {
    this->energy -= other.energy;
    this->virial -= other.virial;
    return *this;
  };
};

static inline EnergyVirial operator-(const EnergyVirial& a, const EnergyVirial& b)
{
  return {a.energy - b.energy, a.virial - b.virial};
};

static inline EnergyVirial operator+(const EnergyVirial& a, const EnergyVirial& b)
{
  return {a.energy + b.energy, a.virial + b.virial};
};

static inline EnergyVirial operator/(const EnergyVirial& a, const EnergyVirial& b)
{
  return {a.energy / b.energy, a.virial / b.virial};
};

/**
 * \brief Calculates the energy and virial for a particle.
 *
 * Computes the energy and virial contributions of a particle at a given position.
 *
 * \param position Position of the particle.
 * \param particleIdx Index of the particle.
 * \param startIndex Start loop over other particles from this index, allowing lower triangular computation
 * \return EnergyVirial object containing energy and virial.
 */
static EnergyVirial particleEnergyVirial(const std::vector<double3>& positions, const double3& particlePosition,
                                         int particleIdx, const double& boxSize, const double& cutOff)
{
  EnergyVirial particleEnergyVirial;
  double cutOffSquared = cutOff * cutOff;
  // Compute energy and virial contributions from interactions with other particles
  for (int i = 0; i < positions.size(); i++)
  {
    if (i != particleIdx)
    {
      double3 dr = particlePosition - positions[i];
      dr = wrap(dr, boxSize);
      double r2 = double3::dot(dr, dr);
      if (r2 < cutOffSquared)
      {
        double r2i = 1.0 / r2;
        double r6i = r2i * r2i * r2i;

        particleEnergyVirial.energy += 4.0 * (r6i * r6i - r6i);
        particleEnergyVirial.virial += 48.0 * (r6i * r6i - 0.5 * r6i);
      }
    }
  }
  return particleEnergyVirial;
}

static EnergyVirial cutOffEnergyVirial(const int numberOfParticles, const double& boxSize,
                                       const EnergyVirial& cutOffPrefactor)
{
  double volume = boxSize * boxSize * boxSize;
  double cutOffEnergy = cutOffPrefactor.energy * numberOfParticles * numberOfParticles / volume;
  double cutOffVirial = cutOffPrefactor.virial * numberOfParticles * numberOfParticles / (volume * volume);
  return EnergyVirial(cutOffEnergy, cutOffVirial);
}

/**
 * \brief Calculates the total energy and virial of the system.
 *
 * Computes the total energy and virial contributions of all particles.
 *
 * \return EnergyVirial object containing total energy and virial.
 */
static EnergyVirial systemEnergyVirial(const std::vector<double3>& positions, const double& boxSize,
                                       const double& cutOff, const EnergyVirial& cutOffPrefactor)
{
  double volume = boxSize * boxSize * boxSize;
  double density = positions.size() / volume;

  EnergyVirial systemEnergyVirial;
  for (int i = 0; i < positions.size(); i++)
  {
    systemEnergyVirial += particleEnergyVirial(positions, positions[i], i, boxSize, cutOff);
  }
  systemEnergyVirial.energy /= 2.0;
  systemEnergyVirial.virial /= 2.0;

  // Calculate ignored cutoff tail energy and virial
  systemEnergyVirial += cutOffEnergyVirial(positions.size(), boxSize, cutOffPrefactor);
  return systemEnergyVirial;
}