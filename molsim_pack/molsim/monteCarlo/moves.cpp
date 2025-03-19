#include <cmath>
#include <vector>

#include "mc.h"

void MonteCarlo::translationMove()
{
  translationAttempted++;
  // Generate random displacement
  int particleIdx = static_cast<int>(uniform() * numberOfParticles);
  double3 displacement = maxDisplacement * (double3(uniform(), uniform(), uniform()) - 0.5);
  double3 trialPosition = positions[particleIdx] + displacement;

  // Calculate old and new energy and virial
  EnergyVirial oldEnergyVirial = particleEnergyVirial(positions, positions[particleIdx], particleIdx, boxSize, cutOff);
  EnergyVirial newEnergyVirial = particleEnergyVirial(positions, trialPosition, particleIdx, boxSize, cutOff);

  // Accept or reject the move based on Metropolis criterion
  if (uniform() < std::exp(-beta * (newEnergyVirial.energy - oldEnergyVirial.energy)))
  {
    translationAccepted++;
    positions[particleIdx] = trialPosition;
    runningEnergyVirial += (newEnergyVirial - oldEnergyVirial);
  }
}

void MonteCarlo::optimizeMaxDisplacement()
{
  // Adjust max displacement to maintain optimal acceptance ratio
  if (translationAttempted > 100)
  {
    double acceptance = translationAccepted / translationAttempted;
    double scaling = std::clamp(2.0 * acceptance, 0.5, 1.5);
    maxDisplacement = std::clamp(scaling * maxDisplacement, 0.0001, 0.49 * boxSize);
    translationAccepted = 0;
    translationAttempted = 0;
  }
}

void MonteCarlo::volumeMove()
{
  volumeAttempted++;

  double volumeChange = (uniform() - 0.5) * maxVolumeChange;

  double newVolume = volume + volumeChange;
  if (newVolume < 0.0)
  {
    return;
  }

  double newBoxSize = std::cbrt(newVolume);
  double scale = newBoxSize / boxSize;

  std::vector<double3> trialPositions(positions);
  for (int i = 0; i < numberOfParticles; i++)
  {
    trialPositions[i] *= scale;
  }
  EnergyVirial newEnergyVirial = systemEnergyVirial(trialPositions, newBoxSize, cutOff, cutOffPrefactor);

  // start refactor
  if (uniform() < std::exp((numberOfParticles + 1.0) * std::log(newVolume / volume) -
                           beta * (pressure * volumeChange + (newEnergyVirial.energy - runningEnergyVirial.energy))))
  // if (uniform() < 0.0)
  // end refactor
  {
    volumeAccepted++;
    positions = trialPositions;
    boxSize = newBoxSize;
    volume = boxSize * boxSize * boxSize;
    density = numberOfParticles / volume;
    runningEnergyVirial = newEnergyVirial;
  }
}

void MonteCarlo::optimizeVolumeChange()
{
  if (volumeAttempted > 100)
  {
    double acceptance = volumeAccepted / volumeAttempted;
    double scaling = std::clamp(2.0 * acceptance, 0.5, 1.5);
    maxVolumeChange = std::clamp(scaling * maxVolumeChange, 0.001 * volume, 0.5 * volume);
    volumeAccepted = 0;
    volumeAttempted = 0;
  }
}

void MonteCarlo::swapMove()
{
  if (uniform() < 0.5)
  {
    insertionAttempted++;
    // trial insertion of new particle
    double3 newParticle = boxSize * double3(uniform(), uniform(), uniform());
    EnergyVirial diffEnergyVirial = particleEnergyVirial(positions, newParticle, numberOfParticles, boxSize, cutOff);

    // add increase tail energy (N+1)**2 - N**2
    diffEnergyVirial += (cutOffEnergyVirial(numberOfParticles + 1, boxSize, cutOffPrefactor) -
                         cutOffEnergyVirial(numberOfParticles, boxSize, cutOffPrefactor));

    if (uniform() <
        (beta * volume / (numberOfParticles + 1)) * std::exp(-beta * (diffEnergyVirial.energy - chemicalPotential)))
    {
      // accept
      insertionAccepted++;
      runningEnergyVirial += diffEnergyVirial;
      numberOfParticles++;
      density = numberOfParticles / volume;
      positions.push_back(newParticle);
    }
  }
  else
  {
    if (numberOfParticles == 1)
    {
      // can not delete if there are no particles
      return;
    }

    deletionAttempted++;
    // trial deletion
    int selectedParticle = static_cast<int>(uniform() * numberOfParticles);
    EnergyVirial diffEnergyVirial =
        particleEnergyVirial(positions, positions[selectedParticle], selectedParticle, boxSize, cutOff);
    diffEnergyVirial.energy *= -1.0;
    diffEnergyVirial.virial *= -1.0;

    // removed tail energy
    diffEnergyVirial += (cutOffEnergyVirial(numberOfParticles - 1, boxSize, cutOffPrefactor) -
                         cutOffEnergyVirial(numberOfParticles, boxSize, cutOffPrefactor));

    if (uniform() <
        (numberOfParticles / (beta * volume)) * std::exp(-beta * (diffEnergyVirial.energy + chemicalPotential)))
    {
      deletionAccepted++;
      runningEnergyVirial += diffEnergyVirial;
      numberOfParticles--;
      density = numberOfParticles / volume;
      positions.erase(positions.begin() + selectedParticle);
    }
  }
}