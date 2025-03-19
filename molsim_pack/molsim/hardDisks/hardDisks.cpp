#include "hardDisks.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

HardDisks::HardDisks(int numberOfInitCycles, int numberOfProdCycles, int numberOfParticles, double maxDisplacement,
                     int sampleFrequency, double boxSize, int rdfBins, bool periodicBoundary, bool runStatic)
    : numberOfInitCycles(numberOfInitCycles),
      numberOfProdCycles(numberOfProdCycles),
      numberOfParticles(numberOfParticles),
      maxDisplacement(maxDisplacement),
      sampleFrequency(sampleFrequency),
      boxSize(boxSize),
      positions(numberOfParticles),
      rdf(rdfBins),
      rdfBins(rdfBins),
      periodicBoundary(periodicBoundary)
{
  int latticeSites = static_cast<int>(boxSize);
  double latticeSiteScaling = (1 - 1 / latticeSites);
  std::vector<double2> latticePositions(latticeSites * latticeSites);
  for (int ix = 0; ix < latticeSites; ix++)
  {
    for (int iy = 0; iy < latticeSites; iy++)
    {
      latticePositions[ix + latticeSites * iy] = double2(ix * latticeSiteScaling + 0.5, iy * latticeSiteScaling + 0.5);
    }
  }
  std::sample(latticePositions.begin(), latticePositions.end(), positions.begin(), positions.size(), gen);

  delta = 0.5 * boxSize / (static_cast<double>(rdfBins));
  halfBoxSizeSq = 0.25 * boxSize * boxSize;

  method = runStatic ? Method::Static : Method::Dynamic;
}

void HardDisks::run()
{
  switch (method)
  {
    case Method::Dynamic:
      dynamicRun();
      break;
    case Method::Static:
      staticRun();
      break;
    default:
      break;
  }

  acceptanceRatio = static_cast<double>(numberOfAcceptedMoves) / static_cast<double>(numberOfAttemptedMoves);
  std::cout << "Acceptance fraction: " << acceptanceRatio << std::endl;
}

bool HardDisks::checkOverlapWalls(double2& newPosition)
{
  return (newPosition.x < 0.5 || newPosition.y < 0.5 || newPosition.x > (boxSize - 0.5) ||
          newPosition.y > (boxSize - 0.5));
}

bool HardDisks::checkOverlapParticles(double2& newPosition, std::vector<double2>& referencePositions, int particleIdx)
{
  for (int i = 0; i < numberOfParticles; i++)
  {
    // avoid computing for particle itself
    if (i != particleIdx)
    {
      double2 dr = newPosition - referencePositions[i];
      if (periodicBoundary)
      {
        dr = wrap(dr, boxSize);
      }

      double r2 = dot(dr, dr);
      if (r2 < 1.0)
      {
        return true;
      }
    }
  }
  return false;
}

void HardDisks::dynamicRun()
{
  std::uniform_int_distribution<> indexDist(0, numberOfParticles - 1);
  std::uniform_real_distribution<> uniform(-0.5, 0.5);
  for (int cycle = 0; cycle < numberOfInitCycles + numberOfProdCycles; cycle++)
  {
    numberOfAttemptedMoves++;

    // generate new particle position
    int particleIdx = indexDist(gen);
    double2 newPosition = positions[particleIdx];
    newPosition.x += maxDisplacement * uniform(gen);
    newPosition.y += maxDisplacement * uniform(gen);

    bool overlap = false;

    // check if new position overlaps with walls or is outside box
    if (!periodicBoundary)
    {
      overlap = checkOverlapWalls(newPosition);
    }

    // if it does not overlap with walls, check if it overlaps with other particles
    if (!overlap)
    {
      overlap = checkOverlapParticles(newPosition, positions, particleIdx);
    }

    // if it does not overlap, accept
    if (!overlap)
    {
      positions[particleIdx] = newPosition;
      numberOfAcceptedMoves++;
    }
    if (cycle % sampleFrequency == 0 && cycle > numberOfInitCycles)
    {
      sampleRDF();
    }
  }
}

void HardDisks::staticRun()

{
  std::uniform_real_distribution<> uniform(0.0, boxSize);

  for (int cycle = 0; cycle < numberOfInitCycles + numberOfProdCycles; cycle++)
  {
    numberOfAttemptedMoves++;

    // generate new positions
    std::vector<double2> newPositions(positions.size());
    for (int i = 0; i < numberOfParticles; i++)
    {
      newPositions[i] = double2(uniform(gen), uniform(gen));

      // scale positions such that they always fit in the box
      if (!periodicBoundary)
      {
        newPositions[i].x = (1.0 - 1.0 / boxSize) * newPositions[i].x + 0.5;
        newPositions[i].y = (1.0 - 1.0 / boxSize) * newPositions[i].y + 0.5;
      }
    }

    // check if there is overlap between particles
    bool overlap = false;
    for (int i = 0; i < numberOfParticles - 1; i++)
    {
      if (checkOverlapParticles(newPositions[i], newPositions, i))
      {
        overlap = true;
        break;
      }
    }

    // if there is no overlap, accept configuration
    if (!overlap)
    {
      positions = newPositions;
      numberOfAcceptedMoves++;
    }

    // compute rdf
    if (cycle % sampleFrequency == 0 && cycle > numberOfInitCycles)
    {
      sampleRDF();
    }
  }
}

void HardDisks::sampleRDF()
{
  numberOfSamples++;
  for (int i = 0; i < numberOfParticles - 1; i++)
  {
    for (int j = i + 1; j < numberOfParticles; j++)
    {
      double2 dr = positions[i] - positions[j];
      dr.x -= boxSize * std::floor(dr.x / boxSize);
      dr.y -= boxSize * std::floor(dr.y / boxSize);

      double r2 = dot(dr, dr);
      if (r2 < halfBoxSizeSq)
      {
        int idx = static_cast<int>(std::sqrt(r2) / delta);
        rdf[idx] += 2.0;
      }
    }
  }
}

pybind11::array_t<double> HardDisks::getRDF()
{
  std::vector<double> normalizedRDF(rdf.size());
  for (int i = 0; i < rdfBins; i++)
  {
    double areaDiff = M_PI * delta * delta * ((i + 1) * (i + 1) - i * i);
    double invDensitySq = boxSize * boxSize / (numberOfParticles * (numberOfParticles - 1));
    normalizedRDF[i] = 4.0 * rdf[i] * invDensitySq / (areaDiff * static_cast<double>(numberOfSamples));
  }
  return pybind11::array_t<double>(normalizedRDF.size(), normalizedRDF.data());
}
