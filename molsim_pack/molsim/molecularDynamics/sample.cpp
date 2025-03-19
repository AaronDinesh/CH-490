#include "sample.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "double3.h"
#include "utils.h"

/**
  -------------------------------------------------------------
  -------------            RDF               ------------------
  -------------------------------------------------------------
 */

SampleRDF::SampleRDF(int numberOfParticles, double boxSize, double cutOff)
    : histogram(numberOfBins), numberOfParticles(numberOfParticles), boxSize(boxSize), r(numberOfBins), cutOff(cutOff)
{
  delta = cutOff / static_cast<double>(numberOfBins);
  for (int i = 0; i < numberOfBins; ++i)
  {
    r[i] = (i + 0.5) * delta;
  }
};

void SampleRDF::sample(std::vector<double3>& positions)
{
  ++numberOfSamples;

  // Loop over unique particle pairs
  for (int i = 0; i < numberOfParticles - 1; ++i)
  {
    for (int j = i + 1; j < numberOfParticles; ++j)
    {
      double3 dr = positions[i] - positions[j];
      dr = wrap(dr, boxSize);
      double dist = std::sqrt(double3::dot(dr, dr));
      if (dist < cutOff)
      {
        int bin = static_cast<int>(dist / delta);
        histogram[bin] += 2.0;
      }
    }
  }
}

pybind11::array_t<double> SampleRDF::getResults()
{
  std::vector<double> normalizedHistogram(numberOfBins);

  // Function to calculate (i+1)**3 - i**3
  auto cubeDifference = [](int i) -> int { return (i + 1) * (i + 1) * (i + 1) - i * i * i; };

  // Normalization factors
  double volume = boxSize * boxSize * boxSize;
  double density = numberOfParticles / volume;
  double dV = delta * delta * delta;
  const double PI = 3.14159265358979323846;
  double prefactor = (4.0 * PI / 3.0) * density * dV * numberOfSamples * numberOfParticles;

  for (int i = 0; i < numberOfBins; ++i)
  {
    normalizedHistogram[i] = histogram[i] / (cubeDifference(i) * prefactor);
  }

  // convert to numpy array
  pybind11::array_t<double> result({static_cast<ssize_t>(numberOfBins), static_cast<ssize_t>(2)});
  auto result_mutable = result.mutable_unchecked<2>();
  for (int i = 0; i < numberOfBins; ++i)
  {
    result_mutable(i, 0) = r[i];
    result_mutable(i, 1) = normalizedHistogram[i];
  }

  return result;
}

/**
  -------------------------------------------------------------
  -------------             MSD              ------------------
  -------------------------------------------------------------
 */

SampleMSD::SampleMSD(int numberOfParticles, double boxSize, double sampleTime)
    : numberOfParticles(numberOfParticles),
      boxSize(boxSize),
      sampleTime(sampleTime),
      originTimes(maxOriginTimes),
      sampleCounts(numberOfCorrelationTimes),
      meanSquareDisplacements(numberOfCorrelationTimes),
      velocityAutocorrelation(numberOfCorrelationTimes),
      velocityAtOrigin(maxOriginTimes, std::vector<double3>(numberOfParticles)),
      positionAtOrigin(maxOriginTimes, std::vector<double3>(numberOfParticles))
{
}

void SampleMSD::sample(std::vector<double3>& unwrappedPositions, std::vector<double3>& velocities)
{
  ++time;

  // Periodically reset origin times
  if (time % resetOriginInterval == 0)
  {
    int originIndex = originTimeCounter % maxOriginTimes;
    originTimes[originIndex] = time;
    ++originTimeCounter;

    std::copy(unwrappedPositions.begin(), unwrappedPositions.end(), positionAtOrigin[originIndex].begin());
    std::copy(velocities.begin(), velocities.end(), velocityAtOrigin[originIndex].begin());
  }

  for (int i = 0; i < std::min(originTimeCounter, maxOriginTimes); ++i)
  {
    double correlationTime = time - originTimes[i];
    if (correlationTime < numberOfCorrelationTimes)
    {
      ++sampleCounts[correlationTime];
      for (int j = 0; j < numberOfParticles; ++j)
      {
        velocityAutocorrelation[correlationTime] += double3::dot(velocities[j], velocityAtOrigin[i][j]);

        // Assumes particles do not move more than half the box length
        double3 dr = wrap(unwrappedPositions[j] - positionAtOrigin[i][j], boxSize);
        meanSquareDisplacements[correlationTime] += double3::dot(dr, dr);
      }
    }
  }
}

pybind11::array_t<double> SampleMSD::getResults()
{
  std::vector<double> normalizedMSD(numberOfCorrelationTimes);
  std::vector<double> normalizedVACF(numberOfCorrelationTimes);
  std::vector<double> cumulativeVACF(numberOfCorrelationTimes);

  for (int i = 0; i < numberOfCorrelationTimes; ++i)
  {
    normalizedMSD[i] = meanSquareDisplacements[i] / (numberOfParticles * sampleCounts[i]);
    normalizedVACF[i] = velocityAutocorrelation[i] / (numberOfParticles * sampleCounts[i]);

    cumulativeVACF[i] = sampleTime * normalizedVACF[i] / 3.0 + (i > 0 ? cumulativeVACF[i - 1] : 0.0);
  }

  pybind11::array_t<double> result({static_cast<ssize_t>(numberOfCorrelationTimes), static_cast<ssize_t>(5)});
  auto result_mutable = result.mutable_unchecked<2>();
  for (int i = 0; i < numberOfCorrelationTimes; ++i)
  {
    result_mutable(i, 0) = sampleTime * i;
    result_mutable(i, 1) = normalizedMSD[i];
    result_mutable(i, 2) = normalizedMSD[i] / (6.0 * i * sampleTime);
    result_mutable(i, 3) = normalizedVACF[i];
    result_mutable(i, 4) = cumulativeVACF[i];
  }

  return result;
}
