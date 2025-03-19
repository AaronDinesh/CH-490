#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "double3.h"

/**
 * \brief Represents a sampler for calculating the Radial Distribution Function (RDF).
 *
 * The SampleRDF struct is responsible for accumulating pairwise distance measurements
 * between particles in a system to compute the RDF. It maintains a histogram of distances,
 * manages the sampling process, and provides the results in a numpy format suitable for
 * further analysis or visualization.
 */
struct SampleRDF
{
  int numberOfSamples = 0;
  int numberOfBins = 100;
  std::vector<double> histogram;

  int numberOfParticles;
  double boxSize;

  double cutOff;
  double delta;

  std::vector<double> r;

  /**
   * @brief Constructs a SampleRDF object for computing the radial distribution function.
   *
   * Initializes the histogram bins and computes the radial positions for each bin.
   *
   * @param numberOfParticles The number of particles in the system.
   * @param boxSize The length of the simulation box (assuming cubic box).
   */
  SampleRDF(int numberOfParticles, double boxSize, double cutOff);

  /**
   * @brief Samples particle positions to compute the radial distribution function (RDF).
   *
   * Updates the histogram by counting the number of particle pairs at each distance bin.
   *
   * @param positions A vector of particle positions.
   */
  void sample(std::vector<double3>& positions);

  /**
   * @brief Computes and returns the normalized radial distribution function as a numpy array.
   *
   * Normalizes the histogram to obtain the RDF g(r) and returns the radial positions and g(r) values.
   *
   * @return A pybind11 numpy array of shape (numberOfBins, 2) where the first column is r and the second is g(r).
   */
  pybind11::array_t<double> getResults();
};

struct SampleMSD
{
  // Time tracking variables
  int time = 0;
  int originTimeCounter = 0;
  int numberOfCorrelationTimes = 7500;
  int maxOriginTimes = 250;
  int resetOriginInterval = 50;

  int numberOfParticles;
  double boxSize;
  double sampleTime;

  // Data storage vectors
  std::vector<int> sampleCounts;
  std::vector<int> originTimes;
  std::vector<double> meanSquareDisplacements;
  std::vector<double> velocityAutocorrelation;
  std::vector<std::vector<double3>> velocityAtOrigin;
  std::vector<std::vector<double3>> positionAtOrigin;

  /**
   * @brief Constructs a SampleMSD object for computing the mean square displacement (MSD) and velocity autocorrelation
   * function (VACF).
   *
   * Initializes storage for origin times, sample counts, MSD, VACF, and particle positions and velocities at origin
   * times.
   *
   * @param numberOfParticles The number of particles in the system.
   * @param boxSize The length of the simulation box (assuming cubic box).
   * @param sampleTime The time interval between samples.
   */
  SampleMSD(int numberOfParticles, double boxSize, double sampleTime);

  /**
   * @brief Samples particle positions and velocities to compute MSD and VACF.
   *
   * Updates the mean square displacement and velocity autocorrelation function based on current positions and
   * velocities.
   *
   * @param unwrappedPositions The current unwrapped positions of particles.
   * @param velocities The current velocities of particles.
   */
  void sample(std::vector<double3>& unwrappedPositions, std::vector<double3>& velocities);

  /**
   * @brief Computes and returns the normalized MSD and VACF results as a numpy array.
   *
   * Calculates the mean square displacement, diffusion coefficient, velocity autocorrelation, and cumulative VACF.
   *
   * @return A pybind11 numpy array of shape (numberOfCorrelationTimes, 5) where columns are:
   *         time, MSD, diffusion coefficient, VACF, cumulative VACF.
   */
  pybind11::array_t<double> getResults();
};
