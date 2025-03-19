#pragma once

#include <pybind11/iostream.h>

#include "double3.h"

/**
 * \brief Provides logging functionalities with different log levels.
 *
 * The Logger class allows logging messages at various levels (debug, info, error)
 * using pybind11 to print the messages. The log level determines which messages
 * are displayed based on their severity.
 */
struct Logger
{
  int logLevel{0};  ///< The current log level determining message verbosity.

  /**
   * \brief Constructs a Logger object with the specified log level.
   *
   * \param logLevel The initial log level. Lower values enable more verbose logging.
   */
  Logger(int logLevel) : logLevel(logLevel) {}

  /**
   * \brief Logs a debug message if the log level allows.
   *
   * \param message The debug message to be logged.
   */
  void debug(const std::string &message)
  {
    if (logLevel < 1)
    {
      pybind11::print("[DEBUG]: ", message);
    }
  }

  /**
   * \brief Logs an info message if the log level allows.
   *
   * \param message The info message to be logged.
   */
  void info(const std::string &message)
  {
    if (logLevel < 2)
    {
      pybind11::print("[INFO]: ", message);
    }
  }

  /**
   * \brief Logs an error message.
   *
   * \param message The error message to be logged.
   */
  void error(const std::string &message) { pybind11::print("[ERROR]: ", message); }
};

static double average(std::vector<double> &data)
{
  return (data.size()) ? std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size()) : 0.0;
}

static double variance(std::vector<double> &data)
{
  double size = static_cast<double>(data.size());
  if (size == 0.0)
  {
    return 0.0;
  }
  double mean = std::accumulate(data.begin(), data.end(), 0.0) / size;
  double meanOfSquares = std::inner_product(data.begin(), data.end(), data.begin(), 0.0) / size;
  return meanOfSquares - mean * mean;
}

/**
 * @brief Calculates the block average and variance of a dataset.
 *
 * Divides the data into 5 blocks, computes the average for each block, and then
 * calculates the overall average and variance from these block averages.
 *
 * @param data The dataset to be averaged.
 * @return A pair containing the total average and the variance between block
 * averages.
 */
static std::pair<double, double> blockAverage(std::vector<double> &data)
{
  std::vector<double> averages(5);
  std::vector<int> counts(5);

  // Sum data into blocks
  for (int i = 0; i < data.size(); ++i)
  {
    int bin = std::floor(i * 5 / data.size());
    averages[bin] += data[i];
    ++counts[bin];
  }

  double totalAverage = 0.0;
  // Calculate average for each block
  for (int i = 0; i < 5; ++i)
  {
    averages[i] /= counts[i];
    totalAverage += averages[i] / 5;
  }

  double totalVariance = 0.0;
  // Compute variance from block averages
  for (int i = 0; i < 5; ++i)
  {
    totalVariance += (averages[i] - totalAverage) * (averages[i] - totalAverage);
  }
  totalVariance /= 4;  // Using N-1 for variance calculation
  double confidence = 2.776 * std::sqrt(totalVariance / 5);

  return std::make_pair(totalAverage, confidence);
};
