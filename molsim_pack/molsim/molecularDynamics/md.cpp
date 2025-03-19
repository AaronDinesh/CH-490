#include "md.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "writePDB.h"

MolecularDynamics::MolecularDynamics(int numberOfParticles, double temperature, double dt, double boxSize,
                                     int numberOfEquilibrationSteps, int numberOfProductionSteps, bool outputPDB,
                                     int sampleFrequency, int logLevel, int seed, bool useNoseHoover,
                                     int noseHooverTimeScaleParameter)
    : numberOfParticles(numberOfParticles),
      temperature(temperature),
      dt(dt),
      boxSize(boxSize),
      numberOfEquilibrationSteps(numberOfEquilibrationSteps),
      numberOfProductionSteps(numberOfProductionSteps),
      sampleFrequency(sampleFrequency),
      outputPDB(outputPDB),
      useNoseHoover(useNoseHoover),
      degreesOfFreedom(3 * numberOfParticles - 3),
      velocityScaling(temperature, degreesOfFreedom),
      noseHoover(temperature, degreesOfFreedom, noseHooverTimeScaleParameter * dt, dt, seed),
      positions(numberOfParticles),
      momenta(numberOfParticles),
      forces(numberOfParticles),
      mt(seed),
      uniform_dist(0.0, 1.0),
      normal_dist(0.0, 1.0),
      numberOfSamples(static_cast<int>(numberOfProductionSteps / sampleFrequency)),
      time(numberOfSamples),
      pressures(numberOfSamples),
      kineticEnergies(numberOfSamples),
      potentialEnergies(numberOfSamples),
      conservedEnergies(numberOfSamples),
      observedTemperatures(numberOfSamples),
      rdfSampler(numberOfParticles, boxSize, cutOff),
      msdSampler(numberOfParticles, boxSize, dt * sampleFrequency),
      logger(logLevel)

{
  // Calculate cutoff radius, cutoff energy, and system properties

  // $U(r_c) = 4.0 * (r_c^{-12} - r_c^{-6})$
  double cutOffi6 = std::pow(cutOff, -6.0);
  cutOffEnergy = 4.0 * (cutOffi6 * cutOffi6 - cutOffi6);

  volume = boxSize * boxSize * boxSize;
  density = numberOfParticles / volume;

  // empty movie.pdb file
  std::ofstream file("movie.pdb");
  file.close();

  // Initialize MD simulation
  logger.info("Class MD created.");
  logger.debug(repr());

  // Initialize momenta with random Gaussian distribution and compute total momentum
  kineticEnergy = 0.0;
  for (int i = 0; i < numberOfParticles; ++i)
  {
    momenta[i] = double3(normal(), normal(), normal()) * std::sqrt(temperature);
    totalMomentum += momenta[i];
    kineticEnergy += 0.5 * double3::dot(momenta[i], momenta[i]);
  }
  totalMomentum /= static_cast<double>(numberOfParticles);
  logger.debug("(Init) initialized momenta.");

  // Zero total momentum
  for (int i = 0; i < numberOfParticles; ++i)
  {
    momenta[i] -= totalMomentum;
  }
  observedTemperature = 2.0 * kineticEnergy / degreesOfFreedom;
  logger.debug("(Init) Zeroed momentum.");

  latticeInitialization();
  writePDB("movie.pdb", positions, boxSize, frameNumber);
  ++frameNumber;
  gradientDescent();

  logger.info("(Init) completed. Final energy: " + std::to_string(potentialEnergy));
};

void MolecularDynamics::latticeInitialization()
{
  // Calculate number of grid points and grid spacing based on number of particles

  // $n_grids = \lceil \cbrt{N} + 1.5 \rceil$
  int numGrids = static_cast<int>(std::round(std::pow(numberOfParticles, 1.0 / 3.0) + 1.5));

  // $l = L / (n + 2)$
  gridSize = boxSize / (static_cast<double>(numGrids) + 2.0);
  logger.info("numGrids " + std::to_string(numGrids) + " gridSize " + std::to_string(gridSize));

  int counter = 0;
  for (int i = 0; i < numGrids; ++i)
  {
    for (int j = 0; j < numGrids; ++j)
    {
      for (int k = 0; k < numGrids; ++k)
      {
        if (counter < numberOfParticles)
        {
          // $(x_i, y_i, z_i) = (l * i, l * j, l * z)$
          positions[counter] = double3(i * gridSize, j * gridSize, k * gridSize);
          ++counter;
        }
      }
    }
  }
}

void MolecularDynamics::gradientDescent()
{
  // Initialize step size and variables for gradient descent
  double dx = 0.2 * gridSize;
  double previousEnergy = 0.0;
  std::vector<double3> oldForces(numberOfParticles);
  std::vector<double3> oldPositions(numberOfParticles);

  for (int step = 0; step < 1000; ++step)
  {
    if (step == 0)
    {
      // On first step, calculate initial forces and energy
      calculateForceFast();
      previousEnergy = potentialEnergy;
      logger.debug("(Init) initial energy: " + std::to_string(previousEnergy));
    }

    // Save current positions and forces before attempting step
    std::copy(positions.begin(), positions.end(), oldPositions.begin());
    std::copy(forces.begin(), forces.end(), oldForces.begin());

    // Calculate maximum force component to determine scaling factor
    // $F_{max} = max(F_{0x}, F_{0y}, ... F_{Nz})$
    double maxForce;
    for (double3& force : forces)
    {
      maxForce = std::max(maxForce, abs(force.x));
      maxForce = std::max(maxForce, abs(force.y));
      maxForce = std::max(maxForce, abs(force.z));
    }
    // $s = dx / F_{max}$
    double scaleGradient = dx / maxForce;

    // Update positions in the direction of forces (gradient descent step)
    // q_{i+1} = q_i + s * F_i$
    for (int i = 0; i < numberOfParticles; ++i)
    {
      positions[i] += scaleGradient * forces[i];
      positions[i] = wrap(positions[i], boxSize);
    }

    // Recalculate forces and potential energy after position update
    calculateForceFast();

    // Accept or reject the new positions based on energy change, adjust step size accordingly
    if (potentialEnergy < previousEnergy)
    {
      logger.debug("(Init) accepted gradient descent at step " + std::to_string(step) +
                   " energy difference = " + std::to_string(potentialEnergy - previousEnergy));
      previousEnergy = potentialEnergy;
      dx = std::min(0.5 * boxSize, 1.2 * dx);
    }
    else
    {
      logger.debug("(Init) rejected gradient descent at step " + std::to_string(step) +
                   " energy difference = " + std::to_string(potentialEnergy - previousEnergy));
      std::copy(oldPositions.begin(), oldPositions.end(), positions.begin());
      std::copy(oldForces.begin(), oldForces.end(), forces.begin());
      dx *= 0.1;
    }
  }
  logger.debug(repr());
}

void MolecularDynamics::calculateForceFast()
{
  // Initialize forces to zero and reset potential energy and pressure
  double cutOffSquared = cutOff * cutOff;

  for (int i = 0; i < numberOfParticles; ++i)
  {
    forces[i] = double3();
  }

  potentialEnergy = 0.0;
  pressure = 0.0;

  // Loop over all unique particle pairs to compute interactions
  for (int i = 0; i < positions.size() - 1; ++i)
  {
    for (int j = i + 1; j < positions.size(); ++j)
    {
      // Calculate minimum image distance between particles
      double3 dr = positions[i] - positions[j];
      dr = wrap(dr, boxSize);

      // Get squared distance $r^2 = (x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2$
      double r2 = double3::dot(dr, dr);

      // If within cutoff radius, compute Lennard-Jones potential and force
      if (r2 < cutOffSquared)
      {
        double r2i = 1.0 / r2;
        double r6i = r2i * r2i * r2i;

        // Update potential energy and pressure
        // $U(r) = 4 (r^{-12} - r^{-6}) - U(r_{c})$
        potentialEnergy += 4.0 * r6i * (r6i - 1.0) - cutOffEnergy;

        // $F(r) \dot r = -r \frac{\partial U}{\partial r} = 48.0 * (r^{-12} - \frac{1}{2} r^{-6})$
        double virial = 48.0 * r6i * (r6i - 0.5);
        pressure += virial;

        // Add forces to particles i and j
        forces[i] += virial * r2i * dr;
        forces[j] -= virial * r2i * dr;
      }
    }
  }

  // Normalize pressure by dividing by 3 times the volume
  pressure /= 3.0 * volume;
}

void MolecularDynamics::calculateForceSlow()
{
  // Initialize forces to zero and reset potential energy and pressure
  double cutOffSquared = cutOff * cutOff;

  for (int i = 0; i < numberOfParticles; ++i)
  {
    forces[i] = double3();
  }

  potentialEnergy = 0.0;
  pressure = 0.0;

  // Loop over all unique particle pairs to compute interactions
  for (int i = 0; i < positions.size(); ++i)
  {
    for (int j = 0; j < positions.size(); ++j)
    {
      if (i != j)
      {
        // Calculate minimum image distance between particles
        double3 dr = positions[i] - positions[j];
        dr = wrap(dr, boxSize);

        // Get squared distance $r^2 = (x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2$
        double r2 = double3::dot(dr, dr);
        double r = std::sqrt(r2);

        // Update potential energy and pressure
        // $U(r) = 4 (r^{-12} - r^{-6}) - U(r_{c})$
        potentialEnergy += 0.5 * 4.0 * (std::pow(r, -12) - std::pow(r, -6)) - cutOffEnergy;

        // $F(r) \dot r = -r \frac{\partial U}{\partial r} = 48.0 * (r^{-12} - \frac{1}{2} r^{-6})$
        double virial = 0.5 * 48.0 * (std::pow(r, -12) - 0.5 * std::pow(r, -6));
        pressure += virial;

        // Add forces to particles i and j
        forces[i] += virial * dr / r2;
        forces[j] -= virial * dr / r2;
      }
    }
  }

  // Normalize pressure by dividing by 3 times the volume
  pressure /= 3.0 * volume;
}

void MolecularDynamics::thermostat()
{
  if (useNoseHoover)
  {
    noseHoover.scale(momenta, kineticEnergy);
    conservedEnergy += noseHoover.getEnergy();
  }
  else
  {
    // only use hard velocity scaling in equilibration when not using Nose Hoover
    if (step <= numberOfEquilibrationSteps)
    {
      velocityScaling.scale(momenta, kineticEnergy);
    }
  }
}

void MolecularDynamics::integrate()
{
  // Update momenta half step
  // $p(t + 0.5 \Delta t) = p(t) + 0.5 * F(t) * \Delta t$
  for (int i = 0; i < numberOfParticles; ++i)
  {
    momenta[i] += 0.5 * forces[i] * dt;
  }

  // Update positions
  // $q(t + \Delta t) = q(t) + p(t) * \Delta t + 0.5 * F(t) * (\Delta t)^2$
  for (int i = 0; i < numberOfParticles; ++i)
  {
    positions[i] += momenta[i] * dt;
  }

  // start refactor
  calculateForceFast();
  // calculateForceSlow();
  // end refactor

  // Update momenta half step and compute kinetic energy
  kineticEnergy = 0.0;
  totalMomentum = double3();

  // $p(t + 0.5 \Delta t) = p(t) + 0.5 * F(t) * \Delta t$
  for (int i = 0; i < numberOfParticles; ++i)
  {
    momenta[i] += 0.5 * forces[i] * dt;
    kineticEnergy += 0.5 * double3::dot(momenta[i], momenta[i]);
    totalMomentum += momenta[i];
  }
  conservedEnergy = kineticEnergy + potentialEnergy;
  thermostat();

  // kinetic part of the pressure (virial part computed in calculateForce)
  pressure += 2.0 * kineticEnergy / (volume * 3.0);
}

void MolecularDynamics::run()
{
  int sampleCounter = 0;
  // Main simulation loop
  for (step = 0; step < numberOfEquilibrationSteps + numberOfProductionSteps + 1; ++step)
  {
    // Update forces and integrate positions and momenta
    integrate();
    if (step == numberOfEquilibrationSteps) baselineEnergy = conservedEnergy;

    observedTemperature = 2.0 * kineticEnergy / degreesOfFreedom;

    // Log and sample data every 100 steps
    if (step % sampleFrequency == 0)
    {
      logger.debug(repr());
      if (outputPDB)
      {
        writePDB("movie.pdb", positions, boxSize, frameNumber);
        ++frameNumber;
      }
      if (step > numberOfEquilibrationSteps)
      {
        time[sampleCounter] = (step - numberOfEquilibrationSteps) * dt;
        observedTemperatures[sampleCounter] = observedTemperature;
        pressures[sampleCounter] = pressure;
        potentialEnergies[sampleCounter] = potentialEnergy;
        kineticEnergies[sampleCounter] = kineticEnergy;
        conservedEnergies[sampleCounter] = conservedEnergy;
        sampleCounter++;

        rdfSampler.sample(positions);
        msdSampler.sample(positions, momenta);
      }
    }
  }

  logThermodynamicalAverages();
}

void MolecularDynamics::logThermodynamicalAverages()
{
  std::string s;
  std::pair<double, double> aveTemperature = blockAverage(observedTemperatures);
  std::pair<double, double> avePressure = blockAverage(pressures);
  std::pair<double, double> avePotentialEnergy = blockAverage(potentialEnergies);
  std::pair<double, double> aveKineticEnergy = blockAverage(kineticEnergies);
  std::pair<double, double> aveConservedEnergy = blockAverage(conservedEnergies);

  s += "Thermodynamical averages\n";
  s += "----------------------------\n";
  s += "Temperature          : " + std::to_string(aveTemperature.first) + " ± " +
       std::to_string(aveTemperature.second) + "\n";
  s +=
      "Pressure             : " + std::to_string(avePressure.first) + " ± " + std::to_string(avePressure.second) + "\n";
  s += "Potential energy     : " + std::to_string(avePotentialEnergy.first) + " ± " +
       std::to_string(avePotentialEnergy.second) + "\n";
  s += "Kinetic energy       : " + std::to_string(aveKineticEnergy.first) + " ± " +
       std::to_string(aveKineticEnergy.second) + "\n";
  s += "Conserved energy     : " + std::to_string(aveConservedEnergy.first) + " ± " +
       std::to_string(aveConservedEnergy.second) + "\n";

  logger.info(s);
}

std::string MolecularDynamics::repr()
{
  std::string s;
  s += "Molecular Dynamics program\n";
  s += "----------------------------\n";
  s += "Number of particles  : " + std::to_string(numberOfParticles) + "\n";
  s += "Temperature          : " + std::to_string(temperature) + "\n";
  s += "delta t              : " + std::to_string(dt) + "\n";
  s += "Box length           : " + std::to_string(boxSize) + "\n";
  s += "Density              : " + std::to_string(density) + "\n";
  s += "CutOff radius        : " + std::to_string(cutOff) + "\n";
  s += "CutOff energy        : " + std::to_string(cutOffEnergy) + "\n";
  s += "Steps run            : " + std::to_string(step) + "\n";
  s += "Observed temperature : " + std::to_string(observedTemperature) + "\n";
  s += "Pressure             : " + std::to_string(pressure) + "\n";
  s += "Potential energy     : " + std::to_string(potentialEnergy) + "\n";
  s += "Kinetic energy       : " + std::to_string(kineticEnergy) + "\n";
  s += "Conserved energy     : " + std::to_string(conservedEnergy) + "\n";
  if (useNoseHoover)
  {
    s += "Thermostat energy    : " + std::to_string(noseHoover.getEnergy()) + "\n";
  }
  if (step > numberOfEquilibrationSteps)
  {
    s += "Drift energy         : " + std::to_string(std::abs((conservedEnergy - baselineEnergy) / baselineEnergy)) +
         "\n";
  }
  s += "\n";
  return s;
}
