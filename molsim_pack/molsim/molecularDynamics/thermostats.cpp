#include "thermostats.h"

VelocityScaling::VelocityScaling(double temperature, int degreesOfFreedom)
    : temperature(temperature), degreesOfFreedom(degreesOfFreedom)
{
}

void VelocityScaling::scale(std::vector<double3>& velocities, double& kineticEnergy)
{
  // $s = \sqrt{T * g / (2K)}
  double velocityScaling = std::sqrt(temperature * degreesOfFreedom / (2.0 * kineticEnergy));

  // $v(t) = s * v'(t)
  for (int i = 0; i < velocities.size(); ++i)
  {
    velocities[i] *= velocityScaling;
  }
  kineticEnergy *= velocityScaling * velocityScaling;
}

NoseHooverNVT::NoseHooverNVT(double temperature, int degreesOfFreedom, double timescaleParameter, double timeStep,
                             int seed)
    : temperature(temperature),
      degreesOfFreedom(degreesOfFreedom),
      timescaleParameter(timescaleParameter),
      timeStep(timeStep),
      mt(seed),
      normal_dist(0.0, 1.0)
{
  // $Q = g \tau^2$
  thermostatMass = degreesOfFreedom * timescaleParameter * timescaleParameter;

  // $\xi(0) = N(0, 1) * \sqrt{T / Q}
  thermostatVelocity = normal_dist(mt) * std::sqrt(temperature / thermostatMass);
}

void NoseHooverNVT::scale(std::vector<double3>& velocities, double& kineticEnergy)
{
  // $\dot{\xi} = \frac{2K - g k_B T}{Q}$
  thermostatForce = (2.0 * kineticEnergy - degreesOfFreedom * temperature) / thermostatMass;

  // $\xi(t + \Delta t) = \xi(t) + \dot{\xi} \Delta t$
  thermostatVelocity += thermostatForce * timeStep;

  // $s = \exp(-\xi \Delta t)
  double velocityScaling = std::exp(-thermostatVelocity * timeStep);

  // $\eta(t + \Delta t) = \eta(t) + \xi \Delta t$
  thermostatPosition += thermostatVelocity * timeStep;

  // $v(t) = s * v'(t)
  for (int i = 0; i < velocities.size(); i++)
  {
    velocities[i] *= velocityScaling;
  }
  kineticEnergy *= velocityScaling * velocityScaling;
}

double NoseHooverNVT::getEnergy()
{
  // $E_{NH} = \frac{1}{2} Q \xi^2 + g k_B T \eta$
  return 0.5 * thermostatMass * thermostatVelocity * thermostatVelocity +
         degreesOfFreedom * temperature * thermostatPosition;
}