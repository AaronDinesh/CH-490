#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hardDisks.h"
#include "mc.h"
#include "md.h"
#include "sample.h"

PYBIND11_MODULE(_molsim, m)
{
  pybind11::class_<HardDisks>(m, "HardDisks")
      .def(pybind11::init<int, int, int, double, int, double, int, bool, bool>(), pybind11::arg("numberOfInitCycles"),
           pybind11::arg("numberOfProdCycles"), pybind11::arg("numberOfParticles"), pybind11::arg("maxDisplacement"),
           pybind11::arg("sampleFrequency"), pybind11::arg("boxSize"), pybind11::arg("rdfBins"),
           pybind11::arg("periodicBoundary"), pybind11::arg("runStatic"))
      .def("run", &HardDisks::run)
      .def("getRDF", &HardDisks::getRDF)
      .def_readonly("acceptanceRatio", &HardDisks::acceptanceRatio);

  pybind11::class_<MonteCarlo>(m, "MonteCarlo")
      .def(pybind11::init<int, int, int, double, double, double, double, bool, double, double, double, double, double,
                          int, int, bool, int>(),
           pybind11::arg("numberOfParticles"), pybind11::arg("numberOfInitCycles"), pybind11::arg("numberOfProdCycles"),
           pybind11::arg("temperature"), pybind11::arg("boxSize"), pybind11::arg("maxDisplacement"),
           pybind11::arg("translationProbability") = 0.0, pybind11::arg("optimizeMCMoves") = true,
           pybind11::arg("pressure") = 0.0, pybind11::arg("volumeProbability") = 0.0,
           pybind11::arg("maxVolumeChange") = 1.0, pybind11::arg("chemicalPotential") = 0.0,
           pybind11::arg("swapProbability") = 0.0, pybind11::arg("sampleFrequency") = 100,
           pybind11::arg("logLevel") = 0, pybind11::arg("outputPDB") = true, pybind11::arg("seed") = 12)
      .def("__repr__", &MonteCarlo::repr)
      .def("run", &MonteCarlo::run)
      .def_readonly("insertionAcceptance", &MonteCarlo::insertionAcceptance)
      .def_readonly("deletionAcceptance", &MonteCarlo::deletionAcceptance)
      .def_readonly("translationAcceptance", &MonteCarlo::translationAcceptance)
      .def_readonly("volumeAcceptance", &MonteCarlo::volumeAcceptance)
      .def_readonly("maxDisplacement", &MonteCarlo::maxDisplacement)
      .def_readonly("maxVolumeChange", &MonteCarlo::maxVolumeChange)
      .def_readonly("pressures", &MonteCarlo::pressures)
      .def_readonly("energies", &MonteCarlo::energies)
      .def_readonly("particleCounts", &MonteCarlo::particleCounts)
      .def_readonly("volumes", &MonteCarlo::volumes)
      .def_readonly("driftEnergies", &MonteCarlo::driftEnergies)
      .def_readonly("densities", &MonteCarlo::densities)
      .def_readonly("chemicalPotentials", &MonteCarlo::chemicalPotentials);

  pybind11::class_<MolecularDynamics>(m, "MolecularDynamics")
      .def(pybind11::init<int, double, double, double, int, int, bool, int, int, int, bool, int>(),
           pybind11::arg("numberOfParticles"), pybind11::arg("temperature"), pybind11::arg("dt"),
           pybind11::arg("boxSize"), pybind11::arg("numberOfEquilibrationSteps"),
           pybind11::arg("numberOfProductionSteps"), pybind11::arg("outputPDB") = false,
           pybind11::arg("sampleFrequency") = 100, pybind11::arg("logLevel") = 0, pybind11::arg("seed") = 12,
           pybind11::arg("useNoseHoover") = false, pybind11::arg("noseHooverTimeScaleParameter") = 500)
      .def_readonly("rdfSampler", &MolecularDynamics::rdfSampler)
      .def_readonly("msdSampler", &MolecularDynamics::msdSampler)
      .def_readonly("time", &MolecularDynamics::time)
      .def_readonly("observedTemperatures", &MolecularDynamics::observedTemperatures)
      .def_readonly("pressures", &MolecularDynamics::pressures)
      .def_readonly("potentialEnergies", &MolecularDynamics::potentialEnergies)
      .def_readonly("kineticEnergies", &MolecularDynamics::kineticEnergies)
      .def_readonly("conservedEnergies", &MolecularDynamics::conservedEnergies)
      .def("__repr__", &MolecularDynamics::repr)
      .def("run", &MolecularDynamics::run);
  pybind11::class_<SampleRDF>(m, "SampleRDF").def("getResults", &SampleRDF::getResults);
  pybind11::class_<SampleMSD>(m, "SampleMSD").def("getResults", &SampleMSD::getResults);
}
