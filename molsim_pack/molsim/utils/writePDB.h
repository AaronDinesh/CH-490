#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "double3.h"

/**
 * \brief Writes atomic positions and simulation box information to a PDB file.
 *
 * The writePDB function appends a new frame to the specified PDB (Protein Data Bank) file,
 * including the model number, crystal parameters, and atomic coordinates. It formats the
 * data according to PDB file standards and ensures atomic positions are wrapped within
 * the simulation box boundaries.
 *
 * \param fileName    The name of the PDB file to write to. If the file does not exist, it will be created.
 * \param positions   A vector of double3 structures representing the positions of atoms in the current frame.
 * \param boxSize     The size of the simulation box. Used to wrap atomic positions to ensure they are within bounds.
 * \param frameNumber The current frame number to be written to the PDB file.
 */
static void writePDB(const std::string &fileName, std::vector<double3> &positions, double &boxSize, int &frameNumber)
{
  std::ofstream file(fileName, std::ios::app);

  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file " << fileName << std::endl;
    return;
  }

  file << "MODEL " << std::setw(9) << frameNumber << "\n";
  file << "CRYST1   " << std::setw(6) << std::fixed << std::setprecision(3) << boxSize << "   " << std::setw(6)
       << boxSize << "   " << std::setw(6) << boxSize << "  90.00  90.00  90.00 P 1         1\n";

  for (int i = 0; i < positions.size(); ++i)
  {
    double3 wrapped = wrapFloor(positions[i], boxSize);
    file << std::left << "ATOM" << std::right << std::setw(7) << i << "  H" << std::setw(16) << i << std::fixed
         << std::setprecision(3) << std::setw(8) << wrapped.x << std::setw(8) << wrapped.y << std::setw(8) << wrapped.z
         << std::setw(21) << " "  // To maintain space between the coordinates and "H"
         << "H" << std::endl;
  }

  file << "ENDMDL\n\n";
  file.close();
}
