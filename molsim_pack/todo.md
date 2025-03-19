
| Day             | Exercise              | Code type   | Source code | Exercises in jupyter | Pretty plots | Manual updated | Code reviewed |
| --------------- | --------------------- | ----------- | ----------- | -------------------- | ------------ | -------------- | ------------- |
| 1 (StatMech)    | Distributions         | python      | ✅            | ✅                     |   ✅           |                |               |
|                 | Random walk (1D & 2D) | numba       | ✅           |  ✅                    |    ✅          |                |               |
| 2 (Basic MC)    | Pi                    | python      | ✅           | ✅                     | ✅            |                |               |
|                 | Hard Disks            | numba & c++ | ✅           | ✅                     | ✅          |                |               |
|                 | LJ - MC               | c++         | ✅           |                ✅      |   ✅           |                |               |
| 3 (Basic MD)    | Muller Brown          | numba       | ✅           | ✅                    |  ✅             |                |               |
|                 | LJ - MD               | c++         | ✅           | ✅                    | ✅             |                |               |
|                 | LJ code optimization  | c++         | ✅           | ✅                    | ✅             |                |               |
| 4 (Advanced MC) | Parallel Tempering    | numba       | ✅           |    ✅                  | ✅            |                |               |
|                 | $\mu VT$ LJ       | c++         |  ✅          |                      |              |                |               |
|                 | NPT LJ                | c++         | ✅           |                      |              |                |               |
| 5 (Advanced MD) | Umbrella sampling     | numba       |  ✅           |   ✅                   |          ✅    |                |               |
|                 | Nosé-Hoover           | c++         |  ✅        |                      |              |                |               |
|                 | Ewald (opt.)          | numba       | ✅           |                      |              |                |               |
Other:
- Manual how to use code
- Manual figure 1.8 and 1.9 are the same



Errors or unknowns:
- LJ movie.pdb loading issue using mdtraj (loading ok in vmd)
- Day 1:  random walk 2d markdown cell contains undefined image
- Day 2: calculation of Pi states references [Frenkel and Smit, 2002; Dubbeldam, 2013], where are those from? You could refer to page 33, configurational integral, of molsim book?
- progress bar in hard
- 3d visualizatie at first visualizaiton of muller brown 
- ops tutorial see if we can add it to molsim env.yml
- All notebooks should start after the import with a text of the exercises. not directly some python code.


Larger tasks:
- learning key points in manual for each day also for TPS. 
- notebook for plotting path metadybamics day 8/9 



Exercises:
Day 1 (StatMech):
- Distributions
    - non interacting old exercises 3.1
    - non interacting with energy levels old 3.2.1
    - non interacting with degenerate energy levels old 3.2.3
    - Boltzmann Q1 refactor incomplete

    - TODO: unfinished interacitng oscillators
    - interacting oscillators old 3.3

- Random walk (1d & 2d)
    - wiener process 
    - exerise run at different lengts and find different variances which we can relate to the diffusion constant.
    - exercise to determline diffusion through both the rmsd and the variance  old 3.4

    - particles on a lattice (2d walk) same questions as old 3.5
    - 2d: implementation sections contain both working code and commented code



Day 2 (Basic MC):
- PI
    - exercises same as odl 4.1 
    - convergence of 1/sqrt(n)


- Hard disk:
    - periodic boundaries
    - static and dynamic mC
    - simulation efficiency (python, numba jit, c++ pybind)

- LJ MC
    - run at different temperatuers and densities and look if ideal gas law \beta p = \rho is satisfied
    - try to visiualize using ovito (nglview did not work) isntall ovito locally (TODO: this needs guide)
    - heat capactity calculation old 4.4 exercise 4 run at different temperatures and plottting. odl was implementation of the calculation 
    fill in the method part of calculating the heat capacitiy with the hind there is a variance function.
    - translation move that optimized for 50% acceptance, plot maximum displacement vs density 
    - turn of the translation move optimziation and look at the acceptance ratio as a funciton of the density (e.g. run at fixed N and change the volume.)
    - look at decorralation times between different acceptance 
    - plot noth acceptance vs density and acceptance vs decorralation at the same time.

Day 3 (Basic MD):
- Muller brown
    - implement two integratos 
        - implement verlet integrator 
        - implemenet langevin itnegrator (random integrator with Wiener process) (so this is clear for the umbrellas sampling)
    - interpretation question: what do we see as a difference btween the tow integrators and why is this happening. part about deterministic integratos and wiener processies also discussed in statistics before at random walks.
    - how to modify the temperature in the (initial total energy aka kinteit energy) and in the langevin you give a temp. 
    - implement drift calculate for velocity verlet. question why does calculating drift for langevin not make sense.

- LJ - MD
    - visualize the simulation 
    - how do you set the tmeperature and plot the kinetic and obsered tempertures
    - calculate the drift and look at the conserved energy
    - symplecitc integrator ()
    - mean square displacement and velcoity autocorrelation
    - old exercise 5.1 e 7 compute the potential of mean froce from the radial distribution function
    TODO: possible error in msd calculation

- LJ code optimization
    - modify the c++ code and compile using pip install . to speed up the mD simulations. remove pow, cut off and pair calclation of distance only on one side of diagonal.

Day 4 (Advanced MC):
- Parallel Tempering
    - TODO input what the exercies do
<!-- - NVT coexistence (removed) -->
- insertion move! TODO implement
- NPT LJ
    - Volume move 
    - TOOD uniform smaller then 0
    - exercise implement acceptance rule and detailed balance!
    - 

Day 5 (Advanced MD):
- Umbrella sampling
    - in langevin integrator impelement bias both for x and y 
    - interpretation question on stable state run
    - what happens for different kappa values and bias x and or bias y 
    - devin path along y
    - perform US on path
    optional use NEB as path forthe US


- Nose-Hoover
    - clear explanation on why nose hooover and the difficulty of conserved quantities in md.
    - energy conservation what if shadow hamiltonian is conserved
    - analyze time scale parameter change of the nose hoover (thermostat mass)
    - set mass of 
    maybe do nose hoover hamiltonaian part and look at energy (nosehovernvt function in c++ src)
    - let them do a few analysiss runs

<!-- - Ewald (aux?) -->

a