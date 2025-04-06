import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


@numba.njit
def checkOverlapWalls(newPosition: np.ndarray, boxSize: float):
    """
    Checks if placing a particle outside the allowed box (if no PBC are applied).

    Parameters:
        newPosition (np.ndarray): The proposed position of the particle.
        boxSize (float): The size of the simulation box along each dimension.

    Returns:
        bool: True if there is an overlap (or invalid placement), False otherwise.
    """
    # Define boundary limits return whether x and y are within limits
    # start refactor
    radius = 0.5
    return np.any(newPosition < -boxSize - radius) or np.any(newPosition > boxSize + radius)
    # end refactor


@numba.njit
def checkOverlapParticles(
    newPosition: np.ndarray, positions: np.ndarray, boxSize: float, particleIdx: int, periodicBoundary: bool
):
    """
    Checks if placing a particle at a new position causes overlap with any other particle.

    Parameters:
        newPosition (np.ndarray): The proposed position of the particle.
        positions (np.ndarray): Current positions of all particles, shape (N, dim).
        boxSize (float): The size of the simulation box along each dimension.
        particleIdx (int): Index of the particle that is being moved.
        periodicBoundary (bool): Whether to apply periodic boundary conditions (PBC).

    Returns:
        bool: True if there is an overlap (or invalid placement), False otherwise.
    """
    # Compute displacement vectors to all other particles
    dr = newPosition - positions

    # If periodic boundaries are active, wrap into main simulation cell
    if periodicBoundary:
        # start refactor
        dr = dr - boxSize*int(dr/boxSize)
        pass
        # end refactor

    # Compute distances to all particles
    d = np.sum(dr**2, axis=-1)

    # Ignore the particle itself
    d[particleIdx] = np.inf

    # Check if any particle is too close (overlap)s
    return np.any(d < 1.0)


@numba.njit
def lowerTriangularDistMatrix(positions: np.ndarray, numberOfParticles: int, boxSize: float, periodicBoundary: bool):
    """
    Computes the lower triangular portion of the pairwise distance matrix for a set of particles.

    Parameters:
        positions (np.ndarray): A 2D array of shape (numberOfParticles, dim), where dim is the dimensionality
                                of the space. Each row represents the coordinates of a particle.
        numberOfParticles (int): The total number of particles in the system.
        boxSize (float): The size of the simulation box along each dimension (assumes a square or cubic box).
        periodicBoundary (bool): Whether to apply periodic boundary conditions (PBC) to compute distances.

    Returns:
        np.ndarray: A 1D array containing the pairwise distances in the lower triangular part of the distance matrix
                    (excluding the diagonal). The length of the array is 0.5 * numberOfParticles * (numberOfParticles - 1).
    """

    # Initialize an array to store pairwise distances in the lower triangular format
    dist = np.zeros(int(0.5 * numberOfParticles * (numberOfParticles - 1)))

    # Counter to track the position in the `dist` array
    counter = 0

    # Loop over all unique pairs (i, j) with i < j
    for i in range(numberOfParticles - 1):
        for j in range(i + 1, numberOfParticles):
            # start refactor
            # Vector difference between particle positions
            dr = positions[i] - positions[j]

            # Apply periodic boundary conditions if requested
                # Shift distances into the minimal image convention range
            if periodicBoundary:
                dr = dr - boxSize*int(dr/boxSize)

            # Compute Euclidean distance
            d = np.sqrt(np.sum(dr**2))
            dist[counter] = d
            pass
            # end refactor

            # Move to the next pair for storing in dist array
            counter += 1

    return dist


@numba.njit
def normalizeRdf(
    unnormalizedRdf: np.ndarray, edges: np.ndarray, numberOfParticles: int, boxSize: float, sampleCounter: int
):
    """
    Normalizes the radial distribution function (RDF).

    Parameters:
        unnormalizedRdf (np.ndarray): The RDF histogram values to be normalized.
        edges (np.ndarray): Bin edges from np.histogram.
        numberOfParticles (int): Total number of particles.
        boxSize (float): The size of the simulation box.
        sampleCounter (int): The total number of RDF samples collected.

    Returns:
        np.ndarray: The normalized RDF.
    """
    # start implementation
    #unnormalizedRdf = [ unnormalizedRdf[i] * boxSize**2 / ((np.pi * (edges[i + 1] ** 2 - edges[i] ** 2) * numberOfParticles*(numberOfParticles-1) * sampleCounter)) for i in range(unnormalizedRdf.shape[0])]


    for i in range(unnormalizedRdf.shape[0]):
        unnormalizedRdf[i] /= (np.pi * (edges[i + 1] ** 2 - edges[i] ** 2))
        unnormalizedRdf[i] /= (numberOfParticles*(numberOfParticles-1))/(boxSize**2)
        unnormalizedRdf[i] /= sampleCounter


    # end implementation
    return unnormalizedRdf


@numba.njit
def dynamicHardDisks(
    numberOfInitCycles: int,
    numberOfProdCycles: int,
    numberOfParticles: int,
    maxDisplacement: float,
    sampleFrequency: int,
    boxSize: float,
    rdfBins: int,
    periodicBoundary: bool,
):
    """
    Performs a dynamic Monte Carlo simulation of hard disks in a 2D box.

    Parameters:
        numberOfInitCycles (int): Number of equilibration (initialization) cycles.
        numberOfProdCycles (int): Number of production cycles after initialization.
        numberOfParticles (int): Number of particles in the system.
        maxDisplacement (float): Maximum displacement for random moves.
        sampleFrequency (int): Frequency of sampling particle positions and RDF.
        boxSize (float): Size of the simulation box along each dimension.
        rdfBins (int): Number of bins for the radial distribution function (RDF).
        periodicBoundary (bool): Whether to apply periodic boundary conditions (PBC).

    Returns:
        samplePositions (np.ndarray): Array of shape (numberOfSamples, numberOfParticles, 2) with sampled positions.
        rdf (np.ndarray): The computed radial distribution function.
        moveAccepted (np.ndarray): Array of shape (numberOfSamples, 2) indicating the particle move acceptance info.
    """

    # Initialize a lattice of positions
    latticeSites = int(boxSize)
    nLattice = latticeSites**2
    positions = np.zeros((nLattice, 2))

    # Populate lattice positions evenly spaced in x and y
    count = 0
    for x in np.linspace(0.5, (1 - 1 / latticeSites) * boxSize - 0.5, latticeSites):
        for y in np.linspace(0.5, (1 - 1 / latticeSites) * boxSize - 0.5, latticeSites):
            positions[count] = np.array([x, y])
            count += 1

    # Randomly select 'numberOfParticles' unique positions from the lattice
    positions = positions[np.random.choice(nLattice, nLattice, replace=False)[:numberOfParticles]]

    # Set up sampling arrays
    numberOfSamples = numberOfProdCycles // sampleFrequency
    samplePositions = np.zeros((numberOfSamples, numberOfParticles, 2))
    moveAccepted = np.zeros((numberOfSamples, 2))
    sampleCounter = 0
    unnormalizedRdf = np.zeros(rdfBins)

    # Track move attempts and acceptances
    numberOfAttemptedMoves = 0
    numberOfAcceptedMoves = 0

    # Main Monte Carlo loop
    for cycle in range(numberOfInitCycles + numberOfProdCycles):
        numberOfAttemptedMoves += 1

        # Random displacement and particle selection
        displacement = (np.random.rand(2) - 0.5) * maxDisplacement
        particleIdx = np.random.choice(numberOfParticles)
        newPosition = positions[particleIdx] + displacement

        overlap = False
        if not periodicBoundary:
            overlap = checkOverlapWalls(newPosition, boxSize)

        # Check if the proposed move causes overlap
        if not overlap:
            overlap = checkOverlapParticles(newPosition, positions, boxSize, particleIdx, periodicBoundary)

        if not overlap:
            # Accept the move
            positions[particleIdx] = newPosition
            numberOfAcceptedMoves += 1

        # Perform sampling after initialization
        if cycle >= numberOfInitCycles and cycle % sampleFrequency == 0:
            samplePositions[sampleCounter] = positions
            moveAccepted[sampleCounter, 0] = particleIdx
            moveAccepted[sampleCounter, 1] = overlap

            # Compute RDF from the sample
            dist = lowerTriangularDistMatrix(positions, numberOfParticles, boxSize, periodicBoundary)
            hist, edges = np.histogram(dist, bins=rdfBins, range=(0, 0.5 * boxSize))
            unnormalizedRdf += 2 * hist

            sampleCounter += 1

    # Normalize the RDF
    normalizedRdf = normalizeRdf(unnormalizedRdf, edges, numberOfParticles, boxSize, sampleCounter)

    # Print the acceptance fraction
    acceptanceRatio = numberOfAcceptedMoves / numberOfAttemptedMoves
    print("The acceptance fraction:", acceptanceRatio)

    return acceptanceRatio, samplePositions % boxSize, [(edges[1:] + edges[:-1]) / 2, normalizedRdf], moveAccepted


@numba.njit
def staticHardDisks(
    numberOfInitCycles: int,
    numberOfProdCycles: int,
    numberOfParticles: int,
    maxDisplacement: float,
    sampleFrequency: int,
    boxSize: float,
    rdfBins: int,
    periodicBoundary: bool,
):
    """
    Simulates a static Monte Carlo hard disk system.
    In this mode, the positions of particles are reset at each move attempt rather
    than performing a perturbation from the previous positions.

    Parameters:
        numberOfInitCycles (int): Number of initialization (equilibration) cycles.
        numberOfProdCycles (int): Number of production cycles.
        numberOfParticles (int): Number of particles in the system.
        maxDisplacement (float): Maximum displacement for random moves (not utilized in static mode).
        sampleFrequency (int): Frequency of sampling positions and RDF.
        boxSize (float): Size of the simulation box.
        rdfBins (int): Number of bins for the radial distribution function (RDF).
        periodicBoundary (bool): Whether to apply periodic boundary conditions (PBC).

    Returns:
        samplePositions (np.ndarray): Sampled positions of shape (numberOfSamples, numberOfParticles, 2).
        rdf (np.ndarray): The computed radial distribution function.
    """

    # Set up a lattice of positions
    latticeSites = int(boxSize)
    nLattice = latticeSites**2
    positions = np.zeros((nLattice, 2))

    # Fill the lattice positions
    count = 0
    for x in np.linspace(0.5, (1 - 1 / latticeSites) * boxSize - 0.5, latticeSites):
        for y in np.linspace(0.5, (1 - 1 / latticeSites) * boxSize - 0.5, latticeSites):
            positions[count] = np.array([x, y])
            count += 1

    # Randomly choose initial particle positions
    positions = positions[np.random.choice(nLattice, nLattice, replace=False)[:numberOfParticles]]

    # Set up arrays for sampling
    numberOfSamples = numberOfProdCycles // sampleFrequency
    samplePositions = np.zeros((numberOfSamples, numberOfParticles, 2))
    sampleCounter = 0
    unnormalizedRdf = np.zeros(rdfBins)

    numberOfAttemptedMoves = 0
    numberOfAcceptedMoves = 0

    # Main Monte Carlo loop
    for cycle in range(numberOfInitCycles + numberOfProdCycles):
        numberOfAttemptedMoves += 1

        # In static mode, new positions are completely random each move
        newPositions = np.random.rand(numberOfParticles, 2) * boxSize

        # scale positions such that they always fit within the box
        if not periodicBoundary:
            newPositions = (1 - 1 / boxSize) * newPositions + 0.5

        # Check for overlaps in the new configuration
        overlap = False
        for i in range(numberOfParticles - 1):
            if checkOverlapParticles(newPositions[i], newPositions, boxSize, i, periodicBoundary):
                overlap = True
                break

        # If no overlap, accept this new configuration
        if not overlap:
            positions = newPositions
            numberOfAcceptedMoves += 1

        # Sample after initialization
        if cycle >= numberOfInitCycles and cycle % sampleFrequency == 0:
            samplePositions[sampleCounter] = positions

            # Compute RDF
            dist = lowerTriangularDistMatrix(positions, numberOfParticles, boxSize, periodicBoundary)
            hist, edges = np.histogram(dist, bins=rdfBins, range=(0, 0.5 * boxSize))
            unnormalizedRdf += 2 * hist

            sampleCounter += 1

    # Normalize the RDF
    normalizedRdf = normalizeRdf(unnormalizedRdf, edges, numberOfParticles, boxSize, sampleCounter)

    # Print acceptance ratio
    acceptanceRatio = numberOfAcceptedMoves / numberOfAttemptedMoves
    print("The acceptance fraction:", acceptanceRatio)

    return (
        acceptanceRatio,
        samplePositions % boxSize,
        [(edges[1:] + edges[:-1]) / 2, normalizedRdf],
    )


def hardDiskTrajectory(trajectory, moveAccepted, boxSize, periodicBoundary=True, dynamic=True):
    """
    Displays an animation of particle trajectories over time.

    Parameters:
        trajectory (np.ndarray): Particle positions at each frame of shape (n_frames, n_particles, 2).
                                 trajectory[frame, particle] = (x, y).
        moveAccepted (np.ndarray): Array of shape (n_frames, 2). Used to determine which particle
                                   to highlight and whether to fill it at a given frame.
                                   For example, moveAccepted[frame, 0] could indicate a particle index,
                                   and moveAccepted[frame, 1] could be a boolean for fill state.
        boxSize (float): Size of the simulation box.
        periodicBoundary (bool, optional): Whether periodic boundary conditions are considered in the visualization.
        dynamic (bool, optional): Whether to dynamically change particle colors based on move acceptance.
                                  Default is True.

    Returns:
        HTML: An HTML object containing the animation.
    """

    # Extract the shape of the trajectory
    x = trajectory
    n_frames = x.shape[0]
    n_particles = x.shape[1]

    # Internal helper function to compute positions for periodic images
    def getPeriodicOffsets(boxSize):
        return [
            np.array([0, 0]),
            np.array([boxSize, 0]),
            np.array([-boxSize, 0]),
            np.array([0, boxSize]),
            np.array([0, -boxSize]),
            np.array([boxSize, boxSize]),
            np.array([-boxSize, boxSize]),
            np.array([boxSize, -boxSize]),
            np.array([-boxSize, -boxSize]),
        ]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.25 * boxSize, boxSize * 1.25)
    ax.set_ylim(-0.25 * boxSize, boxSize * 1.25)

    # Draw box boundaries as dashed lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(boxSize, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(boxSize, color="gray", linestyle="--", linewidth=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Particle Trajectory Animation")
    ax.grid(False)

    # Determine offsets for periodic boundary visualization
    offsets = getPeriodicOffsets(boxSize) if periodicBoundary else [np.array([0, 0])]
    n_offsets = len(offsets)

    # Create circle patches for all particles and their periodic images
    circle_patches = []
    for _ in range(n_particles * n_offsets):
        # Each circle represents one particle in one periodic image
        # Radius is 0.5 (from overlap check), chosen for visualization
        circle = Circle((0, 0), 0.5, edgecolor="blue", fill=False, linewidth=1)
        circle_patches.append(circle)
        ax.add_patch(circle)

    def update(frame):
        # Update particle positions for the given frame
        positions = x[frame] % boxSize  # Wrap positions into main box

        # Identify which particles to highlight
        modifiedIdx = []
        if moveAccepted is not None:
            # Extract particle index and overlap info from moveAccepted
            # moveAccepted[frame,0] = particle index, moveAccepted[frame,1] = fill state
            particles_to_modify = int(moveAccepted[frame, 0])
            if 0 <= particles_to_modify < n_particles:
                modifiedIdx = [particles_to_modify * n_offsets + i for i in range(n_offsets)]

        # Update circles for each particle and each periodic image
        index = 0
        for pos in positions:
            for offset in offsets:
                # Set the circle center
                circle_patches[index].center = pos + offset

                # If dynamic coloring is enabled and particle was recently moved
                if dynamic and index in modifiedIdx:
                    circle_patches[index].set_edgecolor("red")
                    circle_patches[index].set_fill(moveAccepted[frame, 1])
                    circle_patches[index].set_facecolor("red")
                else:
                    # Default styling
                    circle_patches[index].set_edgecolor("blue")
                    circle_patches[index].set_fill(False)

                index += 1

    # Create the animation
    ani = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=True)

    # Close the figure to avoid static display
    plt.close(fig)

    # Return HTML object
    return HTML(ani.to_jshtml())
