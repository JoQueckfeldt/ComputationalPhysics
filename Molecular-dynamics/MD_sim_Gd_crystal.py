import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def init_Gd_super_cell(a=3.304):
    """"
    Create a gadolinium super-cell of 9x5x5 unit cells.
    The supercell is created by repeating the unit cell in 3D space.
    a=3.304Å is the lattice constant.

    Returns the positions of the atoms in the supercell and the dimensions of the supercell.
    """
    # defines unit cell dimensions
    unit_cell_dims = np.array([a, np.sqrt(3)*a, np.sqrt(8/3)*a])
    
    # the unit cell contains four Gd atoms, which sit at fractional coordinates
    unit_cell_frac_coords = np.array([
        [1/4, 5/6, 1/4],
        [1/4, 1/6, 3/4],
        [3/4, 1/3, 1/4],
        [3/4, 2/3, 3/4]
    ])

    # 4 atoms in the unit cell, 4 * 9x5x5 = 900
    total_atoms = 900

    # number of unit cells in each direction
    super_cell_size = np.array([9, 5, 5])

    # calculate super-cell dimensions
    super_cell_dims = unit_cell_dims * super_cell_size

    # initialize the positions array
    positions = np.zeros((total_atoms, 3))

    # loop over each unit cell and fill in the positions of the atoms
    atom_index = 0
    for i in range(super_cell_size[0]):
        for j in range(super_cell_size[1]):
            for k in range(super_cell_size[2]):
                # calculate the position of the unit cell
                unit_cell_pos = np.array([i, j, k]) * unit_cell_dims
                # loop over each atom in the unit cell
                for atom in unit_cell_frac_coords:
                    positions[atom_index] = unit_cell_pos + atom * unit_cell_dims
                    atom_index += 1

    return positions, super_cell_dims


def init_velocities(positions, temp=300.0, mass=157.25):
    """
    Initialize the velocities of the atoms in the supercell using Maxwell-Boltzmann distribution.
    The velocities are initialized based on the temperature and mass of the atoms.

    Parameters
    ----------
    positions: array, positions of the atoms in the supercell
    temperature: float, temperature in Kelvin
    mass: float, mass of Gd atom in amu

    Returns
    -------
    velocities: array, initialized velocities of the atoms
    """
    # Boltzmann constant in eV/K
    k_B = 8.61734e-5

    amu_to_eV_fs2_A2 = 103.6427  # Convert amu to eV·fs²/Å²

    # Convert mass from amu to eV·fs²/Å²
    mass *= amu_to_eV_fs2_A2 

    # calculate the standard deviation of the velocity distribution
    std_dev = np.sqrt(k_B * temp / mass)

    # initialize velocities using normal distribution. [Å/fs]
    velocities = np.random.normal(loc=0, scale=std_dev, size=positions.shape)

    # Remove net momentum (zero center-of-mass velocity)
    v_cm = np.mean(velocities, axis=0)
    velocities -= v_cm

    return velocities
    

def calculate_distance(pos1, pos2, super_cell_dims):
    """
    Calculate distance between two atoms with periodic boundary conditions
    
    Parameters
    -----------
    pos_i, pos_j : ndarray
        Positions of the two atoms
    super_cell_dims : ndarray
        Dimensions of the super-cell
        
    Returns
    --------
    float
        Distance between the atoms
    """
    # Calculate the difference in positions
    delta = pos1 - pos2
    
    # Apply periodic boundary conditions
    delta -= np.round(delta / super_cell_dims) * super_cell_dims
    
    # Calculate the distance using Euclidean norm
    distance = np.linalg.norm(delta)
    
    return distance


def build_neighbor_list(positions, super_cell_dims, cutoff=12.0):
    """
    Build a neighbor list for the atoms in the supercell.
    Based on the assumption that the atoms does not move much during the simulation.

    Parameters
    ----------
    positions: array, positions of the atoms in the supercell
    super_cell_dims: array, dimensions of the supercell
    cutoff: float, cutoff distance for neighbors

    Returns
    -------
    neighbor_list: list of lists, where each inner list contains the indices of neighboring atoms for each atom
    """
    
    n_atoms = positions.shape[0]
    neighbor_list = [[] for _ in range(n_atoms)]

    # loop over each atom
    # calculate distance from atom i to all other atoms j
    # if distance < cutoff, add j to neighbor_list[i]and i to neighbor_list[j]
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            delta = positions[i] - positions[j] # vector from j to i
            delta -= np.round(delta / super_cell_dims) * super_cell_dims  # PBC
            r = np.linalg.norm(delta)

            if r < cutoff:
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)

    return neighbor_list


def LJ_potential(positions, super_cell_dims, neighbor_list, eps=0.1136, sigma=3.304, cutoff=10.0):
    """
    Calculate the Lennard-Jones pair potential.

    Parameters
    ----------
    positions: array, positions of the atoms in the super-cell
    eps: float, depth of the potential well (in eV)
    sigma: float, finite distance at which the potential is zero (in Å)
    cutoff: float, cutoff distance for the potential (in Å)

    Returns
    -------
    float, Lennard-Jones potential energy (in eV)
    """
    
    # number of atoms
    n_atoms = positions.shape[0]

    # initialize potential energy
    V = 0.0

    for i in range(n_atoms):
        for j in neighbor_list[i]:
            # calculate distance between atoms i and j
            delta = positions[i] - positions[j] # vector from j to i
            delta -= np.round(delta / super_cell_dims) * super_cell_dims  # PBC
            r = np.linalg.norm(delta) # distance between atoms i and j

            if r < cutoff and i < j: # avoid double counting 
                # calculate Lennard-Jones potential
                V += 4 * eps * ((sigma / r)**12 - (sigma / r)**6)

    return V


def LJ_force(positions, super_cell_dims, neighbor_list, eps=0.1136, sigma=3.304, cutoff=10.0):
    """
    Calculate the Lennard-Jones force on each atom.

    Parameters
    ----------
    positions: array, positions of the atoms in the super-cell
    eps: float, depth of the potential well (in eV)
    sigma: float, finite distance at which the potential is zero (in Å)
    cutoff: float, cutoff distance for the potential (in Å)

    Returns
    -------
    forces: array, Lennard-Jones forces on each atom (in eV/Å)
    """

    n_atoms = positions.shape[0]
    forces = np.zeros_like(positions)

    # loop over all pairs of atoms
    for i in range(n_atoms-1):
        for j in neighbor_list[i]: 
            delta = positions[i] - positions[j] # vector from j to i
            delta -= np.round(delta / super_cell_dims) * super_cell_dims  # PBC

            r = np.linalg.norm(delta) # distance between atoms i and j

            if r < cutoff and i < j: # avoid double counting
                # calculate force using Lennard-Jones potential derivative (F = -dV/dr)
                force_mag = 24 * eps * (2 * (sigma / r)**12 - (sigma / r)**6) / r
                # update forces on atoms i and j
                force_vec = force_mag * delta / r  # normalize the force vector
                forces[i] += force_vec  
                forces[j] -= force_vec 
    
    return forces

def generate_random_forces(num_atoms, temp, gamma, dt=1, mass=157.25):
    """
    Generate random forces for Langevin dynamics using the Box-Muller transform.
    
    Parameters:
    -----------
    num_atoms : int
        Number of atoms
    temperature : float
        Temperature in Kelvin
    gamma : float
        Friction coefficient
    dt : float
        Time step
    mass : float
        Atomic mass
        
    Returns:
    --------
    random_force1 : ndarray
        Random forces for the first half of the step
    random_force2 : ndarray
        Random forces for the second half of the step
    """

    # Boltzmann constant in eV/K
    k_B = 8.61734e-5

    mass *= 103.6427  # Convert amu to eV·fs²/Å²

    sigma = np.sqrt(2.0 * mass * gamma * k_B * temp / dt)  # standard deviation of the random force
    # np.sqrt((1 - np.exp(-gamma * dt)) * mass / (k_B* temp))

    # Vectorized random normal sampling using Box-Muller
    u1 = np.random.rand(num_atoms, 3)
    u2 = np.random.rand(num_atoms, 3)

    # Box-Muller transform to generate normally distributed random variables
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    return sigma * z1, sigma * z2


def velocity_verlet_step(super_cell_dims, positions, velocities, forces, gamma, 
                        neighbor_list, temp=300, dt=1, mass=157.25):
    """
    Perform one step of the Velocity Verlet algorithm and Langevin dynamics.
    """    

    mass *= 103.6427  # Convert amu to eV·fs²/Å²
    # Boltzmann constant in eV/K
    k_B = 8.61734e-5
    random_force1, random_force2 = generate_random_forces(len(positions), temp, gamma, dt=dt, mass=mass)
    exp_term = np.exp(-gamma * dt / 2)  # damping factor
    
    # Step 1: Apply first half of stochastic velocity update
    velocities = exp_term * velocities + random_force1 / mass

    # Step 2: Apply first half of velocity update due to forces
    velocities += 0.5 * dt * forces / mass
    
    # Step 3: Update positions
    positions += velocities * dt
    positions %= super_cell_dims # PBC. x = x % L -> x if xi < L else x - L

    new_forces = LJ_force(positions, super_cell_dims, neighbor_list)
    # Step 4: Apply second half of velocity update due to new forces
    velocities += 0.5 * dt * new_forces / mass

    # Step 5: Apply second half of stochastic velocity update
    velocities = exp_term * velocities + random_force2 / mass

    return positions, velocities, new_forces


def run_simulation(initial_positions, super_cell_dims, neighbor_list, temp=300, gamma=0.01, 
                    dt=1, thermalization_steps=1000, production_steps=5000, sample_rate=1):
    """
    Run a complete molecular dynamics simulation with thermalization and production phases.
    
    Parameters:
    -----------
    initial_positions : ndarray
        Initial positions of atoms
    super_cell_dims : ndarray
        Dimensions of the supercell
    temperature : float
        Temperature (K)
    gamma : float
        Friction coefficient (fs^-1)
    dt : float
        Time step (fs)
    thermalization_steps : int
        Number of thermalization steps
    production_steps : int
        Number of production steps
        
    Returns:
    --------
    trajectory : ndarray
        Array of positions for each atom at each time step during production phase
    """
    # Create a copy of the initial positions
    positions = np.copy(initial_positions)
    
    # Initialize velocities
    velocities = init_velocities(positions, temp=temp)
    
    # Calculate initial forces
    forces = LJ_force(positions, super_cell_dims, neighbor_list)
    
    # Storage for trajectory during production phase
    trajectory = np.zeros((production_steps, len(positions), 3))
    
    # Thermalization phase
    print(f"Starting thermalization phase ({thermalization_steps} steps)...")
    for step in range(thermalization_steps):
        if step % 100 == 0:
            print(f"Thermalization step {step}/{thermalization_steps}")
            
        positions, velocities, forces = velocity_verlet_step(super_cell_dims, positions, velocities, 
                                                            forces, gamma, neighbor_list, temp=temp, dt=dt)
    
    print("Thermalization complete. Starting production phase...")
    
    # Production phase
    for step in range(production_steps):
        if step % 100 == 0:
            print(f"Production step {step}/{production_steps}")
            
        positions, velocities, forces = velocity_verlet_step(super_cell_dims, positions, velocities, 
                                                            forces, gamma, neighbor_list, temp=temp, dt=dt, mass=157.25)
        
        # Store the positions in the trajectory array
        if step % sample_rate == 0:
            trajectory[step // sample_rate] = positions

    print("Production phase complete.")
    return trajectory


def calculate_autocorrelation(trajectory, max_delay=800):
    """
    Calculate the position-position autocorrelation function C(t) from trajectory data.
    
    Parameters:
    -----------
    trajectory : ndarray
        Array of shape (time_steps, n_atoms, 3) containing atom positions at each time step
    max_delay : int
        Maximum time delay to calculate correlation for
        
    Returns:
    --------
    corr : ndarray
        Autocorrelation function C(t) for time delays from 0 to max_delay
    """
    n_steps = trajectory.shape[0]
    n_atoms = trajectory.shape[1]
    
    # Limit max_delay to be less than total trajectory length
    max_delay = min(max_delay, n_steps-1)
    
    # Initialize autocorrelation array
    corr = np.zeros(max_delay+1)
    
    # Loop through each possible time delay
    for t in range(max_delay+1):
        if t % 100 == 0:
            print(f"Calculating autocorrelation for time delay {t}/{max_delay}")
        
        # Number of time origins we can use
        n_origins = n_steps - t
        
        # Initialize numerator and denominator sums for each atom
        atom_corr = np.zeros(n_atoms)
        
        # Calculate mean positions for normalization
        # Mean over all possible time origins for each atom
        mean_pos = np.mean(trajectory[:n_origins], axis=0)
        
        for i in range(n_atoms):
            # Calculate <r_i(t') · r_i(t'+t)> term
            # This is the average over all time origins t'
            pos_corr = 0
            for t_prime in range(n_origins):
                pos_corr += np.dot(trajectory[t_prime, i], trajectory[t_prime+t, i])
            pos_corr /= n_origins
            
            # Calculate <r_i(t')> · <r_i(t'+t)> term
            mean_pos_t = np.mean(trajectory[t:, i][:n_origins], axis=0)
            mean_corr = np.dot(mean_pos[i], mean_pos_t)
            
            # Calculate variance for normalization (denominator)
            # This is <r_i(t') · r_i(t')> - <r_i(t')> · <r_i(t')>
            var_i = 0
            for t_prime in range(n_origins):
                var_i += np.dot(trajectory[t_prime, i], trajectory[t_prime, i])
            var_i /= n_origins
            var_i -= np.dot(mean_pos[i], mean_pos[i])
            
            # Calculate normalized correlation for this atom
            if var_i > 1e-10:  # Avoid division by zero
                atom_corr[i] = (pos_corr - mean_corr) / var_i
            
        # Average over all atoms
        corr[t] = np.mean(atom_corr)
    
    return corr


def correlation_time(corr, dt=1.0):
    """
    Calculate the correlation time from the autocorrelation function using simpson's rule.
    """

    time = np.arange(len(corr)) * dt
    corr_time = simps(corr, time)
    
    return corr_time


def MSD(trajectory, initial_positions, super_cell_dims, dt=1.0):
    """
    Calculate the Mean Squared Displacement (MSD) from trajectory data.

    trajectory : ndarray
        Array of shape (time_steps, n_atoms, 3) containing atom positions at each time step
    
    initial_positions : ndarray
        Initial positions of atoms in the supercell, shape (n_atoms, 3)
    """

    n_steps = trajectory.shape[0]
    n_atoms = trajectory.shape[1]

    squared_displacements = np.zeros((n_steps, n_atoms))
    
    for t in range(n_steps):
        # Calculate displacement from initial positions
        delta = trajectory[t] - initial_positions
        
        # Apply minimum image convention for PBC
        delta -= np.round(delta / super_cell_dims) * super_cell_dims
        
        # Compute squared displacement per atom
        squared_displacements[t] = np.sum(delta**2, axis=1)
    
    msd = np.mean(squared_displacements)

    return msd
    


def plot_super_cell(positions, super_cell_dims):
    """
    Plot the supercell dimensions and atoms in 3D.
    """
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o')

    # Plot the edges of the supercell
    edges = np.array([
        [0, 0, 0],
        [super_cell_dims[0], 0, 0],
        [super_cell_dims[0], super_cell_dims[1], 0],
        [0, super_cell_dims[1], 0],
        [0, 0, super_cell_dims[2]],
        [super_cell_dims[0], 0, super_cell_dims[2]],
        [super_cell_dims[0], super_cell_dims[1], super_cell_dims[2]],
        [0, super_cell_dims[1], super_cell_dims[2]]
    ])
    # Define the edges to connect
    edge_connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    for start, end in edge_connections:
        ax.plot(
            [edges[start, 0], edges[end, 0]],
            [edges[start, 1], edges[end, 1]],
            [edges[start, 2], edges[end, 2]],
            'r-'
        )

    ax.set_xlabel('X-axis (Å)')
    ax.set_ylabel('Y-axis (Å)')
    ax.set_zlabel('Z-axis (Å)')
    ax.set_title('Gd Crystal Supercell Structure')
    plt.show()


if __name__ == "__main__":
     initial_positions, super_cell_dims = init_Gd_super_cell()
     print("super_cell_dims:", super_cell_dims)
     run_simulation(initial_positions, super_cell_dims, temp=300, gamma=0.01, 
                    dt=1, thermalization_steps=1000, production_steps=5000)
