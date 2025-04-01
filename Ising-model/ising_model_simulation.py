import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.special import ellipk, ellipe  # Import elliptic integrals
from tqdm import tqdm

@jit
def hamiltonian(lattice, J, B):
        """
        Compute the total energy of the system.
        """
        L = lattice.shape[0]
        
        H = 0.0

        # sum over all spins
        for i in range(L):
            for j in range(L):
                # Considering the right and below neighbors so that each interaction is counted only once.
                right_neighbor = lattice[i, (j + 1) % L] # '% L' for periodic boundary conditions
                below_neighbor = lattice[(i + 1) % L, j]

                # 1st term: interaction with right and below neighbors
                # 2nd term: external magnetic field
                H += -J * lattice[i, j] * (right_neighbor + below_neighbor) - B * lattice[i, j]
        return H


@jit
def delta_energy(lattice, i, j, J, B):
    """
    Compute the change in energy when flipping a spin at position (i, j).
    """
    L = lattice.shape[0]

    # nearest neighbors
    right= lattice[i, (j + 1) % L]
    left = lattice[i, (j - 1) % L]
    above = lattice[(i - 1) % L, j]
    below = lattice[(i + 1) % L, j]

    # Calculate energy difference
    dE = 2 * J * lattice[i, j] * (right + left + above + below) + 2 * B * lattice[i, j]

    return dE


@jit
def metropolis_step(lattice, J, T, B):
    L = lattice.shape[0]
    for _ in range(L**2): # L^2 trials
        # randomly select a spin
        i, j = np.random.randint(0, L, 2)

        # calculate energy difference
        dE = delta_energy(lattice, i, j, J, B)

        # flip spin if Delta E <= 0 or with probability exp(-ΔE / T)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1


@jit
def heat_bath_step(lattice, J, T):
    L = lattice.shape[0]
    for _ in range(L**2):
        # randomly select a spin
        i, j = np.random.randint(0, L, 2)

        # nearest neighbors
        right = lattice[i, (j + 1) % L]
        left = lattice[i, (j - 1) % L]
        above = lattice[(i - 1) % L, j]
        below = lattice[(i + 1) % L, j]

        # probability of spin up (+1).
        p = 1.0 / (1.0 + np.exp(-2.0 * J / T * (right + left + above + below)))

        # update spin based on heat bath probability
        new_spin = 1 if np.random.rand() < p else -1
        
        # flip spin if different
        if new_spin != lattice[i, j]:
            lattice[i, j] = new_spin


@jit(nopython=True, parallel=True)
def monte_carlo_simulation(L, J, T, B, thermalization_steps, measurement_steps, measurement_interval, algo, num_runs):
    """
    Perform Monte Carlo simulation of the 2D Ising model using Metropolis or heat bath algorithm.
    Return the average energy, average absolute magnetization, susceptibility, heat capacity, and Binder cumulant.
    """

    E = np.zeros(num_runs)
    M = np.zeros(num_runs)
    chi = np.zeros(num_runs)
    C_B = np.zeros(num_runs)
    U_L = np.zeros(num_runs)

    # parallelize over runs (each run is independent)
    for run in prange(num_runs):

        # set random seed
        np.random.seed(22 + run)

        # initialize lattice with random spin configuration
        lattice = np.random.choice(np.array([-1, 1]), size=(L, L))

        energies = np.zeros(measurement_steps//measurement_interval)
        magnetizations = np.zeros(measurement_steps//measurement_interval)

        index = 0
        for step in range(thermalization_steps + measurement_steps):
            if algo == "metropolis":
                metropolis_step(lattice, J, T, B)
            else:
                heat_bath_step(lattice, J, T)

            # Collect data after thermalization. Record measurements at intervals to reduce correlation
            if step >= thermalization_steps and step % measurement_interval == 0:
                energies[index] = hamiltonian(lattice, J, B)
                magnetizations[index] = np.sum(lattice)
                index += 1

        # calculate mean energy and magnetization^2 (used twice in the calculations below) 
        E_mean = np.mean(energies)
        M_squared_mean = np.mean(magnetizations**2)

        # calculate observables
        E[run] = E_mean / L**2 # mean energy per spin
        M[run] = np.mean(np.abs(magnetizations)) / L**2 # mean absolute magnetization per spin
        chi[run] = (M_squared_mean - np.mean(magnetizations)**2) / (T * L**2) # susceptibility per spin
        C_B[run] = (np.mean(energies**2) - E_mean**2) / (T**2 * L**2) # heat capacity per spin
        
        # Binder cumulant: U = 1 - <M^4>/(3 <M^2>^2)
        M4 = np.mean(magnetizations**4)
        U_L[run] = 1 - M4 / (3 * (M_squared_mean**2)) if M_squared_mean>1e-8 else 0 # binder cumulant

    # return average over the given number of runs
    return np.mean(E), np.mean(M), np.mean(chi), np.mean(C_B), np.mean(U_L)


def exact_M_and_CB(T, J):
    """
    Compute the exact analytical magnetization and specific heat for the 2D Ising model.

    Parameters
    ----------
    T: array-like
        Temperatures
    J: float
        Interaction strength

    Returns
    -------
    tuple:
        Magnetization (array-like), Specific heat (array-like)
    """
    beta = 1 / np.array(T)
    x = 2 * J / T
    z = np.exp(-2 * beta * J)  # z = exp(-2J/kT)

    # Critical temperature for the 2D Ising model
    Tc = 2 * J / np.log(1 + np.sqrt(2))

    # Magnetization (M/N)
    M = np.zeros_like(T)
    M[T < Tc] = (1 + z[T < Tc]**2)**(1/4) * (1 - 6 * z[T < Tc]**2 + z[T < Tc]**4)**(1/8) / (1 - z[T < Tc]**2)**(1/2)

    # Specific heat (C_B/N)
    kappa = np.minimum(2 * np.sinh(x) / np.cosh(x)**2, 1)  # Ensure kappa <= 1
    kappa_prime = 2 * np.tanh(x)**2 - 1                    # kappa' = 2 tanh^2(2J) - 1
    coth_2J = 1 / np.tanh(x)                               # coth(2J)

    C_B = (2 / np.pi) * (J * coth_2J)**2 * (
        2 * ellipk(kappa) - 2 * ellipe(kappa) - (1 - kappa_prime) * (np.pi / 2 + kappa_prime * ellipk(kappa))
    )

    return M, C_B


def run_simulation(lattice_sizes, temps, J=1, B=0, thermalization_steps=2000, measurement_steps=10000, 
                    measurement_interval=10, algo="metropolis", num_MCS=4):
    """
    Run the Ising model simulation for different lattice sizes and temperatures.

    Parameters
    ----------
    lattice_sizes: list[int]
        List of lattice sizes to simulate. e.g. [10, 20, 30] -> 10x10, 20x20, 30x30 lattices
    temps: list[float]
        List of temperatures to simulate
    thermalization_steps: int
        Number of Monte Carlo steps for thermalization
    measurement_steps: int
        Number of Monte Carlo steps for measurements
    measurement_interval: int
        Interval to record measurements
    J: float
        Interaction strength
    B: float
        External magnetic field
    algo: str
        "metropolis" or "heatbath" for the Monte Carlo algorithm to use
    num_MCS: int
        Number of independent Monte Carlo simulations to run and average over

    Returns
    -------
    dict:
        Dictionary containing the simulation results
    """

    results = {"T": temps}  # Store temperatures at the top level
    for L in lattice_sizes:
        avg_E = np.zeros(len(temps))
        abs_avg_M = np.zeros(len(temps))
        susceptibilities = np.zeros(len(temps))
        specific_heats = np.zeros(len(temps))
        cumulants = np.zeros(len(temps))

        print(f"Running simulation for lattice size: {L}x{L}")
        temp_iter = tqdm(enumerate(temps), total=len(temps), desc=f"L={L}")
        for i, T in temp_iter:
            E, M, chi, C_B, U_L = monte_carlo_simulation(
                L, J, T, B, thermalization_steps, measurement_steps, measurement_interval, algo, num_MCS
            )

            avg_E[i] = E
            abs_avg_M[i] = M
            susceptibilities[i] = chi
            specific_heats[i] = C_B
            cumulants[i] = U_L
        
        results[L] = {
            "E": avg_E, 
            "M_abs": abs_avg_M,
            "chi": susceptibilities, 
            "C_B": specific_heats, 
            "U_L": cumulants
        }
    return results


def plot_results(results, J=1, include_exact=True):
    """
    Plot the simulation results for different lattice sizes and compare with exact analytical results.
    """
    observables = ["E", "M_abs", "chi", "C_B", "U_L"]
    labels = {
        "E": "Energy",
        "M_abs": "Absolute Magnetization",
        "chi": "Susceptibility",
        "C_B": "Specific Heat",
        "U_L": "Binder Cumulant"
    }

    if include_exact:
        # Use the top-level "T" key for temperatures
        T_ex = np.linspace(min(results["T"]), max(results["T"]), 100)
        M_exact, C_B_exact = exact_M_and_CB(T_ex, J)

    for obs in observables:
        plt.figure(figsize=(7, 5))
        for L, data in results.items():
            if L == "T":  # Skip the top-level "T" key
                continue
            plt.plot(results["T"], data[obs], "o--", label=f"L={L}")
        
        # Add exact analytical results for magnetization and specific heat
        if include_exact and obs == "M_abs":
            plt.plot(T_ex, M_exact, "k--", label="Exact")
        elif include_exact and obs == "C_B":
            plt.plot(T_ex, C_B_exact, "k--", label="Exact")
        
        plt.xlabel(r"Temperature [$\text{J/k}_{\text{B}}$]")
        plt.ylabel(labels[obs])
        #plt.title(f"{labels[obs]} vs Temperature")
        plt.grid()
        plt.legend()
        plt.show()


def visualize_lattice_evolution(L, J, T, B, steps, algo="metropolis", interval=100, initial_state='random'):
    """
    Visualize the evolution of the lattice during thermalization.

    Parameters
    ----------
    L: int
        Lattice size (LxL).
    J: float
        Interaction strength.
    T: float
        Temperature.
    B: float
        External magnetic field.
    thermalization_steps: int
        Number of thermalization steps.
    algo: str
        Algorithm to use ("metropolis" or "heatbath").
    interval: int
        Interval at which to capture snapshots of the lattice.
    initial_state: str
        Initial state of the lattice ("random", "pos" or "neg").
    """
    if initial_state == 'neg':
        lattice = np.ones((L, L), dtype=np.int8) * -1
    elif initial_state == 'pos':
        lattice = np.ones((L, L), dtype=np.int8)
    elif initial_state == 'random':
        lattice = np.random.choice([-1, 1], size=(L, L))
    else:
        raise ValueError("initial_state must be 'random' or 'ordered'")
    snapshots = [lattice.copy()]  # Store the initial lattice configuration

    for step in range(interval, steps + 1, interval):
        for _ in range(interval):
            if algo == "metropolis":
                # Inline the logic of metropolis_step to avoid JIT issues
                i, j = np.random.randint(0, L, 2)
                dE = 2 * J * lattice[i, j] * (
                    lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L] +
                    lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j]
                ) + 2 * B * lattice[i, j]
                if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                    lattice[i, j] *= -1
            else:
                # Inline the logic of heat_bath_step to avoid JIT issues
                i, j = np.random.randint(0, L, 2)
                neighbors = (
                    lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L] +
                    lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j]
                )
                p = 1.0 / (1.0 + np.exp(-2.0 * J / T * neighbors))
                lattice[i, j] = 1 if np.random.rand() < p else -1

        snapshots.append(lattice.copy())  # Store the lattice at each interval

    # Plot the snapshots
    num_snapshots = len(snapshots)
    plt.figure(figsize=(15, 5))
    for idx, snapshot in enumerate(snapshots):
        plt.subplot(1, num_snapshots, idx + 1)
        plt.imshow(snapshot, cmap="coolwarm", interpolation="nearest")
        plt.title(f"Step {idx * interval}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()