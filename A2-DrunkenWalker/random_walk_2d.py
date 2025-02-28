import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def single_walk(N):
    """
    Simulate a single random walk in 2D space with step size 1.

    Returns the squared distance end-to-end, given the number of steps N.
    """
    angles = np.random.uniform(0, 2*np.pi, N) # random angles for each step
    x = np.sum(np.cos(angles)) # x component of the final position
    y = np.sum(np.sin(angles)) # y component of the final position
    return x**2 + y**2 # squared distance from starting point


def random_walk_2d(N, num_samples):
    """
    Generate multiple independent random walks in 2D space with step size 1.
    
    Returns the average squared distance end-to-end distance, 
    given the number of steps N and the number of samples.
    """

    R2_values = [single_walk(N) for _ in range(num_samples)]
    avg_R2 = np.mean(R2_values)
    return avg_R2


def main():
    N_values = np.arange(10, 1011, 50)
    num_samples = 10000

    # simulating the average squared distance of for each N
    avg_R2_values = [random_walk_2d(N, num_samples)[0] for N in N_values]

    # linear fit
    slope, intercept, _, _, _ = linregress(np.log(N_values), np.log(avg_R2_values))
    print(f"Linear fit: log(R^2) = {slope} log(N) + {intercept}")

    plt.figure(figsize=(8, 6))
    plt.plot(N_values, avg_R2_values, 'bo-', label='Simulated ⟨R^2⟩') 
    plt.plot(N_values, N_values, 'r--', label='Theoretical ⟨R^2⟩ = N')
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Mean Squared End-to-End Distance ⟨R^2⟩')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
    
    