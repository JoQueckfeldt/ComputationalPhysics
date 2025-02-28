"""
This script is used to plot the end positions of 2d random walks given a number of steps N
and a number of independent realizations num_samples. Every walk starts at the origin (0, 0).

The average distance from the origin is also calculated and displayed.
"""

import numpy as np
import matplotlib.pyplot as plt

def random_walk_endpoints(N, num_samples):
    """
    Generate multiple independent random walks in 2D space with step size 1.
    """
    end_positions = np.zeros((num_samples, 2))
    ete_distances = np.zeros(num_samples)
    for i in range(num_samples):
        angles = np.random.uniform(0, 2*np.pi, N) # random angles for each step
        x = np.sum(np.cos(angles)) # x component of the final position
        y = np.sum(np.sin(angles)) # y component of the final position
        end_positions[i] = [x, y]
        ete_distances[i] = np.sqrt(x**2 + y**2)

    # Average squared distance from origin
    avg_R2 = np.mean(end_positions[:, 0]**2 + end_positions[:, 1]**2)
    return avg_R2, ete_distances, end_positions



N = 1000
num_samples = 1000000

avg_R2, ete_distances, end_positions = random_walk_endpoints(N, num_samples)
print(f"Average squared distance from origin: {avg_R2:.4f}")
print(f"Average distance from origin: {np.mean(ete_distances)**2:.4f}")

# plot end positions
plt.figure(figsize=(5, 5))
plt.plot(end_positions[:, 0], end_positions[:, 1], 'bo', alpha=0.2)
plt.plot(0, 0, 'ro', label='Origin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
    

# histogram of end-to-end distances
plt.figure(figsize=(6, 4))
plt.hist(ete_distances, bins=100, color='b', alpha=0.7)
plt.axvline(x=np.mean(ete_distances), color='r', linestyle='--', label='Average Distance')
plt.axvline(x=np.sqrt(avg_R2), color='g', linestyle='--', label='sqrt(Average R^2)')
plt.xlabel('Distance from Origin')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()