import numpy as np
import matplotlib.pyplot as plt
import pdb

def generate_hypersphere_points(N, k,radius=1,noise_std_dev=0.1):
    # Generate N k-dimensional Gaussian random vectors
    points = np.random.normal(size=(N, k))

    # Normalize each point to the hypersphere of given radius
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= radius

    # Add Gaussian noise to the points
    noise = np.random.normal(scale=noise_std_dev, size=(N, k))
    points += noise

    
    # Normalize the points so they fall within the range [0, 1]
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)

    normalized_points = (points - min_values) / (max_values - min_values)

    return normalized_points

# Example usage:
N = 1000  # Number of points
k = 2  # Dimensionality of the space
radius = 1  # Radius of the hypersphere

points = generate_hypersphere_points(N, k) * 1


# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter the points
#ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.scatter(points[:, 0], points[:, 1], np.zeros(N))


# Show the plot
plt.show()
