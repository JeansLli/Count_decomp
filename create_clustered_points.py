import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_clustered_points(N, d, c, cluster_std_dev):
    # Initialize an empty array to hold the points
    points = np.empty((0, d))

    # Calculate the number of points per cluster
    points_per_cluster = N // c

    for _ in range(c):
        # Generate a random center for this cluster
        center = np.random.rand(d)

        # Generate points around this center with Gaussian noise
        cluster_points = np.random.normal(loc=center, scale=cluster_std_dev, size=(points_per_cluster, d))

        # Append the points to the array
        points = np.vstack([points, cluster_points])

    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)

    # Normalize the points so they fall within the range [0, 1]
    normalized_points = (points - min_values) / (max_values - min_values)

    return normalized_points


# Set the parameters
N = 1000  # total number of points
d = 3     # number of dimensions
c = 4     # number of clusters
cluster_std_dev = 0.05  # standard deviation of Gaussian noise

# Generate the points
points = generate_clustered_points(N, d, c, cluster_std_dev)

# Now points is a N x d numpy array containing the generated points


# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# Show the plot
plt.show()


# # Scatter the points
# plt.scatter(points[:, 0], points[:, 1])

# # Show the plot
# plt.show()