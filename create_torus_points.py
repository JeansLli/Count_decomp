import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_torus_points(N, R, r, noise_std_dev):
    # Generate N uniform random angles for the circle and tube
    theta = 2 * np.pi * np.random.rand(N)  # angle for the circle
    phi = 2 * np.pi * np.random.rand(N)  # angle for the tube

    # Convert the toroidal coordinates to Cartesian coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    # Add some Gaussian noise to the points
    x += np.random.normal(scale=noise_std_dev, size=N)
    y += np.random.normal(scale=noise_std_dev, size=N)
    z += np.random.normal(scale=noise_std_dev, size=N)
    
    # Scale the coordinates to fit in a [0, 1] x [0, 1] x [0, 1] cube
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    #z = (z - z.min()) / (z.max() - z.min())
    z = z + 0.5

    return np.column_stack([x, y, z])

# Usage:
N = 1000  # number of points
R = 1  # distance from the center of the tube to the center of the torus
r = 0.1  # radius of the tube (distance from the center of the tube to the torus surface)
noise_std_dev = 0.05  # standard deviation of the Gaussian noise
pts_range = 10
# Generate the points
points = generate_torus_points(N, R, r, noise_std_dev)
points = points * pts_range
#import pdb
#pdb.set_trace()

# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# Set the labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the same scale for all axes
ax.axis('equal')

# Set the same limits for all axes
ax.set_xlim([0, pts_range])
ax.set_ylim([0, pts_range])
ax.set_zlim([0, pts_range])

# Display the plot
plt.show()
