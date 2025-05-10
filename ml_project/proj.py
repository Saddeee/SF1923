import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


w = [-1.2, 0.9] # Weights w1, w2

x = np.linspace(-1, 1, 201)

pairs = np.concatenate([w, x])

t = np.array([1, 2, 3, 4])

t = w[0] + t**w[1]
my = 0.0
t = np.random.normal(loc= t - my, scale=1.0, size=None)
##z = z ** 2
product = np.prod(t)

print(product)


import math

def normal_distribution(ti, mu, sigma):
    """
    Computes the value of the normal distribution N(ti | µ, σ).

    Parameters:
    ti (float): The input value.
    mu (float): The mean of the distribution.
    sigma (float): The standard deviation of the distribution.

    Returns:
    float: The value of the normal distribution at ti.
    """
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive.")
    
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((ti - mu) / sigma) ** 2
    return coefficient * math.exp(exponent)

# Example usage
result = normal_distribution(ti=1.0, mu=0.0, sigma=1.0)
print(f"Normal distribution value: {result}")


prior = normal_distribution(ti=1.0, mu=0.0, sigma=1.0)



# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 3.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

# Create a figure
fig = plt.figure()

# Create a 3D axis using add_subplot
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

# Contour plot under the surface
cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks, and view angle
ax.set_zlim(-0.15, 0.2)
ax.set_zticks(np.linspace(0, 0.2, 5))
ax.view_init(27, -21)

# Show the plot
plt.show()

# Save the plot to a file
plt.savefig('image.png')
