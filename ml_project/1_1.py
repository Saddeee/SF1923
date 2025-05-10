import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set the precision of the prior
alpha = 2
beta= 1.0
mean = np.array([0, 0])  # mean vector for prior
cov = (1 / alpha) * np.eye(2)  # covariance matrix (diagonal)


x_input = np.arange(-1, 1.01, 0.01)  # Adding 0.01 to the upper limit ensures inclusion

t = x_input
w = [-1.2, 0.9]
t = w[0] + t*w[1]
print(t)

test = [t[0], t[1], t[2]]




# Create a grid over w0 and w1
w0 = np.linspace(-2, 2, 100)
w1 = np.linspace(-2, 2, 100)
W0, W1 = np.meshgrid(w0, w1)
pos = np.dstack((W0, W1))

# Create multivariate normal distribution
prior = multivariate_normal(mean, cov)

posterior = multivariate_normal(mean, cov)

# Evaluate the density at each point on the grid
Z = prior.pdf(pos)

# Plot the 2D contour of the prior distribution
plt.figure(figsize=(8, 6))
contour = plt.contour(W0, W1, Z, levels=15, cmap='viridis')
plt.clabel(contour, inline=1, fontsize=8)
plt.title('Contour Plot of Prior Distribution over $w_0$ and $w_1$')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.grid(True)
plt.axis('equal')
plt.show()


plt.savefig('image1.png')