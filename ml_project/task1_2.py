import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal


def compute_likelihood(w, X, t, beta):
    """Computes the likelihood P(t | X, w) as product of Gaussians."""
    mu = X @ w  # Mean predictions for each x_i
    var = 1 / beta
    likelihoods = norm.pdf(t, loc=mu, scale=np.sqrt(var))  # Vector of length N
    return np.prod(likelihoods)  # Scalar likelihood


# Parameters
alpha = 2.0  # Prior precision
beta = 1.0   # Likelihood precision
mean = np.array([0, 0])  # Prior mean
cov = (1 / alpha) * np.eye(2)  # Prior covariance

# Generate synthetic data
x_input = np.arange(-1, 1.01, 0.01)  # Total 201 points
true_w = np.array([-1.2, 0.9])       # True weights
X_full = np.vstack((np.ones_like(x_input), x_input)).T  # Design matrix (201 x 2)
# Generate synthetic data WITH NOISE
noise = np.random.normal(loc=0, scale=np.sqrt(0.2), size=len(x_input))
t_full = X_full @ true_w + noise  # Add Gaussian noise

# Select N training samples RANDOMLY
N = 3
indices = np.random.choice(len(X_full), size=N, replace=False)
X_train = X_full[indices]
t_train = t_full[indices]

# Evaluate prior over w0 × w1 grid
w0 = np.linspace(-2, 2, 100)
w1 = np.linspace(-2, 2, 100)
W0, W1 = np.meshgrid(w0, w1)
grid_points = np.dstack((W0, W1))  # Shape: (100, 100, 2)

# PRIOR plot
prior = multivariate_normal(mean=mean, cov=cov)
Z_prior = prior.pdf(grid_points)

plt.figure(figsize=(8, 6))
contour_prior = plt.contour(W0, W1, Z_prior, levels=15, cmap='Blues')
plt.clabel(contour_prior, inline=1, fontsize=8)
plt.title('Prior over $w_0$ and $w_1$')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.grid(True)
plt.axis('equal')
plt.show()


# LIKELIHOOD plot across w0 × w1
Z_likelihood = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w_candidate = np.array([W0[i, j], W1[i, j]])
        Z_likelihood[i, j] = compute_likelihood(w_candidate, X_train, t_train, beta)


plt.figure(figsize=(8, 6))
contour_likelihood = plt.contour(W0, W1, Z_likelihood, levels=15, cmap='plasma')
plt.clabel(contour_likelihood, inline=1, fontsize=8)
plt.title('Likelihood $P(t | X, w)$ using N=3 samples')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.grid(True)
plt.axis('equal')
plt.show()

plt.savefig('image2.png')