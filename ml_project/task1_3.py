import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# === Parameters ===
alpha = 2.0  # Prior precision
beta = 25.0  # Likelihood precision (inverse of noise variance)
mean_prior = np.array([0.0, 0.0])
cov_prior = (1 / alpha) * np.eye(2)


# Generate synthetic data
x_input = np.arange(-1, 1.01, 0.01)  # Total 201 points
true_w = np.array([-1.2, 0.9])       # True weights
X_full = np.vstack((np.ones_like(x_input), x_input)).T  # Design matrix (201 x 2)
# Generate synthetic data WITH NOISE
noise = np.random.normal(loc=0, scale=np.sqrt(1/beta), size=len(x_input))
t_full = X_full @ true_w + noise  # Add Gaussian noise

# Select N training samples RANDOMLY
N = 10
indices = np.random.choice(len(X_full), size=N, replace=False)
X = X_full[indices]
t = t_full[indices]

# === Grid over weight space ===
w0 = np.linspace(-2, 2, 100)
w1 = np.linspace(-2, 2, 100)
W0, W1 = np.meshgrid(w0, w1)
grid = np.dstack((W0, W1))
grid_shape = W0.shape

# === Compute Prior over grid ===
prior = multivariate_normal(mean_prior, cov_prior)
Z_prior = prior.pdf(grid)

# === Compute Likelihood over grid ===
def compute_likelihood_grid(X, t, beta, W0, W1):
    Z_likelihood = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w = np.array([W0[i, j], W1[i, j]])
            mu = X @ w
            likelihoods = norm.pdf(t, loc=mu, scale=np.sqrt(1/beta))
            Z_likelihood[i, j] = np.prod(likelihoods)
    return Z_likelihood

Z_likelihood = compute_likelihood_grid(X, t, beta, W0, W1)

# === Compute Posterior ===
S_N_inv = alpha * np.eye(2) + beta * X.T @ X
S_N = np.linalg.inv(S_N_inv)
m_N = beta * S_N @ X.T @ t
posterior = multivariate_normal(m_N, S_N)
Z_posterior = posterior.pdf(grid)

# === Plot All Three Contours ===
plt.figure(figsize=(10, 8))
contour_prior = plt.contour(W0, W1, Z_prior, levels=10, cmap='Greens', linestyles='dotted')
contour_likelihood = plt.contour(W0, W1, Z_likelihood, levels=10, cmap='Blues', linestyles='dashed')
contour_posterior = plt.contour(W0, W1, Z_posterior, levels=10, cmap='Reds')

plt.clabel(contour_prior, fontsize=8)
plt.clabel(contour_likelihood, fontsize=8)
plt.clabel(contour_posterior, fontsize=8)

plt.title(f'Prior (green dotted), Likelihood (blue dashed), Posterior (red)\nBased on N = {N} samples')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.legend(['Prior', 'Likelihood', 'Posterior'], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()

plt.savefig('posterior.png')