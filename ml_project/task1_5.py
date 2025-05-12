import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# === Parameters ===
alpha = 2.0  # Prior precision
beta = 25.0  # Likelihood precision (inverse of noise variance)
mean_prior = np.array([0.0, 0.0])
cov_prior = (1 / alpha) * np.eye(2)

true_w = np.array([-1.2, 0.9])  # True weights

N = 4 # len(x_all)  # Use all samples for this example


# === Generate Synthetic Data ===
x_all = np.linspace(-1, 1, N)
X_all = np.vstack((np.ones_like(x_all), x_all)).T  # Xest - Design matrix (100, 2)
t_all = X_all @ true_w + np.random.normal(0, np.sqrt(0.2), size=x_all.shape)

# === Choose N samples for training ===
X = X_all[:N]
t = t_all[:N]


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
# contour_likelihood = plt.contour(W0, W1, Z_likelihood, levels=10, cmap='Blues', linestyles='dashed')
contour_posterior = plt.contour(W0, W1, Z_posterior, levels=10, cmap='Reds')

# Sample 5 random points from the posterior distribution

curves = [posterior.rvs() for _ in range(5)] 

plt.clabel(contour_prior, fontsize=8)
# plt.clabel(contour_likelihood, fontsize=8)
plt.clabel(contour_posterior, fontsize=8)

plt.title(f'Prior (green dotted), Likelihood (blue dashed), Posterior (red)\nBased on N = {N} samples')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.legend(['Prior', 'Likelihood', 'Posterior'], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()


tests_all= np.linspace(-1.5, 1.5, 30)
Test_all = np.vstack((np.ones_like(tests_all), tests_all)).T
t_tests = Test_all @ true_w + np.random.normal(0, np.sqrt(0.2), size=tests_all.shape)

# Plot y = w0 + w1x for sampled weights
plt.figure(figsize=(10, 8))
plt.scatter(x_all, t_all, label="Data", color="black", alpha=0.6)  # Plot the data points
# plt.scatter(t_tests, tests_all, label="Tests", color="red", alpha=0.6)
# Generate x values for the line
#x_all = np.linspace(-1, 1, 100)
plt.scatter(tests_all, t_tests, label="Tests", color="red", alpha=0.6)

# Plot lines for each sampled weight
for curve in curves:
    w0_sample, w1_sample = curve
    y_line = w0_sample + w1_sample * x_all
    plt.plot(x_all, y_line, label=f"Sampled line: w0={w0_sample:.2f}, w1={w1_sample:.2f}", alpha=0.7)

plt.title("Lines Sampled from Posterior Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True)

plt.savefig('posterior.png')
plt.show()





predictive_means = []
predictive_stds = []

for x in x_all:
    phi_x = np.array([1, x])  # Design vector for input x
    mean = m_N.T @ phi_x        # Predictive mean
    variance = (1 / beta) + phi_x @ S_N @ phi_x.T  # Predictive variance
    predictive_means.append(mean)
    predictive_stds.append(np.sqrt(variance))


# Convert to numpy arrays for easier handling
predictive_means = np.array(predictive_means)
predictive_stds = np.array(predictive_stds)

# Plot data points with error bars
plt.figure(figsize=(10, 8))
plt.errorbar(x_all, predictive_means, yerr=predictive_stds, fmt='o', color='blue', label="Predictive Mean ± Std Dev", alpha=0.7)
plt.scatter(x_all, t_all, color='black', label="Data Points", alpha=0.6)

# Add title and labels
plt.title("Predictive Distribution with Error Bars")
plt.xlabel("x")
plt.ylabel("t")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True)

# Save and show the plot
plt.savefig('predictive_error_bars.png')
plt.show()

plt.savefig('prediction_task_1_5.png')