import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# === Parameters ===
alpha = 2.0  # Prior precision
beta = 25.0  # Likelihood precision (inverse of noise variance)
mean_prior = np.array([0.0, 0.0])
cov_prior = (1 / alpha) * np.eye(2)

true_w = np.array([0.0, 2.5, -0.5])  # True weights

N = 10 # len(x_all)  # Use all samples for this example

# === Generate Synthetic Data ===


x_1_all = np.linspace(-1, 1, N)
x_2_all = np.linspace(-1, 1, N)

sigma = 0.3
X_all = np.vstack((np.ones_like(x_1_all), x_1_all**2, x_2_all**3))  # Design vector
t_all = true_w.T @ X_all + np.random.normal(0, sigma, size=x_1_all.shape)





def plot_feature_space_3d(x1, x2):
    """
    Plots a 3D point cloud of the feature space where:
    x-axis: x1^2
    y-axis: x2^3
    z-axis: target t (optionally synthetic)

    Parameters:
        x1 (np.ndarray): Array of x1 values
        x2 (np.ndarray): Array of x2 values
    """
    # Transform inputs into feature space
    x1_squared = x1**2
    x2_cubed = x2**3

    # Optionally compute synthetic target or just visualize feature transform
    # For example: z = 2.5 * x1^2 - 0.5 * x2^3 (matches true_w)
    z = 2.5 * x1_squared - 0.5 * x2_cubed + np.random.normal(0, 0.1, size=x1.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1_squared, x2_cubed, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax.set_title('3D Feature Space (x1^2, x2^3, t)')
    ax.set_xlabel('$x_1^2$')
    ax.set_ylabel('$x_2^3$')
    ax.set_zlabel('Target $t$')

    plt.tight_layout()
    plt.show()
    plt.savefig('task_2_2.png')



plot_feature_space_3d(x_1_all, x_2_all)