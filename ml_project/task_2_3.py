import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Parameters ===
alpha = 2.0
beta = 25.0
mean_prior = np.array([0.0, 0.0])
cov_prior = (1 / alpha) * np.eye(2)
true_w = np.array([0.0, 2.5, -0.5])  # w0, w1, w2

# === Test Domain (excluding center region) ===
x1_parts_test = [np.linspace(-1, -0.3, 50), np.linspace(0.3, 1, 50)]
x2_parts_test = [np.linspace(-1, -0.3, 50), np.linspace(0.3, 1, 50)]

x1_parts_train = np.linspace(-0.3, 0.3,50)
x2_parts_train = np.linspace(-0.3, 0.3,50)
x_1_all_train, x_2_all_train = np.meshgrid(x1_parts_train, x2_parts_train)
x_1_all_train = x_1_all_train.flatten()
x_2_all_train = x_2_all_train.flatten()

x1_full_test = np.concatenate(x1_parts_test)
x2_full_test = np.concatenate(x2_parts_test)

x_1_all_test, x_2_all_test = np.meshgrid(x1_full_test, x2_full_test)
x_1_all_test = x_1_all_test.flatten()
x_2_all_test = x_2_all_test.flatten()

sigma = 0.8

# === Design matrix: φ(x) = [1, x1^2, x2^3] ===
X_all_test = np.vstack((np.ones_like(x_1_all_test), x_1_all_test**2, x_2_all_test**3))  # Shape: (3, N)
t_all_test = true_w.T @ X_all_test + np.random.normal(0, sigma, size=x_1_all_test.shape)


# === Design matrix: φ(x) = [1, x1^2, x2^3] ===
X_all_train = np.vstack((np.ones_like(x_1_all_train), x_1_all_train**2, x_2_all_train**3))  # Shape: (3, N)
t_all_train = true_w.T @ X_all_train + np.random.normal(0, sigma, size=x_1_all_train.shape)
'''
# === Plot the noisy test data ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_1_all_test, x_2_all_test, t_all_test, c=t_all_test, cmap='viridis')
ax.set_title(f"Generated Test Data (σ = {sigma})")
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.tight_layout()
plt.savefig('task_2_test.png')  # Save BEFORE show
plt.show()
'''

print(X_all_train)
# Ml skattning av w0, w1
# w_ml = (np.linalg.inv(X_all_train.T @ X_all_train))@ (X_all_train.T @ t_all_train)
w_ml = np.linalg.inv(X_all_train @ X_all_train.T) @ (X_all_train @ t_all_train)
summa = sum((t_all_train - (X_all_train.T @ w_ml))**2)

# summa = [(t_all_train[i]-w_ml.T*X_all_train[i])**2 for i in range(0,len(t_all_train))]

beta = 1/(summa / len(x1_parts_train))
print("Your little beta is here: " + str(beta))




# Predict using ML weights on the test set
# t_pred_test = w_ml @ X_all_test.T  # (3,) @ (3, N) → (N,)
t_pred_train = X_all_train.T  @ w_ml

mean_square_error = sum((t_pred_train - t_all_train)**2) / len(t_pred_train)
print("you mse is " + str(mean_square_error))
# Plot predicted surface over test points
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_1_all_train, x_2_all_train, t_pred_train, c=t_pred_train, cmap='viridis', label="ML prediction")

ax.set_title("Function Prediction Using ML Weights (Test Region)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$t_{pred}$")
plt.legend()
plt.tight_layout()
plt.savefig('task_2_3.png')
plt.show()

