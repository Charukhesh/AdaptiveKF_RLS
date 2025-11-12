import numpy as np

# System & Simulation Parameters
# Physical Parameters
m = 10.0
k = 5.0
c = 3.0

# Simulation Parameters
Ts = 0.1
t_start = 0
t_end = 25
wall_pos = 2.0

# Initial Conditions for true system
z0 = -1.0
z_dot0 = 1.0
x0_true = np.array([z0, z_dot0])

# Filter Parameters
x0_hat = np.array([[0.0], [0.0]])
P0 = 0.1 * np.eye(2)

INNOVATION_THRESHOLD = 9.0
P_RESET_VALUE = 0.1 * np.eye(2)

# Noise Characteristics
measurement_noise_var = 0.01
R_k = np.array([[measurement_noise_var]])
process_noise = 0.01 * np.eye(2)

# Discretized System Matrices
Ak = np.array([
    [0.9975, 0.09843],
    [-0.04922, 0.9680]
])

Bk = np.array([
    [4.948e-4],
    [9.843e-3]
])

Ck = np.array([[1.0, 1.0]])

# Adaptive KF Parameters
sigma_Kalman_KFS = 0.01 * np.eye(2)
K_alpha = 2.0
K_beta = 10.0
n_states = 2
alpha = 1 - 1 / K_alpha
beta = 1 - 1 / K_beta
lambda_min = 0.5
lambda_max = 1.0
eps = 1e-6


# Targeted Injection KF Parameters
RESET_INNOVATION_THRESHOLD = 9.0
VELOCITY_UNCERTAINTY_INJECTION = 0.1
Q_INJECTION_MATRIX = np.array([
    [0.0, 0.0],
    [0.0, VELOCITY_UNCERTAINTY_INJECTION]
])

