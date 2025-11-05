import numpy as np
import config as cfg

class StandardKF:
    def __init__(self, x0, P0):
        self.x_hat = x0.copy()
        self.P = P0.copy()

    def predict(self, u_k):
        # Standard Prediction
        self.x_hat = cfg.Ak @ self.x_hat + cfg.Bk * u_k
        self.P = cfg.Ak @ self.P @ cfg.Ak.T + cfg.process_noise

    def update(self, y_k):
        # Standard Update
        innovation = y_k - cfg.Ck @ self.x_hat
        S_k = cfg.Ck @ self.P @ cfg.Ck.T + cfg.R_k
        K_k = self.P @ cfg.Ck.T @ np.linalg.inv(S_k)

        self.x_hat = self.x_hat + K_k @ innovation
        self.P = (np.eye(cfg.n_states) - K_k @ cfg.Ck) @ self.P

class AdaptiveKF:
    def __init__(self, x0, P0):
        self.x_hat = x0.copy()
        self.P = P0.copy()
        self.lambda_k = cfg.lambda_max

        # Initial values
        self.sigma_e_sq = 1.0
        self.sigma_v_sq = 1.0
        self.sigma_q_sq = 1.0

    def _calculate_lambda(self, y_k):
        pred_error_sq = float((y_k - cfg.Ck @ self.x_hat)**2)
        self.sigma_e_sq = cfg.alpha * self.sigma_e_sq + (1 - cfg.alpha) * pred_error_sq

        x_Pk_x = float(self.x_hat.T @ self.P @ self.x_hat)
        self.sigma_v_sq = cfg.alpha * self.sigma_v_sq + (1 - cfg.alpha) * x_Pk_x

        self.sigma_q_sq = cfg.beta * self.sigma_q_sq + (1 - cfg.beta) * pred_error_sq

        sigma_e = np.sqrt(self.sigma_e_sq)
        sigma_v = np.sqrt(self.sigma_v_sq)

        if sigma_e <= sigma_v:
            self.lambda_k = cfg.lambda_max
        else:
            ratio = (self.sigma_q_sq + cfg.eps) / (self.sigma_e_sq - self.sigma_v_sq + cfg.eps)
            self.lambda_k = max(min(ratio, cfg.lambda_max), cfg.lambda_min)

    def predict(self, u_k, y_k):
        self._calculate_lambda(y_k)

        sigma_forget_k = ((1 / self.lambda_k) - 1) * self.P
        P_forget_k = self.P + sigma_forget_k

        self.x_hat = cfg.Ak @ self.x_hat + cfg.Bk * u_k
        self.P = cfg.Ak @ P_forget_k @ cfg.Ak.T + cfg.sigma_Kalman_KFS

    def update(self, y_k):
        innovation = y_k - cfg.Ck @ self.x_hat
        S_k = cfg.Ck @ self.P @ cfg.Ck.T + cfg.R_k
        K_k = self.P @ cfg.Ck.T @ np.linalg.inv(S_k)

        self.x_hat = self.x_hat + K_k @ innovation
        self.P = (np.eye(cfg.n_states) - K_k @ cfg.Ck) @ self.P

class TargetedInjectionKF(StandardKF):
    """
    Implements a targeted covariance injection algorithm.
    When a large innovation is detected, it injects uncertainty
    ONLY into the state variables affected by the unmodeled event (velocity).
    """
    def __init__(self, x0, P0):
        super().__init__(x0, P0)
        self.injection_log = []

    def update(self, y_k):
        # 1. Calculate innovation and check for outliers (same as before)
        innovation = y_k - cfg.Ck @ self.x_hat
        S_k = cfg.Ck @ self.P @ cfg.Ck.T + cfg.R_k
        S_k_inv = np.linalg.inv(S_k)
        
        d_squared = innovation.T @ S_k_inv @ innovation
        
        # 2. Implement the *TARGETED* action
        if float(d_squared) > cfg.RESET_INNOVATION_THRESHOLD:
            # Instead of a full reset, just add to the current P
            self.P = self.P + cfg.Q_INJECTION_MATRIX
            self.injection_log.append(1)
        else:
            self.injection_log.append(0)

        # 3. Perform the standard Kalman update
        K_k = self.P @ cfg.Ck.T @ S_k_inv
        self.x_hat = self.x_hat + K_k @ innovation
        self.P = (np.eye(cfg.n_states) - K_k @ cfg.Ck) @ self.P

