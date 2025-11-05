import numpy as np
import config as cfg
from simulation import run_ground_truth_simulation
from filters import StandardKF, AdaptiveKF, TargetedInjectionKF
from plot import plot_all_figures

def main():
    """ Main function to run the simulation and filtering """
    print("Starting simulation...")
    
    # Run the ground truth simulation to get data
    t_span, x_true_hist, yk_hist, uk_hist = run_ground_truth_simulation()
    num_steps = len(t_span)
    
    # Initialize filters
    kf = StandardKF(cfg.x0_hat, cfg.P0)
    kfs = AdaptiveKF(cfg.x0_hat, cfg.P0)
    tikf = TargetedInjectionKF(cfg.x0_hat, cfg.P0)

    # Prepare storage for results
    x_hat_kf_hist = np.zeros((cfg.n_states, num_steps))
    P_kf_hist = np.zeros((num_steps, cfg.n_states, cfg.n_states))
    
    x_hat_kfs_hist = np.zeros((cfg.n_states, num_steps))
    P_kfs_hist = np.zeros((num_steps, cfg.n_states, cfg.n_states))
    lambda_k_hist = np.zeros(num_steps)

    x_hat_tikf_hist = np.zeros((cfg.n_states, num_steps))
    P_tikf_hist = np.zeros((num_steps, cfg.n_states, cfg.n_states))
    injection_log_hist = np.zeros(num_steps)

    # Set initial values
    x_hat_kf_hist[:, 0] = kf.x_hat.flatten()
    P_kf_hist[0, :, :] = kf.P
    x_hat_kfs_hist[:, 0] = kfs.x_hat.flatten()
    P_kfs_hist[0, :, :] = kfs.P
    lambda_k_hist[0] = kfs.lambda_k
    x_hat_tikf_hist[:, 0] = tikf.x_hat.flatten()
    P_tikf_hist[0, :, :] = tikf.P
    
    print("Running filters...")
    # Main loop
    for k in range(num_steps - 1):
        y_k = yk_hist[:, k+1]
        u_k = uk_hist[k]
        
        # Standard KF
        kf.predict(u_k)
        kf.update(y_k)
        x_hat_kf_hist[:, k+1] = kf.x_hat.flatten()
        P_kf_hist[k+1, :, :] = kf.P
        
        # Adaptive KF
        kfs.predict(u_k, y_k)
        kfs.update(y_k)
        x_hat_kfs_hist[:, k+1] = kfs.x_hat.flatten()
        P_kfs_hist[k+1, :, :] = kfs.P
        lambda_k_hist[k+1] = kfs.lambda_k

        # Covariance Reseting KF
        tikf.predict(u_k)
        tikf.update(y_k)
        x_hat_tikf_hist[:, k+1] = tikf.x_hat.flatten()
        P_tikf_hist[k+1, :, :] = tikf.P
        injection_log_hist[k+1] = tikf.injection_log[-1]
        
    print("Simulation complete. Generating plots...")
    results = {
        't_span': t_span,
        'x_true': x_true_hist,
        'x_kf': x_hat_kf_hist,
        'P_kf': P_kf_hist,
        'x_kfs': x_hat_kfs_hist,
        'P_kfs': P_kfs_hist,
        'lambda_k': lambda_k_hist,
        'x_tikf': x_hat_tikf_hist,
        'P_tikf': P_tikf_hist,
        'injection_log': injection_log_hist,
    }
    plot_all_figures(results)

if __name__ == "__main__":
    main()