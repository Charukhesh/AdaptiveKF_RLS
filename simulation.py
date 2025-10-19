import numpy as np
import config as cfg

def run_ground_truth_simulation():
    """
    Simulates the mass-spring-damper system with wall collisions.
    This provides the "true" state and the noisy measurements for the filters.
    """
    t_span = np.arange(cfg.t_start, cfg.t_end, cfg.Ts)
    num_steps = len(t_span)

    dt = 0.001
    sim_steps = int(cfg.t_end / dt)
    t_sim = np.linspace(cfg.t_start, cfg.t_end, sim_steps)

    x_true = np.zeros((2, sim_steps))
    x_true[:, 0] = cfg.x0_true

    x_true_sampled = np.zeros((2, num_steps))
    yk_sampled = np.zeros((1, num_steps))

    measurement_noise = np.random.normal(0, np.sqrt(cfg.measurement_noise_var), num_steps)

    current_sample_idx = 0
    for i in range(1, sim_steps):
        t = t_sim[i-1]
        x_current = x_true[:, i-1]

        # Continuous dynamics
        F_t = 10 * np.sin(t)
        z_ddot = (F_t - cfg.k * x_current[0] - cfg.c * x_current[1]) / cfg.m
        x_dot = np.array([x_current[1], z_ddot])

        x_next = x_current + x_dot * dt

        # Unmodeled collision logic
        if x_next[0] >= cfg.wall_pos:
            x_next[0] = cfg.wall_pos
            x_next[1] = -x_current[1]

        x_true[:, i] = x_next

        if current_sample_idx < num_steps and t >= t_span[current_sample_idx]:
            x_true_sampled[:, current_sample_idx] = x_true[:, i]
            yk = cfg.Ck @ x_true[:, i] + measurement_noise[current_sample_idx]
            yk_sampled[0, current_sample_idx] = yk
            current_sample_idx += 1

    uk_sequence = 10 * np.sin(t_span)

    return t_span, x_true_sampled, yk_sampled, uk_sequence










