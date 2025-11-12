import matplotlib.pyplot as plt

def plot_all_figures(results):
    """ Generates all figures based on the simulation results """
    t_span = results['t_span']
    x_true = results['x_true']
    x_kf, P_kf = results['x_kf'], results['P_kf']
    x_kfs, P_kfs, lambda_k = results['x_kfs'], results['P_kfs'], results['lambda_k']
    x_tikf, P_tikf, injection_log = results['x_tikf'], results['P_tikf'], results['injection_log']

    # State Estimation and Forgetting Factor
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("State Estimation and Forgetting Factor", fontsize=16)

    axes2[0].plot(t_span, x_true[0, :], 'b-', label='True State')
    axes2[0].plot(t_span, x_kf[0, :], 'r--', label='KF')
    axes2[0].plot(t_span, x_kfs[0, :], 'k:', lw=2, label='KF* (Var. Forget)')
    axes2[0].plot(t_span, x_tikf[0, :], 'g-.', label='TI-KF (Targeted)')
    axes2[0].set_ylabel('Position z(t)'); axes2[0].legend(); axes2[0].grid(True); axes2[0].set_ylim(-4, 4)

    axes2[1].plot(t_span, x_true[1, :], 'b-', label='True State')
    axes2[1].plot(t_span, x_kf[1, :], 'r--', label='KF')
    axes2[1].plot(t_span, x_kfs[1, :], 'k:', lw=2, label='KF* (Var. Forget)')
    axes2[1].plot(t_span, x_tikf[1, :], 'g-.', label='TI-KF (Targeted)')
    axes2[1].set_ylabel('Velocity ż(t)'); axes2[1].grid(True); axes2[1].set_ylim(-4, 3)

    axes2[2].plot(t_span, lambda_k, 'm-', label='λk (KF*)')
    axes2[2].stem(t_span, injection_log, linefmt='g-', markerfmt='go', basefmt=" ", label='Injection Trigger (TI-KF)')
    axes2[2].set_ylabel('Adaptation Trigger'); axes2[2].set_xlabel('Time (s)'); axes2[2].legend(); axes2[2].grid(True);
    plt.savefig("./figures/State_Estimation_and_Forgetting_Factor.png") 

    # Estimation Error
    error_kf, error_kfs, error_tikf = x_true - x_kf, x_true - x_kfs, x_true - x_tikf

    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig3.suptitle("Estimation Error", fontsize=16)

    axes3[0].plot(t_span, error_kf[0, :], 'r-', label='KF Error')
    axes3[0].plot(t_span, error_kfs[0, :], 'b-', label='KF* Error')
    axes3[0].plot(t_span, error_tikf[0, :], 'g-.', label='TI-KF (Targeted)')
    axes3[0].set_ylabel('Position Error'); axes3[0].legend(); axes3[0].grid(True); axes3[0].set_ylim(-1, 1)

    axes3[1].plot(t_span, error_kf[1, :], 'r-', label='KF Error')
    axes3[1].plot(t_span, error_kfs[1, :], 'b-', label='KF* Error')
    axes3[1].plot(t_span, error_tikf[1, :], 'g-.', label='TI-KF (Targeted)')
    axes3[1].set_ylabel('Velocity Error'); axes3[1].set_xlabel('Time (s)'); axes3[1].legend(); axes3[1].grid(True); axes3[1].set_ylim(-2, 2)
    plt.savefig("./figures/Estimation_Error.png")

    # Marginal Variance
    var_z_kf = P_kf[:, 0, 0]; var_zdot_kf = P_kf[:, 1, 1]
    var_z_kfs = P_kfs[:, 0, 0]; var_zdot_kfs = P_kfs[:, 1, 1]
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig4.suptitle("Marginal Variance (Diagonal of P)", fontsize=16)

    axes4[0].plot(t_span, var_z_kf, 'r-', label='KF')
    axes4[0].plot(t_span, var_z_kfs, 'b-', label='KF*')
    axes4[0].plot(t_span, P_tikf[:, 0, 0], 'g-.', label='TI-KF')
    axes4[0].set_ylabel('σ_z^2 (Position)'); axes4[0].legend(); axes4[0].grid(True); axes4[0].set_ylim(0, 0.16)
    
    axes4[1].plot(t_span, var_zdot_kf, 'r-', label='KF')
    axes4[1].plot(t_span, var_zdot_kfs, 'b-', label='KF*')
    axes4[1].plot(t_span, P_tikf[:, 1, 1], 'g-.', label='TI-KF')
    axes4[1].set_ylabel('σ_ż^2 (Velocity)'); axes4[1].set_xlabel('Time (s)'); axes4[1].legend(); axes4[1].grid(True); axes4[1].set_ylim(0, 0.16)
    plt.savefig("./figures/Marginal_Variance.png")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()