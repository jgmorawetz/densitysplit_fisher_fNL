import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def derivative_convergence_fit(x, C):
    """The analytic function to fit to the derivative convergence as a function
       of the number of realizations averaged over (normalized to 500 realizations).

    Args:
        x (array): Number of realizations averaged over (including all three LOS directions 
                   for each).
        C (float): The constant denoting level of convergence.

    Returns:
        array: The value of the derivative convergence at those number of realizations.
    """
    return ((1+C)/(1+500*C/x))**0.5


if __name__ == '__main__':
    
    # Makes plot of the normalized constraints as a function of number of derivative realizations averaged over
    # (holding the number of covariance matrix realizations constant at 15000)
    query_type, n_randoms, n_quantiles, filter_radius = 'lattice', None, 5, 10
    data_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}'
    nderiv_list = np.arange(10, 510, 10)
    ncov_list = np.arange(2100, 15300, 300)
    ncov_max, nderiv_max = 15000, 500
    # Left side of plot are the covariance constraints while right hand side are derivative convergence
    fig, ax = plt.subplots(3, 2, dpi=400, figsize=(8,8))
    fig.subplots_adjust(hspace=0.15, wspace=0.25, left=0.09, right=0.97, top=0.96, bottom=0.07)
    # Iterates through each function combination
    param_list = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    param_list_labels = [r'$f_{NL}^{loc}$', r'$f_{NL}^{equil}$', r'$f_{NL}^{ortho}$', r'$M_{min}$', r'$h$', r'$n_s$', r'$\Omega_m$', r'$\sigma_8$']
    param_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    combs = ['Halo', 'DS', 'Joint']
    # Initiates arrays to save to file for the raw and corrected constraints for each parameter and function combination
    raw_constraints = np.zeros((len(param_list), len(combs)))
    corrected_constraints = np.zeros((len(param_list), len(combs)))
    for i in range(3):
        comb = combs[i]
        for j in range(len(param_list)):
            param = param_list[j]
            param_comb_data_cov = [np.sqrt(np.load(os.path.join(data_folder, f'marginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv_max}.npy'))[-1][j][j]) for ncov in ncov_list]
            param_comb_data_deriv = [np.sqrt(np.load(os.path.join(data_folder, f'marginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov_max}_{nderiv}.npy'))[-1][j][j]) for nderiv in nderiv_list]
            fit_ind = 9 # sets minimum fitting index to be at N_deriv=100 (such that sufficiently number of samples so that central limit accurately applies in fitting formula)
            ax[i][0].plot(ncov_list, np.array(param_comb_data_cov)/param_comb_data_cov[-1], '-o', label=param_list_labels[j], color=param_colors[j], markersize=2)
            ax[i][1].plot(nderiv_list, np.array(param_comb_data_deriv)/param_comb_data_deriv[-1], '-o', label=param_list_labels[j], color=param_colors[j], markersize=2)
            best_fit_param = curve_fit(f=derivative_convergence_fit, xdata=nderiv_list[fit_ind:], ydata=(np.array(param_comb_data_deriv)/param_comb_data_deriv[-1])[fit_ind:], p0=[0.1])[0][0]
            print(comb, param_list[j], 'corrected: ', round(param_comb_data_deriv[-1]*((1+best_fit_param)**0.5), 4))
            raw_constraints[j][i] = param_comb_data_deriv[-1]
            corrected_constraints[j][i] = param_comb_data_deriv[-1]*((1+best_fit_param)**0.5)
    ax[0][0].set_xticks([])
    ax[1][0].set_xticks([])
    ax[0][1].set_xticks([])
    ax[1][1].set_xticks([])
    ax[2][0].set_xlabel(r'$N_{cov}$')
    ax[2][1].set_xlabel(r'$N_{deriv}$')
    ax[0][0].set_ylabel(r'$\sigma(N_{cov})/\sigma(N_{cov}=15000)$')
    ax[1][0].set_ylabel(r'$\sigma(N_{cov})/\sigma(N_{cov}=15000)$')
    ax[2][0].set_ylabel(r'$\sigma(N_{cov})/\sigma(N_{cov}=15000)$')
    ax[0][1].set_ylabel(r'$\sigma(N_{deriv})/\sigma(N_{deriv}=500)$')
    ax[1][1].set_ylabel(r'$\sigma(N_{deriv})/\sigma(N_{deriv}=500)$')
    ax[2][1].set_ylabel(r'$\sigma(N_{deriv})/\sigma(N_{deriv}=500)$')
    ax[0][1].axhline(y=1, linestyle='--', linewidth=0.5, color='grey')
    ax[1][1].axhline(y=1, linestyle='--', linewidth=0.5, color='grey')
    ax[2][1].axhline(y=1, linestyle='--', linewidth=0.5, color='grey')
    ax[0][0].set_title('Halo')
    ax[1][0].set_title('DSC')
    ax[2][0].set_title('Joint')
    ax[0][1].set_title('Halo')
    ax[1][1].set_title('DSC')
    ax[2][1].set_title('Joint')
    ax[0][0].legend()
    ax[0][0].set_ylim(0.9, 1.1)
    ax[1][0].set_ylim(0.9, 1.1)
    ax[2][0].set_ylim(0.9, 1.1)
    ax[0][1].set_ylim(0, 1.05)
    ax[1][1].set_ylim(0, 1.05)
    ax[2][1].set_ylim(0, 1.05)
    np.save(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/constraints_raw_{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}_including_all_auto.npy', raw_constraints)
    np.save(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/constraints_corrected_{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}_including_all_auto.npy', corrected_constraints) 
    fig.savefig(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/covariance_derivative_convergence_{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}_including_all_auto.png')