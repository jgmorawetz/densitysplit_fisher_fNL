import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
      
    # Plots constraints as a function of maximum wavenumber (both marginalized and unmarginalized
    # for halo only, DS only or joint)
    fig, ax = plt.subplots(4, 2, sharex=True, sharey=False, dpi=400, figsize=(6,9))
    fig.subplots_adjust(hspace=0.18, wspace=0.21, left=0.13, right=0.97, top=0.97, bottom=0.06)
    # Specific hyperparameters
    k_avg = np.load('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA/power_fiducial_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy', allow_pickle=True).item()['k_avg']
    k_upper = 0.5
    kmax_vals = k_avg[(k_avg >= k_avg[2]) & (k_avg <= k_upper)][::4]
    query_type, n_randoms, n_quantiles, filter_radius = 'lattice', None, 5, 10
    ncov, nderiv = 15000, 500
    data_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}'
    param_list = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    param_list_labels = [r'$f_{NL}^{loc}$', r'$f_{NL}^{equil}$', r'$f_{NL}^{ortho}$', r'$M_{min}$', r'$h$', r'$n_s$', r'$\Omega_m$', r'$\sigma_8$']
    combs = ['Halo', 'DS', 'Joint']
    comb_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(3):
        comb = combs[i]
        if comb == 'Halo':
            comb_label = 'Halo'
        elif comb == 'DS':
            comb_label = 'DSC'
        elif comb == 'Joint':
            comb_label = 'Halo+DSC'
        for j in range(len(param_list)):
            if j == 0: row, col = 0, 0
            elif j == 1: row, col = 0, 1
            elif j == 2: row, col = 1, 0
            elif j == 3: row, col = 1, 1
            elif j == 4: row, col = 2, 0
            elif j == 5: row, col = 2, 1
            elif j == 6: row, col = 3, 0
            elif j == 7: row, col = 3, 1
            param = param_list[j]
            marginalized_data = np.load(os.path.join(data_folder, f'marginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv}.npy'))
            unmarginalized_data = np.load(os.path.join(data_folder, f'unmarginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv}.npy'))
            param_marginalized = [np.sqrt(marginalized_data[kmax_ind][j][j]) for kmax_ind in range(len(kmax_vals))]
            param_unmarginalized = [np.sqrt(unmarginalized_data[kmax_ind][j][j]) for kmax_ind in range(len(kmax_vals))]
            min_ind = 2
            ax[row][col].plot(kmax_vals[min_ind:], param_marginalized[min_ind:], '-', color=comb_colors[i], label=comb_label)
            ax[row][col].plot(kmax_vals[min_ind:], param_unmarginalized[min_ind:], '--', color=comb_colors[i])
            ax[row][col].set_title(param_list_labels[j])
    ax[0][0].set_yscale('log'); ax[0][1].set_yscale('log')
    ax[1][0].set_yscale('log'); ax[1][1].set_yscale('log')
    ax[2][0].set_yscale('log'); ax[2][1].set_yscale('log')
    ax[3][0].set_yscale('log'); ax[3][1].set_yscale('log')
    ax[3][0].set_xlabel(r'$k_{max} \ [hMpc^{-1}]$')
    ax[3][1].set_xlabel(r'$k_{max} \ [hMpc^{-1}]$')
    ax[0][0].set_ylabel(r'$\sigma$')
    ax[1][0].set_ylabel(r'$\sigma$')
    ax[2][0].set_ylabel(r'$\sigma$')
    ax[3][0].set_ylabel(r'$\sigma$')
    fig.savefig('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/constraints_max_wavenumber.png')