import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
      
    # Plots constraints as a function of maximum wavenumber (both marginalized and unmarginalized
    # for halo only, DS only or joint)
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=False, dpi=500, figsize=(9,6))
    fig.subplots_adjust(hspace=0.13, wspace=0.3, left=0.15, right=0.97, top=0.95, bottom=0.12)
    k_avg = np.load('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA/power_fiducial_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy', allow_pickle=True).item()['k_avg']
    k_upper = 0.5
    kmax_vals = k_avg[(k_avg >= k_avg[2]) & (k_avg <= k_upper)][::4]
    query_type, n_randoms, n_quantiles, filter_radius = 'lattice', None, 5, 10
    ncov, nderiv = 15000, 500
    data_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}'
    param_list = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    param_list_labels = [r'$f_{\mathrm{NL}}^{\mathrm{loc}}$', r'$f_{\mathrm{NL}}^{\mathrm{equil}}$', r'$f_{\mathrm{NL}}^{\mathrm{ortho}}$', r'$M_{\mathrm{min}}$', r'$h$', r'$n_{\mathrm{s}}$', r'$\Omega_{\mathrm{m}}$', r'$\sigma_8$']
    combs = ['Halo', 'DS', 'Joint']
    comb_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(3):
        comb = combs[i]
        if comb == 'Halo':
            comb_label = 'Halo'
        elif comb == 'DS':
            comb_label = 'DSC'
        elif comb == 'Joint':
            comb_label = 'Joint'
        for j in range(len(param_list)):
            if j == 0: row, col = 0, 0
            elif j == 1: row, col = 0, 1
            elif j == 2: row, col = 0, 2
            elif j == 3: row, col = 0, 3
            elif j == 4: row, col = 1, 0
            elif j == 5: row, col = 1, 1
            elif j == 6: row, col = 1, 2
            elif j == 7: row, col = 1, 3
            param = param_list[j]
            marginalized_data = np.load(os.path.join(data_folder, f'marginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv}.npy'))
            unmarginalized_data = np.load(os.path.join(data_folder, f'unmarginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv}.npy'))
            param_marginalized = [np.sqrt(marginalized_data[kmax_ind][j][j]) for kmax_ind in range(len(kmax_vals))]
            param_unmarginalized = [np.sqrt(unmarginalized_data[kmax_ind][j][j]) for kmax_ind in range(len(kmax_vals))]
            min_ind = 2
            ax[row][col].plot(kmax_vals[min_ind:], param_marginalized[min_ind:], '-', color=comb_colors[i], label=comb_label)
            ax[row][col].plot(kmax_vals[min_ind:], param_unmarginalized[min_ind:], '--', color=comb_colors[i])
            if j == 0 and i in [0, 2]:
                print(kmax_vals[min_ind:], param_unmarginalized[min_ind:])
            ax[row][col].set_title(param_list_labels[j])
    ax[0][0].legend()
    ax[0][0].set_yscale('log'); ax[1][0].set_yscale('log')
    ax[0][1].set_yscale('log'); ax[1][1].set_yscale('log')
    ax[0][2].set_yscale('log'); ax[1][2].set_yscale('log')
    ax[0][3].set_yscale('log'); ax[1][3].set_yscale('log')
    ax[1][0].set_xlabel(r'$k_{\mathrm{max}} \ [h\mathrm{Mpc}^{-1}]$')
    ax[1][1].set_xlabel(r'$k_{\mathrm{max}} \ [h\mathrm{Mpc}^{-1}]$')
    ax[1][2].set_xlabel(r'$k_{\mathrm{max}} \ [h\mathrm{Mpc}^{-1}]$')
    ax[1][3].set_xlabel(r'$k_{\mathrm{max}} \ [h\mathrm{Mpc}^{-1}]$')
    ax[0][0].set_ylabel(r'$\sigma$')
    ax[1][0].set_ylabel(r'$\sigma$')
    fig.savefig('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/constraints_max_wavenumber_including_all_auto.pdf')