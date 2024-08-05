import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Reads in the produced constraint data
    data_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results'
    raw_lattice_5_10 = np.load(os.path.join(data_folder, 'constraints_raw_lattice_None_5_10_including_all_auto.npy'))
    raw_random_5_10 = np.load(os.path.join(data_folder, 'constraints_raw_random_5_5_10_including_all_auto.npy'))
    raw_lattice_3_10 = np.load(os.path.join(data_folder, 'constraints_raw_lattice_None_3_10_including_all_auto.npy'))
    raw_lattice_7_10 = np.load(os.path.join(data_folder, 'constraints_raw_lattice_None_7_10_including_all_auto.npy'))
    raw_lattice_5_7 = np.load(os.path.join(data_folder, 'constraints_raw_lattice_None_5_7_including_all_auto.npy'))
    raw_lattice_5_13 = np.load(os.path.join(data_folder, 'constraints_raw_lattice_None_5_13_including_all_auto.npy'))
    corrected_lattice_5_10 = np.load(os.path.join(data_folder, 'constraints_corrected_lattice_None_5_10_including_all_auto.npy'))
    corrected_random_5_10 = np.load(os.path.join(data_folder, 'constraints_corrected_random_5_5_10_including_all_auto.npy'))
    corrected_lattice_3_10 = np.load(os.path.join(data_folder, 'constraints_corrected_lattice_None_3_10_including_all_auto.npy'))
    corrected_lattice_7_10 = np.load(os.path.join(data_folder, 'constraints_corrected_lattice_None_7_10_including_all_auto.npy'))
    corrected_lattice_5_7 = np.load(os.path.join(data_folder, 'constraints_corrected_lattice_None_5_7_including_all_auto.npy'))
    corrected_lattice_5_13 = np.load(os.path.join(data_folder, 'constraints_corrected_lattice_None_5_13_including_all_auto.npy'))

    # Makes a plot of the relative (corrected and raw) improvement between the joint halo/DSC power spectra vs halo 
    # power spectrum for each of the parameters
    
    x_tick_labels = [r'$f_{NL}^{loc}$', r'$f_{NL}^{equil}$', r'$f_{NL}^{ortho}$', r'$M_{min}$', r'$h$', r'$n_s$', r'$\Omega_m$', r'$\sigma_8$']

    fig, ax = plt.subplots(dpi=400)
    markersize=5
    comb_colors = ['tab:blue', 'tab:green']
    comb_labels = [r'$\sigma_{Halo}/\sigma_{DSC}$', r'$\sigma_{Halo}/\sigma_{Joint}$']
    for i in range(1,3): # three function combinations
        x_vals = np.arange(8)
        # normalizes constraints to the halo power spectrum constraints
        ax.plot(np.array(x_vals)+0.1, corrected_lattice_5_10[:, 0]/corrected_lattice_5_10[:, i], '^', 
                   color=comb_colors[i-1], label=comb_labels[i-1], markersize=markersize)
    print('(Raw) halo-only constraints:', list(np.round(raw_lattice_5_10[:, 0], 6)))
    print('(Corrected) halo-only constraints:', list(np.round(corrected_lattice_5_10[:, 0], 6)))
    print('(Raw) relative improvements:', list(np.round(raw_lattice_5_10[:, 0]/raw_lattice_5_10[:, 2], 6)))
    print('(Corrected) relative improvements:', list(np.round(corrected_lattice_5_10[:, 0]/corrected_lattice_5_10[:, 2], 6)))
    for i in range(1, 3):
        ax.plot(np.array(x_vals)-0.1, raw_lattice_5_10[:, 0]/raw_lattice_5_10[:, i], 'X', 
                   color=comb_colors[i-1], label=comb_labels[i-1]+' (raw)', markersize=markersize)
    ax.set_yscale('log')
    ax.set_yticks([0.5, 1,2,3,4,5,6,7,8,9,10])
    ax.set_yticklabels([0.5, 1,2,3,4,5,6,7,8,9,10])
    for i in np.arange(0.5, 10, 0.5):
        if i == 1:
            ax.axhline(y=i, linestyle='--', color='gray', linewidth=1)
        else:
            ax.axhline(y=i, linestyle='--', color='gray', linewidth=0.1)
    ax.set_ylim(0.5, 10)
    ax.set_xticks([0,1,2,3,4,5,6,7])
    ax.set_xticklabels(x_tick_labels)
    ax.legend(loc=1)
    fig.savefig(os.path.join(data_folder, 'raw_corrected_relative_improvement_constraints_including_all_auto.png'))

    # Makes a plot of the relative improvement (corrected) between the joint halo/DSC power spectra vs halo
    # power spectrum for each the parameters, depending on the hyperparameter choice

    fig2, ax2 = plt.subplots(dpi=400)
    x_vals = np.arange(8)
    ax2.plot(np.array(x_vals), corrected_lattice_3_10[:, 0]/corrected_lattice_3_10[:, 2], '^', label='Lattice, ' + r'$N_{quantile}=$' + '3, ' + r'$R_s=$' + '10', markersize=markersize, color='tab:blue')
    ax2.plot(np.array(x_vals), corrected_lattice_7_10[:, 0]/corrected_lattice_7_10[:, 2], '^', label='Lattice, ' + r'$N_{quantile}=$' + '7, ' + r'$R_s=$' + '10', markersize=markersize, color='tab:orange')
    ax2.plot(np.array(x_vals), corrected_lattice_5_7[:, 0]/corrected_lattice_5_7[:, 2], '^', label='Lattice, ' + r'$N_{quantile}=$' + '5, ' + r'$R_s=$' + '7', markersize=markersize, color='tab:green')
    ax2.plot(np.array(x_vals), corrected_lattice_5_13[:, 0]/corrected_lattice_5_13[:, 2], '^', label='Lattice, ' + r'$N_{quantile}=$' + '5, ' + r'$R_s=$' + '13', markersize=markersize, color='tab:red')
    ax2.plot(np.array(x_vals), corrected_random_5_10[:, 0]/corrected_random_5_10[:, 2], '^', label='Random, ' + r'$N_{quantile}=$' + '5, ' + r'$R_s=$' + '10', markersize=markersize, color='tab:purple')
    ax2.plot(np.array(x_vals), corrected_lattice_5_10[:, 0]/corrected_lattice_5_10[:, 2], '+', label='Lattice, ' + r'$N_{quantile}=$' + '5, ' + r'$R_s=$' + '10', markersize=1.5*markersize, color='black')
    ax2.legend(loc=1)
    ax2.set_ylim(1, 10.01)
    ax2.set_yscale('log')
    ax2.axhline(y=1, linestyle='--', color='gray', linewidth=0.5)
    for i in np.arange(1.5, 10, 0.5):
        if i == 1:
            ax2.axhline(y=i, linestyle='--', color='gray', linewidth=1)
        else:
            ax2.axhline(y=i, linestyle='--', color='gray', linewidth=0.1)
    ax2.set_xticks([0,1,2,3,4,5,6,7])
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_yticks([1,2,3,4,5,6,7,8,9,10])
    ax2.set_yticklabels([1,2,3,4,5,6,7,8,9,10])
    ax2.set_ylabel(comb_labels[1])
    fig2.savefig(os.path.join(data_folder, 'relative_improvement_hyperparameters_including_all_auto.png'))