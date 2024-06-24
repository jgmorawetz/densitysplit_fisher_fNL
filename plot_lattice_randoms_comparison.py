import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Makes plot of the fiducial power spectra for both the lattice and x5 random query positions
    data_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
    q = 5 # the quantile index
    l = 0 # the multipole order
    lattice_data = np.load(os.path.join(data_folder, 'power_fiducial_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'),
                           allow_pickle=True).item()
    random_data = np.load(os.path.join(data_folder, 'power_fiducial_Gaussian_10_5_512_random_5_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'),
                           allow_pickle=True).item()
    k_avg = lattice_data['k_avg']
    k_cut = k_avg <= 0.5
    k_avg = k_avg[k_cut]
    lattice_mean, lattice_std = np.mean(lattice_data['z'][f'{q}-{q}({l})'][:, k_cut], axis=0), np.std(lattice_data['z'][f'{q}-{q}({l})'][:, k_cut], axis=0)
    random_mean, random_std = np.mean(random_data['z'][f'{q}-{q}({l})'][:, k_cut], axis=0), np.std(random_data['z'][f'{q}-{q}({l})'][:, k_cut], axis=0)

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, dpi=400, figsize=(6, 5), height_ratios=[2, 1, 2])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.1, hspace=0.15, wspace=0.1)
    ax[0].plot(k_avg, k_avg**2*random_mean, '-', color='green', linewidth=1, label='Random')
    ax[0].fill_between(x=k_avg, y1=k_avg**2*(random_mean-random_std), y2=k_avg**2*(random_mean+random_std), alpha=0.2, color='green')
    ax[0].plot(k_avg, k_avg**2*lattice_mean, '-', color='red', linewidth=1, label='Lattice')
    ax[0].fill_between(x=k_avg, y1=k_avg**2*(lattice_mean-lattice_std), y2=k_avg**2*(lattice_mean+lattice_std), alpha=0.2, color='red')
    ax[1].plot(k_avg, lattice_mean/random_mean, '-', color='black', linewidth=1)
    ax[2].plot(k_avg, lattice_std/random_std, '-', color='black', linewidth=1)
    ax[1].set_ylim(0.98, 1.02)
    ax[1].axhline(y=1, linestyle='--', linewidth=0.2, color='black')
    ax[2].set_xlabel(r'$k \ [hMpc^{-1}]$')
    ax[0].set_ylabel(r'$k^2 P_0(k) \ [h^{-1}Mpc]$')
    ax[1].set_ylabel(r'$\left<P_{lattice}\right>/\left<P_{random}\right>$')
    ax[2].set_ylabel(r'$\sigma_{lattice}/\sigma_{random}$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(loc=1)
    fig.savefig(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/lattice_randoms_fiducial_power_DS{q}_l={l}.png')