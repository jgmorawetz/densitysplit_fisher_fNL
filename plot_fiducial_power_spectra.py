import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Makes plot of the fiducial power spectra for the halos and the DS quantiles (the mean and standard noise for each)
    data_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
    # The hyperparameters for the given version of the power to plot
    variation = 'fiducial'
    filter_type = 'Gaussian'
    filter_radius = 10
    n_quantiles = 5
    nmesh = 512
    query_type = 'lattice'
    n_randoms = None
    redshift = 0
    mass_cut = 32000000000000.0
    split = 'zsplit'
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    rebin_factor = 1
    # Reads in the data
    fiducial_path = os.path.join(
        data_folder, f'power_{variation}_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_{mass_cut}_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy')
    fiducial_data = np.load(fiducial_path, allow_pickle=True).item()
    fiducial_power = fiducial_data['z']
    k_avg = fiducial_data['k_avg']
    # Creates plot
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=False, dpi=400, figsize=(8, 4.8))
    fig.subplots_adjust(hspace=0.15, wspace=0.1, left=0.07, right=0.98, top=0.95, bottom=0.1)
    # Halo power spectra
    halo_monopole, halo_quadropole = fiducial_power['h-h(0)'], fiducial_power['h-h(2)']
    halo_monopole_mean, halo_monopole_std = np.mean(halo_monopole, axis=0), np.std(halo_monopole, axis=0)
    halo_quadropole_mean, halo_quadropole_std = np.mean(halo_quadropole, axis=0), np.std(halo_quadropole, axis=0)
    ax[0][0].fill_between(x=k_avg, y1=k_avg*(halo_monopole_mean-halo_monopole_std),
                          y2=k_avg*(halo_monopole_mean+halo_monopole_std), alpha=0.2, color='black')
    ax[0][0].plot(k_avg, k_avg*halo_monopole_mean, '-', color='black', label='Halo')
    ax[1][0].fill_between(x=k_avg, y1=k_avg*(halo_quadropole_mean-halo_quadropole_std),
                          y2=k_avg*(halo_quadropole_mean+halo_quadropole_std), alpha=0.2, color='black')
    ax[1][0].plot(k_avg, k_avg*halo_quadropole_mean, '-', color='black')
    # Iterates through the quantile power spectra
    colors = ['mediumblue', 'cornflowerblue', 'grey', 'lightsalmon', 'firebrick'] # need to add/remove colors if using more/less than five quantiles
    for i in range(n_quantiles):
        color = colors[i]
        quantile_auto_monopole, quantile_auto_quadropole = fiducial_power[f'{i+1}-{i+1}(0)'], fiducial_power[f'{i+1}-{i+1}(2)']
        quantile_cross_monopole, quantile_cross_quadropole = fiducial_power[f'{i+1}-h(0)'], fiducial_power[f'{i+1}-h(2)']
        quantile_auto_monopole_mean, quantile_auto_monopole_std = np.mean(quantile_auto_monopole, axis=0), np.std(quantile_auto_monopole, axis=0)
        quantile_auto_quadropole_mean, quantile_auto_quadropole_std = np.mean(quantile_auto_quadropole, axis=0), np.std(quantile_auto_quadropole, axis=0)
        quantile_cross_monopole_mean, quantile_cross_monopole_std = np.mean(quantile_cross_monopole, axis=0), np.std(quantile_cross_monopole, axis=0)
        quantile_cross_quadropole_mean, quantile_cross_quadropole_std = np.mean(quantile_cross_quadropole, axis=0), np.std(quantile_cross_quadropole, axis=0)
        ax[0][0].fill_between(x=k_avg, y1=k_avg*(quantile_auto_monopole_mean-quantile_auto_monopole_std),
                              y2=k_avg*(quantile_auto_monopole_mean+quantile_auto_monopole_std), alpha=0.2, color=color)
        ax[0][0].plot(k_avg, k_avg*quantile_auto_monopole_mean, '-', color=color, label=f'DS{i+1}')
        ax[1][0].fill_between(x=k_avg, y1=k_avg*(quantile_auto_quadropole_mean-quantile_auto_quadropole_std),
                              y2=k_avg*(quantile_auto_quadropole_mean+quantile_auto_quadropole_std), alpha=0.2, color=color)
        ax[1][0].plot(k_avg, k_avg*quantile_auto_quadropole_mean, '-', color=color)
        ax[0][1].fill_between(x=k_avg, y1=k_avg*(quantile_cross_monopole_mean-quantile_cross_monopole_std),
                              y2=k_avg*(quantile_cross_monopole_mean+quantile_cross_monopole_std), alpha=0.2, color=color)
        ax[0][1].plot(k_avg, k_avg*quantile_cross_monopole_mean, '-', color=color)
        ax[1][1].fill_between(x=k_avg, y1=k_avg*(quantile_cross_quadropole_mean-quantile_cross_quadropole_std),
                              y2=k_avg*(quantile_cross_quadropole_mean+quantile_cross_quadropole_std), alpha=0.2, color=color)
        ax[1][1].plot(k_avg, k_avg*quantile_cross_quadropole_mean, '-', color=color)
    # Adjusts plot parameters
    ax[0][0].legend(); ax[1][0].set_xlabel(r'$k \ [hMpc^{-1}]$')
    ax[1][1].set_xlabel(r'$k \ [hMpc^{-1}]$')
    ax[0][0].set_ylabel(r'$k P(k) \ [h^{-2}Mpc^2]$')
    ax[1][0].set_ylabel(r'$k P(k) \ [h^{-2}Mpc^2]$')
    ax[0][0].set_title('Quantile Auto')
    ax[0][1].set_title('Quantile-Halo Cross')
    ax[0][0].set_xscale('log')
    ax[0][0].ticklabel_format(scilimits=(0,0), axis='y')
    ax[0][1].ticklabel_format(scilimits=(0,0), axis='y')
    ax[1][0].ticklabel_format(scilimits=(0,0), axis='y')
    ax[1][1].ticklabel_format(scilimits=(0,0), axis='y')
    fig.savefig('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fiducial_power_spectra.png')