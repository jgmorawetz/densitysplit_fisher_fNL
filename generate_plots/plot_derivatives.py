import os
import numpy as np
import matplotlib.pyplot as plt


def average_realizations(realizations, n_realizations, func, k_cut, fid_or_not):
    """Computes the average of the available realizations.

    Args:
        realizations (dict): Dictionary containin the realizations for the given variation.
        n_realizations (int): The number of realizations to use for derivatives (where a single realization
          includes all three LOS directions for a single simulation).
        func (str): The given qunatile fucntion string.
        k_cut (array); the array to apply the cut on the wavenumbers to include.
    
    Returns:
        array: The mean function.
    """
    if not fid_or_not:
        func_x = realizations['x'][func]
        func_y = realizations['y'][func]
        func_z = realizations['z'][func]
        shape = np.shape(func_x)
        func_all = np.zeros((3*shape[0], shape[1]))
        func_all[0::3, :] = func_x
        func_all[1::3, :] = func_y
        func_all[2::3, :] = func_z
        func_mean = np.mean(func_all[:3*n_realizations, k_cut], axis=0)
    else:
        func_mean = np.mean((realizations['z'][func])[:n_realizations, k_cut], axis=0)
    return func_mean

def derivative_per_noise(low_realizations, high_realizations, fid_realizations, n_realizations, func, k_cut, low_up_diff):
    """Computes the mean of the derivative vector in units of the fiducial noise.

    Args:
        low_realizations (dict): Dictionary containing the realizations for the parameter low variation.
        high_realizations (dict): Dictionary containing the realizations for the parameter high variation.
        fid_realizations (dict): Dictionary containing the realizations for the fiducial variation.
        n_realizations (int): The number of realizations to use for derivatives (where a single realization
                              includes all three LOS directions for a single simulation).
        func (str): The given quantile function string.
        k_cut (array): The array to apply the cut on the wavenumbers to include.
        low_high_diff (float): The change in parameter value between the low and high variations.

    Returns:
        array: The mean derivative function.
    """
    # Stacks the low and high realizations into arrays of realizations
    func_x_low = low_realizations['x'][func]
    func_y_low = low_realizations['y'][func]
    func_z_low = low_realizations['z'][func]
    func_x_high = high_realizations['x'][func]
    func_y_high = high_realizations['y'][func]
    func_z_high = high_realizations['z'][func]
    shape = np.shape(func_x_low)
    func_low = np.zeros((3*shape[0], shape[1]))
    func_low[0::3, :] = func_x_low
    func_low[1::3, :] = func_y_low
    func_low[2::3, :] = func_z_low
    func_low_mean = np.mean(func_low[:3*n_realizations, k_cut], axis=0)
    func_high = np.zeros((3*shape[0], shape[1]))
    func_high[0::3, :] = func_x_high
    func_high[1::3, :] = func_y_high
    func_high[2::3, :] = func_z_high
    func_high_mean = np.mean(func_high[:3*n_realizations, k_cut], axis=0)
    func_diff_mean = func_high_mean-func_low_mean
    deriv_mean = func_diff_mean/low_up_diff
    fiducial_noise = np.std(fid_realizations['z'][func][:, k_cut], axis=0)
    deriv_mean /= fiducial_noise
    return deriv_mean


if __name__ == '__main__':

    # Reads in the data
    filter_type = 'Gaussian'
    nmesh = 512
    redshift = 0
    split = 'zsplit'
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    rebin_factor = 1
    hyperparameters = (5, 10, None, 'lattice')
    n_realizations = 500
    n_quantiles, filter_radius, n_randoms, query_type = hyperparameters
    save_path = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/derivatives'
    start_path = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
    k_avg = np.load(os.path.join(start_path, 'power_fiducial_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'),
                    allow_pickle=True).item()['k_avg']
    k_cut = k_avg < 0.5
    k_plot = k_avg[k_cut]
    param_names = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    param_labels = [r'$f_{\rm{NL}}^{\rm{loc}}$', r'$f_{\rm{NL}}^{\rm{equil}}$', r'$f_{\rm{NL}}^{\rm{ortho}}$', r'$M_{\rm{min}}$', r'$h$', r'$n_{\rm{s}}$', r'$\Omega_{\rm{m}}$', r'$\sigma_8$']
    param_diffs = [200, 200, 200, 2e12, 0.04, 0.04, 0.02, 0.03]
    fiducial_path = os.path.join(start_path,
        f'power_fiducial_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_32000000000000.0_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy')
    low_paths = list(map(lambda x: os.path.join(start_path,
        f'power_{x}_m_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_32000000000000.0_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy')
        if x != 'Mmin' else os.path.join(start_path,
        f'power_Mmin_3.1e13_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_31000000000000.0_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy'), param_names))
    high_paths = list(map(lambda x: os.path.join(start_path,
        f'power_{x}_p_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_32000000000000.0_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy')
        if x != 'Mmin' else os.path.join(start_path,
        f'power_Mmin_3.3e13_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_33000000000000.0_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy'), param_names))
    colors = ['mediumblue', 'cornflowerblue', 'grey', 'lightsalmon', 'firebrick']
    # Generates figure with the derivatives
    fig, ax = plt.subplots(len(param_names)//2, 2, sharex=True, sharey=False, figsize=(7,9), dpi=400)
    # Iterates through each parameter and plots results for each
    ax_indices = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
    for i in range(len(param_names)):
        fiducial_realizations = np.load(fiducial_path, allow_pickle=True).item()
        low_realizations = np.load(low_paths[i], allow_pickle=True).item()
        high_realizations = np.load(high_paths[i], allow_pickle=True).item()
        for j in range(n_quantiles):
            derivative_cross = derivative_per_noise(low_realizations, high_realizations, fiducial_realizations, n_realizations, f'{j+1}-h(0)', k_cut, param_diffs[i])
            ax[ax_indices[i][0]][ax_indices[i][1]].plot(k_plot, derivative_cross, '-', color=colors[j], label=f'Q{j+1}', linewidth=1)
        derivative_halo = derivative_per_noise(low_realizations, high_realizations, fiducial_realizations, n_realizations, f'h-h(0)', k_cut, param_diffs[i])
        ax[ax_indices[i][0]][ax_indices[i][1]].plot(k_plot, derivative_halo, '-', color='black', label=f'Halo', linewidth=1)
        ax[ax_indices[i][0]][ax_indices[i][1]].ticklabel_format(scilimits=(0,0), axis='y')
        ax[ax_indices[i][0]][ax_indices[i][1]].set_title(param_labels[i])
    ax[0][0].legend(loc=1)
    ax[-1][0].set_xlabel(r'$k \ [h^{-1}\rm{Mpc}]$')
    ax[-1][1].set_xlabel(r'$k \ [h^{-1}\rm{Mpc}]$')
    ax[0][0].set_ylabel(r'$\partial P / \partial \theta / \sigma_P$')
    ax[1][0].set_ylabel(r'$\partial P / \partial \theta / \sigma_P$')
    ax[2][0].set_ylabel(r'$\partial P / \partial \theta / \sigma_P$')
    ax[3][0].set_ylabel(r'$\partial P / \partial \theta / \sigma_P$')
    ax[0][0].set_xscale('log')
    fig.suptitle('Quantile Cross Monopole')
    fig.subplots_adjust(bottom=0.07, top=0.92, left=0.08, right=0.98, hspace=0.23, wspace=0.16)
    fig.savefig(os.path.join(save_path, 'cross_derivatives.pdf'))