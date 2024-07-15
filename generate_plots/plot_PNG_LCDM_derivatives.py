import os
import numpy as np
import matplotlib.pyplot as plt


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

    # Makes plot of the numerical derivatives (in units of fiducial noise) for the DS quantiles for fNL of local type

    # Reads in the fiducial data
    data_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
    kstep = 1
    fiducial_data_lattice = np.load(os.path.join(data_folder, 'power_fiducial_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'), allow_pickle=True).item()
    k_avg = fiducial_data_lattice['k_avg']
    k_cut = k_avg < 0.5
    k_plot = k_avg[k_cut]
    linewidth = 1
    n_derivative = 500
    variations = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    for variation in variations:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6,6), dpi=400)
        fig.subplots_adjust(hspace=0.13, wspace=0.15, left=0.14, right=0.95, bottom=0.09, top=0.95)
        if variation == 'LC' or variation == 'EQ' or variation == 'OR_LSS':
            low_up_diff = 200
        if variation == 'h':
            low_up_diff = 0.04
        if variation == 'Om':
            low_up_diff = 0.02
        if variation == 'ns':
            low_up_diff = 0.04
        if variation == 's8':
            low_up_diff = 0.03
        # Reads in the data for each parameter variation
        lower_data_lattice = np.load(os.path.join(data_folder, 'power_LC_m_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'), allow_pickle=True).item()
        upper_data_lattice = np.load(os.path.join(data_folder, 'power_LC_p_Gaussian_10_5_512_lattice_None_0_32000000000000.0_zsplit_tsc_0_True_1kF.npy'), allow_pickle=True).item()
        for q in [1,2,3,4,5]:
            if q == 1: color, label = 'mediumblue', 'DS1'
            elif q == 2: color, label = 'cornflowerblue', 'DS2'
            elif q == 3: color, label = 'grey', 'DS3'
            elif q == 4: color, label = 'lightsalmon', 'DS4'
            elif q == 5: color, label = 'firebrick', 'DS5'
            auto_monopole_deriv = derivative_per_noise(lower_data_lattice, upper_data_lattice, fiducial_data_lattice, n_derivative, f'{q}-{q}(0)', k_cut, low_up_diff)
            cross_monopole_deriv = derivative_per_noise(lower_data_lattice, upper_data_lattice, fiducial_data_lattice, n_derivative, f'{q}-h(0)', k_cut, low_up_diff)
            ax[0].plot(k_plot, auto_monopole_deriv, '-', label=f'DS{q}', color=color, linewidth=linewidth)
            ax[1].plot(k_plot, cross_monopole_deriv, '-', label=f'DS{q}', color=color, linewidth=linewidth)
            if q==5:
                auto_monopole_deriv = derivative_per_noise(lower_data_lattice, upper_data_lattice, fiducial_data_lattice, n_derivative, f'h-h(0)', k_cut, low_up_diff)
                ax[0].plot(k_plot, auto_monopole_deriv, '-', label='Halo', color='black', linewidth=linewidth)
                ax[1].plot(k_plot, auto_monopole_deriv, '-', label='Halo', color='black', linewidth=linewidth)
        ax[0].legend(loc=1)
        ax[1].set_xlabel(r'$k \ [hMpc^{-1}]$')
        ax[0].set_ylabel(r'$\partial P \ / \ \partial f_{NL}^{loc} \ / \ \sigma_P$')
        ax[1].set_ylabel(r'$\partial P \ / \ \partial f_{NL}^{loc} \ / \ \sigma_P$')
        ax[1].set_xscale('log')
        ax[0].ticklabel_format(scilimits=(0,0), axis='y'); ax[1].ticklabel_format(scilimits=(0,0), axis='y')
        ax[0].set_title('Quantile Auto Monopole')
        ax[1].set_title('Quantile-Halo Cross Monopole')
        fig.savefig(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/{variation}_numerical_derivative.png')