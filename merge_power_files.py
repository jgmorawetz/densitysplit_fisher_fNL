import os
import numpy as np


# Merges the fiducial & LCDM/PNG Quijote power spectrum files together appropriately


if __name__ == '__main__':

    # Fixed hyperparameters across all separate versions of the power spectrum
    len_power = 254 # using k bins from 2*np.pi/boxsize to np.pi/(boxsize/nmesh) in steps of 2*np.pi/boxsize
    rebin_factor = 1 # the integer multiple to rebin from the original bin resolution
    len_power_new = len_power // rebin_factor
    filter_type = 'Gaussian'
    nmesh = 512
    redshift = 0
    mass_cut = 32000000000000.0
    split = 'zsplit'
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    # Iterates through the separate hyperparameter cases
    for hyperparameters in [(5, 10, None, 'lattice'), 
                            (3, 10, None, 'lattice'), 
                            (7, 10, None, 'lattice'),
                            (5, 7, None, 'lattice'),
                            (5, 13, None, 'lattice'),
                            (5, 10, 5, 'random')]:
        n_quantiles, filter_radius, n_randoms, query_type = hyperparameters
        ds_funcs = [] # functions include halo power and quantile auto/cross power
        ds_funcs.append('h-h')
        for i in range(n_quantiles):
            ds_funcs.append(f'{i+1}-{i+1}')
            ds_funcs.append(f'{i+1}-h')
        # Iterates through each parameter variation and saves separate file for each
        for variation in ['fiducial', 'LC_m', 'LC_p', 'EQ_m', 'EQ_p', 'OR_LSS_m', 'OR_LSS_p',
                          'Mmin_3.1e13', 'Mmin_3.3e13', 'h_m', 'h_p', 'ns_m', 'ns_p',
                          'Om_m', 'Om_p', 's8_m', 's8_p']:
            # Folder path to retrieve the generated data
            path_load = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/{variation}'
            path_save = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
            if variation == 'fiducial':
                N_sims = 15000
                los_dirs = ['z']
                results = {'z':{}}
            else:
                N_sims = 500
                los_dirs = ['x', 'y', 'z']
                results = {'x':{}, 'y':{}, 'z':{}}
            for ds_func in ds_funcs:
                for los_dir in los_dirs:
                    # Creates empty array to store monopole and quadropole realizations for each function
                    results[los_dir][ds_func + '(0)'] = np.zeros((N_sims, len_power_new))
                    results[los_dir][ds_func + '(2)'] = np.zeros((N_sims, len_power_new))
            # Iterates through all simulation indices and adds results for each
            for i in range(N_sims):
                data = np.load(os.path.join(
                    path_load, f'phase{i}_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_{mass_cut}_{split}_{resampler}_{interlacing}_{compensate}.npy'),
                    allow_pickle=True).item()
                for ds_func in ds_funcs:
                    for los_dir in los_dirs:
                        results[los_dir][ds_func + '(0)'][i] = data[los_dir][ds_func].poles[:len_power_new*rebin_factor:rebin_factor].power.real[0]
                        results[los_dir][ds_func + '(2)'][i] = data[los_dir][ds_func].poles[:len_power_new*rebin_factor:rebin_factor].power.real[1]
                if i % 100 == 0: print(i) # prints out progress to track runtime
            results['k_avg'] = data[los_dir][ds_func].poles[:len_power_new*rebin_factor:rebin_factor].kavg # stores the mean k value in each bin
            np.save(os.path.join(
                path_save, f'power_{variation}_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_{mass_cut}_{split}_{resampler}_{interlacing}_{compensate}_{rebin_factor}kF.npy'),
                results)
            print(hyperparameters, variation, 'DONE')