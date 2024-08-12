import os
import numpy as np


# Merges the fiducial & PNG local Quijote power spectrum files together appropriately


if __name__ == '__main__':

    # Fixed hyperparameters across all separate versions of the power spectrum
    len_power = 253 # using k bins from 2*np.pi/boxsize to np.pi/(boxsize/nmesh) in steps of 2*np.pi/boxsize
    rebin_factor = 1 # the integer multiple to rebin from the original bin resolution
    len_power_new = len_power // rebin_factor
    filter_type = 'Gaussian'
    nmesh = 512
    redshift = 0
    split = 'rsplit'
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    # Iterates through the separate hyperparameter cases
    for hyperparameter in [(False, 5, None, 10, 'lattice', 3.2e13), # baseline DSC settings
                           (True, None, [0.365, 1.305, 2.605, 4.995], 5, 'halo', 1e13)]: # modifications (thresholds manually tested by averaging several realizations of overdensity distribution code)
        mass_weighted, n_quantiles, overdensity_thresholds, filter_radius, query_type, min_mass = hyperparameter
        ds_funcs = [] # includes the matter power spectrum, halo and quantile-matter cross power spectrum and halo and quantile auto power spectrum
        ds_funcs.append('m-m')
        ds_funcs.append('h-m')
        ds_funcs.append('h-h')
        if n_quantiles == None:
            N_QUANTILES = len(overdensity_thresholds)+1
        else:
            N_QUANTILES = n_quantiles
        for i in range(N_QUANTILES):
            ds_funcs.append(f'{i+1}-m')
            ds_funcs.append(f'{i+1}-{i+1}')
        # Iterates through each parameter variation and saves separate file for each
        for variation in ['fiducial', 'LC_m', 'LC_p']:
            # Folder path to retrieve the generated data
            path_load = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/{variation}'
            path_save = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/MERGED_DATA'
            if variation == 'fiducial':
                N_sims = 500
                results = {}
            else:
                N_sims = 500
                results = {}
            mass_cut = min_mass
            for ds_func in ds_funcs:
                # Creates empty array to store monopole realizations for each function
                results[ds_func] = np.zeros((N_sims, len_power_new))
                results[ds_func] = np.zeros((N_sims, len_power_new))
            # Iterates through all simulation indices and adds results for each
            for i in range(N_sims):
                data = np.load(os.path.join(
                    path_load, f'sim{i}_{mass_weighted}_{n_quantiles}_{filter_radius}_{query_type}_{min_mass}.npy'),
                    allow_pickle=True).item()
                for ds_func in ds_funcs:
                    if ds_func == 'm-m':
                        results[ds_func][i] = data['Matter'].power['power'].real
                    elif ds_func == 'h-m':
                        results[ds_func][i] = data['Halo-Matter'].power['power'].real
                    elif ds_func == 'h-h':
                        results[ds_func][i] = data['Halo'].power['power'].real
                    elif 'm' in ds_func:
                        results[ds_func][i] = data['Quantile-Matter'][int(ds_func.split('-')[0])-1].power['power'].real
                    else:
                        results[ds_func][i] = data['Quantile'][int(ds_func.split('-')[0])-1].power['power'].real
                if i % 100 == 0: print('sim', i, 'DONE') # prints out progress to track runtime
            results['k_avg'] = data['Matter'].power['k']
            np.save(os.path.join(
                path_save, f'power_{variation}_{mass_weighted}_{n_quantiles}_{filter_radius}_{query_type}_{min_mass}.npy'),
                results)
            print(hyperparameter, variation, 'DONE')

