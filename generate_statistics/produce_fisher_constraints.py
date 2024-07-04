import os
import pickle
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt


def inverse_covariance_matrix(realizations, n_realizations, functions, k_cut, n_param):
    """Computes the inverse covariance matrix of the summary statistic.

    Args:
        realizations (dict): Dictionary containing the realizations for each function and
                             line-of-sight (LOS) direction. The first layer is the LOS 
                             direction while the second layer are the functions.
        n_realizations (int): The number of realizations to use for covariance matrix.
        functions (list): List of the strings denoting the functions to include.
        k_cut (array): The array to apply the cut on the wavenumbers to include.
        n_param (int): The number of parameters being marginalized over in Fisher matrix.

    Returns:
        array: The inverse covariance matrix.
    """
    # Stacks all the functions together into a single array of realizations
    all_functions = []
    for func in functions: # only one LOS direction for covariance matrix
        all_functions.append(realizations['z'][func][:n_realizations, k_cut].T)
    all_functions = np.concatenate(all_functions).T
    n_sims, n_bins = np.shape(all_functions)
    # Applies Percival correction factor to debias inverse matrix
    cov_mat = np.cov(all_functions.T)*((n_sims-1)/(n_sims-n_bins+n_param-1))
    inv_cov_mat = np.linalg.inv(cov_mat)
    return inv_cov_mat

def derivative_vector(low_realizations, high_realizations, n_realizations, functions, k_cut, low_high_diff):
    """Computes the mean derivative column vector of the summary statistic for a given parameter.

    Args:
        low_realizations (dict): Dictionary containing the realizations for the parameter low variation.
        high_realizations (dict): Dictionary containing the realizations for the parameter high variation.
        n_realizations (int): The number of realizations to use for derivatives (where a single realization
                              includes all three LOS directions for a single simulation).
        functions (list): List of the strings denoting the functions to include.
        k_cut (array): The array to apply the cut on the wavenumbers to include.
        low_high_diff (float): The change in parameter value between the low and high variations.

    Returns:
        array: The mean derivative column vector.
    """
    # Stacks the low and high realizations into arrays of realizations
    full_vector = []
    for func in functions: # three LOS directions for derivatives
        func_x_low = low_realizations['x'][func]
        func_y_low = low_realizations['y'][func]
        func_z_low = low_realizations['z'][func]
        shape = np.shape(func_x_low)
        # Alternates between the three LOS when stacking realizations to ensure that when averaging over
        # many realizations, the three LOS directions for a single realization counts as one sample,
        # instead of having all three LOS directions treated independently
        func_low = np.zeros((3*shape[0], shape[1]))
        func_low[0::3, :] = func_x_low
        func_low[1::3, :] = func_y_low
        func_low[2::3, :] = func_z_low
        func_low_mean = np.mean(func_low[:3*n_realizations, k_cut], axis=0)
        func_x_high = high_realizations['x'][func]
        func_y_high = high_realizations['y'][func]
        func_z_high = high_realizations['z'][func]
        func_high = np.zeros((3*shape[0], shape[1]))
        func_high[0::3, :] = func_x_high
        func_high[1::3, :] = func_y_high
        func_high[2::3, :] = func_z_high
        func_high_mean = np.mean(func_high[:3*n_realizations, k_cut], axis=0)
        func_diff_mean = func_high_mean-func_low_mean
        deriv_mean = func_diff_mean/low_high_diff
        full_vector.append(deriv_mean)
    full_vector = np.concatenate(full_vector)
    full_vector = np.array([full_vector]).T # converts to column vector format
    return full_vector

def fisher_matrix_element(inv_cov_mat, deriv_col_vec_i, deriv_col_vec_j):
    """Computes the (i,j) and (j,i) elements of the Fisher matrix.

    Args:
        inv_cov_mat (array): The inverse covariance matrix.
        deriv_col_vec_i (array): The ith derivative column vector.
        deriv_col_vec_j (array): The jth derivative column vector.

    Returns:
        float: The (i,j)th element of the Fisher matrix.
    """
    # Performs the matrix multiplication
    element = np.matmul(np.matmul(deriv_col_vec_i.T, inv_cov_mat),
                        deriv_col_vec_j)[0][0]
    if element == 0:
        print('Warning, one of the elements in Fisher matrix is 0!')
    return element

def max_wavenumber_constraints(func_combination_list, func_combination_label, param_names, param_diffs, marginalization,
                               fiducial_path, n_realizations_cov, low_paths, high_paths, n_realizations_deriv, k_upper,
                               save_path):
    """Generates parameter covariance matrices (inverse Fisher matrices) for different max fitting wavenumbers.

    Args:
        func_combination_list (list): List containing the function labels.
        func_combination_label (str): String denoting the particular combination of functions.
        param_names (list): List of the strings denoting the parameters included in the Fisher matrix.
        param_diffs (list): List of the differences between the low and high parameter variations.
        marginalization (bool): Whether to marginalize over the other parameters or not.
        fiducial_path (str): The path to the dictionary containing the fiducial realizations.
        n_realizations_cov (int): The number of realizations to include for the covariance matrix.
        low_paths (list): List of the file paths to each of the separate low parameter variations.
        high_paths (list): List of the file paths to each of the separate high parameter variations.
        n_realizations_deriv (int): The number of realizations to include for the derivatives.
        k_upper (float): The upper wavenumber to fit to (h/Mpc).
        save_path (str): The folder path to save the results.
    """
    # Reads in the fiducial data and extracts wavenumbers in each bin
    fiducial_data = np.load(fiducial_path, allow_pickle=True).item()
    k_avg = fiducial_data['k_avg']
    # Initiates list to store all the parameter covariance matrices for each of the kmax values
    param_cov_matrices = []
    k_max_vals = []
    # Iterates through the different max fitting wavenumbers (only samples every fourth
    # wavenumber to save runtime)
    for k_max in k_avg[(k_avg >= k_avg[2]) & (k_avg <= k_upper)][::4]:
        k_max_vals.append(k_max)
        k_cut = k_avg <= k_max
        # Initiates Fisher information matrix
        fisher_mat = np.zeros((len(param_names), len(param_names)))
        # Retrieves inverse covariance matrix (depending on marginalization or not)
        if marginalization:
            inv_cov_mat = inverse_covariance_matrix(
                fiducial_data, n_realizations_cov, func_combination_list, k_cut, len(param_names))
        else:
            inv_cov_mat = inverse_covariance_matrix(
                fiducial_data, n_realizations_cov, func_combination_list, k_cut, 1)
        # Creates list to store the derivative column vectors for each of the parameters
        deriv_col_vecs = []
        # Derivatives for each parameter variation
        for i in range(len(param_names)):
            param_diff = param_diffs[i]
            low_path = low_paths[i]
            high_path = high_paths[i]
            low_data = np.load(low_path, allow_pickle=True).item()
            high_data = np.load(high_path, allow_pickle=True).item()
            deriv_col_vec = derivative_vector(
                low_data, high_data, n_realizations_deriv, func_combination_list, k_cut, param_diff)
            deriv_col_vecs.append(deriv_col_vec)
        # Adds terms to the Fisher matrix
        if marginalization:
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    if j >= i: # avoids double counting
                        elem = fisher_matrix_element(inv_cov_mat, deriv_col_vecs[i], deriv_col_vecs[j])
                        fisher_mat[i][j] = elem
                        fisher_mat[j][i] = elem
        else: # when not marginalizating includes only diagonal terms so inverse simply gives inverse of each element
            for i in range(len(param_names)):
                elem = fisher_matrix_element(inv_cov_mat, deriv_col_vecs[i], deriv_col_vecs[i])
                fisher_mat[i][i] = elem
        # Computes associated parameter covariance matrix (inverse of Fisher matrix)
        param_cov_matrices.append(np.linalg.inv(fisher_mat))
    # Saves away results to file
    if marginalization:
        marginalization_label = 'marginalized'
    else:
        marginalization_label = 'unmarginalized'
    param_label = '-'.join(param_names)
    np.save(os.path.join(save_path,
            f'{marginalization_label}_{func_combination_label}_{param_label}_{n_realizations_cov}_{n_realizations_deriv}.npy'),
            param_cov_matrices)
    return None


if __name__ == '__main__':

    # Obtains Fisher constraints (marginalized and unmarginalized) as a function of the maximum
    # fitting wavenumber for different function combinations

    # Iterates through the different hyperparameter scenarios
    filter_type = 'Gaussian'
    nmesh = 512
    redshift = 0
    split = 'zsplit'
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    rebin_factor = 1
    for hyperparameters in [(5, 10, None, 'lattice'), 
                            (3, 10, None, 'lattice'), 
                            (7, 10, None, 'lattice'),
                            (5, 7, None, 'lattice'),
                            (5, 13, None, 'lattice'),
                            (5, 10, 5, 'random')]:
        n_quantiles, filter_radius, n_randoms, query_type = hyperparameters
        # Unique folder for each hyperparameter combination
        save_path = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}'
        # All the function combinations to produce results for
        halo_terms = ['h-h(0)', 'h-h(2)']
        DS_terms = list(np.concatenate([[f'{i+1}-h(0)', f'{i+1}-h(2)', f'{i+1}-{i+1}(0)', f'{i+1}-{i+1}(2)']
                        for i in list(range(n_quantiles//2))+list(range(n_quantiles//2+1, n_quantiles))])) # skips middle quantile
        joint_terms = halo_terms + DS_terms
        functions_list = [halo_terms, DS_terms, joint_terms]
        function_labels = ['Halo', 'DS', 'Joint']
        # Parameters to marginalize over
        param_names = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
        param_diffs = [200, 200, 200, 2e12, 0.04, 0.04, 0.02, 0.03]
        # Paths to read existing data
        start_path = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
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
        n_realizations_cov = 15000#n_realizations_deriv = 500
        k_upper = 0.5
        # Varies the number of derivative realizations used
        for n_realizations_deriv in np.arange(10, 510, 10):#for n_realizations_cov in np.arange(1500, 15300, 300):
            for i in range(len(functions_list)):
                functions = functions_list[i]
                function_label = function_labels[i]
                # Performs for both marginalization and no marginalization
                max_wavenumber_constraints(functions, function_label, param_names, param_diffs, True, fiducial_path, 
                                           n_realizations_cov, low_paths, high_paths, n_realizations_deriv, k_upper, save_path)
                print(hyperparameters, function_label, n_realizations_deriv, 'marginalized', 'DONE')
                max_wavenumber_constraints(functions, function_label, param_names, param_diffs, False, fiducial_path, 
                                           n_realizations_cov, low_paths, high_paths, n_realizations_deriv, k_upper, save_path)
                print(hyperparameters, function_label, n_realizations_deriv, 'unmarginalized', 'DONE')
