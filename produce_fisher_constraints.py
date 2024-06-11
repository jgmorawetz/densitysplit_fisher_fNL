import os
import pickle
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
from getdist import plots, MCSamples


# obtains fisher constraints (marginalized and unmarginalized) as a function of the maximum
# fitting wavenumber for the different function combinations 


def inverse_covariance_matrix(realizations, n_realizations, functions, k_cut, n_param):
    """Computes the inverse covariance matrix, given a dictionary with all the realizations of 
       the various functions, the number of desired realizations to include, the list of all
       the functions to include (in a given order), the cut to apply to the wavenumbers, and 
       the number of parameters involved in the Fisher calculation."""
    all_functions = []
    for func in functions:
        all_functions.append(realizations['z'][func][:n_realizations, k_cut].T) # only z line of sight direction
    all_functions = np.concatenate(all_functions).T
    n_sims, n_bins = np.shape(all_functions)
    # currently applies Hartlap correction factor (need to incorporate the number of parameters if wanting to 
    # use Percival correction instead)
    cov_mat = np.cov(all_functions.T)*((n_sims-1)/(n_sims-n_bins-2))
    if np.shape(cov_mat) != ():
        inv_cov_mat = np.linalg.inv(cov_mat)
    else:
        inv_cov_mat = np.array([[cov_mat**-1]]) # handles the case where it is one dimension only
    return inv_cov_mat
    
def numerical_derivative_vector(low_realizations, high_realizations, n_realizations, functions, k_cut, low_up_diff):
    """Computes the numerical derivative column vector, given the dictionaries with all the realizations of the low and
       high parameter variations, the number of desired realizations to include, the list of all functions to include
       (in a given order), the cut to apply to the wavenumbers, and difference between the high and low parameter 
       values."""
    full_vector = []
    for func in functions:
        func_x_low = low_realizations['x'][func]
        func_y_low = low_realizations['y'][func]
        func_z_low = low_realizations['z'][func]
        shape = np.shape(func_x_low)
        # alternates between the three lines of sight when stacking the realizations such that the x,y,z line of sight
        # versions for each realization are placed together (to avoid preferential grouping when averaging over 
        # different numbers of realizations)
        func_low = np.zeros((3*shape[0], shape[1]))
        func_low[0::3, :] = func_x_low
        func_low[1::3, :] = func_y_low
        func_low[2::3, :] = func_z_low
        func_low_mean = np.mean(func_low[:n_realizations, k_cut], axis=0)
        func_x_high = high_realizations['x'][func]
        func_y_high = high_realizations['y'][func]
        func_z_high = high_realizations['z'][func]
        func_high = np.zeros((3*shape[0], shape[1]))
        func_high[0::3, :] = func_x_high
        func_high[1::3, :] = func_y_high
        func_high[2::3, :] = func_z_high
        func_high_mean = np.mean(func_high[:n_realizations, k_cut], axis=0)
        func_diff_mean = func_high_mean-func_low_mean
        deriv_mean = func_diff_mean/low_up_diff
        full_vector.append(deriv_mean)
    full_vector = np.concatenate(full_vector)
    full_vector = np.array([full_vector]).T # converts to column vector format
    return full_vector

def fisher_matrix_element(inv_cov_mat, num_deriv_col_vec_i, num_deriv_col_vec_j):
    """Computes the (i,j),(j,i) elements of the fisher matrix based on inverse covariance matrix
       and numerical derivative column vectors."""
    element = np.matmul(np.matmul(num_deriv_col_vec_i.T, inv_cov_mat),
                        num_deriv_col_vec_j)[0][0]
    if element == 0:
        print('Warning, one of the elements in Fisher matrix is 0!')
    return element

def max_wavenumber_constraints(func_combination_list, func_combination_label, param_names, param_diffs, marginalization,
                               fiducial_path, n_realizations_cov, low_paths, high_paths, n_realizations_deriv, k_upper, 
                               kstep_interval, save_path):
    """Computes Fisher constraints as a function of max fitting wavenumber, given a list of lists with all the desired
       functions to include, the labels associated with each function combination, the names of the parameters to retrieve
       constraints for, the difference in high-low variation for each of the parameters, whether or not to perform 
       marginalization, the path to the fiducial realizations, the number of realizations to use for the covariance matrix,
       the paths to the low/high variations, the number of realizations to use for the derivatives, the upper wavenumber to
       fit to, the integer multiple of the fundamental wavenumber used for the wavenumber bins, and the save path."""
    # reads in the fiducial data and extracts the wavenumbers in each bin
    fiducial_data = np.load(fiducial_path, allow_pickle=True).item()
    k_avg = fiducial_data['k_avg']
    k_ind = (k_avg >= k_avg[1]) & (k_avg <= k_upper)
    # initiates list to store all the parameter covariance matrices for each of the different max k values
    param_cov_matrices = []
    k_max_vals = []
    # iterates through the different max fitting wavenumbers
    for k_max in k_avg[k_ind]:
        k_max_vals.append(k_max)
        k_cut = k_avg <= k_max
        # initiates a fisher information matrix
        fisher_mat = np.zeros((len(param_names), len(param_names)))
        # retrieves inverse covariance matrix (depending on marginalization or not)
        if marginalization:
            inv_cov_mat = inverse_covariance_matrix(fiducial_data, n_realizations_cov, func_combination_list, k_cut, len(param_names))
        else:
            inv_cov_mat = inverse_covariance_matrix(fiducial_data, n_realizations_cov, func_combination_list, k_cut, 1)
        # creates a list to store the derivative column vectors for each of the parameters
        deriv_col_vecs = []
        # obtains derivatives for each parameter variation
        for i in range(len(param_names)):
            param_diff = param_diffs[i]
            low_path = low_paths[i]
            high_path = high_paths[i]
            low_data = np.load(low_path, allow_pickle=True).item()
            high_data = np.load(high_path, allow_pickle=True).item()
            deriv_col_vec = numerical_derivative_vector(low_data, high_data, n_realizations_deriv, func_combination_list, k_cut, param_diff)
            deriv_col_vecs.append(deriv_col_vec)
        if marginalization: # when marginalizing over other parameters includes cross terms in Fisher matrix
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    if j >= i: # avoids double counting unnecessarily
                        elem = fisher_matrix_element(inv_cov_mat, deriv_col_vecs[i], deriv_col_vecs[j])
                        fisher_mat[i][j] = elem
                        fisher_mat[j][i] = elem
        else: # when not marginalizing includes only diagonal terms so that inverse simply gives inverses of each element as desired
            for i in range(len(param_names)):
                elem = fisher_matrix_element(inv_cov_mat, deriv_col_vecs[i], deriv_col_vecs[i])
                fisher_mat[i][i] = elem
        # computes associated parameter covariance matrix by taking inverse of Fisher matrix
        param_cov_matrices.append(np.linalg.inv(fisher_mat))
    # saves away the results to file
    if marginalization:
        marginalization_label = 'marginalized'
    else:
        marginalization_label = 'unmarginalized'
    np.save(os.path.join(save_path, '{0}_{1}_{2}_{3}_{4}_{5}kF_constraints.npy'.format(marginalization_label, func_combination_label, 
            '-'.join(param_names), n_realizations_cov, n_realizations_deriv, kstep_interval)), param_cov_matrices)
    return None


if __name__ == '__main__':

    functions_list = [['h-h(0)', 'h-h(2)'], # 2PCF
                      ['1-h(0)', '1-h(2)', '1-1(0)', '1-1(2)'], # DS1
                      ['5-h(0)', '5-h(2)', '5-5(0)', '5-5(2)'], # DS5
                      ['1-h(0)', '1-h(2)', '5-h(0)', '5-h(2)', '1-1(0)', '1-1(2)', '5-5(0)', '5-5(2)'], # DS1+5
                      ['1-h(0)', '1-h(2)', '2-h(0)', '2-h(2)', '4-h(0)', '4-h(2)', '5-h(0)', '5-h(2)',
                       '1-1(0)', '1-1(2)', '2-2(0)', '2-2(2)', '4-4(0)', '4-4(2)', '5-5(0)', '5-5(2)'], # DS
                      ['h-h(0)', 'h-h(2)',
                       '1-h(0)', '1-h(2)', '2-h(0)', '2-h(2)', '4-h(0)', '4-h(2)', '5-h(0)', '5-h(2)',
                       '1-1(0)', '1-1(2)', '2-2(0)', '2-2(2)', '4-4(0)', '4-4(2)', '5-5(0)', '5-5(2)']] # DS+2PCF
    functions_labels = ['2PCF',
                        'DS1', 'DS5', 'DS1+5',
                        'DS',
                        'DS+2PCF']
    param_names = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8'] # ['Mmin', 'h', 'ns', 'Om', 's8']
    param_diffs = [200, 200, 200, 2e12, 0.04, 0.04, 0.02, 0.03] # [2e12, 0.04, 0.04, 0.02, 0.03]
    filter_type = 'Gaussian'
    filter_radius = 10
    n_quantiles = 5
    nmesh = 512
    query_type = 'lattice' #'random'
    n_randoms = None #5
    kstep_interval = 1
    redshift = 0
    start_path = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/MERGED_DATA'
    fiducial_path = os.path.join(start_path, 'power_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}kF.npy'.format(
                    'fiducial', filter_type, filter_radius, n_quantiles, nmesh, query_type, n_randoms, redshift, kstep_interval))
    low_paths = list(map(lambda x: os.path.join(start_path, 'power_{0}_m_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}kF.npy'.format(
                                   x, filter_type, filter_radius, n_quantiles, nmesh, query_type, n_randoms, redshift, kstep_interval)) if x != 'Mmin' else 
                                   os.path.join(start_path, 'power_{0}_3.1e13_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}kF.npy'.format(
                                   x, filter_type, filter_radius, n_quantiles, nmesh, query_type, n_randoms, redshift, kstep_interval)), param_names))
    high_paths = list(map(lambda x: os.path.join(start_path, 'power_{0}_p_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}kF.npy'.format(
                                    x, filter_type, filter_radius, n_quantiles, nmesh, query_type, n_randoms, redshift, kstep_interval)) if x != 'Mmin' else 
                                    os.path.join(start_path, 'power_{0}_3.3e13_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}kF.npy'.format(
                                    x, filter_type, filter_radius, n_quantiles, nmesh, query_type, n_randoms, redshift, kstep_interval)), param_names))
    n_realizations_cov = 15000
    k_upper = 0.5
    save_path = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}'
    for n_realizations_deriv in np.arange(100, 1600, 100):
        for i in range(len(functions_list)):
            functions = functions_list[i]
            function_label = functions_labels[i]
            max_wavenumber_constraints(functions, function_label, param_names, param_diffs, True, fiducial_path, n_realizations_cov,
                                       low_paths, high_paths, n_realizations_deriv, k_upper, kstep_interval, save_path) # marginalization
            print(function_label, 'marginalized', 'done')
            #max_wavenumber_constraints(functions, function_label, param_names, param_diffs, False, fiducial_path, n_realizations_cov,
            #                           low_paths, high_paths, n_realizations_deriv, k_upper, kstep_interval, save_path) # no marginalization
            #print(function_label, 'unmarginalized', 'done')
        print(n_realizations_deriv, 'done')