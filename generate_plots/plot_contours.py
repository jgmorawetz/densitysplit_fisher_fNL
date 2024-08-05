import os
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
from getdist import plots, MCSamples


if __name__ == '__main__':
      
    # Makes a triangle (contour) plot of the marginalized constraints fitted to the max wavenumber
    # for halo only, DS only or joint

    # Specific hyperparameters
    query_type, n_randoms, n_quantiles, filter_radius = 'lattice', None, 5, 10
    ncov, nderiv = 15000, 500
    data_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/fisher_convergence/power/{query_type}_{n_randoms}_{n_quantiles}_{filter_radius}'
    param_list = ['LC', 'EQ', 'OR_LSS', 'Mmin', 'h', 'ns', 'Om', 's8']
    param_means = [0, 0, 0, 3.2e13, 0.6711, 0.9624, 0.3175, 0.834]
    param_list_labels = [r'$f_{NL}^{loc}$', r'$f_{NL}^{equil}$', r'$f_{NL}^{ortho}$', r'$M_{min}$', r'$h$', r'$n_s$', r'$\Omega_m$', r'$\sigma_8$']
    combs = ['Halo', 'DS', 'Joint']
    comb_colors = ['tab:blue', 'tab:orange', 'tab:green']
    samples = []
    for i in range(3):
        comb = combs[i]
        if comb == 'Halo':
            comb_label = 'Halo'
        elif comb == 'DS':
            comb_label = 'DSC'
        elif comb == 'Joint':
            comb_label = 'Joint'
        # parameter covariance matrix at the max wavenumber (final index of list of matrices)
        marginalized_data = np.load(os.path.join(data_folder, f'marginalized_{comb}_LC-EQ-OR_LSS-Mmin-h-ns-Om-s8_{ncov}_{nderiv}.npy'))[-1]
        samples_comb = np.random.multivariate_normal(mean=param_means, cov=marginalized_data, size=1000000)
        samples_comb = MCSamples(samples=samples_comb, names=param_list_labels, settings={'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})
        samples.append(samples_comb)

    # Makes the triangle plot
    g = plots.get_subplot_plotter()
    g.settings.axes_labelsize = 25
    g.settings.figure_legend_loc = 'upper right'
    g.settings.axes_fontsize = 20
    g.settings.num_plot_contours = 2
    g.settings.legend_fontsize = 30
    g.settings.legend_colored_text = False
    g.settings.figure_legend_frame = True
    g.settings.linewidth = 3
    g.settings.solid_contour_palefactor = 0.5
    g.triangle_plot(roots=samples, 
                    legend_labels=['Halo', 'DSC', 'Joint'],
                    filled=True, contour_colors=comb_colors)
    plt.savefig('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/constraints_contour_plot_including_all_auto.png', dpi=400)