# densitysplit_fisher_fNL
This repository contains all the codes necessary to reproduce the results in the paper 'Constraining Primordial Non-Gaussianity with Density-Split Clustering'.

1) Begin by downloading the halo catalogs for the different parameter variations ('fiducial', 'LC_m', 'LC_p', 'EQ_m', 'EQ_p', 'OR_LSS_m', 'OR_LSS_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 'Om_m', 'Om_p', 's8_m', 's8_p') found at /Halos/FoF/ on Globus collection 'Quijote_simulations2'. The data for redshift snapshot z=0 are found under 'groups_004' tab. The matter fields (used later in the analysis) can be found by downloading the 'fiducial', 's8_m', 's8_p', 'LC_m', 'LC_p' variations under the /3D_cubes/ folder.

2) Set up bash scripts to run array jobs (iterating through the simulation indices) to execute the python scripts 'densitysplit_power_fiducial.py', 'densitysplit_power_LCDM_PNG.py', 'test_matter_field_indirectly.py', 'test_matter_field_directly.py' under the folder 'generate_statistics', which generate all the necessary power spectra and overdensity distributions used in the analysis and save away to file.

3) Under the folder 'generate_statistics', run 'merge_power_files.py' to combine all the power spectrum results for each simulation (stored in individual files) into one file containing all realizations. This applies to the codes 'densitysplit_power_fiducial.py', 'densitysplit_power_LCDM_PNG.py'. The other codes 'test_matter_field_indirectly.py' and 'test_matter_field_directly.py' have commented out sections of the code which can perform this step.

4) Under the folder 'generate_statistics', run 'produce_fisher_constraints.py' to generate the parameter covariance matrices (inverse of the Fisher matrices) as a function of fitting wavenumber.

5) Under the folder 'generate_plots':
   - Run 'plot_densitysplit_cross_section.py' to generate figure 1.
   - Run 'plot_fiducial_power_spectra.py' to generate figure 2.
   - Run 'plot_constraints_max_wavenumber.py' to generate figure 3.
   - Run 'plot_contours.py' to generate figure 4.
   - Run 'plot_derivative_covariance_convergence.py' to generate figure 5 (and repeat for the different hyperparameter combinations other than the baseline for results in remaining plots).
   - Run 'plot_corrected_constraints.py' to generate figures 6 and 7.
   - Run (the commented out sections of) 'test_matter_field_indirectly.py' and 'test_matter_field_directly.py' to generate figure 8.
   - Run 'plot_lattice_randoms_comparison.py' to generate figure 9.
