# densitysplit_fisher_fNL
This repository contains all the codes necessary to reproduce the results in the paper...

1) Begin by downloading the halo catalogs for the different parameter variations found at /Halos/FoF/ on Globus collection 'Quijote_simulations2'.

2) Under the folder 'generate_statistics', the bash scripts 'power_spectra_fiducial.sh' and 'power_spectra_LCDM_PNG.sh' contain the instructions necessary to run the python scripts 'densitysplit_power_fiducial.py' and 'densitysplit_power_LCDM_PNG.py' which generate all the necessary power spectra and save away to file.

3) Under the folder 'generate_statistics', run 'merge_power_files.py' to combine all the power spectrum results for each simulation (stored in individual files) into one file containing all realizations.

4) Under the folder 'generate_statistics', run 'produce_fisher_constraints.py' to generate the parameter covariance matrices (inverse of the Fisher matrices) as a function of fitting wavenumber, and in doing so produce the results needed for figures 3 and 4.

5) Under the folder 'generate_plots':
   - Run 'plot_densitysplit_cross_section.py' to generate figure 1.
   - Run 'plot_fiducial_power_spectra.py' to generate figure 2.
   - Run 'plot_constraints_max_wavenumber.py' to generate figure 3.
   - Run 'plot_contours.py' to generate figure 4.
