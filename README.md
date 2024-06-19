# densitysplit_fisher_fNL
This repository contains all the codes necessary to reproduce the results in the paper ...

1) Begin by downloading all the Quijote simulation halo catalogs using Globus.

2) Run the codes 'densitysplit_power_fiducial.py' and 'densitysplit_power_LCDM_PNG.py' to generate and save away all power spectrum results.

3) Run 'merge_power_files.py' to combine all the power spectrum results for each simulation (stored individually) into one file containing all realizations.

4) Run 'produce_fisher_constraints.py' to generate the parameter covariance matrices as a function of fitting wavenumber (and in doing so produce the results needed for plots ...).
