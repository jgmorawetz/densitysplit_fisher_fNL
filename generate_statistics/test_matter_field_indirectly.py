import os
import time
import argparse
import readfof
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import ArrayCatalog


def smoothed_overdensity(meshgrid, boxsize, filter_radius, nmesh):
    """Applies Gaussian smoothing to the meshgrid.

    Args:
        meshgrid (array): The (unsmoothed) overdensity field.
        boxsize (float): The simulation box length (Mpc/h).
        filter_radius (float): The Gaussian filter radius (Mpc/h).
        nmesh (int): The mesh grid resolution.

    Returns:
        meshgrid_smooth: The smoothed overdensity field.
    """
    # Applies smoothing in Fourier space
    kstep = np.fft.fftfreq(nmesh)*np.pi/(boxsize/nmesh/2)
    kx,ky,kz=np.meshgrid(kstep, kstep, kstep, indexing='ij')
    k2 = kx**2+ky**2+kz**2
    meshgrid_k = np.fft.fftn(meshgrid)
    gaussian=np.exp(-1/2*k2*filter_radius**2)
    meshgrid_k_smooth = meshgrid_k*gaussian
    meshgrid_smooth = np.fft.ifftn(meshgrid_k_smooth).real
    return meshgrid_smooth

def get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los):
    """Retrieves halo positions and associated halo masses in real or redshift space.

    Args:
        halo_path (str): The folder path of the snapshot.
        boxsize (float): The boxsize of the simulation volume (Mpc/h).
        snapnum (int): The snapshot corresponding to given redshift 
                       {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}.
        redshift (float): The redshift corresponding to the given snapshot.
        omega_m (float): The omega matter parameter.
        min_mass (float): The mass cut to apply to the halos (Msun/h).
        space (str): Real ('r') or redshift ('z') space.
        los (str): The line-of-sight direction ('x', 'y', 'z').

    Returns:
        array: The 3D positions of the halos (dimension (N, 3)) where N is the 
               number of halos.
        array: The masses of the halos (Msun/h).
    """
    # Assumes flat LCDM cosmology
    omega_l = 1-omega_m
    H0 = 100
    az = 1/(1+redshift)
    Hz = H0*np.sqrt(omega_m*(1+redshift)**3+omega_l)
    # Reads in unprocessed catalog
    data = readfof.FoF_catalog(halo_path, snapnum, long_ids=False, swap=False,
                               SFR=False, read_IDs=False)
    # Extracts positions, velocities and masses
    pos = data.GroupPos/1e3
    vel = data.GroupVel*(1+redshift)
    mass = data.GroupMass*1e10
    # Applies mass cut
    mass_cut = mass >= min_mass
    pos = pos[mass_cut]
    vel = vel[mass_cut]
    mass = mass[mass_cut]
    # Applies redshift space distortions if needed
    if space == 'r':
        return pos, mass
    elif space == 'z':
        if los == 'x':
            los_vec = np.array([1, 0, 0])
        elif los == 'y':
            los_vec = np.array([0, 1, 0])
        elif los == 'z':
            los_vec = np.array([0, 0, 1])
        pos = pos + (vel*los_vec)/(az*Hz)
        # Enforces periodic boundary conditions
        pos = pos % boxsize
        return pos, mass

def halo_field(halo_info, boxsize, nmesh, resampler, weighted):
    """Applies the halo positions and masses to obtain the mesh overdensity field.

    Args:
        halo_info (tuple): The halo positions and masses.
        boxsize (float): The boxsize of the simulation volume (Mpc/h).
        nmesh (int): The resolution of the mesh grid.
        resampler (str): The resampler scheme.
        weighted (bool): Whether to weight the overdensity field by halo mass or not.

    Returns:
        array: The mesh overdensity field.
    """
    # Makes field either mass-weighted or not mass-weighted.
    halo_positions, halo_masses = halo_info
    if weighted:
        halo_grid = ArrayCatalog(data={'Position':halo_positions, 'Mass':halo_masses}).to_mesh(
            Nmesh=nmesh, BoxSize=boxsize, resampler=resampler, weight='Mass').preview(Nmesh=nmesh)
    else:
        halo_grid = ArrayCatalog(data={'Position':halo_positions}).to_mesh(
            Nmesh=nmesh, BoxSize=boxsize, resampler=resampler).preview(Nmesh=nmesh)
    halo_grid /= np.mean(halo_grid); halo_grid -= 1
    return halo_grid

def percentile_from_histogram(hist_counts, bin_centres, percentiles):
    """Computes the percentiles in bin centres based on the histogram counts.

    Args:
        hist_counts (array): The counts in each bin of the histogram.
        bin_centres (array): The bin centres.
        percentiles (array): The percentiles of interest.

    Returns:
        array: The percentile values.
    """
    cdf = np.cumsum(hist_counts)
    cdf = cdf/cdf[-1]
    percentile_values = np.interp(percentiles, cdf, bin_centres)
    return percentile_values


if __name__ == '__main__':

    # splits up into array jobs with several phases for each
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    start_idx = args.start_idx
    n = args.n
    phases = np.arange(start_idx, start_idx+n)

    boxsize = 1000
    snapnum = 4
    redshift = 0
    omega_m = 0.3175
    min_mass = 3.2e13
    space = 'r'
    los = 'z'
    nmesh_power = 256
    matter_folder = '/home/jgmorawe/scratch/matter_fields'
    halo_folder = '/home/jgmorawe/scratch/quijote'
    n_quantiles = 5
    resampler = 'cic'
    filter_radius = 10
    save_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect'
    bins = np.arange(-1, 500, 0.0001)
    for phase in phases:
        # matter fields
        fiducial_grid = np.load(os.path.join(matter_folder, 'fiducial', f'{phase}/df_m_256_CIC_z=0.npy'))
        s8_m_grid = np.load(os.path.join(matter_folder, 's8_m', f'{phase}/df_m_256_CIC_z=0.npy'))
        s8_p_grid = np.load(os.path.join(matter_folder, 's8_p', f'{phase}/df_m_256_CIC_z=0.npy'))
        print('No smoothing matter: ', np.percentile(fiducial_grid.flatten(), np.arange(0, 100+100//n_quantiles, 100//n_quantiles)))
        # smoothes the matter fields
        smoothed_fiducial_grid = smoothed_overdensity(fiducial_grid, boxsize, filter_radius, nmesh_power)
        smoothed_s8_m_grid = smoothed_overdensity(s8_m_grid, boxsize, filter_radius, nmesh_power)
        smoothed_s8_p_grid = smoothed_overdensity(s8_p_grid, boxsize, filter_radius, nmesh_power)
        print('Smoothing matter: ', np.percentile(smoothed_fiducial_grid.flatten(), np.arange(0, 100+100//n_quantiles, 100//n_quantiles)))
        # obtains halo catalogs
        halo_info_fiducial = get_halo_info(os.path.join(halo_folder, f'fiducial/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        halo_info_s8_m = get_halo_info(os.path.join(halo_folder, f's8_m/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        halo_info_s8_p = get_halo_info(os.path.join(halo_folder, f's8_p/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        # obtains halo fields
        fiducial_halo_grid_weighted = halo_field(halo_info_fiducial, boxsize, nmesh_power, resampler, True)
        s8_m_halo_grid_weighted = halo_field(halo_info_s8_m, boxsize, nmesh_power, resampler, True)
        s8_p_halo_grid_weighted = halo_field(halo_info_s8_p, boxsize, nmesh_power, resampler, True)
        fiducial_halo_grid_notweighted = halo_field(halo_info_fiducial, boxsize, nmesh_power, resampler, False)
        s8_m_halo_grid_notweighted = halo_field(halo_info_s8_m, boxsize, nmesh_power, resampler, False)
        s8_p_halo_grid_notweighted = halo_field(halo_info_s8_p, boxsize, nmesh_power, resampler, False)
        # smoothes the halo fields
        smoothed_fiducial_halo_grid_weighted = smoothed_overdensity(fiducial_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        smoothed_s8_m_halo_grid_weighted = smoothed_overdensity(s8_m_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        smoothed_s8_p_halo_grid_weighted = smoothed_overdensity(s8_p_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        print('Smoothing halo weighted: ', np.percentile(smoothed_fiducial_halo_grid_weighted.flatten(), np.arange(0, 100+100//n_quantiles, 100//n_quantiles)))
        smoothed_fiducial_halo_grid_notweighted = smoothed_overdensity(fiducial_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        smoothed_s8_m_halo_grid_notweighted = smoothed_overdensity(s8_m_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        smoothed_s8_p_halo_grid_notweighted = smoothed_overdensity(s8_p_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        print('Smoothing halo unweighted: ', np.percentile(smoothed_fiducial_halo_grid_notweighted.flatten(), np.arange(0, 100+100//n_quantiles, 100//n_quantiles)))
        # bins into histogram and stores away for later
        fiducial_grid_histogram = np.histogram(fiducial_grid.flatten(), bins=bins)[0]
        s8_m_grid_histogram = np.histogram(s8_m_grid.flatten(), bins=bins)[0]
        s8_p_grid_histogram = np.histogram(s8_p_grid.flatten(), bins=bins)[0]
        smoothed_fiducial_grid_histogram = np.histogram(smoothed_fiducial_grid.flatten(), bins=bins)[0]        
        smoothed_s8_m_grid_histogram = np.histogram(smoothed_s8_m_grid.flatten(), bins=bins)[0]
        smoothed_s8_p_grid_histogram = np.histogram(smoothed_s8_p_grid.flatten(), bins=bins)[0]
        smoothed_fiducial_halo_grid_weighted_histogram = np.histogram(smoothed_fiducial_halo_grid_weighted.flatten(), bins=bins)[0]
        smoothed_s8_m_halo_grid_weighted_histogram = np.histogram(smoothed_s8_m_halo_grid_weighted.flatten(), bins=bins)[0]
        smoothed_s8_p_halo_grid_weighted_histogram = np.histogram(smoothed_s8_p_halo_grid_weighted.flatten(), bins=bins)[0]
        smoothed_fiducial_halo_grid_notweighted_histogram = np.histogram(smoothed_fiducial_halo_grid_notweighted.flatten(), bins=bins)[0]
        smoothed_s8_m_halo_grid_notweighted_histogram = np.histogram(smoothed_s8_m_halo_grid_notweighted.flatten(), bins=bins)[0]
        smoothed_s8_p_halo_grid_notweighted_histogram = np.histogram(smoothed_s8_p_halo_grid_notweighted.flatten(), bins=bins)[0]
        np.save(os.path.join(save_folder, f'phase{phase}_fiducial_nosmooth_matter.npy'), fiducial_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_s8_m_nosmooth_matter.npy'), s8_m_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_s8_p_nosmooth_matter.npy'), s8_p_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_fiducial_smooth{filter_radius}_matter.npy'), smoothed_fiducial_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_s8_m_smooth{filter_radius}_matter.npy'), smoothed_s8_m_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_s8_p_smooth{filter_radius}_matter.npy'), smoothed_s8_p_grid_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_fiducial_smooth{filter_radius}_weighted_halo.npy'), smoothed_fiducial_halo_grid_weighted_histogram)      
        np.save(os.path.join(save_folder, f'phase{phase}_s8_m_smooth{filter_radius}_weighted_halo.npy'), smoothed_s8_m_halo_grid_weighted_histogram)   
        np.save(os.path.join(save_folder, f'phase{phase}_s8_p_smooth{filter_radius}_weighted_halo.npy'), smoothed_s8_p_halo_grid_weighted_histogram)
        np.save(os.path.join(save_folder, f'phase{phase}_fiducial_smooth{filter_radius}_notweighted_halo.npy'), smoothed_fiducial_halo_grid_notweighted_histogram)      
        np.save(os.path.join(save_folder, f'phase{phase}_s8_m_smooth{filter_radius}_notweighted_halo.npy'), smoothed_s8_m_halo_grid_notweighted_histogram)   
        np.save(os.path.join(save_folder, f'phase{phase}_s8_p_smooth{filter_radius}_notweighted_halo.npy'), smoothed_s8_p_halo_grid_notweighted_histogram)  


    """
    # Merges the files together into a single histogram for each variation (averaged over all realizations)
    N = 500
    bin_edges = np.arange(-1, 500, 0.0001)
    bin_centres = 1/2*(bin_edges[1:]+bin_edges[:-1])
    filter_radius = 10
    load_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect'
    save_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect_merged'
    for title in ['fiducial_nosmooth_matter', 's8_m_nosmooth_matter', 's8_p_nosmooth_matter',
                  f'fiducial_smooth{filter_radius}_matter', f's8_m_smooth{filter_radius}_matter', f's8_p_smooth{filter_radius}_matter',
                  f'fiducial_smooth{filter_radius}_weighted_halo', f's8_m_smooth{filter_radius}_weighted_halo', f's8_p_smooth{filter_radius}_weighted_halo',
                  f'fiducial_smooth{filter_radius}_notweighted_halo', f's8_m_smooth{filter_radius}_notweighted_halo', f's8_p_smooth{filter_radius}_notweighted_halo']:
        total_histogram = np.zeros(len(bin_centres))
        for phase in range(N):
            total_histogram += np.load(os.path.join(load_folder, f'phase{phase}_{title}.npy'))
            if phase % 10 == 0: print(phase, title)
        np.save(os.path.join(save_folder, f'{title}_ALL.npy'), total_histogram)
    """


    """
    # Generates the values of dlnn/dlnsigma8 based on the counts of objects within the density thresholds depending on the s8 variation
    bin_edges = np.arange(-1, 500, 0.0001)
    bin_centres = 1/2*(bin_edges[1:]+bin_edges[:-1])
    for title in ['nosmooth_matter_ALL', 'smooth10_matter_ALL', 'smooth10_weighted_halo_ALL', 'smooth10_notweighted_halo_ALL']:
        fiducial_histogram = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect_merged/fiducial_{title}.npy')
        s8_m_histogram = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect_merged/s8_m_{title}.npy')
        s8_p_histogram = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/indirect_merged/s8_p_{title}.npy')
        # finds the average thresholds for the fiducial quantiles across all realizations
        percentile_cutoffs_fiducial = percentile_from_histogram(fiducial_histogram, bin_centres, [0.2, 0.4, 0.6, 0.8])
        print(title, 'percentile cutoffs', percentile_cutoffs_fiducial)
        # obtains the bin indices associated with these thresholds
        bin0_ind_fiducial = bin_centres <= percentile_cutoffs_fiducial[0]
        bin1_ind_fiducial = (bin_centres > percentile_cutoffs_fiducial[0]) & (bin_centres <= percentile_cutoffs_fiducial[1])
        bin2_ind_fiducial = (bin_centres > percentile_cutoffs_fiducial[1]) & (bin_centres <= percentile_cutoffs_fiducial[2])
        bin3_ind_fiducial = (bin_centres > percentile_cutoffs_fiducial[2]) & (bin_centres <= percentile_cutoffs_fiducial[3])
        bin4_ind_fiducial = bin_centres > percentile_cutoffs_fiducial[3]
        # finds the counts of objects in each bin and computes the ratio directly
        s8_m_histogram_sums = np.array([np.sum(s8_m_histogram[ind]) for ind in [bin0_ind_fiducial, bin1_ind_fiducial, bin2_ind_fiducial, bin3_ind_fiducial, bin4_ind_fiducial]])#[bin0_ind_s8_m, bin1_ind_s8_m, bin2_ind_s8_m, bin3_ind_s8_m, bin4_ind_s8_m]])
        s8_p_histogram_sums = np.array([np.sum(s8_p_histogram[ind]) for ind in [bin0_ind_fiducial, bin1_ind_fiducial, bin2_ind_fiducial, bin3_ind_fiducial, bin4_ind_fiducial]])#[bin0_ind_s8_p, bin1_ind_s8_p, bin2_ind_s8_p, bin3_ind_s8_p, bin4_ind_s8_p]])
        print(title, 'scale-dependent bias coefficient', np.log(s8_p_histogram_sums/s8_m_histogram_sums)/np.log(0.849/0.819))
    """