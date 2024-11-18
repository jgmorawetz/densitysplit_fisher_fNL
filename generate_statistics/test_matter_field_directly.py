import os
import time
import argparse
import readfof
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import ArrayCatalog, ArrayMesh, FFTPower


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

def quantile_field(grid, quantile, n_quantiles, thresholds, boxsize):
    """Generates the overdensity mesh field for the quantiles given the smoothed or unsmoothed
       matter or halo overdensity field.

    Args:
        grid (array): The overdensity grid of the matter or halo field.
        quantile (int): The quantile index (starting from 0).
        n_quantiles (int): The total number of quantiles.
        thresholds (list): The overdensity thresholds used to split by environment (can
                           be fixed or allowed to vary depending on the scenario). These
                           exclude the bottom or uppermost boundary.
        boxsize (float): The box size of the simulation volume.

    Returns:
        ArrayMesh: The new overdensity field object associated with pixels associated with a given quantile.
    """
    if quantile == 0:
        new_grid = np.where((grid <= thresholds[quantile]), n_quantiles-1, -1).astype('f8')
    elif quantile == n_quantiles-1:
        new_grid = np.where((grid > thresholds[quantile-1]), n_quantiles-1, -1).astype('f8')
    else:
        new_grid = np.where((grid <= thresholds[quantile]) & (grid > thresholds[quantile-1]),
                            n_quantiles-1, -1).astype('f8')
    new_mesh = ArrayMesh(new_grid, BoxSize=boxsize)
    return new_mesh

def soln(arr1, arr2):
    """Averages the fNL local response between fNL=-100 and fNL=100."""
    avg_soln = 1/2*(arr1-arr2)
    return avg_soln


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
    kmin = 2*np.pi/boxsize
    kmax = np.pi/(boxsize/nmesh_power)
    dk = kmin/2
    K_LEN = 253
    matter_folder = '/home/jgmorawe/scratch/matter_fields'
    halo_folder = '/home/jgmorawe/scratch/quijote'
    n_quantiles = 5
    resampler = 'cic'
    filter_radius = 10
    fixed = False
    save_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct'
    thresholds_nosmoothing_matter = [-0.74003942, -0.58800351, -0.34882247, 0.19935823] # mean percentile cutoffs as calculated from indirect code
    thresholds_smoothing10_matter = [-0.30762673, -0.15279646, 0.01475099, 0.26071456]
    thresholds_smoothing10_halos_weighted = [-0.72337907, -0.46775241, -0.11420917, 0.51917617]
    thresholds_smoothing10_halos_notweighted = [-0.62599347, -0.33496785, 0.01418675, 0.53934997]
    for phase in phases:
        # matter fields (unsmoothed and smoothed)
        fiducial_grid = np.load(os.path.join(matter_folder, 'fiducial', f'{phase}/df_m_256_CIC_z=0.npy'))
        LC_m_grid = np.load(os.path.join(matter_folder, 'LC_m', f'{phase}/df_m_256_CIC_z=0.npy'))
        LC_p_grid = np.load(os.path.join(matter_folder, 'LC_p', f'{phase}/df_m_256_CIC_z=0.npy'))
        smoothed_fiducial_grid = smoothed_overdensity(fiducial_grid, boxsize, filter_radius, nmesh_power)
        smoothed_LC_m_grid = smoothed_overdensity(LC_m_grid, boxsize, filter_radius, nmesh_power)
        smoothed_LC_p_grid = smoothed_overdensity(LC_p_grid, boxsize, filter_radius, nmesh_power)
        # matter meshes
        fiducial_mesh = ArrayMesh(fiducial_grid, BoxSize=boxsize)
        LC_m_mesh = ArrayMesh(LC_m_grid, BoxSize=boxsize)
        LC_p_mesh = ArrayMesh(LC_p_grid, BoxSize=boxsize)
        smoothed_fiducial_mesh = ArrayMesh(smoothed_fiducial_grid, BoxSize=boxsize)
        smoothed_LC_m_mesh = ArrayMesh(smoothed_LC_m_grid, BoxSize=boxsize)
        smoothed_LC_p_mesh = ArrayMesh(smoothed_LC_p_grid, BoxSize=boxsize)
        # obtains halo catalogs
        halo_info_fiducial = get_halo_info(os.path.join(halo_folder, f'fiducial/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        halo_info_LC_m = get_halo_info(os.path.join(halo_folder, f'LC_m/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        halo_info_LC_p = get_halo_info(os.path.join(halo_folder, f'LC_p/{phase}'), boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        # obtains halo fields (smoothed mass weighted and not mass weighted)
        fiducial_halo_grid_weighted = halo_field(halo_info_fiducial, boxsize, nmesh_power, resampler, True)
        LC_m_halo_grid_weighted = halo_field(halo_info_LC_m, boxsize, nmesh_power, resampler, True)
        LC_p_halo_grid_weighted = halo_field(halo_info_LC_p, boxsize, nmesh_power, resampler, True)
        fiducial_halo_grid_notweighted = halo_field(halo_info_fiducial, boxsize, nmesh_power, resampler, False)
        LC_m_halo_grid_notweighted = halo_field(halo_info_LC_m, boxsize, nmesh_power, resampler, False)
        LC_p_halo_grid_notweighted = halo_field(halo_info_LC_p, boxsize, nmesh_power, resampler, False)
        smoothed_fiducial_halo_grid_weighted = smoothed_overdensity(fiducial_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        smoothed_LC_m_halo_grid_weighted = smoothed_overdensity(LC_m_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        smoothed_LC_p_halo_grid_weighted = smoothed_overdensity(LC_p_halo_grid_weighted, boxsize, filter_radius, nmesh_power)
        smoothed_fiducial_halo_grid_notweighted = smoothed_overdensity(fiducial_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        smoothed_LC_m_halo_grid_notweighted = smoothed_overdensity(LC_m_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        smoothed_LC_p_halo_grid_notweighted = smoothed_overdensity(LC_p_halo_grid_notweighted, boxsize, filter_radius, nmesh_power)
        # obtains halo meshes
        fiducial_halo_mesh_weighted = ArrayMesh(fiducial_halo_grid_weighted, BoxSize=boxsize)
        LC_m_halo_mesh_weighted = ArrayMesh(LC_m_halo_grid_weighted, BoxSize=boxsize)
        LC_p_halo_mesh_weighted = ArrayMesh(LC_p_halo_grid_weighted, BoxSize=boxsize)
        fiducial_halo_mesh_notweighted = ArrayMesh(fiducial_halo_grid_notweighted, BoxSize=boxsize)
        LC_m_halo_mesh_notweighted = ArrayMesh(LC_m_halo_grid_notweighted, BoxSize=boxsize)
        LC_p_halo_mesh_notweighted = ArrayMesh(LC_p_halo_grid_notweighted, BoxSize=boxsize)
        smoothed_fiducial_halo_mesh_weighted = ArrayMesh(smoothed_fiducial_halo_grid_weighted, BoxSize=boxsize)
        smoothed_LC_m_halo_mesh_weighted = ArrayMesh(smoothed_LC_m_halo_grid_weighted, BoxSize=boxsize)
        smoothed_LC_p_halo_mesh_weighted = ArrayMesh(smoothed_LC_p_halo_grid_weighted, BoxSize=boxsize)
        smoothed_fiducial_halo_mesh_notweighted = ArrayMesh(smoothed_fiducial_halo_grid_notweighted, BoxSize=boxsize)
        smoothed_LC_m_halo_mesh_notweighted = ArrayMesh(smoothed_LC_m_halo_grid_notweighted, BoxSize=boxsize)
        smoothed_LC_p_halo_mesh_notweighted = ArrayMesh(smoothed_LC_p_halo_grid_notweighted, BoxSize=boxsize)
        # computes power spectra for the different scenarios
        for scenario in ['nosmooth_matter', 'smooth10_matter', 'smooth10_weighted_halo', 'smooth10_notweighted_halo']: # the four different ways in which the quantile field can be defined
            result_fiducial = {'Matter':None, 'Halo-Matter':None, 'Quantile-Matter':[None for i in range(n_quantiles)]}
            result_LC_m = {'Matter':None, 'Halo-Matter':None, 'Quantile-Matter':[None for i in range(n_quantiles)]}
            result_LC_p = {'Matter':None, 'Halo-Matter':None, 'Quantile-Matter':[None for i in range(n_quantiles)]}
            # first calculates the matter (unsmoothed) and halo-matter (unsmoothed) power spectra
            result_fiducial['Matter'] = FFTPower(first=fiducial_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            result_LC_m['Matter'] = FFTPower(first=LC_m_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            result_LC_p['Matter'] = FFTPower(first=LC_p_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            result_fiducial['Halo-Matter'] = FFTPower(first=fiducial_halo_mesh_notweighted, second=fiducial_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            result_LC_m['Halo-Matter'] = FFTPower(first=LC_m_halo_mesh_notweighted, second=LC_m_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            result_LC_p['Halo-Matter'] = FFTPower(first=LC_p_halo_mesh_notweighted, second=LC_p_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0,0,1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            if scenario == 'nosmooth_matter':
                fiducial_mesh_split = fiducial_grid
                LC_m_mesh_split = LC_m_grid
                LC_p_mesh_split = LC_p_grid
                if fixed == False: # depending on whether or not fixed thresholds or quantiles are used
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = (np.percentile(fiducial_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_m_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_p_mesh_split.flatten(), [20, 40, 60, 80]))
                else:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = thresholds_nosmoothing_matter, thresholds_nosmoothing_matter, thresholds_nosmoothing_matter
            elif scenario == 'smooth10_matter':
                fiducial_mesh_split = smoothed_fiducial_grid
                LC_m_mesh_split = smoothed_LC_m_grid
                LC_p_mesh_split = smoothed_LC_p_grid
                if fixed == False:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = (np.percentile(fiducial_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_m_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_p_mesh_split.flatten(), [20, 40, 60, 80]))
                else:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = thresholds_smoothing10_matter, thresholds_smoothing10_matter, thresholds_smoothing10_matter
            elif scenario == 'smooth10_weighted_halo':
                fiducial_mesh_split = smoothed_fiducial_halo_grid_weighted
                LC_m_mesh_split = smoothed_LC_m_halo_grid_weighted
                LC_p_mesh_split = smoothed_LC_p_halo_grid_weighted
                if fixed == False:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = (np.percentile(fiducial_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_m_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_p_mesh_split.flatten(), [20, 40, 60, 80]))
                else:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = thresholds_smoothing10_halos_weighted, thresholds_smoothing10_halos_weighted, thresholds_smoothing10_halos_weighted
            elif scenario == 'smooth10_notweighted_halo':
                fiducial_mesh_split = smoothed_fiducial_halo_grid_notweighted
                LC_m_mesh_split = smoothed_LC_m_halo_grid_notweighted
                LC_p_mesh_split = smoothed_LC_p_halo_grid_notweighted
                if fixed == False:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = (np.percentile(fiducial_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_m_mesh_split.flatten(), [20, 40, 60, 80]), 
                                                                             np.percentile(LC_p_mesh_split.flatten(), [20, 40, 60, 80]))
                else:
                    thresholds_fiducial, thresholds_LC_m, thresholds_LC_p = thresholds_smoothing10_halos_notweighted, thresholds_smoothing10_halos_notweighted, thresholds_smoothing10_halos_notweighted
            # generates quantile fields from the smoothed mesh grids
            for i in range(n_quantiles):
                fiducial_mesh_quantile = quantile_field(fiducial_mesh_split, i, n_quantiles, thresholds_fiducial, boxsize)
                LC_m_mesh_quantile = quantile_field(LC_m_mesh_split, i, n_quantiles, thresholds_LC_m, boxsize)
                LC_p_mesh_quantile = quantile_field(LC_p_mesh_split, i, n_quantiles, thresholds_LC_p, boxsize)
                result_fiducial['Quantile-Matter'][i] = FFTPower(first=fiducial_mesh_quantile, second=fiducial_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                result_LC_m['Quantile-Matter'][i] = FFTPower(first=LC_m_mesh_quantile, second=LC_m_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                result_LC_p['Quantile-Matter'][i] = FFTPower(first=LC_p_mesh_quantile, second=LC_p_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
            if fixed == False:
                np.save(os.path.join(save_folder, f'phase{phase}_fiducial_{scenario}_quantiles.npy'), result_fiducial)
                np.save(os.path.join(save_folder, f'phase{phase}_LC_m_{scenario}_quantiles.npy'), result_LC_m)
                np.save(os.path.join(save_folder, f'phase{phase}_LC_p_{scenario}_quantiles.npy'), result_LC_p)
            else:
                np.save(os.path.join(save_folder, f'phase{phase}_fiducial_{scenario}.npy'), result_fiducial)
                np.save(os.path.join(save_folder, f'phase{phase}_LC_m_{scenario}.npy'), result_LC_m)
                np.save(os.path.join(save_folder, f'phase{phase}_LC_p_{scenario}.npy'), result_LC_p)
            print(phase, scenario, 'DONE')
    

    """
    # Merges all the power spectra into common files for each scenario
    N = 500
    filter_radius = 10
    fixed = False
    K_LEN = 253
    n_quantiles = 5
    load_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct'
    save_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged'
    for scenario in ['nosmooth_matter', 'smooth10_matter', 'smooth10_weighted_halo', 'smooth10_notweighted_halo']:
        result_fiducial = {'Matter':np.zeros((N, K_LEN)), 'Halo-Matter':np.zeros((N, K_LEN)), 'Quantile-Matter':[np.zeros((N, K_LEN)) for i in range(n_quantiles)]}
        result_LC_m = {'Matter':np.zeros((N, K_LEN)), 'Halo-Matter':np.zeros((N, K_LEN)), 'Quantile-Matter':[np.zeros((N, K_LEN)) for i in range(n_quantiles)]}
        result_LC_p = {'Matter':np.zeros((N, K_LEN)), 'Halo-Matter':np.zeros((N, K_LEN)), 'Quantile-Matter':[np.zeros((N, K_LEN)) for i in range(n_quantiles)]}
        for phase in range(N):
            if fixed == False:
                fiducial_data = np.load(os.path.join(load_folder, f'phase{phase}_fiducial_{scenario}_quantiles.npy'), allow_pickle=True).item()
                LC_m_data = np.load(os.path.join(load_folder, f'phase{phase}_LC_m_{scenario}_quantiles.npy'), allow_pickle=True).item()
                LC_p_data = np.load(os.path.join(load_folder, f'phase{phase}_LC_p_{scenario}_quantiles.npy'), allow_pickle=True).item()
            else:
                fiducial_data = np.load(os.path.join(load_folder, f'phase{phase}_fiducial_{scenario}.npy'), allow_pickle=True).item()
                LC_m_data = np.load(os.path.join(load_folder, f'phase{phase}_LC_m_{scenario}.npy'), allow_pickle=True).item()
                LC_p_data = np.load(os.path.join(load_folder, f'phase{phase}_LC_p_{scenario}.npy'), allow_pickle=True).item()
            result_fiducial['Matter'][phase] = fiducial_data['Matter'].power['power'].real
            result_LC_m['Matter'][phase] = LC_m_data['Matter'].power['power'].real
            result_LC_p['Matter'][phase] = LC_p_data['Matter'].power['power'].real
            result_fiducial['Halo-Matter'][phase] = fiducial_data['Halo-Matter'].power['power'].real 
            result_LC_m['Halo-Matter'][phase] = LC_m_data['Halo-Matter'].power['power'].real
            result_LC_p['Halo-Matter'][phase] = LC_p_data['Halo-Matter'].power['power'].real 
            for i in range(n_quantiles):
                result_fiducial['Quantile-Matter'][i][phase] = fiducial_data['Quantile-Matter'][i].power['power'].real
                result_LC_m['Quantile-Matter'][i][phase] = LC_m_data['Quantile-Matter'][i].power['power'].real
                result_LC_p['Quantile-Matter'][i][phase] = LC_p_data['Quantile-Matter'][i].power['power'].real
            if phase % 10 == 0: print(phase, scenario, 'DONE')
        result_fiducial['k_avg'] = fiducial_data['Matter'].power['k']
        result_LC_m['k_avg'] = LC_m_data['Matter'].power['k']
        result_LC_p['k_avg'] = LC_p_data['Matter'].power['k']
        if fixed == False:
            np.save(os.path.join(save_folder, f'{scenario}_fiducial_quantiles_ALL.npy'), result_fiducial)
            np.save(os.path.join(save_folder, f'{scenario}_LC_m_quantiles_ALL.npy'), result_LC_m)
            np.save(os.path.join(save_folder, f'{scenario}_LC_p_quantiles_ALL.npy'), result_LC_p)
        else:   
            np.save(os.path.join(save_folder, f'{scenario}_fiducial_ALL.npy'), result_fiducial)
            np.save(os.path.join(save_folder, f'{scenario}_LC_m_ALL.npy'), result_LC_m)
            np.save(os.path.join(save_folder, f'{scenario}_LC_p_ALL.npy'), result_LC_p)
    """


    """
    # obtains the dln(n)/dln(sigma8) predictions from the code 'test_matter_field_indirectly.py'
    fixed = False
    dlnn_dlnsigma8_nosmooth_matter = np.array([2.24940221, -0.11110986, -0.76148008, -0.95784285, -0.41674286])
    dlnn_dlnsigma8_smooth10_matter = np.array([1.62286659, -0.44305718, -0.89965544, -0.82278199, 0.54402084])
    dlnn_dlnsigma8_smooth10_notweighted_halo = np.array([-0.58322923, 0.04753868, 0.27895424, 0.37382723, -0.11690663])
    dlnn_dlnsigma8_smooth10_weighted_halo = np.array([-0.15674471, 0.0840107, 0.08664489, 0.04099359, -0.05490029])

    # plots the responses (and overlays predicted responses for each of the four scenarios)
    fig,ax=plt.subplots(2, 2, sharex=True, sharey=False, dpi=500, figsize=(8,7))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, hspace=0.13, wspace=0.16)
    for title in ['nosmooth_matter', 'smooth10_matter', 'smooth10_weighted_halo', 'smooth10_notweighted_halo']:
        if title == 'nosmooth_matter': dlnn_dlnsigma8, limits, ax_ind = dlnn_dlnsigma8_nosmooth_matter, (-0.85, 0.7), (0, 0)
        elif title == 'smooth10_matter': dlnn_dlnsigma8, limits, ax_ind = dlnn_dlnsigma8_smooth10_matter, (-0.2, 0.3), (0, 1)
        elif title == 'smooth10_weighted_halo': dlnn_dlnsigma8, limits, ax_ind = dlnn_dlnsigma8_smooth10_weighted_halo, (-0.25, 0.23), (1, 1)
        elif title == 'smooth10_notweighted_halo': dlnn_dlnsigma8, limits, ax_ind = dlnn_dlnsigma8_smooth10_notweighted_halo, (-0.05, 0.27), (1, 0)
        if fixed == False:
            data_fiducial = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_fiducial_quantiles_ALL.npy', allow_pickle=True).item()
            data_LC_m = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_LC_m_quantiles_ALL.npy', allow_pickle=True).item()
            data_LC_p = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_LC_p_quantiles_ALL.npy', allow_pickle=True).item()
        else:
            data_fiducial = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_fiducial_ALL.npy', allow_pickle=True).item()
            data_LC_m = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_LC_m_ALL.npy', allow_pickle=True).item()
            data_LC_p = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/matterfield_tests/direct_merged/{title}_LC_p_ALL.npy', allow_pickle=True).item()
        k = data_fiducial['k_avg']
        k_cut = k <= 0.1
        k_cut2 = k <= 0.03
        k = k[k_cut]
        ax[ax_ind[0]][ax_ind[1]].set_ylim(limits)
        colors = ['mediumblue', 'cornflowerblue', 'grey', 'lightsalmon', 'firebrick']
        linear_biases = [] # appends all the linear biases (as measured by mean ratio on large scales k <= 0.03 h/Mpc)
        for i in range(5):
            linear_biases.append(np.mean(np.mean(data_fiducial['Quantile-Matter'][i][:, k_cut2], axis=0)/np.mean(data_fiducial['Matter'][:, k_cut2], axis=0)))
            if title in ['smooth10_weighted_halo', 'smooth10_notweighted_halo']: # skips DS3 for the bottom two panels since it is too noisy to properly visualize
                if i == 2:
                    continue
            ax[ax_ind[0]][ax_ind[1]].plot(k, soln(np.mean(data_LC_p['Quantile-Matter'][i][:, k_cut], axis=0)/np.mean(data_fiducial['Quantile-Matter'][i][:, k_cut], axis=0),
                                                  np.mean(data_LC_m['Quantile-Matter'][i][:, k_cut], axis=0)/np.mean(data_fiducial['Quantile-Matter'][i][:, k_cut], axis=0)), 
                                                  '-', color=colors[i], linewidth=0.7, label=f'Q{i+1}')
        ratios = dlnn_dlnsigma8 / np.array(linear_biases) # computes bphi/b ratios of the quantiles
        print(title, ratios)
        for i in range(5): # plots the DS1 quantile rescaled according to the relative values of bphi/b for the remaining quantiles
            if i != 0:
                if title in ['smooth10_weighted_halo', 'smooth10_notweighted_halo']:
                    if i == 2:
                        continue
                ax[ax_ind[0]][ax_ind[1]].plot(k, ratios[i]/ratios[0]*soln(np.mean(data_LC_p['Quantile-Matter'][0][:, k_cut], axis=0)/np.mean(data_fiducial['Quantile-Matter'][0][:, k_cut], axis=0),
                                                                          np.mean(data_LC_m['Quantile-Matter'][0][:, k_cut], axis=0)/np.mean(data_fiducial['Quantile-Matter'][0][:, k_cut], axis=0)), 
                                                                          '--', color=colors[i], linewidth=1)
    ax[1][0].plot(k, soln(np.mean(data_LC_p['Halo-Matter'][:, k_cut], axis=0)/np.mean(data_fiducial['Halo-Matter'][:, k_cut], axis=0),
                          np.mean(data_LC_m['Halo-Matter'][:, k_cut], axis=0)/np.mean(data_fiducial['Halo-Matter'][:, k_cut], axis=0)), '-', color='black', linewidth=0.7)
    ax[0][0].plot(k, [None for i in range(len(k))], '-', color='black', linewidth=0.7, label=f'Halo')
    ax[0][0].axhline(y=0, linestyle='--', linewidth=0.2, color='black')
    ax[0][1].axhline(y=0, linestyle='--', linewidth=0.2, color='black')
    ax[1][0].axhline(y=0, linestyle='--', linewidth=0.2, color='black')
    ax[1][1].axhline(y=0, linestyle='--', linewidth=0.2, color='black')
    ax[0][0].set_ylabel(r'$\Delta b(k) f_{\mathrm{NL}}^{\mathrm{loc}} / b$')
    ax[1][0].set_ylabel(r'$\Delta b(k) f_{\mathrm{NL}}^{\mathrm{loc}} / b$')
    ax[1][0].set_xlabel(r'$k \ [h^{-1}\mathrm{Mpc}]$')
    ax[1][1].set_xlabel(r'$k \ [h^{-1}\mathrm{Mpc}]$')    
    ax[0][0].set_title('Unsmoothed, Matter')
    ax[0][1].set_title(r'Smoothed, Matter')
    ax[1][0].set_title(r'Smoothed, Halo, Non-mass weighted')
    ax[1][1].set_title(r'Smoothed, Halo, Mass weighted')
    ax[0][0].set_xscale('log')
    ax[0][0].legend(loc=1)
    fig.savefig('/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results/responses_different_fields.pdf')
    """