import os
import time
import readfof
import argparse
import numpy as np
from pypower import CatalogFFTPower
from densitysplit.pipeline import DensitySplit


def get_halo_positions(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, 
                       space, los):
    """Retrieves halo positions in real or redshift space.

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
    # Applies redshift space distortions if needed
    if space == 'r':
        return pos
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
        return pos
    
def get_quantile_positions(halo_positions, n_quantiles, filter_type, filter_radius, 
                           n_randoms, boxsize, cellsize_ds, cellsize_lattice, query_type, 
                           resampler, interlacing, compensate, sim_index):
    """Retrieves quantile positions based on smoothed halo field.

    Args:
        halo_positions (array): The halo positions (dimension (N, 3)) where N is the 
                                number of halos.
        n_quantiles (int): The number of quantiles.
        filter_type (str): The smoothing filter type ('TopHat', 'Gaussian').
        filter_radius (float): The smoothing filter radius (Mpc/h).
        n_randoms (int): The number of random query positions as a multiple of the 
                         total number of halos (or None if lattice positions used).
        boxsize (float): The boxsize of the simulation volume (Mpc/h).
        cellsize_ds (float): The size of the mesh cells used to obtain the smoothed
                             density field (Mpc/h).
        cellsize_lattice (float): The size of the mesh cells used to obtain the 
                                  lattice query positions (or None if random positions
                                  are applied) (Mpc/h).
        query_type (str): The type of query positions ('random', 'lattice', 'halo').
        resampler (str): The resampler ('ngp', 'cic', 'tsc', 'pcs').
        interlacing (int): The interlacing order (0, 2).
        compensate (bool): Whether to apply compensation.
        sim_index (int): The index of the simulation.

    Returns:
        list: The list of the quantiles, each of which is an array of positions.
    """
    # Initiates random seed to be different for each simulation
    np.random.seed(sim_index)
    if query_type == 'random':
        # Absolute number of randoms
        n_randoms_abs = int(n_randoms*len(halo_positions))
        query_positions = np.random.uniform(0, boxsize, (n_randoms_abs, 3))
    elif query_type == 'lattice':
        # Assigns 3d lattice with equally spaced query points
        edges = np.arange(0, boxsize+cellsize_lattice, cellsize_lattice)
        centres = 1/2*(edges[:-1]+edges[1:])
        lattice_x, lattice_y, lattice_z = np.meshgrid(centres, centres, centres)
        lattice_x = lattice_x.flatten()
        lattice_y = lattice_y.flatten()
        lattice_z = lattice_z.flatten()
        query_positions = np.vstack((lattice_x, lattice_y, lattice_z)).T
        # Shuffles positions randomly so no preferential ordering
        np.random.shuffle(query_positions)
    elif query_type == 'halo':
        # Halo positions themselves used
        query_positions = halo_positions
        np.random.shuffle(query_positions)
    # Initiates density-split object
    ds_obj = DensitySplit(data_positions=halo_positions, boxsize=boxsize)
    # Gets density at the query locations using meshgrid with finite cells
    ds_obj.get_density_mesh(smooth_radius=filter_radius, cellsize=cellsize_ds,
                            sampling_positions=query_positions,
                            filter_shape=filter_type, resampler=resampler,
                            interlacing=interlacing, compensate=compensate)
    # Splits query positions into quantiles based on density
    quantiles = ds_obj.get_quantiles(nquantiles=n_quantiles)
    return quantiles

def generate_statistics(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, 
                        los_dirs, n_quantiles, filter_type, filter_radius, n_randoms, nmesh, 
                        query_type, resampler, interlacing, compensate, sim_index, k_edges,
                        output_folder):
    """Generates the relevant power spectrum results (halo autocorrelation, quantile-halo
       cross-correlation and quantile autocorrelation) and stores away to file.

    Args:
        halo_path (str): The folder path of the snapshot.
        boxsize (float): The boxsize of the simulation volume (Mpc/h).
        snapnum (int): The snapshot corresponding to given redshift 
                       {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}.
        redshift (float): The redshift corresponding to the given snapshot.
        omega_m (float): The omega matter parameter.
        min_mass (float): The mass cut to apply to the halos (Msun/h).
        space (str): Real ('r') or redshift ('z') space.
        los_dirs (list): The line-of-sight directions ('x', 'y', 'z').
        n_quantiles (int): The number of quantiles.
        filter_type (str): The smoothing filter type ('TopHat', 'Gaussian').
        filter_radius (float): The smoothing filter radius (Mpc/h).
        n_randoms (int): The number of random query positions as a multiple of the 
                         total number of halos (or None if lattice positions used).
        nmesh (int): The number of mesh cells per dimension of grid.
        query_type (str): The type of query positions ('random', 'lattice', 'halo').
        resampler (str): The resampler ('ngp', 'cic', 'tsc', 'pcs').
        interlacing (int): The interlacing order (0, 2).
        compensate (bool): Whether to apply compensation.
        sim_index (int): The index of the simulation.
        k_edges (array): The bin edges for wavenumber (h/Mpc).
        output_folder (str): The folder path to store the results.
    """
    # Cellsize to use for density-split and power spectrum mesh grids
    cellsize_ds = boxsize/nmesh
    cellsize_lattice = boxsize/nmesh
    # Dictionary to store results for each line of sight direction
    if los_dirs == ['z']:
        result = {'z':{}}
    elif los_dirs == ['x', 'y', 'z']:
        result = {'x':{}, 'y':{}, 'z':{}}
    # Keeps track of the time taken to run each portion of the code
    time_process_halos = 0
    time_densitysplit = 0
    time_power = 0
    # Iterates through each line-of-sight direction
    for los in los_dirs:
        t0 = time.time()
        # Retrieves halo positions
        halo_positions = get_halo_positions(
            halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los)
        time_process_halos += time.time()-t0
        t0 = time.time()
        # Retrieves quantile positions
        ds_quantiles = get_quantile_positions(
            halo_positions, n_quantiles, filter_type, filter_radius, n_randoms, boxsize,
            cellsize_ds, cellsize_lattice, query_type, resampler, interlacing, compensate,
            sim_index)
        time_densitysplit += time.time()-t0
        t0 = time.time()
        # Halo autocorrelation
        result[los]['h-h'] = CatalogFFTPower( # Uses 'CIC' for halos for consistency with others
            data_positions1=halo_positions, edges=k_edges, ells=(0,2,4), los=los, nmesh=nmesh, 
            boxsize=boxsize, resampler='cic', interlacing=0, position_type='pos')
        # Iterates through the different quantiles
        for i in range(n_quantiles):
            # Quantile-halo cross-correlation
            result[los][f'{i+1}-h'] = CatalogFFTPower(
                data_positions1=ds_quantiles[i], data_positions2=halo_positions, edges=k_edges,
                ells=(0,2,4), los=los, nmesh=nmesh, boxsize=boxsize, resampler=resampler,
                interlacing=interlacing, position_type='pos')
            # Quantile autocorrelation
            if query_type == 'lattice':
                shotnoise=0
            else:
                shotnoise=None
            result[los][f'{i+1}-{i+1}'] = CatalogFFTPower(
                data_positions1=ds_quantiles[i], data_positions2=None, edges=k_edges, ells=(0,2,4),
                los=los, nmesh=nmesh, boxsize=boxsize, resampler=resampler, interlacing=interlacing,
                position_type='pos', shotnoise=shotnoise)
        time_power += time.time()-t0
    # Adds different times to dictionary
    result['TIME_PROCESS_HALOS'] = time_process_halos
    result['TIME_DENSITYSPLIT'] = time_densitysplit
    result['TIME_POWER'] = time_power
    # Saves away functions to file
    np.save(
        os.path.join(
        output_folder,
        f'phase{sim_index}_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}_{min_mass}_{space}split_{resampler}_{interlacing}_{compensate}.npy'),
        result)
    return None


if __name__ == '__main__':

    # Since doing parallel processing, specifies the particular range of 
    # simulations to run through (array jobs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    start_idx = args.start_idx
    n = args.n
    sim_indices = np.arange(start_idx, start_idx+n)

    # Parameters for density-split and power spectrum algorithms
    boxsize = 1000
    snapnum = 4
    redshift = 0
    space = 'z'
    filter_type = 'Gaussian'
    nmesh = 512
    resampler = 'tsc'
    interlacing = 0
    compensate=True
    k_edges = np.arange(2*np.pi/boxsize, np.pi/(boxsize/nmesh), 2*np.pi/boxsize)
    # Iterates through different combinations of the hyperparameters
    for hyperparameters in [(5, 10, None, 'lattice'), 
                            (3, 10, None, 'lattice'), 
                            (7, 10, None, 'lattice'),
                            (5, 7, None, 'lattice'),
                            (5, 13, None, 'lattice'),
                            (5, 10, 5, 'random')]:
        n_quantiles, filter_radius, n_randoms, query_type = hyperparameters
        # Iterates through the different parameter variations
        for variation in ['fiducial']:
            # Selects appropriate omega matter parameter
            if variation == 'Om_m':
                omega_m = 0.3075
            elif variation == 'Om_p':
                omega_m = 0.3275
            else:
                omega_m = 0.3175
            # Selects appropriate mass cut parameter
            if variation == 'Mmin_3.1e13':
                min_mass = 3.1e13
            elif variation == 'Mmin_3.3e13':
                min_mass = 3.3e13
            else:
                min_mass = 3.2e13
            if variation == 'fiducial':
                los_dirs = ['z']
            else:
                los_dirs = ['x', 'y', 'z']
            # Selects appropraite halo folder label
            if variation in ['fiducial', 'Mmin_3.1e13', 'Mmin_3.3e13']:
                halo_folder_label = 'fiducial'
            else:
                halo_folder_label = variation
            # Folders where the halos and output are/will be stored
            halo_folder = f'/home/jgmorawe/scratch/quijote/{halo_folder_label}'
            output_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/{variation}'
            # Iterates through the different simulation indices
            for sim_index in sim_indices:
                halo_path = os.path.join(halo_folder, f'{sim_index}')
                generate_statistics(
                    halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los_dirs, n_quantiles, filter_type,
                    filter_radius, n_randoms, nmesh, query_type, resampler, interlacing, compensate, sim_index,
                    k_edges, output_folder)
