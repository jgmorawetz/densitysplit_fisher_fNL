import os
import time
import readfof
import argparse
import numpy as np
import matplotlib.pyplot as plt
from densitysplit.pipeline import DensitySplit
from nbodykit.lab import ArrayCatalog, ArrayMesh, FFTPower


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
    
def get_quantile_positions(halo_info, mass_weighted, n_quantiles, overdensity_thresholds, 
                           filter_type, filter_radius, n_randoms, boxsize, cellsize_ds, cellsize_lattice, 
                           query_type, resampler, interlacing, compensate, sim_index):
    """Retrieves quantile positions based on smoothed halo field.

    Args:
        halo_info (list): The halo positions (dimension (N, 3)) where N is the number of halos,
                          and the masses of the halos.
        mass_weighted (bool): Whether to weight the halo field according to mass (True) or
                              just use number density (False).
        n_quantiles (int): The number of quantiles (set to None if fixed thresholds are used).
        overdensity_thresholds (list): The overdensity thresholds to use to split into
                                       bins (set to None if quantiles are used).
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
        n_randoms_abs = int(n_randoms*len(halo_info[0]))
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
        query_positions = halo_info[0]
        np.random.shuffle(query_positions)
    # Initiates density-split object
    if mass_weighted:
        ds_obj = DensitySplit(
            data_positions=halo_info[0], boxsize=boxsize, data_weights=halo_info[1])
    else:
        ds_obj = DensitySplit(
            data_positions=halo_info[0], boxsize=boxsize)
    # Gets density at the query locations using meshgrid with finite cells
    ds_obj.get_density_mesh(smooth_radius=filter_radius, cellsize=cellsize_ds,
                            sampling_positions=query_positions,
                            filter_shape=filter_type, resampler=resampler,
                            interlacing=interlacing, compensate=compensate)
    # Splits query positions into quantiles based on density
    if overdensity_thresholds == None:
        quantiles = ds_obj.get_quantiles(nquantiles=n_quantiles)
    else: # if fixed thresholds are used instead
        n_quantiles = len(overdensity_thresholds)+1
        quantiles = []
        for i in range(n_quantiles):
            if i == 0:
                quantiles.append(ds_obj.sampling_positions[ds_obj.density <= overdensity_thresholds[i]])
            elif i == n_quantiles-1:
                quantiles.append(ds_obj.sampling_positions[ds_obj.density > overdensity_thresholds[i-1]])
            else:
                quantiles.append(ds_obj.sampling_positions[(ds_obj.density > overdensity_thresholds[i-1]) &
                                                           (ds_obj.density <= overdensity_thresholds[i])])
    return quantiles


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

    # Generates meshes of the halo and quantile fields and measures cross power spectra with matter field
    boxsize = 1000
    snapnum = 4
    redshift = 0
    omega_m = 0.3175
    space = 'r'
    los = 'z'
    nmesh_ds = 512
    nmesh_power = 256
    filter_type = 'Gaussian'
    n_randoms = None
    resampler = 'cic'
    interlacing = 0
    compensate = True
    kmin = 2*np.pi/boxsize
    kmax = np.pi/(boxsize/nmesh_power)
    dk = np.pi/boxsize
    hyperparameters = [(False, 5, None, 10, 'lattice', 3.2e13), # baseline DSC settings
                       (True, None, [0.365, 1.305, 2.605, 4.995], 5, 'halo', 1e13)] # modifications (thresholds manually tested by averaging several realizations of overdensity distribution code)
    # iterates through the different hyperparameter combinations
    for hyperparameter in hyperparameters:
        mass_weighted, n_quantiles, overdensity_thresholds, filter_radius, query_type, min_mass = hyperparameter
        for variation in ['fiducial', 'LC_m', 'LC_p']:
            matter_folder = f'/home/jgmorawe/scratch/matter_fields/{variation}'
            halo_folder = f'/home/jgmorawe/scratch/quijote/{variation}'
            save_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/{variation}'
            for sim_index in sim_indices:
                matter_path = os.path.join(matter_folder, f'{sim_index}/df_m_256_CIC_z=0.npy')
                halo_path = os.path.join(halo_folder, f'{sim_index}')
                # retrieves the halo positions and associated quantile positions
                halo_info = get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los)
                quantile_positions = get_quantile_positions(halo_info, mass_weighted, n_quantiles, overdensity_thresholds, filter_type, 
                                                            filter_radius, n_randoms, boxsize, boxsize/nmesh_ds, boxsize/nmesh_ds, query_type,
                                                            'tsc', interlacing, compensate, sim_index)
                # generates mesh grids for the matter, halo and quantile fields
                matter_mesh = ArrayMesh(np.load(matter_path), BoxSize=boxsize)
                halo_array = np.empty(len(halo_info[0]), dtype=[('Position', ('f8', 3))]); halo_array['Position'] = halo_info[0]
                halo_mesh = ArrayCatalog(data=halo_array).to_mesh(Nmesh=nmesh_power, BoxSize=boxsize, interlaced=False, 
                                                                  resampler=resampler)
                if n_quantiles == None:
                    N_QUANTILES = len(overdensity_thresholds)+1
                else:
                    N_QUANTILES = n_quantiles
                quantile_meshes = []
                for i in range(N_QUANTILES):
                    quantile_array = np.empty(len(quantile_positions[i]), dtype=[('Position', ('f8', 3))]); quantile_array['Position'] = quantile_positions[i]
                    quantile_mesh = ArrayCatalog(data=quantile_array).to_mesh(Nmesh=nmesh_power, BoxSize=boxsize, interlaced=False,
                                                                              resampler=resampler)
                    quantile_meshes.append(quantile_mesh)
                # obtains power spectrum for the matter auto, halo-matter cross, and quantile-matter cross
                matter_power = FFTPower(first=matter_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]),
                                        dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                halo_matter_power = FFTPower(first=halo_mesh, second=matter_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]),
                                             dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                halo_power = FFTPower(first=halo_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]), dk=dk, kmin=kmin, 
                                      kmax=kmax, poles=[0])
                quantile_matter_powers = []
                for i in range(N_QUANTILES):
                    quantile_matter_power = FFTPower(first=quantile_meshes[i], second=matter_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize,
                                                    los=np.array([0, 0, 1]), dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                    quantile_matter_powers.append(quantile_matter_power)
                quantile_powers = []
                for i in range(N_QUANTILES):
                    quantile_power = FFTPower(first=quantile_meshes[i], mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]), dk=dk,
                                              kmin=kmin, kmax=kmax, poles=[0])
                    quantile_powers.append(quantile_power)
                results = {}
                results['Matter'] = matter_power
                results['Halo-Matter'] = halo_matter_power
                results['Halo'] = halo_power
                results['Quantile-Matter'] = quantile_matter_powers
                results['Quantile'] = quantile_powers
                # saves away the results to file
                np.save(os.path.join(save_folder, f'sim{sim_index}_{mass_weighted}_{n_quantiles}_{filter_radius}_{query_type}_{min_mass}.npy'), results)
                print(hyperparameters, variation, sim_index, 'done')