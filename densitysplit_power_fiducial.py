import os
import time
import readfof
import argparse
import numpy as np
from astropy.table import Table
from pypower import CatalogFFTPower
from densitysplit.pipeline import DensitySplit


# generates power spectra using the density-split algorithm for the fiducial Quijote mocks


def process_halo_catalog(halo_path, boxsize, snapnum, redshift, omega_m, min_mass):
    """Reads in the halo catalog and outputs an astropy table with all the halo
       positions, velocities and masses."""
    # LCDM parameters
    omega_l = 1 - omega_m
    H0 = 100
    # reads in unprocessed data
    data = readfof.FoF_catalog(halo_path, snapnum, long_ids=False, swap=False,
                               SFR=False, read_IDs=False)
    # extracts positions, velocities and masses
    pos = data.GroupPos/1e3
    vel = data.GroupVel*(1+redshift)
    mass = data.GroupMass*1e10
    # applies a mass cut
    mass_cut = mass > min_mass
    pos = pos[mass_cut]
    vel = vel[mass_cut]
    mass = mass[mass_cut]
    # extracts x,y,z positions and vx,vy,vz velocities
    xpos, ypos, zpos = pos[:, 0], pos[:, 1], pos[:, 2]
    xvel, yvel, zvel = vel[:, 0], vel[:, 1], vel[:, 2]
    # applies RSDs to the positions
    az = 1/(1+redshift)
    Hz = H0*np.sqrt(omega_m*(1+redshift)**3+omega_l)
    xpos_rsd = xpos + xvel/(Hz*az)
    ypos_rsd = ypos + yvel/(Hz*az)
    zpos_rsd = zpos + zvel/(Hz*az)
    # enforces periodic boundary conditions in case points placed outside box
    xpos_rsd = xpos_rsd % boxsize
    ypos_rsd = ypos_rsd % boxsize
    zpos_rsd = zpos_rsd % boxsize
    # outputs table which contains the positions, velocities and masses
    output = np.vstack((xpos, ypos, zpos, xpos_rsd, ypos_rsd, zpos_rsd, xvel, yvel, zvel, mass)).T
    output = Table(output, names=['X', 'Y', 'Z', 'X_RSD', 'Y_RSD', 'Z_RSD', 'VX', 'VY', 'VZ', 'MASS'])
    return output

def get_halo_positions(halo_table, space, los):
    """Takes in the processed halo table and outputs the halo positions, either
       in real or redshift space. If in redshift space, uses the specified line
       of sight direction."""
    # extracts the real/redshift space positions
    xpos = np.array(halo_table['X'])
    ypos = np.array(halo_table['Y'])
    zpos = np.array(halo_table['Z'])
    xpos_rsd = np.array(halo_table['X_RSD'])
    ypos_rsd = np.array(halo_table['Y_RSD'])
    zpos_rsd = np.array(halo_table['Z_RSD'])
    if space == 'r': # real space (no LOS direction needed)
        pos = np.vstack((xpos, ypos, zpos)).T
    elif space == 'z': # redshift space
        if los == 'x':
            pos = np.vstack((xpos_rsd, ypos, zpos)).T
        elif los == 'y':
            pos = np.vstack((xpos, ypos_rsd, zpos)).T
        elif los == 'z':
            pos = np.vstack((xpos, ypos, zpos_rsd)).T
    return pos

def get_quantile_positions(halo_positions, n_quantiles, filter_type, filter_radius,
                           n_randoms, boxsize, cellsize_ds, cellsize_lattice,
                           sim_index, query_type, resampler, interlacing, compensate):
    """Applies the density-split algorithm on the halo positions to generate quantile
       query positions (can specify either random positions, lattice/equally spaced
       positions or the halo positions themselves, depending on the context)."""
    # initiates random seed to be different for any given simulation
    np.random.seed(sim_index)
    if query_type == 'random':
        # computes the absolute number of randoms needed (as a multiple of the 
        # number of halos)
        n_randoms_abs = int(n_randoms*len(halo_positions))
        query_positions = np.random.uniform(0, boxsize, (n_randoms_abs, 3))
    elif query_type == 'lattice':
        # assigns a 3d lattice with equally spaced query points
        edges = np.arange(0, boxsize+cellsize_lattice, cellsize_lattice)
        centres = 1/2 * (edges[:-1] + edges[1:])
        lattice_x, lattice_y, lattice_z = np.meshgrid(centres, centres, centres)
        lattice_x = lattice_x.flatten()
        lattice_y = lattice_y.flatten()
        lattice_z = lattice_z.flatten()
        query_positions = np.vstack((lattice_x, lattice_y, lattice_z)).T
        # shuffles up the query positions randomly so that when densities are
        # computed and divided into bins there is no preferential ordering
        np.random.shuffle(query_positions)
    elif query_type == 'halo':
        # the halo positions themselves are used for query positions
        query_positions = halo_positions
        np.random.shuffle(query_positions)
    # initiates density-split object
    ds_obj = DensitySplit(data_positions=halo_positions, boxsize=boxsize)
    # gets density at the query locations using a meshgrid with finite cells
    ds_obj.get_density_mesh(smooth_radius=filter_radius, cellsize=cellsize_ds,
                            sampling_positions=query_positions, 
                            filter_shape=filter_type, resampler=resampler, 
                            interlacing=interlacing, compensate=compensate)
    # splits query positions into quantiles based on density
    quantiles = ds_obj.get_quantiles(nquantiles=n_quantiles)
    return quantiles

def compute_statistics(halo_table, space, los_dirs, n_quantiles, filter_type, filter_radius, n_randoms, 
                       boxsize, nmesh, sim_index, query_type, k_edges, resampler, interlacing, compensate, 
                       redshift, output_folder):
    """Takes in the processed halo table and the desired parameters, and outputs to file the requested
       density-split functions in redshift-space (power spectra)."""
    # the cellsizes to use for the density-split and power spectrum mesh grids
    cellsize_ds = boxsize/nmesh
    cellsize_lattice = boxsize/nmesh
    result = {'z':{}} # dictionary to store results for each line of sight direction (only z in this case)
    # keeps track of the time taken to run each portion of the code
    time_process_halos = 0
    time_densitysplit = 0
    time_power = 0
    for los in los_dirs: # iterates through each line of sight direction
        t0 = time.time()
        halo_positions = get_halo_positions(halo_table, space, los)
        time_process_halos += time.time() - t0
        t0 = time.time()
        ds_quantiles = get_quantile_positions(
            halo_positions, n_quantiles, filter_type, filter_radius, n_randoms, boxsize, cellsize_ds, 
            cellsize_lattice, sim_index, query_type, resampler, interlacing, compensate)
        time_densitysplit += time.time() - t0
        t0 = time.time()
        # halo autocorrelation (uses cic only for halo functions for consistency with other paper)
        result[los]['h-h'] = CatalogFFTPower(
            data_positions1=halo_positions, data_positions2=None, edges=k_edges, ells=(0, 2, 4), 
            los=los, nmesh=nmesh, boxsize=boxsize, resampler='cic', interlacing=0, 
            position_type='pos')
        for i in range(n_quantiles):
            # quantile autocorrelation
            if query_type == 'lattice':
                shotnoise=0 # no shot noise for quantile autocorrelation if lattice positions used
            else:
                shotnoise=None
            result[los][f'{i+1}-{i+1}'] = CatalogFFTPower(
                data_positions1=ds_quantiles[i], data_positions2=None, edges=k_edges, ells=(0, 2, 4), 
                los=los, nmesh=nmesh, boxsize=boxsize, resampler=resampler, interlacing=interlacing,
                position_type='pos', shotnoise=shotnoise)
            # quantile-halo cross-correlation
            result[los][f'{i+1}-h'] = CatalogFFTPower(
                data_positions1=ds_quantiles[i], data_positions2=halo_positions, edges=k_edges, 
                ells=(0, 2, 4), los=los, nmesh=nmesh, boxsize=boxsize, resampler=resampler,
                interlacing=interlacing, position_type='pos')
        time_power += time.time() - t0
    # adds the different times to the dictionary
    result['TIME_PROCESS_HALOS'] = time_process_halos
    result['TIME_DENSITYSPLIT'] = time_densitysplit
    result['TIME_POWER'] = time_power
    # saves away the functions to file
    np.save(
        os.path.join(
        output_folder, 
        f'phase{sim_index}_{filter_type}_{filter_radius}_{n_quantiles}_{nmesh}_{query_type}_{n_randoms}_{redshift}.npy'),
        result)
    return None
            

if __name__ == '__main__':

    # since doing parallel processing, specifies the particular range of 
    # simulations to run through (array jobs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    start_idx = args.start_idx
    n = args.n
    sim_indices = np.arange(start_idx, start_idx+n)

    # parameters for density-split and power spectrum algorithms
    space = 'z'
    los_dirs = ['z'] # only use one line-of-sight for purposes of covariance matrix
    n_quantiles = 5
    filter_type = 'Gaussian'
    filter_radius = 10
    snapnum = 4
    redshift = 0
    omega_m = 0.3175
    min_mass = 3.2e13
    boxsize = 1000
    nmesh = 512
    k_edges = np.arange(2*np.pi/boxsize, np.pi/(boxsize/nmesh), 2*np.pi/boxsize)
    resampler = 'tsc'
    interlacing = 0
    compensate = True
    variation = 'fiducial'
    halo_folder = f'/home/jgmorawe/scratch/quijote/{variation}'
    output_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/ds_functions/power/{variation}'
    # iterates through different scenarios (lattice, random and halo query positions)
    for parameters in [(None, 'lattice'), (5, 'random'), (None, 'halo')]:
        n_randoms, query_type = parameters
        # iterates through the different simulation indices
        for sim_index in sim_indices:
            halo_path = os.path.join(halo_folder, f'{sim_index}')
            halo_table = process_halo_catalog(halo_path, boxsize, snapnum, redshift, omega_m, min_mass)
            compute_statistics(halo_table, space, los_dirs, n_quantiles, filter_type, filter_radius,
                               n_randoms, boxsize, nmesh, sim_index, query_type, k_edges, resampler,
                               interlacing, compensate, redshift, output_folder)