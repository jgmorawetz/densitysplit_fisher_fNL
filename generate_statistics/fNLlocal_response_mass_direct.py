import os
import time
import readfof
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import ArrayCatalog, ArrayMesh, FFTPower


def get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, max_mass, space, los):
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
    mass_cut = (mass >= min_mass) & (mass < max_mass)
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
    nmesh_power = 256
    resampler = 'cic'
    interlacing = 0
    compensate = True
    kmin = 2*np.pi/boxsize
    kmax = np.pi/(boxsize/nmesh_power)
    dk = np.pi/boxsize
    mass_ranges = [(13, 13.5), (13.5, 14), (14, 14.5), (14.5, 15)]
    # iterates through the different hyperparameter combinations
    for mass_range in mass_ranges:
        min_mass, max_mass = 10**mass_range[0], 10**mass_range[1]
        for variation in ['fiducial', 'LC_m', 'LC_p']:
            matter_folder = f'/home/jgmorawe/scratch/matter_fields/{variation}'
            halo_folder = f'/home/jgmorawe/scratch/quijote/{variation}'
            save_folder = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/{variation}_halomass'
            for sim_index in sim_indices:
                matter_path = os.path.join(matter_folder, f'{sim_index}/df_m_256_CIC_z=0.npy')
                halo_path = os.path.join(halo_folder, f'{sim_index}')
                # retrieves the halo positions and associated quantile positions
                halo_info = get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, max_mass, space, los)
                # generates mesh grids for the matter, halo and quantile fields
                matter_mesh = ArrayMesh(np.load(matter_path), BoxSize=boxsize)
                halo_array = np.empty(len(halo_info[0]), dtype=[('Position', ('f8', 3))]); halo_array['Position'] = halo_info[0]
                halo_mesh = ArrayCatalog(data=halo_array).to_mesh(Nmesh=nmesh_power, BoxSize=boxsize, interlaced=False, 
                                                                  resampler=resampler)
                # obtains power spectrum for the matter auto and halo-matter cross
                matter_power = FFTPower(first=matter_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]),
                                        dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                halo_matter_power = FFTPower(first=halo_mesh, second=matter_mesh, mode='1d', Nmesh=nmesh_power, BoxSize=boxsize, los=np.array([0, 0, 1]),
                                             dk=dk, kmin=kmin, kmax=kmax, poles=[0])
                results = {}
                results['Matter'] = matter_power
                results['Halo-Matter'] = halo_matter_power
                # saves away the results to file
                np.save(os.path.join(save_folder, f'sim{sim_index}_{mass_range[0]}_{mass_range[1]}.npy'), results)
                print(min_mass, max_mass, variation, sim_index, 'DONE')
    
    
#    for mass_range in [(13, 13.5), (13.5, 14), (14, 14.5), (14.5, 15)]:
#        for variation in ['fiducial', 'LC_m', 'LC_p']:
#            results = {'Matter':np.zeros((500, 253)), 'Halo-Matter':np.zeros((500, 253)), 'k_avg':np.zeros(253)}
#            for i in range(500):
#                data = np.load(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/fNLloc_functions/{variation}_halomass/sim{i}_{mass_range[0]}_{mass_range[1]}.npy', allow_pickle=True).item()
#                results['Matter'][i] = data['Matter'].power['power'].real
#                results['Halo-Matter'][i] = data['Halo-Matter'].power['power'].real
#                results['k_avg'] = data['Matter'].power['k']
#                if i % 10 == 0:
#                    print(mass_range, variation, i, 'DONE')
#            np.save(f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/halomass_{variation}_{mass_range[0]}_{mass_range[1]}.npy', results)
    

    