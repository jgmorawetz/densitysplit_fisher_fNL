import os
import time
import readfof
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    
    sim_indices = np.arange(0, 500)
    boxsize = 1000
    snapnum = 4
    redshift = 0
    omega_m = 0.3175
    mass_cut = 1e13 # below the actual minimum to ensure all halos are received
    space = 'r'
    los = 'z'
    for variation in ['fiducial', 's8_m', 's8_p']:
        halo_folder = f'/home/jgmorawe/scratch/quijote/{variation}'
        save_path = f'/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/{variation}_halomasses.npy'
        all_masses = []
        for sim_index in sim_indices:
            halo_path = os.path.join(halo_folder, f'{sim_index}')
            halo_info = get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, mass_cut, space, los)
            halo_masses = halo_info[1]
            all_masses.append(halo_masses)
        all_masses = np.concatenate(all_masses)
        np.save(save_path, all_masses)