import os
import time
import readfof
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.table import Table
from densitysplit.pipeline import DensitySplit


def get_halo_info(halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los):
    """Retrieves halo positions and masses in real or redshift space.

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

    # Makes plots for the first simulation index
    phase = 0
    boxsize = 1000
    nmesh = 512
    cellsize_ds = boxsize/nmesh
    cellsize_lattice = boxsize/nmesh/2
    snapnum = 4
    redshift = 0
    omega_m = 0.3175
    min_mass = 3.2e13
    filter_type = 'Gaussian'
    filter_radius = 10
    filter_radius_eff = filter_radius*np.sqrt(5) # converts to effective TopHat radius
    n_quantiles = 5
    halo_folder = '/home/jgmorawe/scratch/quijote/fiducial'
    plot_folder = '/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/quijote/plot_results'
    halo_path = os.path.join(halo_folder, f'{phase}')
    space = 'z'
    los = 'z'
    halo_positions, halo_masses = get_halo_info(
        halo_path, boxsize, snapnum, redshift, omega_m, min_mass, space, los)
    halo_positions_x = halo_positions[:, 0]
    halo_positions_y = halo_positions[:, 1]
    halo_positions_z = halo_positions[:, 2]
    # Slices for halos found within a 500x500 X-Y cross section and found within twice the
    # smoothing radius in the z direction
    halo_positions_slice = halo_positions[(halo_positions_x >= 0) & (halo_positions_x <= 500) &
                                          (halo_positions_y >= 0) & (halo_positions_y <= 500) &
                                          (halo_positions_z >= 500-filter_radius_eff) &
                                          (halo_positions_z <= 500+filter_radius_eff)]
    edges = np.arange(0, boxsize+cellsize_lattice, cellsize_lattice)
    centres = 1/2*(edges[:-1]+edges[1:])
    x_cells, y_cells = np.meshgrid(centres, centres)
    x_cells = x_cells.flatten()
    y_cells = y_cells.flatten()
    z_cells = 500*np.ones(len(x_cells)) # sets z=500 (middle of box) for X-Y cross section
    query_positions = np.vstack((x_cells, y_cells, z_cells)).T
    ds_object = DensitySplit(data_positions=halo_positions, boxsize=boxsize, data_weights=None)
    density = ds_object.get_density_mesh(smooth_radius=filter_radius, cellsize=cellsize_ds,
                                         sampling_positions=query_positions, filter_shape=filter_type)
    # Finds quantiles associated with the density values with which to split into groups
    cutoffs = np.percentile(density, np.arange(0, 100+100/n_quantiles, 100/n_quantiles))
    overdensity_label = np.zeros(len(density))
    for i in range(len(overdensity_label)):
        for j in range(1, n_quantiles+1):
            if density[i] <= cutoffs[j]:
                overdensity_label[i] = j
                break
    # Now that each pixel has an overdensity group we can reshape back to meshgrid format and plot
    overdensity_label = overdensity_label.reshape(len(centres), len(centres))
    density = density.reshape(len(centres), len(centres))
    # Slices to only plot a 500x500 X-Y cross section
    dim = np.shape(overdensity_label)[0]
    overdensity_label = overdensity_label[0:dim//2, 0:dim//2]
    overdensity_label = np.flip(overdensity_label, axis=0) # flips so x runs left to right, y bottom to top
    density = density[0:dim//2, 0:dim//2]
    density = np.flip(density, axis=0)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, dpi=500, figsize=(7, 3.5))
    fig.subplots_adjust(wspace=0.075, left=0.09, right=0.97, top=0.97, bottom=0.13)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].imshow(X=density, cmap='turbo', extent=(0, 500, 0, 500), alpha=0.6)
    ax[0].plot(halo_positions_slice[:, 0], halo_positions_slice[:, 1], 'o', color='black', markersize=1)
    im = ax[1].imshow(X=overdensity_label, cmap='turbo', extent=(0, 500, 0, 500))
    colors = [im.cmap(im.norm(value)) for value in np.arange(1, n_quantiles+1)]
    patches = [mpatches.Patch(color=colors[i], label='Q{}'.format(np.arange(1, n_quantiles+1)[i])) for 
               i in range(len(np.arange(1, n_quantiles+1)))]
    ax[0].set_xlabel(r'$\mathrm{x} \ [h^{-1}\mathrm{Mpc}]$')
    ax[0].set_ylabel(r'$\mathrm{y} \ [h^{-1}\mathrm{Mpc}]$')
    ax[1].set_xlabel(r'$\mathrm{x} \ [h^{-1}\mathrm{Mpc}]$')
    ax[1].legend(handles=patches)
    fig.savefig(os.path.join(plot_folder, f'phase{phase}_cross_section.pdf'))