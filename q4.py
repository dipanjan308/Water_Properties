import numpy as np
from ase.io import read
from scipy.spatial import KDTree
import pickle
import matplotlib.pyplot as plt

density=1.1603
temperature=400

path = '/leonardo_scratch/fast/IscrB_HPHDQMC/Dipanjan/pure_water/M2_LJ/T400/mol_500/den_1.1603_02'
traj = read(f'{path}/nvt_1.dump', format='lammps-dump-text', index=':100000', specorder=['O', 'H'])

def sample_positions(ase_trajectory, symbol):
    positions = []
    for i in range(len(ase_trajectory)):
        symbols = ase_trajectory[i].get_chemical_symbols()
        indices = [j for j, sym in enumerate(symbols) if sym == symbol]
        positions.append(ase_trajectory[i].get_positions()[indices])
    return np.array(positions)

pos_H = sample_positions(traj, 'H')
pos_O = sample_positions(traj, 'O')

number_of_oxygens = pos_O.shape[1]
number_of_hydrogens = pos_H.shape[1]
number_of_frames = pos_O.shape[0]

box_dim = traj[0].get_cell()
box_dim = np.array(box_dim)
box_dim = box_dim.diagonal()

def tetrahedral_order_vectorized(oxygen_positions, oxygen_tree, box_size):

    distances, neighbors = oxygen_tree.query(oxygen_positions, k = 5)  # k=5 gives the atom itself + 4 neighbors
    neighbors = neighbors[:, 1:]  # Remove self-neighbor (first column)

    # Compute displacement vectors from each atom to its 4 nearest neighbors
    displacements = oxygen_positions[neighbors] - oxygen_positions[:, np.newaxis, :]
    displacements = displacements - box_size * np.rint(displacements / box_size)  # Apply periodic boundary conditions

    # Compute norms of displacement vectors
    norms = np.linalg.norm(displacements, axis = 2)

    # Vectorized computation of pairwise dot products for all atoms
    # Using np.einsum to compute dot products between pairs of displacement vectors
    dot_products = np.einsum('ijk,ilk->ijl', displacements, displacements)

    # Step 5: Calculate cosines of angles between displacement vectors
    # Normalize by the product of norms for all pairs
    norm_products = norms[:, :, np.newaxis] * norms[:, np.newaxis, :]
    cos_theta = dot_products / norm_products

    # We only need the upper triangle (6 unique pairs) for the cosines
    cos_theta_pairs = np.array([
        cos_theta[:, 0, 1], cos_theta[:, 0, 2], cos_theta[:, 0, 3],  # First row (0-1, 0-2, 0-3)
        cos_theta[:, 1, 2], cos_theta[:, 1, 3],  # Second row (1-2, 1-3)
        cos_theta[:, 2, 3]  # Third row (2-3)
    ]).T  # Transpose to get (N, 6) shape

    # Compute tetrahedral order parameter for each atom
    q_values = np.sum((cos_theta_pairs + 1/3) ** 2, axis=1)
    q_values = 1 - (3 / 8) * q_values

    return q_values


q_results = []

for frame_number in range(number_of_frames):

    pos_O_1 = pos_O[frame_number, :, :] # Extracting the positions of the selected frame
    pos_O_wrapped = np.mod(pos_O_1, box_dim)

    Oxygen_wrapped_tree = KDTree(pos_O_wrapped, boxsize = box_dim)
    q_results.append(tetrahedral_order_vectorized(pos_O_wrapped, Oxygen_wrapped_tree, box_dim))


q_results_array = np.array(q_results)

averages_q = np.array([np.mean(q_array) for q_array in q_results])

np.save(f'./averages_q.npy', averages_q)
np.save(f'./q_results.npy', q_results_array)
