import numpy as np
from ase.io import read
from scipy.spatial import KDTree
import pickle

path = '/leonardo_scratch/fast/IscrB_HPHDQMC/Dipanjan/pure_water/M2_LJ/T400/mol_500/den_1.1603_02'
traj = read(f'{path}/nvt_1.dump', format='lammps-dump-text', index='4000:34000', specorder=['O', 'H'])

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

def angle_between_points_atan2(A, B, C, box_dim): # atan method more stable than cos method
    BA = A - B
    BC = C - B

    BA = BA - box_dim * np.rint(BA / box_dim) #PBC for distances
    BC = BC - box_dim * np.rint(BC / box_dim)

    # Cross product and dot product
    cross_product = np.linalg.norm(np.cross(BA, BC))
    dot_product = np.dot(BA, BC)

    # Angle in radians using atan2
    angle_radians = np.arctan2(cross_product, dot_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_angles_each_hydrogen(idx_ho, hydrogen_index, pos_O_wrapped, pos_H_wrapped):
    angle = angle_between_points_atan2(pos_O_wrapped[idx_ho[hydrogen_index][0]], pos_H_wrapped[hydrogen_index], pos_O_wrapped[idx_ho[hydrogen_index][1]], box_dim)
    return angle

def calculate_angles_each_frame(idx_ho, pos_O_wrapped, pos_H_wrapped):
    list_angles = []
    for i in range(number_of_hydrogens):
        angles=calculate_angles_each_hydrogen(idx_ho, i, pos_O_wrapped, pos_H_wrapped)
        list_angles.append(angles)
    return list_angles

all_frame=[]
for i in range(number_of_frames):
    pos_H_wrapped = np.mod(pos_H[i], box_dim)
    pos_O_wrapped = np.mod(pos_O[i], box_dim)
    Oxygen_wrapped_tree = KDTree(pos_O_wrapped, boxsize = box_dim)
    #idx_ho shape is (number_of_hydrogens, 2) because we are finding 2 nearest oxygens for each hydrogen
    #dist_ho shape is (number_of_hydrogens, 2) and contains the distances to the 2 nearest oxygens
    dist_ho, idx_ho = Oxygen_wrapped_tree.query(pos_H_wrapped, k=2)
    all_angles = calculate_angles_each_frame(idx_ho, pos_O_wrapped, pos_H_wrapped)
    all_frame.append(all_angles)
flat_all = [x for sublist in all_frame for x in sublist]
flat_all = np.array(flat_all)

histogram_angles = flat_all
counts, bin_edges = np.histogram(histogram_angles, bins=160, range=(100, 180), density=True)

# Compute bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Save to text file
data = np.column_stack((bin_centers, counts))
np.savetxt("OHO_angles.dat", data, header="#Angle(deg)  ProbabilityDensity")

