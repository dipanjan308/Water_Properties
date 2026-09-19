import numpy as np
from ase.io import read
from scipy.spatial import KDTree

path = '/path_to_the_folder'
traj = read(f'{path}/pw_npt.dump', format='lammps-dump-text', index='700:', specorder=['O', 'H'])

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


def box_dimension_npt(ase_trajectory):
    box_dim = []
    for i in range(len(ase_trajectory)):
        cell = ase_trajectory[i].get_cell()
        box_dim.append(cell.diagonal())
    return np.array(box_dim)

box_dimension_npt = box_dimension_npt(traj)

def distance_periodic(p1, p2, box_dim):
    """
    Minimum-image distance between p1 and p2 with periodic boundaries.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    delta = p2 - p1
    box_dim = np.asarray(box_dim, dtype=float)
    delta -= box_dim * np.rint(delta / box_dim)
    return np.linalg.norm(delta, axis=-1)

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

def calculate_angles_each_hydrogen(idx_ho, hydrogen_index, pos_O_wrapped, pos_H_wrapped, box_dim):
    angle = angle_between_points_atan2(pos_O_wrapped[idx_ho[hydrogen_index][1]], pos_O_wrapped[idx_ho[hydrogen_index][0]], pos_H_wrapped[hydrogen_index], box_dim)
    return angle

def calculate_angles_each_frame(idx_ho, pos_O_wrapped, pos_H_wrapped, box_dim):
    list_angles = []
    for i in range(number_of_hydrogens):
        angles=calculate_angles_each_hydrogen(idx_ho, i, pos_O_wrapped, pos_H_wrapped, box_dim)
        list_angles.append(angles)
    return list_angles

def calculate_number_of_hydrogen_bonds(pos_O_1, pos_H_1, box_dim_npt, ho_cutoff, oo_cutoff, angle_cutoff):
    all_frame=[]
    dist_all = []
    dist_all_2 = []
    for i in range(len(pos_O_1)):
        box_dim = box_dim_npt[i]
        pos_H_wrapped = np.mod(pos_H_1[i], box_dim)
        pos_O_wrapped = np.mod(pos_O_1[i], box_dim)
        Oxygen_wrapped_tree = KDTree(pos_O_wrapped, boxsize = box_dim)
        #idx_ho shape is (number_of_hydrogens, 2) because we are finding 2 nearest oxygens for each hydrogen
        #dist_ho shape is (number_of_hydrogens, 2) and contains the distances to the 2 nearest oxygens
        dist_ho, idx_ho = Oxygen_wrapped_tree.query(pos_H_wrapped, k=2)
        all_angles = calculate_angles_each_frame(idx_ho, pos_O_wrapped, pos_H_wrapped, box_dim)
        dist_oo = distance_periodic(pos_O_wrapped[idx_ho[:, 0]], pos_O_wrapped[idx_ho[:, 1]], box_dim)
        all_frame.append(all_angles)
        dist_all.append(dist_ho[:, 1])
        dist_all_2.append(dist_oo)
    flat_all = [x for sublist in all_frame for x in sublist]
    flat_all = np.array(flat_all)
    flat_dist_all = [x for sublist in dist_all for x in sublist]
    flat_dist_all = np.array(flat_dist_all)
    dist_all_2 = np.array(dist_all_2)
    flat_dist_all_2 = [x for sublist in dist_all_2 for x in sublist]
    flat_dist_all_2 = np.array(flat_dist_all_2)
    all_dist_angle = np.column_stack((flat_dist_all, flat_dist_all_2, flat_all))
    count = 0
    for i in range(len(all_dist_angle)):
        if all_dist_angle[i][0] <= ho_cutoff and all_dist_angle[i][1] <= oo_cutoff and all_dist_angle[i][2] <= angle_cutoff:
            count += 1
    return 2.0*count/(len(pos_O_1)*number_of_oxygens)

number_of_blocks = 10
block_size = number_of_frames // number_of_blocks
block_averages = []
for i in range(number_of_blocks):
    start = i * block_size
    end = (i + 1) * block_size if i < number_of_blocks - 1 else number_of_frames
    block_pos_O = pos_O[start:end]
    block_pos_H = pos_H[start:end]
    box_dimension_reduced = box_dimension_npt[start:end]
    avg_hbonds = calculate_number_of_hydrogen_bonds(block_pos_O, block_pos_H, box_dimension_reduced, 2.5, 3.4, 30)
    block_averages.append(avg_hbonds)
mean=np.mean(block_averages)
error=np.std(block_averages) / np.sqrt(number_of_blocks)
np.savetxt('./block_average_values_h_bonds.txt', block_averages, header='Average number of hydrogen bonds per water molecule for each block', fmt='%.6f')
np.savetxt('./block_averages_h_bonds.txt', np.array([mean, error])[None, :],
           header='mean error', fmt='%.6f')
