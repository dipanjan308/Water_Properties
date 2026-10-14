import numpy as np
from ase.io import read
from scipy.spatial import KDTree

path = '/path_to_the_folder'
traj = read(f'{path}/nvt_1.dump', format='lammps-dump-text', index=':', specorder=['O', 'H'])

def sample_positions(ase_trajectory, symbol):
    positions = []
    for i in range(len(ase_trajectory)):
        symbols = ase_trajectory[i].get_chemical_symbols()
        indices = [j for j, sym in enumerate(symbols) if sym == symbol]
        positions.append(ase_trajectory[i].get_positions()[indices])
    return np.array(positions)

def angle_between_points_atan2(A, B, C, box_dim): # atan method more stable than cos method
    BA = A - B
    BC = C - B

    #print(f"BA: {BA}, BC: {BC}")

    BA = BA - box_dim * np.rint(BA / box_dim) #PBC for distances
    BC = BC - box_dim * np.rint(BC / box_dim)

    # Cross product and dot product
    cross_product = np.linalg.norm(np.cross(BA, BC))
    dot_product = np.dot(BA, BC)

    # Angle in radians using atan2
    angle_radians = np.arctan2(cross_product, dot_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

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

def is_hydrogen_bonded_acceptor_donor(acceptor_oxygen_index, donor_oxygen_index, pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim):
    acceptor_oxygen_pos = pos_O_wrapped_oneframe[acceptor_oxygen_index]
    donor_oxygen_pos = pos_O_wrapped_oneframe[donor_oxygen_index]
    angle_ADH1 = angle_between_points_atan2(acceptor_oxygen_pos, donor_oxygen_pos, pos_H_wrapped_oneframe[idx_oh[donor_oxygen_index][0]], box_dim)
    angle_ADH2 = angle_between_points_atan2(acceptor_oxygen_pos, donor_oxygen_pos, pos_H_wrapped_oneframe[idx_oh[donor_oxygen_index][1]], box_dim)

    dist_A_D = distance_periodic(acceptor_oxygen_pos, donor_oxygen_pos, box_dim)

    dist_AH1 = distance_periodic(acceptor_oxygen_pos, pos_H_wrapped_oneframe[idx_oh[donor_oxygen_index][0]], box_dim)
    dist_AH2 = distance_periodic(acceptor_oxygen_pos, pos_H_wrapped_oneframe[idx_oh[donor_oxygen_index][1]], box_dim)

     # Check the hydrogen bond criteria
    if (dist_A_D < 3.4 and dist_AH1 < 2.5 and angle_ADH1 < 30) or (dist_A_D < 3.4 and dist_AH2 < 2.5 and angle_ADH2 < 30):
        return True
    else:
        return False

def identify_donor_acceptor_bonds_oneframe(pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim, neighbors_oo, dist_oo, number_of_oxygens, max_nbr):
    array = np.zeros((max_nbr, max_nbr), dtype=int)
    delta_r = []
    for ind_o in range(number_of_oxygens):
        count1 = 0
        count2 = 0
        ind_tt = 0
        ind_ff = max_nbr-1
        flg = 0
        for i in range(max_nbr-1):
            ans1 = is_hydrogen_bonded_acceptor_donor(ind_o, neighbors_oo[ind_o][i+1], pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim)
            if ans1 == True:
                count1 += 1

            ans2 = is_hydrogen_bonded_acceptor_donor(neighbors_oo[ind_o][i+1], ind_o, pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim)
            if ans2 == True:
                count2 += 1

            if ans1 == True or ans2 == True:
                ind_tt = i+1

            if ans1 == False and ans2 == False and flg == 0:
                ind_ff = i+1
                flg = 1

        if ind_ff == max_nbr-1 and ind_tt == max_nbr-1:
            ind_ff = max_nbr

        r2 = dist_oo[ind_o][ind_ff]
        r1 = dist_oo[ind_o][ind_tt]
        dr = r2 - r1
        if count1 == 0 and count2 == 0:
            dr = 8.0

        delta_r.append(dr)
        array[count1, count2] += 1
    return array, delta_r

pos_H = sample_positions(traj, 'H')
pos_O = sample_positions(traj, 'O')

number_of_oxygens = pos_O.shape[1]
number_of_hydrogens = pos_H.shape[1]
number_of_frames = pos_O.shape[0]

box_dim = traj[0].get_cell()
box_dim = np.array(box_dim)
box_dim = box_dim.diagonal()

pos_H_wrapped = np.mod(pos_H, box_dim)
pos_O_wrapped = np.mod(pos_O, box_dim)

acceptor_donor_array = []
delta_r_array = []
max_nbr=12
for i in range(number_of_frames):
    pos_O_wrapped_oneframe = pos_O_wrapped[i]
    pos_H_wrapped_oneframe = pos_H_wrapped[i]
    Oxygen_wrapped_tree = KDTree(pos_O_wrapped_oneframe, boxsize = box_dim)
    dist_oo, neighbors_oo = Oxygen_wrapped_tree.query(pos_O_wrapped_oneframe, k = max_nbr+1)  # k=13 gives the atom itself + 12 neighbors
    Hydrogen_wrapped_tree = KDTree(pos_H_wrapped_oneframe, boxsize = box_dim)
    dist_oh, idx_oh = Hydrogen_wrapped_tree.query(pos_O_wrapped_oneframe, k=2)
    arr, delta_r = identify_donor_acceptor_bonds_oneframe(pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim, neighbors_oo, dist_oo, number_of_oxygens, max_nbr+1)
    acceptor_donor_array.append(arr)
    delta_r_array.append(delta_r)

acceptor_donor_array = np.array(acceptor_donor_array)
average_acceptor_donor_array = np.sum(acceptor_donor_array, axis=0)
delta_r_array = np.array(delta_r_array)

flattened_delta_r = delta_r_array.flatten()
range_min = -5.0
range_max = 5.0
bins = 400
bin_size = (range_max - range_min) / bins
# histogram (density)
counts_bin, bin_edges = np.histogram(flattened_delta_r, bins=bins, range=(range_min, range_max), density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

np.savetxt(f'{path}/zeta_hist.dat', np.column_stack((bin_centers, counts_bin)), header='bin_center density', fmt='%.6e')
np.savetxt(f'{path}/average_acceptor_donor_array.dat', average_acceptor_donor_array/(number_of_frames * number_of_oxygens), fmt='%0.6f')
