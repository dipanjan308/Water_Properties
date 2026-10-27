import numpy as np
from ase.io import read
from scipy.spatial import KDTree

path = '/leonardo_scratch/fast/IscrB_HPHDQMC/Dipanjan/pure_water/M2_LJ/T400/mol_500/den_1.3860_02'
traj = read(f'{path}/nvt_1.dump', format='lammps-dump-text', index='100000:300000:10', specorder=['O', 'H'])
path2 = f'{path}/results_adhb_final'

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
    if (dist_A_D < 3.5 and dist_AH1 < 2.5 and angle_ADH1 < 30) or (dist_A_D < 3.5 and dist_AH2 < 2.5 and angle_ADH2 < 30):
        return True
    else:
        return False

def identify_donor_acceptor_bonds_oneframe(pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim, neighbors_oo, dist_oo, number_of_oxygens, max_nbr):
    array = np.zeros((max_nbr, max_nbr), dtype=int)
    delta_r1 = []
    delta_r2 = []
    delta_r3 = []
    delta_r4 = []
    delta_r5 = []
    delta_r6 = []
    delta_r7 = []
    delta_r8 = []
    delta_r9 = []
    delta_r10 = []
    r_hb_list = []
    r_hb_list1 = []
    r_hb_list2 = []
    r_hb_list3 = []
    r_hb_list4 = []
    r_hb_list5 = []
    r_nhb_list = []
    r_nhb_list4 = []
    r_nhb_list5 = []
    r_nhb_list6 = []
    r_nhb_list7 = []
    r_nhb_list8 = []
    tot_hb = []
    for ind_o in range(number_of_oxygens):
        count1 = 0
        count2 = 0
        count_hb = 0
        ind_tt = 0
        ind_ff1 = max_nbr-1
        ind_ff2 = max_nbr-1
        ind_ff3 = max_nbr-1
        ind_ff4 = max_nbr-1
        ind_ff5 = max_nbr-1
        ind_ff6 = max_nbr-1
        ind_ff7 = max_nbr-1
        ind_ff8 = max_nbr-1
        ind_ff9 = max_nbr-1
        ind_ff10 = max_nbr-1
        flg1 = 0
        count_ff = 0
        for i in range(max_nbr-1):
            ans1 = is_hydrogen_bonded_acceptor_donor(ind_o, neighbors_oo[ind_o][i+1], pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim)
            if ans1 == True:
                count1 += 1

            ans2 = is_hydrogen_bonded_acceptor_donor(neighbors_oo[ind_o][i+1], ind_o, pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim)
            if ans2 == True:
                count2 += 1

            if ans1 == True or ans2 == True:
                count_hb += 1
                ind_tt = i+1
                r_hb = dist_oo[ind_o][ind_tt]
                r_hb_list.append(r_hb)
                if i+1 == 1:
                    r_hb_list1.append(r_hb)
                if i+1 == 2:
                    r_hb_list2.append(r_hb)
                if i+1 == 3:
                    r_hb_list3.append(r_hb)
                if i+1 == 4:
                    r_hb_list4.append(r_hb)
                if i+1 == 5:
                    r_hb_list5.append(r_hb)
            else:
                r_nhb = dist_oo[ind_o][i+1]
                r_nhb_list.append(r_nhb)
                if i+1 == 4:
                    r_nhb_list4.append(r_nhb)
                if i+1 == 5:
                    r_nhb_list5.append(r_nhb)
                if i+1 == 6:
                    r_nhb_list6.append(r_nhb)
                if i+1 == 7:
                    r_nhb_list7.append(r_nhb)
                if i+1 == 8:
                    r_nhb_list8.append(r_nhb)
            # Check for the first non-bonded neighbor
            if ans1 == False and ans2 == False and flg1 == 0:
                ind_ff1 = i+1
                flg1 = 1
            #Check for second, third and fourth non-bonded neighbor
            if ans1 == False and ans2 == False and flg1 == 1:
                count_ff += 1
                if count_ff == 2:
                    ind_ff2 = i+1
                if count_ff == 3:
                    ind_ff3 = i+1
                if count_ff == 4:
                    ind_ff4 = i+1
                if count_ff == 5:
                    ind_ff5 = i+1
                if count_ff == 6:
                    ind_ff6 = i+1
                if count_ff == 7:
                    ind_ff7 = i+1
                if count_ff == 8:
                    ind_ff8 = i+1
                if count_ff == 9:
                    ind_ff9 = i+1
                if count_ff == 10:
                    ind_ff10 = i+1
        r21 = dist_oo[ind_o][ind_ff1]
        r22 = dist_oo[ind_o][ind_ff2]
        r23 = dist_oo[ind_o][ind_ff3]
        r24 = dist_oo[ind_o][ind_ff4]
        r25 = dist_oo[ind_o][ind_ff5]
        r26 = dist_oo[ind_o][ind_ff6]
        r27 = dist_oo[ind_o][ind_ff7]
        r28 = dist_oo[ind_o][ind_ff8]
        r29 = dist_oo[ind_o][ind_ff9]
        r210 = dist_oo[ind_o][ind_ff10]
        r1 = dist_oo[ind_o][ind_tt]
        dr1 = r21 - r1
        dr2 = r22 - r1
        dr3 = r23 - r1
        dr4 = r24 - r1
        dr5 = r25 - r1
        dr6 = r26 - r1
        dr7 = r27 - r1
        dr8 = r28 - r1
        dr9 = r29 - r1
        dr10 = r210 - r1
        if count1 == 0 and count2 == 0:
            dr1 = 8.0
            dr2 = 8.0
            dr3 = 8.0
            dr4 = 8.0
            dr5 = 8.0
            dr6 = 8.0
            dr7 = 8.0
            dr8 = 8.0
            dr9 = 8.0
            dr10 = 8.0
        delta_r1.append(dr1)
        delta_r2.append(dr2)
        delta_r3.append(dr3)
        delta_r4.append(dr4)
        delta_r5.append(dr5)
        delta_r6.append(dr6)
        delta_r7.append(dr7)
        delta_r8.append(dr8)
        delta_r9.append(dr9)
        delta_r10.append(dr10)
        array[count1, count2] += 1
        tot_hb.append(count_hb)
    return array, delta_r1, delta_r2, delta_r3, delta_r4, delta_r5, delta_r6, delta_r7, delta_r8, delta_r9, delta_r10, r_hb_list1, r_hb_list2, r_hb_list3, r_hb_list4, r_hb_list5, r_hb_list, r_nhb_list4, r_nhb_list5, r_nhb_list6, r_nhb_list7, r_nhb_list8, r_nhb_list, tot_hb

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

max_nbr=12
acceptor_donor_array = []
delta_r1_array = []
delta_r2_array = []
delta_r3_array = []
delta_r4_array = []
delta_r5_array = []
delta_r6_array = []
delta_r7_array = []
delta_r8_array = []
delta_r9_array = []
delta_r10_array = []
r_hb_list_array = []
r_hb_list1_array = []
r_hb_list2_array = []
r_hb_list3_array = []
r_hb_list4_array = []
r_hb_list5_array = []
r_nhb_list_array = []
r_nhb_list4_array = []
r_nhb_list5_array = []
r_nhb_list6_array = []
r_nhb_list7_array = []
r_nhb_list8_array = []
tot_hb_array = []

for i in range(number_of_frames):
    pos_O_wrapped_oneframe = pos_O_wrapped[i]
    pos_H_wrapped_oneframe = pos_H_wrapped[i]
    Oxygen_wrapped_tree = KDTree(pos_O_wrapped_oneframe, boxsize = box_dim)
    dist_oo, neighbors_oo = Oxygen_wrapped_tree.query(pos_O_wrapped_oneframe, k = max_nbr+1)  # k=8 gives the atom itself + 7 neighbors
    Hydrogen_wrapped_tree = KDTree(pos_H_wrapped_oneframe, boxsize = box_dim)
    dist_oh, idx_oh = Hydrogen_wrapped_tree.query(pos_O_wrapped_oneframe, k=2)
    arr, delta_r1, delta_r2, delta_r3, delta_r4, delta_r5, delta_r6, delta_r7, delta_r8, delta_r9, delta_r10, r_hb_list1, r_hb_list2, r_hb_list3, r_hb_list4, r_hb_list5, r_hb_list, r_nhb_list4, r_nhb_list5, r_nhb_list6, r_nhb_list7, r_nhb_list8, r_nhb_list, tot_hb = identify_donor_acceptor_bonds_oneframe(pos_O_wrapped_oneframe, pos_H_wrapped_oneframe, idx_oh, box_dim, neighbors_oo, dist_oo, number_of_oxygens, max_nbr+1)
    acceptor_donor_array.append(arr)
    delta_r1_array.append(delta_r1)
    delta_r2_array.append(delta_r2)
    delta_r3_array.append(delta_r3)
    delta_r4_array.append(delta_r4)
    delta_r5_array.append(delta_r5)
    delta_r6_array.append(delta_r6)
    delta_r7_array.append(delta_r7)
    delta_r8_array.append(delta_r8)
    delta_r9_array.append(delta_r9)
    delta_r10_array.append(delta_r10)
    r_hb_list1_array.append(r_hb_list1)
    r_hb_list2_array.append(r_hb_list2)
    r_hb_list3_array.append(r_hb_list3)
    r_hb_list4_array.append(r_hb_list4)
    r_hb_list5_array.append(r_hb_list5)
    r_hb_list_array.append(r_hb_list)
    r_nhb_list4_array.append(r_nhb_list4)
    r_nhb_list5_array.append(r_nhb_list5)
    r_nhb_list6_array.append(r_nhb_list6)
    r_nhb_list7_array.append(r_nhb_list7)
    r_nhb_list8_array.append(r_nhb_list8)
    r_nhb_list_array.append(r_nhb_list)
    tot_hb_array.append(tot_hb)

acceptor_donor_array = np.array(acceptor_donor_array)
average_acceptor_donor_array = np.sum(acceptor_donor_array, axis=0)

flattened_r_hb_array = [item for sublist in r_hb_list_array for item in sublist]
flattened_r1_hb_array = [item for sublist in r_hb_list1_array for item in sublist]
flattened_r2_hb_array = [item for sublist in r_hb_list2_array for item in sublist]
flattened_r3_hb_array = [item for sublist in r_hb_list3_array for item in sublist]
flattened_r4_hb_array = [item for sublist in r_hb_list4_array for item in sublist]
flattened_r5_hb_array = [item for sublist in r_hb_list5_array for item in sublist]
flattened_r_nhb_array = [item for sublist in r_nhb_list_array for item in sublist]
flattened_r4_nhb_array = [item for sublist in r_nhb_list4_array for item in sublist]
flattened_r5_nhb_array = [item for sublist in r_nhb_list5_array for item in sublist]
flattened_r6_nhb_array = [item for sublist in r_nhb_list6_array for item in sublist]
flattened_r7_nhb_array = [item for sublist in r_nhb_list7_array for item in sublist]
flattened_r8_nhb_array = [item for sublist in r_nhb_list8_array for item in sublist]
flattened_tot_hb_array = [item for sublist in tot_hb_array for item in sublist]

flattened_r_hb_array = np.array(flattened_r_hb_array)
flattened_r1_hb_array = np.array(flattened_r1_hb_array)
flattened_r2_hb_array = np.array(flattened_r2_hb_array)
flattened_r3_hb_array = np.array(flattened_r3_hb_array)
flattened_r4_hb_array = np.array(flattened_r4_hb_array)
flattened_r5_hb_array = np.array(flattened_r5_hb_array)
flattened_r_nhb_array = np.array(flattened_r_nhb_array)
flattened_r4_nhb_array = np.array(flattened_r4_nhb_array)
flattened_r5_nhb_array = np.array(flattened_r5_nhb_array)
flattened_r6_nhb_array = np.array(flattened_r6_nhb_array)
flattened_r7_nhb_array = np.array(flattened_r7_nhb_array)
flattened_r8_nhb_array = np.array(flattened_r8_nhb_array)
flattened_tot_hb_array = np.array(flattened_tot_hb_array)

av_hb_per_oxygen = np.sum(flattened_tot_hb_array) / (number_of_frames * number_of_oxygens)
np.savetxt(f"{path2}/av_hb.txt", np.array([av_hb_per_oxygen]), fmt="%.2f", header="Average number of hydrogen bonds per oxygen")

range_min = 0.0
range_max = 7.0
bins = 7
bin_size = (range_max - range_min) / bins
# histogram (density)
counts_bin_hb, bin_edges_hb = np.histogram(flattened_tot_hb_array, bins, range=(range_min, range_max), density=True)
bin_centres_hb = (bin_edges_hb[:-1] + bin_edges_hb[1:]) / 2 - 0.5  # Adjusting the bin centers to represent the number of HBs

np.savetxt(f'{path2}/hb_distribution.txt', np.column_stack((bin_centres_hb, counts_bin_hb)), fmt='%.2f', header='Number of hydrogen bonds distribution')

range_min = -5.0
range_max = 7.0
bins = 900
bin_size = (range_max - range_min) / bins
# histogram (density)
counts_bin_r, bin_edges_r = np.histogram(flattened_r_hb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r1, bin_edges_r1 = np.histogram(flattened_r1_hb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r2, bin_edges_r2 = np.histogram(flattened_r2_hb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r3, bin_edges_r3 = np.histogram(flattened_r3_hb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r4, bin_edges_r4 = np.histogram(flattened_r4_hb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r5, bin_edges_r5 = np.histogram(flattened_r5_hb_array, bins, range=(range_min, range_max), density=True)
bin_centers_r = 0.5 * (bin_edges_r[:-1] + bin_edges_r[1:])
bin_centers_r1 = 0.5 * (bin_edges_r1[:-1] + bin_edges_r1[1:])
bin_centers_r2 = 0.5 * (bin_edges_r2[:-1] + bin_edges_r2[1:])
bin_centers_r3 = 0.5 * (bin_edges_r3[:-1] + bin_edges_r3[1:])
bin_centers_r4 = 0.5 * (bin_edges_r4[:-1] + bin_edges_r4[1:])
bin_centers_r5 = 0.5 * (bin_edges_r5[:-1] + bin_edges_r5[1:])

np.savetxt(f'{path2}/hb_neighbour_dist.txt', np.column_stack((bin_centers_r, counts_bin_r, counts_bin_r1, counts_bin_r2, counts_bin_r3, counts_bin_r4, counts_bin_r5)), header='r HB 1N 2N 3N 4N 5N', fmt='%.6e')

counts_bin_nr, bin_edges_nr = np.histogram(flattened_r_nhb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r4, bin_edges_r4 = np.histogram(flattened_r4_nhb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r5, bin_edges_r5 = np.histogram(flattened_r5_nhb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r6, bin_edges_r6 = np.histogram(flattened_r6_nhb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r7, bin_edges_r7 = np.histogram(flattened_r7_nhb_array, bins, range=(range_min, range_max), density=True)
counts_bin_r8, bin_edges_r8 = np.histogram(flattened_r8_nhb_array, bins, range=(range_min, range_max), density=True)
bin_centers_nr = 0.5 * (bin_edges_nr[:-1] + bin_edges_nr[1:])
bin_centers_r4 = 0.5 * (bin_edges_r4[:-1] + bin_edges_r4[1:])
bin_centers_r5 = 0.5 * (bin_edges_r5[:-1] + bin_edges_r5[1:])
bin_centers_r6 = 0.5 * (bin_edges_r6[:-1] + bin_edges_r6[1:])
bin_centers_r7 = 0.5 * (bin_edges_r7[:-1] + bin_edges_r7[1:])
bin_centers_r8 = 0.5 * (bin_edges_r8[:-1] + bin_edges_r8[1:])

np.savetxt(f'{path2}/nhb_neighbour_dist.txt', np.column_stack((bin_centers_nr, counts_bin_nr, counts_bin_r4, counts_bin_r5, counts_bin_r6, counts_bin_r7, counts_bin_r8)), header='r NHB 4N 5N 6N 7N 8N', fmt='%.6e')

delta_r1_array = np.array(delta_r1_array)
delta_r2_array = np.array(delta_r2_array)
delta_r3_array = np.array(delta_r3_array)
delta_r4_array = np.array(delta_r4_array)
delta_r5_array = np.array(delta_r5_array)
delta_r6_array = np.array(delta_r6_array)
delta_r7_array = np.array(delta_r7_array)
delta_r8_array = np.array(delta_r8_array)
delta_r9_array = np.array(delta_r9_array)
delta_r10_array = np.array(delta_r10_array)
flattened_delta_r1 = delta_r1_array.flatten()
flattened_delta_r2 = delta_r2_array.flatten()
flattened_delta_r3 = delta_r3_array.flatten()
flattened_delta_r4 = delta_r4_array.flatten()
flattened_delta_r5 = delta_r5_array.flatten()
flattened_delta_r6 = delta_r6_array.flatten()
flattened_delta_r7 = delta_r7_array.flatten()
flattened_delta_r8 = delta_r8_array.flatten()
flattened_delta_r9 = delta_r9_array.flatten()
flattened_delta_r10 = delta_r10_array.flatten()
range_min = -5.0
range_max = 6.0
bins = 800
bin_size = (range_max - range_min) / bins
# histogram (density)
counts_bin1, bin_edges1 = np.histogram(flattened_delta_r1, bins, range=(range_min, range_max), density=True)
counts_bin2, bin_edges2 = np.histogram(flattened_delta_r2, bins, range=(range_min, range_max), density=True)
counts_bin3, bin_edges3 = np.histogram(flattened_delta_r3, bins, range=(range_min, range_max), density=True)
counts_bin4, bin_edges4 = np.histogram(flattened_delta_r4, bins, range=(range_min, range_max), density=True)
counts_bin5, bin_edges5 = np.histogram(flattened_delta_r5, bins, range=(range_min, range_max), density=True)
counts_bin6, bin_edges6 = np.histogram(flattened_delta_r6, bins, range=(range_min, range_max), density=True)
counts_bin7, bin_edges7 = np.histogram(flattened_delta_r7, bins, range=(range_min, range_max), density=True)
counts_bin8, bin_edges8 = np.histogram(flattened_delta_r8, bins, range=(range_min, range_max), density=True)
counts_bin9, bin_edges9 = np.histogram(flattened_delta_r9, bins, range=(range_min, range_max), density=True)
counts_bin10, bin_edges10 = np.histogram(flattened_delta_r10, bins, range=(range_min, range_max), density=True)
bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
bin_centers3 = 0.5 * (bin_edges3[:-1] + bin_edges3[1:])
bin_centers4 = 0.5 * (bin_edges4[:-1] + bin_edges4[1:])
bin_centers5 = 0.5 * (bin_edges5[:-1] + bin_edges5[1:])
bin_centers6 = 0.5 * (bin_edges6[:-1] + bin_edges6[1:])
bin_centers7 = 0.5 * (bin_edges7[:-1] + bin_edges7[1:])
bin_centers8 = 0.5 * (bin_edges8[:-1] + bin_edges8[1:])
bin_centers9 = 0.5 * (bin_edges9[:-1] + bin_edges9[1:])
bin_centers10 = 0.5 * (bin_edges10[:-1] + bin_edges10[1:])
np.savetxt(f'{path2}/zeta_hist.txt', np.column_stack((bin_centers1, counts_bin1, counts_bin2, counts_bin3, counts_bin4, counts_bin5, counts_bin6, counts_bin7, counts_bin8, counts_bin9, counts_bin10)), header='dr P(dr)', fmt='%.6e')
np.savetxt(f'{path2}/average_acceptor_donor_array.txt', average_acceptor_donor_array/(number_of_frames * number_of_oxygens), fmt='%0.6f')
