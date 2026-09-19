import numpy as np
from ase.io import read

path = '/path_to_the_folder'
traj = read(f'{path}/filename.dump', format='lammps-dump-text', index='20000:40000', specorder=['O', 'H'])

#traj = read('../trajectory_nvt_mw_comb.xyz', '10000:', format='extxyz')

def sample_positions(ase_trajectory, symbol):
    positions = []
    for i in range(len(ase_trajectory)):
        symbols = ase_trajectory[i].get_chemical_symbols()
        indices = [j for j, sym in enumerate(symbols) if sym == symbol]
        positions.append(ase_trajectory[i].get_positions()[indices])
    return np.array(positions)

pos_H = sample_positions(traj, 'H')
pos_O = sample_positions(traj, 'O')

num_H= len(pos_H[0])
num_O= len(pos_O[0])

H = traj[0].get_cell()
Hinv = np.linalg.inv(H)

def list_of_kvector(nmax, nmin, kvec):
    kvec = np.zeros((nvec,3))
    ix=-1
    index=np.concatenate((np.arange(-nmax,-nmin+1),np.arange(nmin,nmax+1)))
    for nx in range(nmin,nmax+1):
        for ny in index:
            for nz in index:
                ix += 1
                kvec[ix,0] = 2 * np.pi * nx
                kvec[ix,1] = 2 * np.pi * ny
                kvec[ix,2] = 2 * np.pi * nz

    return kvec

timelen=20000 # Number of time steps in the trajectory
nmax=18   #nmax is the number of maximum k-vec in each direction
nmin=0
nvec = (nmax-nmin+1)*(2*(nmax-nmin+1))**2
kvec=list_of_kvector(nmax, nmin, nvec)

def get_rhokr_rhoki_sofk(number_of_atoms, position_array, kvec, timelen, nvec):
    rpos=np.zeros((number_of_atoms,3))
    sofk=np.zeros(nvec)
    csum=np.zeros(nvec)
    ssum=np.zeros(nvec)
    rhokr=np.zeros((timelen,nvec))
    rhoki=np.zeros((timelen,nvec))
    count=0
    sofk= np.zeros(nvec)
    results = []
    for i in range(timelen):
        count+=1
        rpos[:]= position_array[i,:] @ Hinv
        sp=np.inner(kvec,rpos)
        csum=np.cos(sp).sum(axis=1)
        ssum=np.sin(sp).sum(axis=1)
        sofk+=(csum**2+ssum**2)/number_of_atoms
        rhokr[i,]=csum
        rhoki[i,]=ssum
    sofk=sofk/count

    results = [rhokr, rhoki]
    array_of_results = np.asarray(results)

    return array_of_results, sofk

res_o, sofk_o=get_rhokr_rhoki_sofk(num_O, pos_O, kvec, timelen, nvec)
res_h, sofk_h=get_rhokr_rhoki_sofk(num_H, pos_H, kvec, timelen, nvec)

def dynamic_structure_factor(rhok_r, rhok_i, time_step_fs, number_of_selected_particles, block_size):
    #Reshaping array
    number_of_blocks = rhok_r.shape[0] // block_size
    rhok_r = rhok_r.reshape([number_of_blocks, block_size])
    rhok_i = rhok_i.reshape([number_of_blocks, block_size])

    #Time array generation
    time_step_fs = time_step_fs * 1e-15 # convert timestep in seconds!
    time = np.arange(int(2**(np.ceil(np.log2(block_size)) + 1))) * time_step_fs

    arr_r = np.zeros([number_of_blocks, len(time)])
    arr_r[:,:rhok_r.shape[1]] = rhok_r
    arr_i = np.zeros([number_of_blocks, len(time)])
    arr_i[:,:rhok_i.shape[1]] = rhok_i

    nt = len(time)
    #print(nt)
    rhokhat = np.zeros((nt),dtype=complex)
    PSD = np.zeros((nt))
    fqt = np.zeros((nt))
    fqt2 = np.zeros((nt))

    results = []

    for nb in range(number_of_blocks):
        rhokhat = np.fft.fft(arr_r[nb, :]+1j*arr_i[nb, :], nt)       # Compute the FFT
        PSD = rhokhat * np.conj(rhokhat) / (number_of_selected_particles*block_size)     # Power spectrum (power per freq)
        fqt += np.real(np.fft.ifft(PSD))
        fqt2 += np.real(np.fft.ifft(PSD))**2

    fqt /= (number_of_blocks)
    fqt2 /= (number_of_blocks)
    fqt2 = np.sqrt((fqt2 - fqt**2)/(number_of_blocks)) # standard error

    results = [time, fqt, fqt2]
    array_of_results = np.asarray(results)

    return array_of_results

def dsf_all_kvec(kvec, res_arr, timestep, number_of_particles, block_size):
    results_full_kvec = []
    for k in range(len(kvec)):
        rhokr = res_arr[0, :, k]
        rhoki = res_arr[1, :, k]
        result=dynamic_structure_factor(rhokr[:], rhoki[:], timestep, number_of_particles, block_size)
        results_full_kvec.append(result)
    result_array = np.asarray(results_full_kvec)

    return result_array

block_size = 5000
timestep = 5*0.5
result_array_o=dsf_all_kvec(kvec, res_o, timestep, num_O, block_size)
result_array_h=dsf_all_kvec(kvec, res_h, timestep, num_H, block_size)

sofk_f_o= result_array_o[:,1,0]
err_sofk_f_o = result_array_o[:,2,0]
sofk_f_h = result_array_h[:,1,0]
err_sofk_f_h = result_array_h[:,2,0]

def sorting_of_sofk(arr, modk, unikap, inverse):
    arr_sort=arr[np.argsort(modk)][0:]
    arr_spherical=np.zeros(len(unikap), dtype=float)
    sums=np.zeros_like(unikap, dtype=float)
    counts = np.zeros_like(unikap, dtype=int)
    np.add.at(sums, inverse, arr_sort)
    np.add.at(counts, inverse, 1)
    arr_spherical  = sums / counts
    return arr_spherical

kvec_pr=kvec @ Hinv
modk=np.sqrt((kvec_pr**2).sum(axis=1))
tolerance=1e-8
kappa=modk[np.argsort(modk)][0:]
kappa_binned=np.round(kappa / tolerance) * tolerance
unikap, inverse = np.unique(kappa_binned, return_inverse=True)

sofk_f_sph_o=sorting_of_sofk(sofk_f_o, modk, unikap, inverse)
err_sofk_f_sph_o=sorting_of_sofk(err_sofk_f_o, modk, unikap, inverse)
sofk_avg_o = sorting_of_sofk(sofk_o, modk, unikap, inverse)

sofk_f_sph_h=sorting_of_sofk(sofk_f_h, modk, unikap, inverse)
err_sofk_f_sph_h=sorting_of_sofk(err_sofk_f_h, modk, unikap, inverse)
sofk_avg_h = sorting_of_sofk(sofk_h, modk, unikap, inverse)

def description_file(total_atoms, density, temperature):
    filename= f'atoms{total_atoms}_density{density:.4f}_temp{temperature}'
    return filename

density=1.1603
temperature=400
first_name=description_file(num_H+num_O, density, temperature)

file_ssf = f'{first_name}_static_sf.dat'
data_ssf= np.column_stack((unikap[:], sofk_f_sph_o[:], err_sofk_f_sph_o[:], sofk_avg_o[:], sofk_f_sph_h[:], err_sofk_f_sph_h[:], sofk_avg_h[:]))
np.savetxt(file_ssf, data_ssf, fmt='%.8e', delimiter='\t', header='|k| sofk_o Error sofk2_o sofk_h Error sofk2_h')
