import os.path as osp
from os.path import join
import mdtraj as mdj
import numpy as np
import pickle as pkl
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around

from wepy.hdf5 import WepyHDF5
from wepy.util.util import box_vectors_to_lengths_angles
import time
from termcolor import colored, cprint


#Change for only lig: Add the ref_pos and center pose idxs
def ensemble_extractor(hdf5, ensemble_labels, ensemble_msm_wts, cluster_labels, all_ancestors, not_water_idx, bs_idx, lig_idx):

    idx_list = []
    pos_list = []
    wts_list = []
    WE_wts_list = []

    wepy_h5 = WepyHDF5(hdf5, mode='r')
    wepy_h5.open()
    #print(len(ensemble_labels))

    for state in ensemble_labels:

        count = 0

        # loop over all the runs in h5 (same as lenght of full cluster label list)
        for run_idx in range(len(cluster_labels)):
            assert ((cluster_labels[run_idx].shape[0]) == wepy_h5.num_run_trajs(run_idx) ) , f'Number of walkers do not match for {lig}, run {run_idx}'
            #loop over all the walkers in each run
            for walker_idx in range(cluster_labels[run_idx].shape[0]):

                assert ((cluster_labels[run_idx].shape[1]) == wepy_h5.num_run_cycles(run_idx) ) , f'Number of cycles do not match for {lig}, run {run_idx}, walker {walker_idx}'

                # store the state labels of each cycles in a walker
                all_states_per_walker = cluster_labels[run_idx][walker_idx]

                # match the states of the cycles with the ensemble states
                cycle_idx_list = np.where(all_states_per_walker == state)[0]

                # proceed only when there is a match
                if len(cycle_idx_list) > 0:

                    for cycle_idx in cycle_idx_list:
                        index = [run_idx,walker_idx,cycle_idx]
                        redund_count = 0
                        for i in range(len(all_ancestors)):
                            for ancestors in all_ancestors[i]:
                                if ancestors == index and redund_count == 0:
                                    #Get the position and box vectors and do the mathematical ops to get the superimposed geom of the ligands
                                    pos = np.asarray(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/positions'][cycle_idx][not_water_idx])
                                    box_vectors = np.asarray(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/box_vectors'][cycle_idx])
                                    unitcell_lengths, _ = box_vectors_to_lengths_angles(box_vectors)
                                    grouped_pos = group_pair(pos, unitcell_lengths,bs_idx, lig_idx)

                                    if len(idx_list) == 0:
                                        ref_pos = grouped_pos
                                        centered_ref_pos = center_around(ref_pos, idxs=bs_idx)
                                        pos_list.append(centered_ref_pos)
                                    if len(idx_list) > 0:
                                        centered_target_pos = center_around(grouped_pos, idxs=bs_idx)
                                        superimposed_pos, _, _ = superimpose(centered_ref_pos, centered_target_pos, idxs=bs_idx, weights=None)
                                        pos_list.append(superimposed_pos)
                                    idx_list.append(index)
                                    wts_list.append(ensemble_msm_wts[ensemble_labels.index(state)])
                                    WE_wts_list.append(np.asarray(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/weights/'][cycle_idx]))

                                    count += 1
                                    redund_count += 1
                                    
                                    ### In case one needs the WE weights instead of the MSM weights ##########
                                    #wts_list.append(np.asarray(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/weights/'][cyc_idx]))
                                    ##########################################################################

                                    #print([run_idx,walker_idx,cycle_idx], ancestors, i, len(all_ancestors[i]), redund_count)
        
        print(colored(f'For ensemble label {state}, for ligand {lig}, the number of snaps chosen (non-weighted) are {count}', 'red'))

    wepy_h5.close()

    return(idx_list , pos_list, wts_list, WE_wts_list)

base_dir = f'/dickson/s1/bosesami/sEH_combined/'
out_dcd_pdbs = f'{base_dir}/lag1_TSE/dcds/NoTICA_full_sys/'
out_pkls = f'{out_dcd_pdbs}/pkls/'
labels_all_cuts_path = f'{base_dir}/lag1_TSE/TS_labels_NOTICA/'

sEH_dir = base_dir
sEH_19_dir = '/dickson/s1/bosesami/KSL_unbinding/KSL_19/simulation_folder/openmm/'


#comm_set = [[0.1,0.3],[0.3, 0.7],[0.7,0.9]]
comm_set = [[0.3, 0.7]]
lig_list = ['10','17','19','29','28','50']
clust_list = [500,800,1200]
lag_time_list = [1]
n_MC_iters = 10000
sam_ligs = ['10','17']
samik_ligs = ['19','28','29','50','51']

h5_path = {}
h5_path['10'] = f'{sEH_dir}/10/all_results.wepy.h5'
h5_path['17'] = f'{sEH_dir}/17/all_results.wepy.h5'
h5_path['28'] = f'{sEH_dir}/28/REVO/clone_all.h5'
h5_path['29'] = f'{sEH_dir}/29/REVO/clone.h5'
h5_path['50'] = f'{sEH_dir}/50/REVO/clone_all.h5'
h5_path['19'] = f'{sEH_19_dir}/clone.h5'

model_path = {}
model_path['10'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['17'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['28'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['29'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['50'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['19'] = f'{base_dir}/MSM_models_withoutTICA'
model_path['51'] = f'{base_dir}/MSM_models_withoutTICA_51/MSM_models_withoutTICA/'

for lig in lig_list:

    all_ancestors = pkl.load(open(f'{base_dir}/TSE/warping_ances_dir/Lig{lig}_all_ancestors.pkl','rb'))
    hdf5 = h5_path[lig]

    if lig in sam_ligs:
        input_dir_pdb  = f'{base_dir}/{lig}/'
        pdb = mdj.load_pdb(join(input_dir_pdb, f'topology_pdb_main_{lig}.pdb'))
        not_water_idx = pdb.top.select('protein or resname UNL')
        lig_idx = pdb.top.select('segname HETA and resname UNL')
        centering_idxs = pdb.top.select('segname PROA and not resname HIS')
        bs_idx = pdb.top.select('segname PROA and (resid 128 to 156) or (resid 178 to 190) or (resid 257 to 274) or (resid 101 to 109) or (resid 33 to 39) or (resid 280 to 285) or (resid 228 to 235)')

    if lig in samik_ligs:
        if lig == '19':
            input_dir = f'{sEH_19_dir}'
            pdb = mdj.load_pdb(osp.join(input_dir, 'step3_input.pdb'))
            not_water_idx = pdb.top.select('protein or resname K19')
            lig_idx = pdb.top.select('segname HETA and resname K19')
            centering_idxs = pdb.top.select('segname PROA and not resname HIS')
            bs_idx = pdb.top.select('segname PROA and (resid 130 to 156) or (resid 178 to 190) or (resid 263 to 274) or (resid 103 to 109) or (resid 35 to 39) or (resid 290 to 294) or (resid 234 to 242)')

        else:
            input_dir_pdb  = f'{base_dir}/{lig}/charmm-gui/openmm/'
            pdb = mdj.load_pdb(join(input_dir_pdb,'step3_charmm2omm.pdb'))
            not_water_idx = pdb.top.select('protein or resname UNK')
            lig_idx = pdb.top.select('resname UNK')
            centering_idxs = pdb.top.select('segname PROA and not resname HIS')
            bs_idx = pdb.top.select('segname PROA and (resid 130 to 156) or (resid 178 to 190) or (resid 263 to 274) or (resid 103 to 109) or (resid 35 to 39) or (resid 290 to 294) or (resid 234 to 242)')

    #loop over num clusters list
    for clust in clust_list:

        #loop over lag time
        for lag_time in lag_time_list:
                comm_path = f'{model_path[lig]}/model_{clust}/committors_lag{lag_time}_nclust{clust}.pkl'
                all_counts_path = f'{model_path[lig]}/model_{clust}/countsmat_lag{lag_time}_nclust{clust}.pkl'
                clusterlabels_path = f'{model_path[lig]}/model_{clust}/reshaped_cluster_labels_{lig}_nclust{clust}.pkl'

                if osp.exists(comm_path) and osp.exists(all_counts_path) and osp.exists(clusterlabels_path):
                    comm = pkl.load(open(comm_path,'rb'))
                    all_counts = pkl.load(open(all_counts_path,'rb'))
                    clusterlabels = pkl.load(open(clusterlabels_path,'rb'))
                    if lig in comm.keys():
                        for comm in comm_set:
                            comm0 = comm[0]
                            comm1 = comm[1]
                            print()
                            print(colored(f'Running for Lig{lig}, lag time {lag_time}, and committor range {comm}...', 'magenta',attrs=['bold']))
                            # get the paths of ts state labels, weights and the full(run*walker*cycle for all the snaps) state labels

                            TS_labels_path = f'{labels_all_cuts_path}/TS_labels_lig{lig}_tau{lag_time}_clust{clust}_comm_cuts_{comm0}_{comm1}.pkl'
                            TS_wts_path = f'{labels_all_cuts_path}/TS_wts_lig{lig}_tau{lag_time}_clust{clust}_comm_cuts_{comm0}_{comm1}.pkl'
                            TS_labels       = pkl.load(open(TS_labels_path, 'rb'))
                            TS_msm_wts      = pkl.load(open(TS_wts_path, 'rb'))
                            t3 = time.time()
                            # proceed only if there exists labels in the ensemble state label list
                            if len(TS_labels) > 0:
                              
                                print(colored(f'For tau {lag_time} and committor range {comm}, the number of states in the committor range is {len(TS_labels)}', 'green', attrs=['bold']))
                                #Change for only lig
                                idxs , pos, wts_list, WE_wts_list = ensemble_extractor(hdf5, TS_labels, TS_msm_wts, clusterlabels, all_ancestors, not_water_idx, bs_idx, lig_idx)

                                with open(join(out_pkls,f'Wts_lig{lig}_clust{clust}_lag{lag_time}_range{comm0}_{comm1}.pkl'),'wb') as f:
                                    pkl.dump(np.array(wts_list),f)
                                with open(join(out_pkls,f'WE_Wts_lig{lig}_clust{clust}_lag{lag_time}_range{comm0}_{comm1}.pkl'),'wb') as f:
                                    pkl.dump(np.array(WE_wts_list),f)


                                weighted_pos = []
                                    
                                sum_wts = np.sum(wts_list)

                                if sum_wts > 0.0:
                                    norm_factor = 1.0/sum_wts
                                    normalized_wts_list = np.array(wts_list)*norm_factor
                                    xyzs_idx=np.arange(0,len(pos),dtype=int)

                                    print(f'Sum of the state weights in committor range {comm} for lig{lig}, n_cluster{clust}, lag {lag_time}, is {sum_wts}')
                                    print(colored(f'Number of frames in the states in committor range {comm} for lig{lig}, n_cluster{clust}, lag {lag_time}, is {len(idxs)}', 'blue', attrs=['bold']))

                                    t4 = time.time()

                                    print('Making the trajectory...')
                                
                                    for i in range(n_MC_iters):
                                        sel_xyzs_idx = np.random.choice(xyzs_idx,p=normalized_wts_list)
                                        weighted_pos.append(pos[sel_xyzs_idx])

                                    trj = mdj.Trajectory(weighted_pos,pdb.atom_slice(not_water_idx).top)
                                    trj_non_wt = mdj.Trajectory(pos,pdb.atom_slice(not_water_idx).top)

                                    trj.save_dcd(f'{out_dcd_pdbs}/FullSys_lig{lig}_{n_MC_iters}_{clust}_{lag_time}_com_range{comm0}_{comm1}.dcd')
                                    trj[0].save_pdb(f'{out_dcd_pdbs}/FullSys_lig{lig}_frame0_{clust}_{lag_time}_com_range{comm0}_{comm1}.pdb')
                                    
                                    t4 = time.time()
                                    print(f'Time taken for the building the TS ensemble and trajectory for lig{lig}, n_cluster{clust}, lag{lag_time}, committor range {comm} is {t4 - t3}')

                                else:
                                    t4 = time.time()
                                    print(f'No of candidates: {len(idxs)}, No candidate for lig {lig}, lag time {lag_time}, committor range {comm}, in  the TS ensemble. Time taken:', t4 -t3)
                            else:
                                t4 = time.time()
                                print(f'No states in TS for lig {lig}, lag time {lag_time}, n_cluster {clust}, committor range {comm} in the TS ensemble. Time taken:', t4 -t3)

