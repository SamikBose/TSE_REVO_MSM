####### Inputs

## 1. committors with a specific TICA dimension and lag time.
## 2. counts matrix to build the MSM with a specific TICA and lag time.
## 3. clusterlabels (with a specific TICA and lag time) to get the fundamental sequences of clusters (states).
## 4. all ancestor information of warping events.
#################
import os.path as osp
import mdtraj as mdj
import numpy as np
import sys
import time
#from termcolor import colored
from os.path import join
from csnanalysis.csn import CSN
from csnanalysis.matrix import *
import pickle as pkl


lig_list = ['10','17','19','28','29','50','51']
#lig_list = ['28','50']

base_dir = '/dickson/s1/bosesami/sEH_combined/'
TS_labels_path = f'{base_dir}/'

cut_list = [(0.1, 0.3), (0.3, 0.7), (0.7, 0.9)]

#cut_list = [(0.3, 0.7)]
#cluster_list = [100,500,800,250,1200]
tica_n_dim_list = [3,5,10]
#lag_time_list = [50,60,80,100,150,200]
cluster_list = [500,800,1200]
lag_time_list = [1]
#lag_time_list = [100]

model_base_path = {}
model_base_path['10'] = f'{base_dir}/MSM_models'
model_base_path['17'] = f'{base_dir}/MSM_models'
model_base_path['28'] = f'{base_dir}/MSM_models'
model_base_path['29'] = f'{base_dir}/MSM_models'
model_base_path['50'] = f'{base_dir}/MSM_models'
model_base_path['19'] = f'{base_dir}/MSM_models'
model_base_path['51'] = f'{base_dir}/MSM_models_51/MSM_models/'

for cutoff in cut_list:
    cut1 = cutoff[0]
    cut2 = cutoff[1]
    
    for lig in lig_list:
        all_ances = pkl.load(open(f'{base_dir}/TSE/warping_ances_dir/Lig{lig}_all_ancestors.pkl','rb')) 

        for clust in cluster_list:
            for tica_n_dim in tica_n_dim_list:
                model_path = f'{model_base_path[lig]}/model_{tica_n_dim}_{clust}/'

                for lag_time in lag_time_list:
                    
                    print(f'Running for lig{lig}, n_clust{clust}, n_tica{tica_n_dim}, lag{lag_time}...')
                    comm_path = f'{model_path}/committors_lag{lag_time}_nclust{clust}.pkl'
                    all_counts_path = f'{model_path}/countsmat_lag{lag_time}_nclust{clust}.pkl'
                    clusterlabels_path = f'{model_path}/reshaped_cluster_labels_{lig}_nclust{clust}.pkl'
                    
                    if osp.exists(comm_path) and osp.exists(all_counts_path) and osp.exists(clusterlabels_path):
                        state_label = []
                        state_wts = []
                        count = 0
                        repeat_FS = 0
                        comm = pkl.load(open(comm_path,'rb'))
                        all_counts = pkl.load(open(all_counts_path,'rb'))
                        clusterlabels = pkl.load(open(clusterlabels_path,'rb'))

                        if lig in comm.keys():
            
                            lig_csn = CSN(all_counts[lig].T)
                            lig_csn.trim()
                            msm_mult_wts = lig_csn.calc_mult_weights()

                            for k  in range(len(all_ances)):
                                #print()
                                #print(f'Running for warping idx: {k}')
                                traj_list = []
                                non_FS = 0
                                repeat_FS =0
                                for ances in all_ances[k]:
                                    run_idx = ances[0]
                                    walker_idx = ances[1]
                                    cycle_idx = ances[2]
                                    microstate_idx = clusterlabels[run_idx][walker_idx,cycle_idx]
                                    traj_list.append(microstate_idx)
                                    unb_comm = comm[lig][microstate_idx,1]
                                    if cut1 <= unb_comm < cut2:
                                        if microstate_idx not in state_label:
                                            count += 1
                                            state_label.append(microstate_idx)
                                            state_wts.append(msm_mult_wts[microstate_idx])
                                            #print(f'State added to the TS list: {microstate_idx}, committor val: {unb_comm}')
                                        #elif microstate_idx in state_label:
                                        #    repeat_FS += 1
                                            #print(f'State {microstate_idx} is already present, committor val: {unb_comm}')
                                #print(f'Committor cutoff: {cut1},{cut2}.. Lig:{lig}, n_clust:{clust}, tica:{tica_n_dim}, lag:{lag_time}, Num of repeat states in the comm range:{repeat_FS}')
                            print(f'Committor cutoff: {cut1},{cut2}.. Lig:{lig}, n_clust:{clust}, tica:{tica_n_dim}, lag:{lag_time}, Num of states in the comm range:{count}\n')
                            #print(f'Committor cutoff: {cut1},{cut2}.. Lig:{lig}, n_clust:{clust}, tica:{tica_n_dim}, lag:{lag_time}, Num of repeat FS states in the comm range:{repeat_FS}\n')
                            #print(state_label)

                            with open(f'{TS_labels_path}/TS_labels_lig{lig}_tau{lag_time}_tica{tica_n_dim}_clust{clust}_comm_cuts_{cut1}_{cut2}.pkl','wb') as f:
                                pkl.dump(state_label, f)
                            with open(f'{TS_labels_path}/TS_wts_lig{lig}_tau{lag_time}_tica{tica_n_dim}_clust{clust}_comm_cuts_{cut1}_{cut2}.pkl','wb') as f:
                                pkl.dump(state_wts, f)
                            '''
                            with open(f'{TS_labels_path}/all_state_wts_lig{lig}_tau{lag_time}_tica{tica_n_dim}_clust{clust}_comm_cuts.pkl','wb') as f:
                                pkl.dump(msm_mult_wts, f)
                            '''
                    else:
                        #print(colored(f'No comm file for lig{lig}_tau{lag_time}_tica{tica_n_dim}_clust{clust} ', 'green', attrs=['bold']))
                        print(f'No comm file for lig{lig}_tau{lag_time}_tica{tica_n_dim}_clust{clust} ')
