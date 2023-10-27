####### Inputs

## 1. committors with a specific TICA dimension and lag time.
## 2. counts matrix to build the MSM with a specific TICA and lag time.
## 3. clusterlabels (with a specific TICA and lag time) to get the fundamental sequences of clusters (states).
## 4. new warping record as per the new definition.
## 5. continuation tree of the wepy runs.


import mdtraj as mdj
import numpy as np
import sys
import time
from termcolor import colored
from os.path import join

from wepy.hdf5 import WepyHDF5
from wepy.boundary_conditions.receptor import UnbindingBC
from wepy.analysis.parents import resampling_panel, parent_panel, net_parent_table, parent_table_discontinuities, sliding_window, ancestors
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.util.util import box_vectors_to_lengths_angles
from geomm.grouping import group_pair

from csnanalysis.csn import CSN
from csnanalysis.matrix import *

import pickle as pkl

## Cant have 51 since 51 h5 files are in hpcc
## Do this separately for 51 in hpcc and get the ancestors from hpcc
#lig_list = ['10','17','19','28','29','50']
lig_list = ['50']

sEH_dir = '/dickson/s1/bosesami/sEH_combined/'
sEH_19_dir = '/dickson/s1/bosesami/KSL_unbinding/KSL_19/simulation_folder/openmm/'
out_dir = f'{sEH_dir}/TSE/warping_ances_dir/'

run_network_tree_all_lig = pkl.load(open(f'{sEH_dir}/TSE/run_network_tree.pkl','rb'))

h5_path = {}
h5_path['10'] = f'{sEH_dir}/10/all_results.wepy.h5'
h5_path['17'] = f'{sEH_dir}/17/all_results.wepy.h5'
h5_path['28'] = f'{sEH_dir}/28/REVO/clone_all.h5'
h5_path['29'] = f'{sEH_dir}/29/REVO/clone.h5'
h5_path['50'] = f'{sEH_dir}/50/REVO/clone_2.h5'
h5_path['19'] = f'{sEH_19_dir}/clone.h5'

for lig in lig_list:
    # Get the the run tree info
    run_network_tree = run_network_tree_all_lig[lig]
    # Get the new warping record (5A cutoff)
    new_warp_record = pkl.load(open(f'LIG{lig}_new_warp_rec.pkl','rb'))
    # Do only when we have wapring...
    if len(new_warp_record) > 0:
        all_ancestors = []
        
        hdf5 = h5_path[lig]
        wepy_h5 = WepyHDF5(hdf5, mode='r')
        wepy_h5.open()
                   
        for warp in new_warp_record:            
            run_idx = warp[0]
            for i in range(len(run_network_tree)):
                if run_idx in run_network_tree[i]:
                    run_set = run_network_tree[i]
                    idx_current_run = run_set.index(run_idx)
            resamp_rec = wepy_h5.resampling_records([run_idx])
            resamp_panel = resampling_panel(resamp_rec)
            par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
            net_par_table = net_parent_table(par_panel)
            walker_idx = warp[1]
            cycle_idx = warp[2]
            mod_ances = []
            tmp_ances = []
            ances1 = ancestors(net_par_table, cycle_idx, walker_idx)
            for element in ances1:
                mod_ances.append([run_idx, element[0], element[1]])
            walker_idx2 = ances1[0][0]
            for i in range(idx_current_run):
                tmp1_ances = []
                run_idx2 = run_set[idx_current_run -(i+1)]
                resamp_rec = wepy_h5.resampling_records([run_idx2])
                resamp_panel = resampling_panel(resamp_rec)
                par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
                net_par_table = net_parent_table(par_panel)
                cycle_idx2 = wepy_h5.num_run_cycles(run_idx2)-1
                ances2 = ancestors(net_par_table, cycle_idx2, walker_idx2)
                for element2 in ances2:
                        tmp1_ances.append([run_idx2 , element2[0], element2[1]])
                tmp_ances = tmp1_ances + tmp_ances
                walker_idx2 = ances2[0][0]
            mod_ances = tmp_ances + mod_ances 
            all_ancestors.append(mod_ances)

        wepy_h5.close()

        with open(f'{out_dir}/Lig{lig}_all_ancestors.pkl','wb') as f:
            pkl.dump(all_ancestors, f)
            
    else:
        print(f'No warping records for ligand {lig}...')

