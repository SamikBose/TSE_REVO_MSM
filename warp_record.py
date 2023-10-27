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

sum_warping_wts = {}
warping_points  = {}

date = '27thOct'
base_dir = '/dickson/s1/bosesami/comp_unbinding/lig19/vac_cavity_rebinding/'
out_dir  =f'{base_dir}/warping_records_diff_sim_length/{date}'
hdf5 = f'{base_dir}/clone_{date}.h5'
run_network_tree = pkl.load(open(f'{base_dir}/run_network_{date}.pkl','rb'))

wepy_h5 = WepyHDF5(hdf5, mode='r')
wepy_h5.open()
new_warp_rec = []
redundant_unb_event = []
num_warp = 0
cutoff = 2.0

for run in run_network_tree:
    
    dict_net_par_table ={}
    # get the resampling panel of this run tree in a dictionary
    for run_idx in run:
        #print(run_idx)
        resamp_rec = wepy_h5.resampling_records([run_idx])
        resamp_panel = resampling_panel(resamp_rec)
        par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
        dict_net_par_table[run_idx] = net_parent_table(par_panel)
        #print(net_par_table)
    counter  = 0

    # Now run over each single run in the run_tree list
    for counter,run_idx in enumerate(run):
        #counter = 0
        print(f'Running for run tree:{run}, and current run index number:{run_idx}')
        idx_current_run = run.index(run_idx)
        for cycle_idx in range(wepy_h5.num_run_cycles(run_idx)):
            rmsd_per_cycle =  [np.array(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/observables/lig_rmsd/'][cycle_idx]) for walker_idx in range(48)]
            walker_idx_list = np.where(np.asarray(rmsd_per_cycle) < cutoff)[0]
            if len(walker_idx_list) > 0:
                for walker_idx in walker_idx_list:
                    mod_ances = []
                    tmp2 = []
                    ances1 =  ancestors(dict_net_par_table[run_idx], cycle_idx, walker_idx)

                    for idx in ances1:
                        mod_ances.append([run_idx, idx[0], idx[1]])
                    walker_idx2 = ances1[0][0]
                    
                    for i in range(counter):
                        tmp_mod_ances = []
                        run_idx2 = run[idx_current_run -(i+1)]
                        cycle_idx2 = wepy_h5.num_run_cycles(run_idx2)
                            
                        ances2 = ancestors(dict_net_par_table[run_idx2], cycle_idx2, walker_idx2)
                        for idx2 in ances2:
                            tmp_mod_ances.append([run_idx2, idx2[0], idx2[1]])
                        walker_idx2 = ances2[0][0]
                        tmp2 = tmp_mod_ances + tmp2
                    mod_ances = tmp2 + mod_ances
                    #print(mod_ances)                     
                    break_flag = False
                    for idx3 in mod_ances:
                         for j in range(len(new_warp_rec)):
                             if ((new_warp_rec[j][0] in run) and idx3[1] == new_warp_rec[j][1] and idx3[2] == new_warp_rec[j][2]):
                             #if (idx3[1] == new_warp_rec[j][1] and idx3[2] == new_warp_rec[j][2]):
                                 #print(f'This unbinding has already been counted: {run_idx},{walker_idx},{cycle_idx}, {new_warp_rec[j][0]}, {new_warp_rec[j][1]}, {new_warp_rec[j][2]},{j}, {run}')
                                 redundant_unb_event.append([run_idx, walker_idx, cycle_idx])
                                 break_flag = True
                                 break
                         if (break_flag == True):
                             break
                             
                    if(break_flag == False):
                        wt = np.array(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/weights'][cycle_idx])[0]        
                        rmsd_new = np.array(wepy_h5.h5[f'runs/{run_idx}/trajectories/{walker_idx}/observables/lig_rmsd/'][cycle_idx])
                        new_warp_rec.append([run_idx, walker_idx, cycle_idx, wt])   
                        print(f'This unbinding event is added to the new warp records: {run_idx},{walker_idx},{cycle_idx}, {wt}, {rmsd_new}')                          
                        num_warp +=1
wepy_h5.close()

sum1 = 0.0
for i in range(len(new_warp_rec)):
    sum1 += new_warp_rec[i][3]
print(f'For cutoff {cutoff}\AA, umber of warp points: {num_warp}, sum of wts: {sum1}')

with open(f'{out_dir}/Binding_warp_rec_RMSD_cut{cutoff}.pkl','wb') as f:
    pkl.dump(new_warp_rec, f)
with open(f'{out_dir}/Redundant_binding_event_cut{cutoff}.pkl','wb') as f:
    pkl.dump(redundant_unb_event, f)
