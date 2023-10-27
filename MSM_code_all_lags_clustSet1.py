## Version update:
import networkx as nx
import mdtraj as mdj
import numpy as np
import pickle as pkl
import sys
import os
from pathlib import Path
import os.path as osp

from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from wepy.hdf5 import WepyHDF5
from os.path import join
from wepy.util.util import box_vectors_to_lengths_angles
import time
from csnanalysis.csn import CSN
from csnanalysis.matrix import *
from wepy.hdf5 import WepyHDF5
from wepy.util.util import traj_box_vectors_to_lengths_angles

from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.analysis.parents import (parent_panel,
                                   net_parent_table,
                                   parent_table_discontinuities)

from wepy.analysis.parents import sliding_window
from wepy.analysis.transitions import run_transition_counts_matrix

from wepy.analysis.contig_tree import ContigTree

from deeptime.decomposition import TICA
from deeptime.util.data import TimeLaggedDataset
from deeptime.clustering import KMeans
import ipdb

start = time.time()


### Paths and etc... 
username = 'bosesami'
base_path = f'/dickson/s1/{username}/sEH_combined/'
sEH_dir = base_path
sEH_19_dir = '/dickson/s1/bosesami/KSL_unbinding/KSL_19/simulation_folder/openmm/'

# If there are different numbers for same ligand (isn't the case now but can happen in future)
all_lig_sets = [['10'],['17'],['19'],['28'],['29'],['50'] ]
all_ligs = ['10','17','19','28','29','50']

# number of tica dimensions (3, 5, 10 ... etc)
n_tica_dim = int(sys.argv[1]) 

#Change this to more optim fashion
n_clusters_set = [250,500,800,1200]

# Can have different values once my time lagged data code is ready
tau_list = [1]

tau_tica = 1

window_length_tica = tau_tica + 1

n_tica_train = 750000
n_walkers = 48
n_eval_points = 5000

rmsd_cutoff_bound = 2.0 # angstroms
mind_cutoff_unbound = 0.5 # nm

nbs_atoms = 56 ## Read pdb and get it from there

## Index mapping of the six same atoms in the central part of the ligands which are 
## used to build the feature set for each ligand unbinding simulation
lig_atom_mappings = {}
lig_atom_mappings['10'] = [2,1,0,3,4,5]
lig_atom_mappings['17'] = [2,1,0,3,4,5]
lig_atom_mappings['28'] = [5,0,1,2,3,4]
lig_atom_mappings['29'] = [5,0,1,2,3,4]
lig_atom_mappings['50'] = [5,0,1,2,3,4]
lig_atom_mappings['19'] = [5,2,0,1,3,4]


# path to ligand h5s
h5_path = {}
h5_path['10'] = f'{sEH_dir}/10/all_results.wepy.h5'
h5_path['17'] = f'{sEH_dir}/17/all_results.wepy.h5'
h5_path['28'] = f'{sEH_dir}/28/REVO/clone_all.h5'
h5_path['29'] = f'{sEH_dir}/29/REVO/clone.h5'
h5_path['50'] = f'{sEH_dir}/50/REVO/clone_all.h5'
h5_path['19'] = f'{sEH_19_dir}/clone.h5'


# Not inputs, these are files computed only once 
pkl_path = f'{base_path}/feature.pkl'
ledger_path = f'{base_path}/ledger.pkl'


# Specific to multi-ligand TSE work, you may not need this...
old_runs = {}
old_runs['10'] = [i for i in range(18)]
old_runs['17'] = [i for i in range(18)]
old_runs['28'] = [0,2,4,6,8]
old_runs['29'] = [0,2,4,6,8]
old_runs['50'] = [0,3,6,9,12]
old_runs['19'] = []

new_runs = {}
new_runs['10'] = []
new_runs['17'] = []
new_runs['19'] = [i for i in range(10)]
new_runs['28'] = [1,3,5,7,9]
new_runs['29'] = [1,3,5,7,9]
new_runs['50'] = [1,2,4,5,7,8,10,11,13,14]

if osp.isfile(pkl_path) and osp.isfile(ledger_path):
    print(time.time()-start,"Loading features and ledger..")
    lig_bs_dists = pkl.load(open(pkl_path,'rb'))
    ledger = pkl.load(open(ledger_path,'rb'))
else:
    print(time.time()-start,"Reading features from hdf5 files..")
    lig_bs_dists = []
    ledger = []
    for lig in all_ligs:
        # determine distance mapping
        dist_idx_map = []
        for lig_idx in lig_atom_mappings[lig]:
            for i in range(nbs_atoms):
                dist_idx_map.append(lig_idx*nbs_atoms + i)

        wepy_h5 = WepyHDF5(h5_path[lig], mode='r')
        wepy_h5.open()
        nruns = len(wepy_h5.runs)
        for run in range(nruns):
            if run in new_runs[lig]:
                scrambled_dists = np.array([wepy_h5.h5[f'/runs/{run}/trajectories/{i}/observables/lig_bs_dist_features'] for i in range(n_walkers)])
            if run in old_runs[lig]:
                scrambled_dists = np.array([wepy_h5.h5[f'/runs/{run}/trajectories/{i}/observables/lig-bs_dists'] for i in range(n_walkers)])

            unscrambled_dists = scrambled_dists[:,:,dist_idx_map]
            lig_bs_dists.append(unscrambled_dists)
            n_cycles = unscrambled_dists.shape[1]
            ledger.append((lig,run,n_cycles))
        wepy_h5.close()

    with open(pkl_path,'wb') as f:
        pkl.dump(lig_bs_dists,f)
    with open(ledger_path,'wb') as f:
        pkl.dump(ledger,f)

# determine the total number of cycles for each ligand
tot_cyc = {}
n_runs = {}
for entry in ledger:
    if entry[0] not in tot_cyc:
        tot_cyc[entry[0]] = 0
        n_runs[entry[0]] = 0
    tot_cyc[entry[0]] += entry[2]
    n_runs[entry[0]] += 1

# Clustering
for n_clusters in  n_clusters_set:
    print(f'Running for n_Clusters: {n_clusters}')
    analysis_folder = osp.join(base_path,f'MSM_models/model_{n_tica_dim}_{n_clusters}')

    cluster_labels = {}
    for lig_set in all_lig_sets:
        label = '_'.join(lig_set)

        for lig in lig_set:
            clust_name = osp.join(analysis_folder,f'cluster_labels_{lig}_nclust{n_clusters}.pkl')

            if osp.exists(clust_name):
                print(time.time()-start,"Reading cluster labels for ligand",lig)
                cluster_labels[lig] = pkl.load(open(clust_name,'rb'))
            
            else:

                feat_name = f'tica_features_{label}_TICAlag{tau_tica}.pkl'
            
                if not osp.exists(osp.join(analysis_folder,feat_name)):
                    print(time.time()-start,f"Building time-lagged dataset for ligand{label}..")     
                    # select runs that belong to this ligand set
                    lig_run_idxs = [i for i in range(len(ledger)) if ledger[i][0] in lig_set]
                    # make a concatenated list of lig_bs_dists that are unique to this ligand set
                    lig_bs_dists_spec = [lig_bs_dists[i] for i in lig_run_idxs]
                    print(time.time()-start,"Reading contig tree..")
                    # build a time-lagged dataset to train TICA
                    all_sws = []
                    n_runs_to_add = 0
                    for lig in lig_set:
                        wepy_h5 = WepyHDF5(h5_path[lig], mode='r')
                        with wepy_h5:
                            ct = ContigTree(wepy_h5,decision_class=MultiCloneMergeDecision,boundary_condition_class=UnbindingBC)

                        lig_sw = np.array(ct.sliding_windows(window_length_tica))

                        # offset the run indices to be compatible with the shape of lig_run_idxs
                        lig_sw[:,:,0] += n_runs_to_add

                        n_runs_to_add += n_runs[lig]
                        all_sws.append(lig_sw)

                    sw = np.concatenate(all_sws,axis=0)

                    # Don't train using transitions between runs (which aren't reliable)
                    valid_sw = [i for i in range(len(sw)) if sw[i,0][0] == sw[i,1][0]]
                    if len(valid_sw) < n_tica_train:
                        sw_tica_train = sw[valid_sw]
                        n_tica_train = len(valid_sw)
                    else:
                        sw_tica_train = sw[np.random.choice(valid_sw,size=n_tica_train,replace=False)]

                    n_dists = lig_bs_dists_spec[0].shape[2]

                    tl_data = np.zeros((2,n_tica_train,n_dists))
                    for i in range(n_tica_train):
                        run1,traj1,cycle1 = sw_tica_train[i,0]
                        run2,traj2,cycle2 = sw_tica_train[i,-1]
                        tl_data[0,i] = lig_bs_dists_spec[run1][traj1,cycle1]
                        tl_data[1,i] = lig_bs_dists_spec[run2][traj2,cycle2]

                    tld = TimeLaggedDataset(tl_data[0],tl_data[1])
                    print(time.time()-start,"Training TICA..")
                    estimator = TICA(dim=n_tica_dim, lagtime=1).fit(tld)
                    feats = [estimator.transform(dists) for dists in lig_bs_dists_spec]

                    # write the features in a pkl file
                    with open(osp.join(analysis_folder,feat_name),'wb') as f:
                        pkl.dump(feats,f)
                else:
                    print(time.time()-start,"Reading TICA features..")
                    with open(osp.join(analysis_folder,feat_name),'rb') as f:
                        feats = pkl.load(f)

                print(time.time()-start,"Clustering features..")
                tmp = np.concatenate(feats,axis=1)
                tot_cycles = tmp.shape[1]
                big_feat_list = tmp.reshape(n_walkers*tot_cycles,n_tica_dim)

                c_labels = KMeans(n_clusters = n_clusters, fixed_seed=True, n_jobs=None).fit_transform(big_feat_list)
                c_labels = c_labels.reshape(n_walkers,tot_cycles)
                cycle_counter = 0

                for lig in lig_set:
                    cluster_labels[lig] = c_labels[:,cycle_counter:cycle_counter+tot_cyc[lig]]
                    cycle_counter += tot_cyc[lig]

                    # write the labels in a pkl file
                    #clust_name = osp.join(analysis_folder,f'tica_cluster_labels_{lig}_TICA_lag{tau_tica}_nclust{n_clusters}.pkl')
                    with open(clust_name,'wb') as f:
                        pkl.dump(cluster_labels[lig],f)
    
    prop_path = osp.join(analysis_folder,f'properties_lag1_nclust{n_clusters}.pkl')
    if not osp.isfile(prop_path):
        properties = {}
        for lig_set in all_lig_sets:
            label = '_'.join(lig_set)

            tot_wts = np.zeros((n_clusters))
            av_rmsds = np.zeros((n_clusters))
            #av_unb_min_dists = np.zeros((n_clusters))
            max_unb_min_dists = np.zeros((n_clusters))
            nav_rmsds = np.zeros((n_clusters))
            #nav_unb_min_dists = np.zeros((n_clusters))
            free_energy_max = 100.

            field_list = ['weights','observables/lig_rmsd','observables/unb_min_dist']
            properties[label] = {}

            for lig in lig_set:
                print(time.time()-start,f"Determining node properties for ligand {lig}..")
                wepy_h5 = WepyHDF5(h5_path[lig], mode='r')

                # reshape the cluster labels again
                lig_run_list = []
                for idx,entry in enumerate(ledger):
                    if entry[0] == lig:
                        # append number of cycles (for this run) to lig_run_list
                        lig_run_list.append(entry[2])

                cl_reshape = []
                cycle_counter = 0
                for run_cyc in lig_run_list:
                    cl_reshape.append(cluster_labels[lig][:,cycle_counter:cycle_counter+run_cyc])
                    cycle_counter += run_cyc

                wts = np.zeros((n_clusters))
                with wepy_h5:
                    for clust in range(n_clusters):
                        trace = []
                        for run, clust_idxs in enumerate(cl_reshape):
                            walkers, cycles = np.where(clust_idxs == clust)
                            n = len(walkers)
                            if n > 0:
                                runs = np.ones((n),dtype=int)*run
                                trace += list(zip(runs,walkers,cycles))

                        if len(trace) > 0:
                            if len(trace) > n_eval_points:
                                sub_trace = [trace[i] for i in np.random.choice(len(trace),n_eval_points)]
                                props = wepy_h5.get_trace_fields(sub_trace, field_list)
                            else:
                                props = wepy_h5.get_trace_fields(trace, field_list)

                            wts[clust] = props['weights'][:,0].mean()
                            av_rmsds[clust] += props['observables/lig_rmsd'].sum()
                            max_unb = props['observables/unb_min_dist'].max()
                            if max_unb > max_unb_min_dists[clust]:
                                max_unb_min_dists[clust] = max_unb

                            nav_rmsds[clust] += len(props['observables/lig_rmsd'])



                tot_wts += wts

            # get free energy from weights
            fe = -np.log(tot_wts,where=tot_wts>0)
            fe -= fe[np.where(tot_wts>0)].min()
            fe[np.where(tot_wts==0)] = free_energy_max
            properties[label]['fe'] = fe

            # add total weight to csn
            properties[label]['tot_weight'] = tot_wts/tot_wts.sum()

            # add averaged attributes to csn
            av_rmsds /= nav_rmsds

            properties[label]['lig_rmsd'] = av_rmsds

            properties[label]['unb_min_dist'] = max_unb_min_dists

            # determine bound and unbound states
            bound = av_rmsds < rmsd_cutoff_bound
            unbound = max_unb_min_dists > mind_cutoff_unbound

            properties[label]['is_bound'] = np.array(bound,dtype=int)
            properties[label]['is_unbound'] = np.array(unbound,dtype=int)

        with open(prop_path,'wb') as f:
            pkl.dump(properties,f)

    else:
        print(f"Loading properties from {prop_path}..")
        properties = pkl.load(open(prop_path,'rb'))

    ### loop for lag_list starts here:
    for tau in tau_list:
        window_length = tau + 1
        print(f'Running for lagtime:{tau}')
        counts_name = osp.join(analysis_folder,f'countsmat_lag{tau}_nclust{n_clusters}.pkl')
            
        if osp.exists(counts_name):
            print(time.time()-start,"Reading all counts matrices..")
            all_counts = pkl.load(open(counts_name,'rb'))

        else:
            print('Countsmat does not exist for lag{tau} and n_cluster:{n_clusters}')
            all_counts = {}
            for lig in all_ligs:
                all_counts[lig] = np.zeros((n_clusters,n_clusters))

            # use the cluster labels to build counts matrices
            for lig in all_ligs:
                print(time.time()-start,f"Building counts matrix for lig {lig}..")
                wepy_h5 = WepyHDF5(h5_path[lig], mode='r')
                with wepy_h5:
                    ct = ContigTree(wepy_h5,decision_class=MultiCloneMergeDecision,boundary_condition_class=UnbindingBC)
                    nruns = len(wepy_h5.runs)
                    all_wts = []
                    for i in range(nruns):
                        all_wts.append(np.array([wepy_h5.h5[f'runs/{i}/trajectories/{j}/weights'][:,0] for j in range(48)]))
    
                sw = np.array(ct.sliding_windows(window_length))

                # reshape the cluster labels for easy lookup
                lig_run_list = []
                for idx,entry in enumerate(ledger):
                    if entry[0] == lig:
                        # append number of cycles (for this run) to lig_run_list
                        lig_run_list.append(entry[2])

                assert np.array(lig_run_list).sum() == cluster_labels[lig].shape[1], \
                'Error! number of cycles in cluster_labels does not agree with ledger'
    
                cl_reshape = []
                cycle_counter = 0
                for run_cyc in lig_run_list:
                    cl_reshape.append(cluster_labels[lig][:,cycle_counter:cycle_counter+run_cyc])
                    cycle_counter += run_cyc
    
                # add transitions to count matrix using mashup table
                for s in sw:
                    run1, traj1, cycle1 = s[0]
                    run2, traj2, cycle2 = s[-1]
                    if run1 == run2:
                        w1 = all_wts[run1][traj1,cycle1]
                        w2 = all_wts[run2][traj2,cycle2]
                        c1 = cl_reshape[run1][traj1,cycle1]
                        c2 = cl_reshape[run2][traj2,cycle2]

                        ## Modification here:::
                        ## Old counts mat:
                        all_counts[lig][c1,c2] += w2
                        ## New attempts:
                        #all_counts[lig][c1,c2] += w2/w1
                        #all_counts[lig][c2,c1] += w1

            with open(counts_name,'wb') as f:
                pkl.dump(all_counts,f)

        #  Great.  Now determine rates!
        mfpts = {}
        fptds = {}
        committors = {}
        csns = {}
        for lig_set in all_lig_sets:
            print(f"Lig set: {lig_set}")
            label = '_'.join(lig_set)
            
            print(f"Building and trimming CSN..")
            lig_csn = CSN(all_counts[label].T) # CSN expects [to][from]
            lig_csn.trim()
            unb = np.where(properties[label]['is_unbound']==1)[0]
            b = np.where(properties[label]['is_bound']==1)[0]
        
            trim_unb = [state for state in unb if state in lig_csn.trim_indices]
            trim_b = [state for state in b if state in lig_csn.trim_indices]
        
            csns[label] = lig_csn
        
            if len(trim_unb) > 0 and len(trim_b) > 0:
                print(f"Determining mfpts..")
                mfpt, fptds[label] = lig_csn.calc_mfpt(trim_unb, maxsteps=200, tol=1e-2, sources=trim_b)
                if fptds[label].shape[1] == 200:
                    unb_frac = fptds[label][0].sum()
                    last_10 = fptds[label][0][-10:].sum()
                    print(f"Max steps reached for {lig_set}.  Unbound fraction = {unb_frac} (last 10: {last_10})")
                mfpts[label] = mfpt*tau

                print("Determining committors..")
                committors[label] = lig_csn.calc_committors([trim_b,trim_unb],maxstep=200)
            else:
                print(f"Not determining mfpts (len(unb) = {len(trim_unb)}, len(b) = {len(trim_b)})")

        with open(osp.join(analysis_folder,f'committors_lag{tau}_nclust{n_clusters}.pkl'),'wb') as f:
            pkl.dump(committors,f)

        with open(osp.join(analysis_folder,f'mfpts_lag{tau}_nclust{n_clusters}.pkl'),'wb') as f:
            pkl.dump(mfpts,f)

        with open(osp.join(analysis_folder,f'fptds_lag{tau}_nclust{n_clusters}.pkl'),'wb') as f:
            pkl.dump(fptds,f)

    #with open(osp.join(analysis_folder,f'csns_{tau}.pkl'),'wb') as f:
    #    pkl.dump(csns,f)

