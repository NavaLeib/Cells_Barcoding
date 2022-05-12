
from functions import *

from sklearn.cluster import KMeans
from sklearn import metrics

import csv


# import requests
import csv
import glob
import shutil
import pathlib
#from fileUtil import writeFile
#from zipperUtil import unzip
import os
#from splitLog import logSplit
import matplotlib.pyplot as plt
#from xlsxwriter.workbook import Workbook

path = str(pathlib.Path(__file__).parent.resolve())
# get folder



barcodes = 10 ** 4
cells = 10 ** 3


p1=[]
p2=[]

wrong_lin_split_hamming = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]
wrong_lin_merge_hamming = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]

wrong_lin_split_drop = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]
wrong_lin_merge_drop = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]

wrong_lin_split_drop_ins = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]
wrong_lin_merge_drop_ins = np.zeros([20, 10, 2000])  # dimensions [p_inseration, p_dropouts, thershold ]

number_true_clusters = np.zeros([20, 10, 2000])
number_measured_clusters = np.zeros([20, 10, 2000])
number_cluster_after_matching_Hamming = np.zeros([20, 10, 2000])
number_cluster_after_matching_drop = np.zeros([20, 10, 2000])
number_cluster_after_matching_drop_ins = np.zeros([20, 10, 2000])

Num_of_measured_lin = np.zeros([20, 10, 2000])

num_clusters = np.zeros([20, 10])

score_split_Hamming = np.zeros([20, 10])
score_merge_Hamming = np.zeros([20, 10])
best_dist_Hamming = np.zeros([20, 10])
score_split_drop = np.zeros([20, 10])
score_merge_drop = np.zeros([20, 10])
best_dist_drop = np.zeros([20, 10])
score_split_drop_ins = np.zeros([20, 10])
score_merge_drop_ins = np.zeros([20, 10])
best_dist_drop_ins = np.zeros([20, 10])

T = np.zeros([20, 10])

epsilon = 10 ** -323

data_full = {}

ins = [0.5, 2, 10]
for p_i in range(3):

    # p_ins=p_i+0.5
    p_ins = ins[p_i]

    system_init = LargeSystem(barcodes=barcodes, cells=cells)
    p = p_ins / barcodes
    system_true = system_init.generate_barcoded_cells(p=p, Poisson=True)

    system_true = system_true[system_true.sum(axis=1) > 0]  # taking only cells with labels (=barcodes inserted)

    df = pd.DataFrame(barcodes_id_list(system_true), columns=['barcodes_id_true'])

    data = {}

    data['barcodes_id_true'] = barcodes_id_list(system_true)

    sys = Simulation(barcodes=barcodes, cells=cells, generations=1)

    gens1 = 5
    system2 = sys.run(system_true, gens1)
    df['propagation_' + str(gens1) + 'geneartions_true'] = barcodes_id_list(np.array(system2))

    gens2 = 5
    system4 = sys.run(system2, gens2 )
    df['propagation_' + str(gens1+gens2) + 'geneartions_true'] = barcodes_id_list(np.array(system4))

    gens3 = 5
    system6 = sys.run(system4, gens3)
    df['propagation_' + str(gens1+gens2+gens3) + 'geneartions_true'] = barcodes_id_list(np.array(system6))

    Total_true = np.concatenate((system2, system4, system6))
    A_total_true = system_init.Rapid_SimilarityMatrix(Total_true)

    df_total = pd.DataFrame(barcodes_id_list(Total_true), columns=['true'])

    clustering_total = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
                                               linkage='complete',
                                               compute_full_tree=True).fit(A_total_true)

    df_total['true_cluster'] = clustering_total.labels_

    print('p_ins=', p_ins)

    outname = 'barcodes_list_id_cells' + str(cells) + '_barcodes' + str(barcodes) +\
              '_moi' + str(p_ins) + '_nodrop.csv'

    root = 'Data'
    subdir = 'cells'+ str(cells) + '_barcodes' + str(barcodes)
    outdir = os.path.join(root, subdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df.to_csv(f"{outdir}/{outname}")

    drop = [0, 0.01, 0.1, 0.5]
    for p_d in range(1,4):

        # p_drop=p_d/10
        p_drop = drop[p_d]

        system_true_drop = system_init.dropouts(np.array(system_true).copy(), p=p_drop)

        df['drop_no_propagation'] = barcodes_id_list(np.array(system_true_drop))

        # A_dist0=LargeSystem.SimilarityMatrix(system_true_drop)

        system2_drop = system_init.dropouts(np.array(system2).copy(), p=p_drop)

        df['drop_' + str(gens1) + 'generation'] = barcodes_id_list(np.array(system2_drop))

        system4_drop = system_init.dropouts(np.array(system4).copy(), p=p_drop)

        df['drop_' + str(gens1 + gens2) + 'generation'] = barcodes_id_list(np.array(system4_drop))

        system6_drop = system_init.dropouts(np.array(system6).copy(), p=p_drop)

        df['drop_' + str(gens1 + gens2+ gens3) + 'generation'] = barcodes_id_list(np.array(system6_drop))

        outname = 'barcodes_list_id_cells' + str(cells) + '_barcodes' + str(barcodes) + \
                  '_moi' + str(p_ins) + '_drop' + str(p_drop) + '.csv'

        df.to_csv(f"{outdir}/{outname}")




