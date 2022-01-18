
from functions import *

barcodes = 10 ** 2
cells = 10 ** 2

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

    gens = 30
    system2 = sys.run(system_true, 5)
    df['propgation' + str(gens) + '_true'] = barcodes_id_list(np.array(system2))

    system4 = sys.run(system2, 5)
    df['propgation' + str(2 * gens) + '_true'] = barcodes_id_list(np.array(system4))

    system6 = sys.run(system4, 10)
    df['propgation' + str(3 * gens) + '_true'] = barcodes_id_list(np.array(system6))

    Total_true = np.concatenate((system2, system4, system6))
    A_total_true = system_init.Rapid_SimilarityMatrix(Total_true)

    df_total = pd.DataFrame(barcodes_id_list(Total_true), columns=['true'])

    clustering_total = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
                                               linkage='complete',
                                               compute_full_tree=True).fit(A_total_true)

    df_total['true_cluster'] = clustering_total.labels_

    print('p_ins=', p_ins)
    drop = [0, 0.01, 0.1, 0.5]
    for p_d in range(4):

        # p_drop=p_d/10
        p_drop = drop[p_d]

        system_true_drop = system_init.dropouts(np.array(system_true).copy(), p=p_drop)

        df['drop0'] = barcodes_id_list(np.array(system_true_drop))

        # A_dist0=LargeSystem.SimilarityMatrix(system_true_drop)

        system2_drop = system_init.dropouts(np.array(system2).copy(), p=p_drop)

        df['drop1'] = barcodes_id_list(np.array(system2_drop))

        # A_dist1=LargeSystem.Rapid_SimilarityMatrix(system2_drop)

        system4_drop = system_init.dropouts(np.array(system4).copy(), p=p_drop)

        df['drop2'] = barcodes_id_list(np.array(system4_drop))

        # A_dist2=LargeSystem.Rapid_SimilarityMatrix(system4_drop)

        system6_drop = system_init.dropouts(np.array(system6).copy(), p=p_drop)

        df['drop3'] = barcodes_id_list(np.array(system6_drop))

        # A_dist3=LargeSystem.Rapid_SimilarityMatrix(system6_drop)

        Total_drop = np.concatenate((system2_drop, system4_drop, system6_drop))

        #######
        A_total_drop_Hamming = system_init.Rapid_SimilarityMatrix(Total_drop)
        A_total_drop_drop = system_init.DistanceMatrix_dropout(Total_drop, dropout_prob=p_drop)
        A_total_drop_ins = system_init.DistanceMatrix_dropout_conditional(Total_drop, dropout_prob=p_drop, p_i=p)
        #######

        A_total_drop_Hamming = A_total_drop_Hamming / (A_total_drop_Hamming.max())
        A_total_drop_drop = A_total_drop_drop / (A_total_drop_drop.max())
        A_total_drop_ins = A_total_drop_ins / (A_total_drop_ins.max())

        df_total['dropped'] = barcodes_id_list(Total_drop)

        A_total_drop_CompleteOverlap = system_init.NumBarcodesDiff(Total_drop)
        clustering_total = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
                                                   linkage='complete',
                                                   compute_full_tree=True).fit(A_total_drop_CompleteOverlap)

        df_total['dropped_cluster'] = clustering_total.labels_

        Clustering_Threshold: int
        for Clustering_Threshold in range(200):
            Thres = Clustering_Threshold / 199 + epsilon

            clustering_total_Hamming = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
                                                               affinity='precomputed', linkage='complete',
                                                               compute_full_tree=True).fit(A_total_drop_Hamming)
            clustering_total_drop = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
                                                            affinity='precomputed', linkage='complete',
                                                            compute_full_tree=True).fit(A_total_drop_drop)
            clustering_total_drop_ins = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
                                                                affinity='precomputed', linkage='complete',
                                                                compute_full_tree=True).fit(A_total_drop_ins)

            # temp= clustering_total.labels_
            # print([temp[x:x+int(len(data['barcodes_id_true']) )] for x in range(0, len(temp)  , int(len(data['barcodes_id_true']) ))])
            # temp1=[temp[x:x+int(len(data['barcodes_id_true']) )] for x in range(0, len(temp)  , int(len(data['barcodes_id_true']) ))]
            # df['cluster1']=temp1[0]
            # df['cluster2']=temp1[1]
            # df['cluster3']=temp1[2]

            df_total['matching_after_drop_cluster_Hamming'] = clustering_total_Hamming.labels_
            df_total['matching_after_drop_cluster_drop'] = clustering_total_drop.labels_
            df_total['matching_after_drop_cluster_drop_ins'] = clustering_total_drop_ins.labels_

            cm_pred_hamming = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster,
                                               df_total[df_total['dropped'] != ''].matching_after_drop_cluster_Hamming)
            cm_pred_drop = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster,
                                            df_total[df_total['dropped'] != ''].matching_after_drop_cluster_drop)
            cm_pred_drop_ins = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster, df_total[
                df_total['dropped'] != ''].matching_after_drop_cluster_drop_ins)

            wrong_lin_split_hamming[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_hamming, axis=1).sum(
                axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
            wrong_lin_merge_hamming[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_hamming, axis=0).sum(
                axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])

            wrong_lin_split_drop[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop, axis=1).sum(axis=0)).sum() / (
                        df_total[df_total['dropped']!=''].shape[0])
            wrong_lin_merge_drop[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop, axis=0).sum(axis=0)).sum() / (
                        df_total[df_total['dropped']!=''].shape[0])

            wrong_lin_split_drop_ins[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop_ins, axis=1).sum(
                axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
            wrong_lin_merge_drop_ins[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop_ins, axis=0).sum(
                axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])

            number_true_clusters[p_i, p_d, Clustering_Threshold] = df_total[
                df_total['dropped'] != ''].true_cluster.nunique()
            number_measured_clusters[p_i, p_d, Clustering_Threshold] = df_total[
                df_total['dropped'] != ''].dropped_cluster.nunique()

            number_cluster_after_matching_Hamming[p_i, p_d, Clustering_Threshold] = df_total[
                df_total['dropped'] != ''].matching_after_drop_cluster_Hamming.nunique()
            number_cluster_after_matching_drop[p_i, p_d, Clustering_Threshold] = df_total[
                df_total['dropped'] != ''].matching_after_drop_cluster_drop.nunique()
            number_cluster_after_matching_drop_ins[p_i, p_d, Clustering_Threshold] = df_total[
                df_total['dropped'] != ''].matching_after_drop_cluster_drop_ins.nunique()

        #  Num_of_measured_lin[p_i,p_d,Clustering_Threshold]=number_cluster_after_matching[p_i,p_d,Clustering_Threshold]/number_true_clusters[p_i,p_d,Clustering_Threshold]-1

        R = list(abs(1 - number_cluster_after_matching_Hamming[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
        best_i = R.index(min(R))
        score_split_Hamming[p_i, p_d] = wrong_lin_split_hamming[p_i, p_d, best_i]
        score_merge_Hamming[p_i, p_d] = wrong_lin_merge_hamming[p_i, p_d, best_i]
        best_dist_Hamming[p_i, p_d] = best_i / 199 + epsilon

        R = list(abs(1 - number_cluster_after_matching_drop[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
        best_i = R.index(min(R))
        score_split_drop[p_i, p_d] = wrong_lin_split_drop[p_i, p_d, best_i]
        score_merge_drop[p_i, p_d] = wrong_lin_merge_drop[p_i, p_d, best_i]
        best_dist_drop[p_i, p_d] = best_i / 199 + epsilon

        R = list(abs(1 - number_cluster_after_matching_drop_ins[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
        best_i = R.index(min(R))
        score_split_drop_ins[p_i, p_d] = wrong_lin_split_drop_ins[p_i, p_d, best_i]
        score_merge_drop_ins[p_i, p_d] = wrong_lin_merge_drop_ins[p_i, p_d, best_i]
        best_dist_drop_ins[p_i, p_d] = best_i / 199 + epsilon

        print(p_i,p_d,score_split_Hamming[p_i,p_d])

#print(score_split_Hamming[:3,:4])
cmap = plt.get_cmap('viridis', 20)
plt.figure(figsize=(25,10))
plt.subplot(3, 3, 1)
plt.pcolormesh(score_split_Hamming[0:3,0:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.xlabel('mean barcodes cassette')
plt.ylabel('dropout prob.')
plt.title('best score split')
plt.colorbar()
plt.subplot(3, 3, 2)
plt.pcolormesh(score_merge_Hamming[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best score merge')
plt.xlabel('mean barcodes cassette')
plt.ylabel('dropout prob.')
plt.colorbar()
plt.subplot(3, 3, 3)
plt.pcolormesh(best_dist_Hamming[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best dist = matching threshold')
plt.xlabel('mean barcodes cassette')
plt.ylabel('dropout prob.')
plt.colorbar()

plt.subplot(3, 3, 4)
plt.pcolormesh(score_split_drop[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.xlabel('mean barcodes cassette')
plt.ylabel('dropout prob.')
plt.title('best score split')
plt.colorbar()
plt.subplot(3, 3, 5)
plt.pcolormesh(score_merge_drop[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best score merge')
plt.xlabel('mean barcodes cassette')
plt.ylabel('droupout prob.')
plt.colorbar()
plt.subplot(3, 3, 6)
plt.pcolormesh(best_dist_drop[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best dist = matching thershold')
plt.xlabel('mean barcodes cassette')
plt.ylabel('droupout prob.')
plt.colorbar()
plt.subplot(3, 3, 7)
plt.pcolormesh(score_split_drop_ins[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.xlabel('mean barcodes cassette')
plt.ylabel('droupout prob.')
plt.title('best scrore split')
plt.colorbar()
plt.subplot(3, 3, 8)
plt.pcolormesh(score_merge_drop_ins[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best scrore merge')
plt.xlabel('mean barcodes cassette')
plt.ylabel('droupout prob.')
plt.colorbar()
plt.subplot(3, 3, 9)
plt.pcolormesh(best_dist_drop_ins[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
plt.title('best dist = matching thershold')
plt.xlabel('mean barcodes cassette')
plt.ylabel('droupout prob.')
plt.colorbar()
plt.show()