
from functions import *

from sklearn.cluster import KMeans
from sklearn import metrics

import os
import csv

import matplotlib.pyplot as plt
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
num_measured_cells = np.zeros([20, 10])

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

path = 'Data'
file_extension = '.csv'
csv_file_list = []
for root, dirs, files in os.walk(path):
    #print(root, dirs, files)
    for name in files:
        if name.endswith(file_extension):
            file_path = os.path.join(root, name)
            csv_file_list.append(file_path)

data={}
for f in csv_file_list:
    file_name=f[f.rfind('\\'):]  ##THIS CHANGE TO RUN ON UOT SEREVER ONLY
    #file_name=f[f.rfind('/'):][1:]
    #print(file_name, '\n')
    data[file_name] = pd.read_csv(f)

# print(data.keys())

print('UPLOAD DATA')


ins=[0.5,2,10]  #MOI array
drop = [0, 0.01, 0.1, 0.5]  #probability drop array

ins=[0.5,2,5,10]  #MOI array  ##ADDED TO 10^3 CELLS ONLY MOI=5!!!

for p_i in range(4):#range(len(ins)):
    p_ins= ins[p_i]

    for p_d in range(1,4):

        p_drop = drop[p_d]

        ##THIS CHANGE TO RUN ON UOT SEREVER ONLY
        file_name = '\\barcodes_list_id_cells' + str(cells) +'_barcodes' +str(barcodes) + '_moi' + str(p_ins) + \
                   '_drop' + str(p_drop) + '.csv'
        if p_drop == 0:
           file_name = '\\barcodes_list_id_cells' + str(cells) + '_barcodes' + str(barcodes) + '_moi' + str(p_ins) + \
                       '_nodrop.csv'
                        
                        
        # file_name = 'barcodes_list_id_cells' + str(cells) +'_barcodes' +str(barcodes) + '_moi' + str(p_ins) + \
        #             '_drop' + str(p_drop) + '.csv'
        # if p_drop == 0:
        #     file_name = 'barcodes_list_id_cells' + str(cells) + '_barcodes' + str(barcodes) + '_moi' + str(p_ins) + \
        #                 '_nodrop.csv'

        df=data[file_name]

        df.applymap(lambda x: str(x).split('.0')[0])

        print('start analysis; p_ins:', p_ins, ', p_drop:', p_drop)
        analysis = Lineages_Analysis(df)
        analysis.num_lineages_between_times(['propagation_5geneartions_true', 'propagation_10geneartions_true','propagation_15geneartions_true'],
        ['drop_5generation', 'drop_10generation','drop_15generation'])
        # analysis.num_lineages_between_times(['drop_5generation','drop_10generation'])
        #
        #analysis = System_Analysis(df)
        #key = 'propagation_5geneartions_true'
        #(v_score, Threshold)= analysis.clustering_score(key_true= 'propagation_5geneartions_true', key_drop = 'drop_5generation')
        #print(v_score)
        #T[p_i,p_d] = v_score

        #perfect_clust= analysis.number_of_perfect_clusters(key_true= 'propagation_5geneartions_true', key_drop = 'drop_5generation')


        #num_measured_cells[p_i,p_d] = df.shape[0]-df[df['drop_5generation'].isnull()].shape[0]
        #print('start analysis; p_ins:', p_ins, 'p_drop', p_drop, 'measured cells', num_measured_cells[p_i,p_d])
        #print(T[:4,:4])


    print(num_measured_cells[:4,:4])
    # clustering = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
    #                                      linkage='complete',
    #                                      compute_full_tree=True).fit(Sim_Matrix)
    # df['clusters'] = clustering.labels_


        # plt.plot(ins,np.array(num_infected_cells)/cells,'o', label='simulation')
        # plt.plot(np.linspace(start=0,stop=10,num=50),[1-np.exp(-i) for i in np.linspace(start=0,stop=10,num=50)],label='analytic')
        # plt.ylabel('% barcoded cells')
        # plt.xlabel('MOI')
        # plt.legend()
plt.show()


# ins = [0.5, 2, 10]
# for p_i in range(3):
#
#     Total_true = np.concatenate((system2, system4, system6))
#     A_total_true = system_init.Rapid_SimilarityMatrix(Total_true)
#
#     df_total = pd.DataFrame(barcodes_id_list(Total_true), columns=['true'])
#
#     clustering_total = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
#                                                linkage='complete',
#                                                compute_full_tree=True).fit(A_total_true)
#
#     df_total['true_cluster'] = clustering_total.labels_
#
#     print('p_ins=', p_ins)
#
#     df.to_csv('true_barcodes_list_id_cells' +str(cells) + '_barcodes' + str(barcodes) +
#               '_moi' + str(p_ins) + '.csv', index=False)
#
#     drop = [0, 0.01, 0.1, 0.5]
#     for p_d in range(2,3):
#
#         # p_drop=p_d/10
#         p_drop = drop[p_d]
#
#         system_true_drop = system_init.dropouts(np.array(system_true).copy(), p=p_drop)
#
#         df['drop0'] = barcodes_id_list(np.array(system_true_drop))
#
#         # A_dist0=LargeSystem.SimilarityMatrix(system_true_drop)
#
#         system2_drop = system_init.dropouts(np.array(system2).copy(), p=p_drop)
#
#         df['drop1'] = barcodes_id_list(np.array(system2_drop))
#
#         # A_dist1=LargeSystem.Rapid_SimilarityMatrix(system2_drop)
#
#         system4_drop = system_init.dropouts(np.array(system4).copy(), p=p_drop)
#
#         df['drop2'] = barcodes_id_list(np.array(system4_drop))
#
#         # A_dist2=LargeSystem.Rapid_SimilarityMatrix(system4_drop)
#
#         system6_drop = system_init.dropouts(np.array(system6).copy(), p=p_drop)
#
#         df['drop3'] = barcodes_id_list(np.array(system6_drop))
#
#         # A_dist3=LargeSystem.Rapid_SimilarityMatrix(system6_drop)
#
#         A_drop1_Hamming = system_init.Rapid_SimilarityMatrix(system2_drop)
#         A_drop1_drop = system_init.DistanceMatrix_dropout(system2_drop, dropout_prob=p_drop)
#         A_drop1_ins = system_init.DistanceMatrix_dropout_conditional(system2_drop, dropout_prob=p_drop, p_i=p)
#
#         Total_drop = np.concatenate((system2_drop, system4_drop, system6_drop))
#
#         #######
#         A_total_drop_Hamming = system_init.Rapid_SimilarityMatrix(Total_drop)
#         A_total_drop_drop = system_init.DistanceMatrix_dropout(Total_drop, dropout_prob=p_drop)
#         A_total_drop_ins = system_init.DistanceMatrix_dropout_conditional(Total_drop, dropout_prob=p_drop, p_i=p)
#         #######
#
#         A_total_drop_Hamming = A_total_drop_Hamming / (A_total_drop_Hamming.max())
#         A_total_drop_drop = A_total_drop_drop / (A_total_drop_drop.max())
#         A_total_drop_ins = A_total_drop_ins / (A_total_drop_ins.max())
#
#         df_total['dropped'] = barcodes_id_list(Total_drop)
#
#         A_total_drop_CompleteOverlap = system_init.NumBarcodesDiff(Total_drop)
#         clustering_total_drop = AgglomerativeClustering(distance_threshold=epsilon, n_clusters=None, affinity='precomputed',
#                                                    linkage='complete',
#                                                    compute_full_tree=True).fit(A_total_drop_CompleteOverlap)
#
#         df_total['dropped_cluster'] = clustering_total_drop.labels_
#
#
#         for Clustering_Threshold in range(200):
#             Thres = Clustering_Threshold / 199 + epsilon
#
#             clustering_total_Hamming = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
#                                                                affinity='precomputed', linkage='complete',
#                                                                compute_full_tree=True).fit(A_total_drop_Hamming)
#             clustering_total_drop = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
#                                                             affinity='precomputed', linkage='complete',
#                                                             compute_full_tree=True).fit(A_total_drop_drop)
#             clustering_total_drop_ins = AgglomerativeClustering(distance_threshold=Thres, n_clusters=None,
#                                                                 affinity='precomputed', linkage='complete',
#                                                                 compute_full_tree=True).fit(A_total_drop_ins)
#
#             # temp= clustering_total.labels_
#             # print([temp[x:x+int(len(data['barcodes_id_true']) )] for x in range(0, len(temp)  , int(len(data['barcodes_id_true']) ))])
#             # temp1=[temp[x:x+int(len(data['barcodes_id_true']) )] for x in range(0, len(temp)  , int(len(data['barcodes_id_true']) ))]
#             # df['cluster1']=temp1[0]
#             # df['cluster2']=temp1[1]
#             # df['cluster3']=temp1[2]
#
#             df_total['matching_after_drop_cluster_Hamming'] = clustering_total_Hamming.labels_
#             df_total['matching_after_drop_cluster_drop'] = clustering_total_drop.labels_
#             df_total['matching_after_drop_cluster_drop_ins'] = clustering_total_drop_ins.labels_
#
#             cm_pred_hamming = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster,
#                                                df_total[df_total['dropped'] != ''].matching_after_drop_cluster_Hamming)
#             cm_pred_drop = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster,
#                                             df_total[df_total['dropped'] != ''].matching_after_drop_cluster_drop)
#             cm_pred_drop_ins = confusion_matrix(df_total[df_total['dropped'] != ''].true_cluster, df_total[
#                 df_total['dropped'] != ''].matching_after_drop_cluster_drop_ins)
#
#             wrong_lin_split_hamming[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_hamming, axis=1).sum(
#                 axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
#             wrong_lin_merge_hamming[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_hamming, axis=0).sum(
#                 axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
#
#             wrong_lin_split_drop[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop, axis=1).sum(axis=0)).sum() / (
#                         df_total[df_total['dropped']!=''].shape[0])
#             wrong_lin_merge_drop[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop, axis=0).sum(axis=0)).sum() / (
#                         df_total[df_total['dropped']!=''].shape[0])
#
#             wrong_lin_split_drop_ins[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop_ins, axis=1).sum(
#                 axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
#             wrong_lin_merge_drop_ins[p_i, p_d, Clustering_Threshold] = (np.amax(cm_pred_drop_ins, axis=0).sum(
#                 axis=0)).sum() / (df_total[df_total['dropped']!=''].shape[0])
#
#             number_true_clusters[p_i, p_d, Clustering_Threshold] = df_total[
#                 df_total['dropped'] != ''].true_cluster.nunique()
#             number_measured_clusters[p_i, p_d, Clustering_Threshold] = df_total[
#                 df_total['dropped'] != ''].dropped_cluster.nunique()
#
#             number_cluster_after_matching_Hamming[p_i, p_d, Clustering_Threshold] = df_total[
#                 df_total['dropped'] != ''].matching_after_drop_cluster_Hamming.nunique()
#             number_cluster_after_matching_drop[p_i, p_d, Clustering_Threshold] = df_total[
#                 df_total['dropped'] != ''].matching_after_drop_cluster_drop.nunique()
#             number_cluster_after_matching_drop_ins[p_i, p_d, Clustering_Threshold] = df_total[
#                 df_total['dropped'] != ''].matching_after_drop_cluster_drop_ins.nunique()
#
#             num=number_true_clusters[p_i, p_d, 0]
#             X=np.array(Total_drop, dtype=np.float64)
#             km=KMeans(n_clusters=int(num)).fit(np.array(X))
#             kmeanlabels=km.labels_
#             #print([metrics.adjusted_mutual_info_score(clustering_total.labels_, kmeanlabels),
#             #      metrics.adjusted_mutual_info_score(clustering_total.labels_, clustering_total_Hamming.labels_)])
#             print(metrics.homogeneity_completeness_v_measure(clustering_total.labels_, clustering_total_Hamming.labels_))
#             print(
#                 metrics.homogeneity_completeness_v_measure(clustering_total.labels_, clustering_total_drop.labels_))
#             p1.append(metrics.homogeneity_completeness_v_measure(clustering_total.labels_, clustering_total_drop.labels_))
#             p2.append(metrics.homogeneity_completeness_v_measure(clustering_total.labels_, clustering_total_Hamming.labels_))
#
#
#
#
#         #  Num_of_measured_lin[p_i,p_d,Clustering_Threshold]=number_cluster_after_matching[p_i,p_d,Clustering_Threshold]/number_true_clusters[p_i,p_d,Clustering_Threshold]-1
#
#         R = list(abs(1 - number_cluster_after_matching_Hamming[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
#         best_i = R.index(min(R))
#         print(best_i)
#         score_split_Hamming[p_i, p_d] = wrong_lin_split_hamming[p_i, p_d, best_i]
#         score_merge_Hamming[p_i, p_d] = wrong_lin_merge_hamming[p_i, p_d, best_i]
#         best_dist_Hamming[p_i, p_d] = best_i / 199 + epsilon
#
#         #plt.plot(p1,label='p1')
#         plt.plot(p1,'--')
#         #plt.plot(wrong_lin_split_hamming[p_i, p_d, :],'.',label='split')
#         #plt.plot(wrong_lin_merge_hamming[p_i, p_d, :],'*',label='merge')
#         #plt.legend()
#         plt.plot(p2, '-')
#         plt.gca().legend(('homogeneity - drop', 'completeness', 'v_measure','homogeneity-hamming', 'completeness', 'v_measure'))
#         plt.show()
#
#         R = list(abs(1 - number_cluster_after_matching_drop[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
#         best_i = R.index(min(R))
#         print(best_i)
#         score_split_drop[p_i, p_d] = wrong_lin_split_drop[p_i, p_d, best_i]
#         score_merge_drop[p_i, p_d] = wrong_lin_merge_drop[p_i, p_d, best_i]
#         best_dist_drop[p_i, p_d] = best_i / 19 + epsilon
#
#         R = list(abs(1 - number_cluster_after_matching_drop_ins[p_i, p_d, :] / number_true_clusters[p_i, p_d, 0]))
#         best_i = R.index(min(R))
#         score_split_drop_ins[p_i, p_d] = wrong_lin_split_drop_ins[p_i, p_d, best_i]
#         score_merge_drop_ins[p_i, p_d] = wrong_lin_merge_drop_ins[p_i, p_d, best_i]
#         best_dist_drop_ins[p_i, p_d] = best_i / 19 + epsilon
#
#         print(p_i,p_d,score_split_Hamming[p_i,p_d])
#
# #print(score_split_Hamming[:3,:4])
# cmap = plt.get_cmap('viridis', 20)
# plt.figure(figsize=(25,10))
# plt.subplot(3, 3, 1)
# plt.pcolormesh(score_split_Hamming[0:3,0:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('dropout prob.')
# plt.title('best score split')
# plt.colorbar()
# plt.subplot(3, 3, 2)
# plt.pcolormesh(score_merge_Hamming[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best score merge')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('dropout prob.')
# plt.colorbar()
# plt.subplot(3, 3, 3)
# plt.pcolormesh(best_dist_Hamming[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best dist = matching threshold')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('dropout prob.')
# plt.colorbar()
#
# plt.subplot(3, 3, 4)
# plt.pcolormesh(score_split_drop[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('dropout prob.')
# plt.title('best score split')
# plt.colorbar()
# plt.subplot(3, 3, 5)
# plt.pcolormesh(score_merge_drop[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best score merge')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('droupout prob.')
# plt.colorbar()
# plt.subplot(3, 3, 6)
# plt.pcolormesh(best_dist_drop[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best dist = matching thershold')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('droupout prob.')
# plt.colorbar()
# plt.subplot(3, 3, 7)
# plt.pcolormesh(score_split_drop_ins[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('droupout prob.')
# plt.title('best scrore split')
# plt.colorbar()
# plt.subplot(3, 3, 8)
# plt.pcolormesh(score_merge_drop_ins[:3,:4].T,vmin=0,vmax=1,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best scrore merge')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('droupout prob.')
# plt.colorbar()
# plt.subplot(3, 3, 9)
# plt.pcolormesh(best_dist_drop_ins[:3,:4].T,cmap=cmap,edgecolors='k', linewidths=0.1)
# plt.title('best dist = matching thershold')
# plt.xlabel('mean barcodes cassette')
# plt.ylabel('droupout prob.')
# plt.colorbar()
# plt.show()