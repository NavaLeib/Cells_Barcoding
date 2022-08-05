# -*- coding: utf-8 -*-

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn import metrics

import numpy as np
from difflib import SequenceMatcher
# import networkx as nx
import random
import math
import scipy.special

import pandas as pd

import itertools

import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix


class SmallSystem:

    def __init__(self, barcodes, cells):
        self.barcodes = barcodes
        self.cells = cells
        self.X = np.empty(cells, dtype="<U100")

    def generate_barcoded_cells(self, p):
        XX = np.empty(self.cells, dtype="<U100")  # U6: 6 differnt letters can be inserted
        for bar in range(65, 65 + int(self.barcodes)):  # just to get 1st barcodes as 'A' (=65 in Ascii)
            for i in range(self.cells):
                if np.random.rand() <= p:  # *(len(X[i])/50)**(1/3)+0.01:
                    XX[i] = XX[i] + chr(bar)
        self.X = XX
        return self.X

    def number_of_lineages(X, cells, number_of_infected):
        flag = 0
        lineage = 0
        Prob_SubSet = 0
        for i in range(cells):
            for j in range(cells):
                if (i != j) & (len(X[i]) > 0) & (flag == 0):
                    if (X[i] in X[j]):
                        Prob_SubSet = Prob_SubSet + (X[i] in X[j])
                        flag = 1
                if (i != j) & (len(X[i]) > 0) & (X[i] == X[j]):
                    lineage = lineage + 1 / 2
        return number_of_infected - lineage

    def prob_complete_subset(X, cells):
        flag = 0
        lineage = 0
        Prob_SubSet = 0
        for i in range(cells):
            for j in range(cells):
                if (i != j) & (len(X[i]) > 0) & (flag == 0):
                    if (X[i] in X[j]):
                        Prob_SubSet = Prob_SubSet + (X[i] in X[j])
                        flag = 1
                if (i != j) & (len(X[i]) > 0) & (X[i] == X[j]):
                    lineage = lineage + 1 / 2
        return Prob_SubSet

    def overlap(X1, X2, cells):
        Prob_overlap = 0
        for i in range(cells):
            for j in range(cells):
                if (len(X1[i]) > 0) & (len(X2[j]) > 0):
                    #          if (X1[i] in X2[j]) or (X2[j] in X1[i]) :
                    if (X1[i] == X2[j]):
                        Prob_overlap = Prob_overlap + 1
        return Prob_overlap

    def number_of_infected(X, cells):
        number_of_infected = 0
        for i in range(cells):
            number_of_infected += (len(X[i]) > 0)
        return number_of_infected

    def drop_out(X, dropout_prob):
        X_afterdrop = X.copy()
        for i in range(len(X)):
            One_Cell = X[i]
            Init_Barcode_Len = len(One_Cell)
            for j in range(Init_Barcode_Len - 1, -1, -1):
                if np.random.rand() <= dropout_prob:
                    #                print(i, j, One_Cell)
                    One_Cell = One_Cell.replace(One_Cell[j], '')
                    #                print(i, j, One_Cell)
                    X_afterdrop[i] = One_Cell
        return X_afterdrop

    def SimilarityMatrix(X):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i, j] = SequenceMatcher(None, X[i], X[j]).ratio()
        return A


def NetworkPlot(A, labels, edge_width=0.1, node_size=10):
    dt = [('len', float)]
    A = A.view(dt)
    G = nx.from_numpy_matrix(A, create_using=nx.MultiGraph)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))
    pos = nx.spring_layout(G)
    d = dict(G.degree)
    nx.draw_networkx(G, pos=pos, width=edge_width, node_size=[v * node_size for v in d.values()],
                     node_color=[(v - 2) ** 0.5 for v in d.values()])
    return True


def sim(list1, list2):
    if float(len(set(list1) | set(list2))) > 0:
        res = len(set(list1) & set(list2)) / float(len(set(list1) | set(list2)))
    else:
        res = 1
    return float(res)


'''
def BarcodesSet(X):
  X_set={}
  cells=X.shape[0]
  for i in range(cells):
    [l for l in np.flatnonzero(X[i,:])
'''


def exponential_generator_system(cells, barcodes, exp_scale):
    p_ins_exp = np.exp(-np.arange(barcodes) / exp_scale) / (np.exp(-np.arange(barcodes) / exp_scale).sum())
    rng = np.random.default_rng()
    system = np.zeros(shape=(cells, barcodes))
    for cell in range(cells):
        L = int(np.random.exponential(scale=exp_scale))
        bar_index = []
        p_ins_temp = p_ins_exp.copy()
        temp = 0
        for i in range(L):
            temp = rng.multinomial(1, p_ins_temp / (p_ins_temp.sum()), size=None)
            new_index = [i for i, x in enumerate(temp) if x == 1]
            p_ins_temp = np.zeros(cells)
            bar_index.extend(new_index)
            for j in range(cells):
                p_ins_temp[j] = 0 if j in bar_index else p_ins_exp[j]
        system[cell, bar_index] = 1
        bar_index = []
    return system


class LargeSystem:

    def __init__(self, barcodes, cells):
        self.barcodes = barcodes
        self.cells = cells
        self.X = np.zeros([cells, barcodes], dtype=int)

    def generate_barcoded_cells(self, p, Poisson):
        cells = self.cells
        barcodes = self.barcodes
        XX = np.zeros([cells, barcodes], dtype=int)
        if Poisson == True:
            for bar in range(barcodes):
                for i in range(cells):
                    if np.random.rand() <= p:
                        XX[i][bar] = 1
        else:
            #     for bar in range(barcodes):
            #         for i in range(cells):
            #             if np.random.rand() <= (XX[i, :].sum() / barcodes) ** (1 / 1) + 1 / (2 * barcodes):
            #                 XX[i][bar] = 1
            # self.X = XX
            XX = exponential_generator_system(cells, barcodes, p * barcodes)
        self.X = XX
        return self.X

    def dropouts(self, X, p):
        barcodes = X.shape[1]
        cells = X.shape[0]
        #    print(barcodes, cells)
        for bar in range(barcodes):
            for i in range(cells):
                if X[i][bar] == 1:
                    if np.random.rand() <= p:  # (X[i,:].sum()/barcodes)**(1/1)+1/(2*barcodes):
                        X[i][bar] = 0
        return X

    def Rapid_SimilarityMatrix(self, X):
        cells = X.shape[0]
        A = np.identity(cells)
        CellsIndexSetS = set(np.arange(cells))
        for CellsPair in itertools.combinations(CellsIndexSetS, r=2):
            i = CellsPair[0]
            j = CellsPair[1]
            A[i, j] = sim(([l for l in np.flatnonzero(X[j, :])]), ([l for l in np.flatnonzero(X[i, :])]))
            A[j, i] = A[i, j]
        return 1 - A

    def SimilarityMatrix(self, X):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i, j] = sim(([l for l in np.flatnonzero(X[j, :])]), ([l for l in np.flatnonzero(X[i, :])]))
        return 1 - A

    def NumBarcodesDiff(self, X):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i, j] = sum(abs(X[j, :] - X[i, :]))
        return A

    def DistanceMatrix_dropout(self, X, dropout_prob):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(i + 1):
                Union_len = len(set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])]))
                Overlap_len = len(set([l for l in np.flatnonzero(X[i, :])]) & set([l for l in np.flatnonzero(X[j, :])]))
                Pij_sim_len = len(
                    (set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])])) & set(
                        [l for l in np.flatnonzero(X[i, :])]))
                Pji_sim_len = len(
                    (set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])])) & set(
                        [l for l in np.flatnonzero(X[j, :])]))
                # A[i,j]=0.5*((1-dropout_prob)**Pij_sim_len*dropout_prob**(Union_len-Pij_sim_len)+(1-dropout_prob)**Pji_sim_len*dropout_prob**(Union_len-Pji_sim_len))
                A[i, j] = (((1 - dropout_prob) ** Pij_sim_len) * (dropout_prob ** (Union_len - Pij_sim_len)) * (
                        (1 - dropout_prob) ** Pji_sim_len) * (dropout_prob ** (Union_len - Pji_sim_len)))
                A[j, i] = A[i, j]
                if j == i:
                    A[i, j] = 1
        return (1 - A)

    def DistanceMatrix_dropout_conditional(self, X, dropout_prob, p_i):
        A = np.zeros([len(X), len(X)])
        cells = X.shape[0]
        barcodes = X.shape[1]
        CellsIndexSetS = set(np.arange(cells))
        for CellsPair in itertools.combinations(CellsIndexSetS, r=2):
            i = CellsPair[0]
            j = CellsPair[1]
            # for i in range(len(X)):
            #    for j in range(i+1):
            Union_len = len(set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])]))
            Overlap_len = len(set([l for l in np.flatnonzero(X[i, :])]) & set([l for l in np.flatnonzero(X[j, :])]))
            Pij_sim_len = len(
                (set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])])) & set(
                    [l for l in np.flatnonzero(X[i, :])]))
            Pji_sim_len = len(
                (set([l for l in np.flatnonzero(X[i, :])]) | set([l for l in np.flatnonzero(X[j, :])])) & set(
                    [l for l in np.flatnonzero(X[j, :])]))
            # A[i,j]=0.5*((1-dropout_prob)**Pij_sim_len*dropout_prob**(Union_len-Pij_sim_len)+(1-dropout_prob)**Pji_sim_len*dropout_prob**(Union_len-Pji_sim_len))
            A[i, j] = ((1 - dropout_prob) ** Pij_sim_len * dropout_prob ** (Union_len - Pij_sim_len) * (
                    1 - dropout_prob) ** Pji_sim_len * dropout_prob ** (Union_len - Pji_sim_len))
            S1 = len(set([l for l in np.flatnonzero(X[i, :])]))
            S2 = len(set([l for l in np.flatnonzero(X[j, :])]))
            # B=(p_i)**(-Overlap_len)*(1-p_i)**(-barcodes+Overlap_len)*scipy.special.binom(barcodes,Union_len)/(scipy.special.binom(barcodes,S1)*scipy.special.binom(barcodes,S2))
            p_double = 1 - ((p_i) ** 1 * (1 - p_i) ** (barcodes - 1) * scipy.special.binom(barcodes, 1) + (p_i) ** 0 * (
                    1 - p_i) ** (barcodes - 0) * scipy.special.binom(barcodes, 0))
            B = (p_double) ** (Overlap_len)
            #  B=scipy.special.binom(barcodes,S1+S2)/(scipy.special.binom(S1+S2,S1)*scipy.special.binom(S1+S2,S2))
            #  print(set([l for l in np.flatnonzero(X[i,:])]),set([l for l in np.flatnonzero(X[j,:])]),B,1-A[i,j], p_double)
            A[i, j] = (1 - A[i, j]) * (B)
            A[j, i] = A[i, j]
            if j == i:
                A[i, j] = 0


class System_Analysis:

    def __init__(self, df):
        self.df = df.applymap(lambda x: str(x).split('.0')[0])  # the data frame to be analyzed

    def Rapid_SimilarityMatrix(self, key):
        """
        This function generate the similarity matrix (Hummin) between every pair of cells within a given
        #cell , #barcodes, MOI and p_drop as specified in data frame name.

        The similarity matrix Aij give the Hummin distance between cell i to cell j.
        The matrix is symmetric Aij=Aji and Aii = 0 (0 distance of cell with itself)

        :key: the column in the df that the analysis is taken (correspond the parameters and generations)
        :return: a matrix of size (cells X cells)
        """
        df = self.df
        data = df[key]

        cells = df.shape[0]
        A = np.identity(cells)
        CellsIndexSetS = set(np.arange(cells))
        for CellsPair in itertools.combinations(CellsIndexSetS, r=2):
            i = CellsPair[0]
            j = CellsPair[1]

            set_i = set([i for i in str(df[key].iloc[[i]].values[0]).split('+ ')])
            set_j = set([i for i in str(df[key].iloc[[j]].values[0]).split('+ ')])
            A[i, j] = sim(set_i, set_j)

            A[j, i] = A[i, j]
        return 1 - A

    def clustering(self, key, Threshold):
        A_dist = self.Rapid_SimilarityMatrix(key=key)

        clustering = AgglomerativeClustering(distance_threshold=Threshold, n_clusters=None, affinity='precomputed',
                                             linkage='complete',
                                             compute_full_tree=True).fit(A_dist)
        return clustering.labels_

    def clustering_score(self, key_true, key_drop):
        # "removing" the undetaected cells form teh analysis:
        df = self.df
        detected_cells_id = ~(df[key_drop].isnull())

        clustering_true = self.clustering(key=key_true, Threshold=10 ** -300)

        V_measured = []
        for Clustering_Threshold in range(20):
            Thres = Clustering_Threshold / 19 + 10 ** -300
            clustering_drop = self.clustering(key=key_drop, Threshold=Thres)
            (homogeneity, completeness, v_measure) = \
                metrics.homogeneity_completeness_v_measure(clustering_true[detected_cells_id],
                                                           clustering_drop[detected_cells_id])
            V_measured.append(v_measure)

        V_index = V_measured.index(max(V_measured))
        Threshold = V_index / 19 + 10 ** -300
        return (max(V_measured), Threshold)

    def number_of_perfect_clusters(self, key_true, key_drop):
        df = self.df
        detected_cells_id = ~(df[key_drop].isnull())

        clustering_true = self.clustering(key=key_true, Threshold=10 ** -300)

        V_measured = []
        for Clustering_Threshold in range(20):
            Thres = Clustering_Threshold / 19 + 10 ** -300
            clustering_drop = self.clustering(key=key_drop, Threshold=Thres)
            (homogeneity, completeness, v_measure) = \
                metrics.homogeneity_completeness_v_measure(clustering_true[detected_cells_id],
                                                           clustering_drop[detected_cells_id])
            V_measured.append(v_measure)
        V_index = V_measured.index(max(V_measured))
        Threshold = V_index / 19 + 10 ** -300
        labels_pred = self.clustering(key=key_drop, Threshold=Threshold)
        label_true = self.clustering(key=key_true, Threshold=10 ** -300)
        # print('labels: \n', label_true, labels_pred)
        # np.set_printoptions(threshold=np.inf)
        set_pred=set(np.sort(labels_pred))
        set_true=set(np.sort(label_true))
        # print(set(np.sort(labels_pred)))
        # print(set(np.sort(label_true)))
        A = contingency_matrix(label_true, labels_pred)
        (num_true_clust, num_pred_clust) = A.shape

        print((num_true_clust, num_pred_clust))

        num = 0
        perfect_clusters_id_in_drop = []
        for row in range(num_true_clust):
            if len(np.where(A[row] > 0)[0]) == 1:
                # print(row)
                #print(A[row])
                col = np.where(A[row] > 0)[0]
                # print(A[:,col])
                # print(sum(A[:,col]))
                if sum(A[:, col])[0] == A[row, col]:
                    num = num + 1
                    perfect_clusters_id_in_drop.append(col)
                    #print(label_true[row], labels_pred[col], 'row', row, col)

        #print("#perfect_clusters=", num)
        return (num, perfect_clusters_id_in_drop)


class Lineages_Analysis:

    def __init__(self, df):
        self.df = df

    def num_lineages_between_times(self, times_true, times_drop):

        # Globaly clustering over all given times
        df = self.df
        # propagated_time_key_true =[]
        # propagated_time_key_drop=[]
        total = []
        total_true = []
        total_drop = []
        for (item_true, item_drop) in zip(times_true, times_drop):
            # propagated_time_key_true.append(item_true)
            # propagated_time_key_drop.append(item_drop)
            total_true = np.concatenate((total_true, df[item_true]))
            total_drop = np.concatenate((total_drop, df[item_drop]))
        df_total = pd.DataFrame(total_true, columns=['cells_id_all_true'])
        df_total['cells_id_all_drop'] = total_drop

        # print(df_total)

        detected_cells_id = df_total[(~(df_total['cells_id_all_drop'].isnull()))]
        detected_cells_id = detected_cells_id.loc[detected_cells_id['cells_id_all_drop'] != '']
        detected_cells_id = detected_cells_id.loc[detected_cells_id['cells_id_all_drop'] != 'NaN']

        df_total_measured = detected_cells_id

        df_total_measured.dropna(axis=0, how='any')
        # detected_cells_id = df_total[(~(df_total['cells_id_all_drop'].isnull()))]
        # detected_cells_id = df_total.loc[df_total['cells_id_all_drop'] != '']

        #df_total_measured = detected_cells_id

        # print(df_total['cells_id_all_true'].dtypes, df_total['cells_id_all_drop'].dtypes)

        df_total.applymap(lambda x: str(x).split('.0')[0])

        sys_all = System_Analysis(df_total_measured)
        (v_score, Thres) = sys_all.clustering_score(key_true='cells_id_all_true', key_drop='cells_id_all_drop')
        df_total_measured['gloabl_cluster'] = sys_all.clustering(key='cells_id_all_drop', Threshold=Thres)

        df_total.applymap(lambda x: str(x).split('.0')[0])

        pd.set_option('display.max_rows', None)

        print(v_score, Thres)
        print(df_total)
        print(df_total_measured)

        sys_all = System_Analysis(df_total_measured)

        # find the perfect global clusters
        (num_clustering_true, perfect_clusters_id_in_drop) = sys_all.number_of_perfect_clusters(
            key_true='cells_id_all_true', key_drop='cells_id_all_drop')
        print('# perfect_clusters = ', (num_clustering_true, perfect_clusters_id_in_drop))

        # count perfect clusters propagate through time
        init = 0
        j = 0
        sets = [set() for i in range(len(times_true))]

        df = self.df

        for (item_true, item_drop) in zip(times_true, times_drop):
            # print(df)
            # num_detected_cells_in_each_prop = (~(df[item_drop].isnull())).shape[0]
            num_detected_cells_in_each_prop = (~df[item_drop].isnull()).sum()
            fini = init + min(num_detected_cells_in_each_prop, df.shape[0])
            print(item_drop, num_detected_cells_in_each_prop, fini)
            # print('\n \n', df_total['gloabl_cluster'].iloc[init:fini - 1])
            sets[j] = set(df_total_measured['gloabl_cluster'].loc[init:fini - 1])
            print(sets[j])
            init = fini
            j = j + 1

        prop_set = sets[0]
        # print(prop_set, sets[0] & sets[1])
        for i in range(len(sets)):
            prop_set = prop_set & sets[i]
        #  print(prop_set)

        print('propagated sets=', prop_set)
        print('prefect sets=', set([item[0] for item in perfect_clusters_id_in_drop]))

        print('# perfect propagated sets=', len(prop_set & set([item[0] for item in perfect_clusters_id_in_drop])))

        return len(prop_set & set([item[0] for item in perfect_clusters_id_in_drop]))


class Simulation:

    def __init__(self, barcodes, cells, generations):
        self.barcodes = barcodes
        self.cells = cells
        self.generations = generations
        self.X = np.zeros([cells, barcodes, generations], dtype=int)

    def propagation(self, system):
        system_prop = system.copy()
        for i in range(self.generations):
            system_prop = np.concatenate((system_prop.copy(), system_prop.copy()))
        return system_prop

    def sampling(self, system):
        return random.sample(list(system.copy()), system.shape[0] // (2 ** self.generations))

    def run(self, system, sampling_times):
        system_temp = system.copy()
        for i in range(sampling_times):
            system_temp_prop = self.propagation(system_temp.copy())
            system_temp = self.sampling(system_temp_prop.copy())
        return system_temp


def barcodes_id_list(system):
    labels = ["" for x in range(system.shape[0])]
    for k in range(system.shape[0]):
        labels[k] = ''.join("+ ").join([str(int(l)) for l in np.flatnonzero(system[k, :])])
    return labels



