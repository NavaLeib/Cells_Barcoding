# -*- coding: utf-8 -*-

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

import numpy as np
from difflib import SequenceMatcher
import networkx as nx
import random
import math
import scipy.special

import pandas as pd

import itertools

import matplotlib.pyplot as plt



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


class LargeSystem:

    def __init__(self, barcodes, cells):
        self.barcodes = barcodes
        self.cells = cells
        self.X = np.zeros([cells, barcodes], dtype=int)

    def generate_barcoded_cells(self, p, Poisson):
        cells=self.cells
        barcodes=self.barcodes
        XX = np.zeros([cells, barcodes], dtype=int)
        if Poisson == True:
            for bar in range(barcodes):
                for i in range(cells):
                    if np.random.rand() <= p:
                        XX[i][bar] = 1
        else:
            for bar in range(barcodes):
                for i in range(cells):
                    if np.random.rand() <= (XX[i, :].sum() / barcodes) ** (1 / 1) + 1 / (2 * barcodes):
                        XX[i][bar] = 1
        self.X = XX
        return self.X

    def dropouts(self,X, p):
        barcodes = X.shape[1]
        cells = X.shape[0]
        #    print(barcodes, cells)
        for bar in range(barcodes):
            for i in range(cells):
                if X[i][bar] == 1:
                    if np.random.rand() <= p:  # (X[i,:].sum()/barcodes)**(1/1)+1/(2*barcodes):
                        X[i][bar] = 0
        return X

    def Rapid_SimilarityMatrix(self,X):
        cells = X.shape[0]
        A = np.identity(cells)
        CellsIndexSetS = set(np.arange(cells))
        for CellsPair in itertools.combinations(CellsIndexSetS, r=2):
            i = CellsPair[0]
            j = CellsPair[1]
            A[i, j] = sim(([l for l in np.flatnonzero(X[j, :])]), ([l for l in np.flatnonzero(X[i, :])]))
            A[j, i] = A[i, j]
        return 1 - A

    def SimilarityMatrix(self,X):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i, j] = sim(([l for l in np.flatnonzero(X[j, :])]), ([l for l in np.flatnonzero(X[i, :])]))
        return 1 - A

    def NumBarcodesDiff(self,X):
        A = np.zeros([len(X), len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i, j] = sum(abs(X[j, :] - X[i, :]))
        return A

    def DistanceMatrix_dropout(self,X, dropout_prob):
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

    def DistanceMatrix_dropout_conditional(self,X, dropout_prob, p_i):
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

        return A





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
        labels[k] = ''.join("+ ").join([str(l) for l in np.flatnonzero(system[k, :])])
    return labels


def clustering(barcodes_in_cells_list, Threshold):
    A_dist = LargeSystem.NumBarcodesDiff(np.array(barcodes_in_cells_list))

    clustering = AgglomerativeClustering(distance_threshold=Threshold, n_clusters=None, affinity='precomputed',
                                         linkage='complete',
                                         compute_full_tree=True).fit(A_dist)
    return clustering.labels_


