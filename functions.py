# -*- coding: utf-8 -*-


import numpy as np
from difflib import SequenceMatcher
import networkx as nx



class SmallSystem:

    def __init__(self, barcodes, cells):
        self.barcodes = barcodes
        self.cells = cells
        self.X=np.empty(cells, dtype="<U100")


    def generate_barcoded_cells(self,p):
        XX = np.empty(self.cells, dtype="<U100")  # U6: 6 differnt letters can be inserted
        for bar in range(65, 65 + int(self.barcodes)):  # just to get 1st barcodes as 'A' (=65 in Ascii)
            for i in range(self.cells):
                if np.random.rand() <= p:#*(len(X[i])/50)**(1/3)+0.01:
                    XX[i] = XX[i] + chr(bar)
        self.X=XX
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

    def overlap(X1,X2,cells):
        Prob_overlap = 0
        for i in range(cells):
            for j in range(cells):
                if  (len(X1[i]) > 0) & (len(X2[j]) > 0) :
      #          if (X1[i] in X2[j]) or (X2[j] in X1[i]) :
                    if (X1[i] == X2[j])  :
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
        A=np.zeros([len(X),len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i,j]=SequenceMatcher(None,X[i],X[j]).ratio()
        return A
    
        
def NetworkPlot(A,labels,edge_width=0.1,node_size=10):
    dt = [('len', float)]
    A = A.view(dt)
    G = nx.from_numpy_matrix(A,create_using=nx.MultiGraph)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))    
    pos=nx.spring_layout(G)
    d = dict(G.degree)
    nx.draw_networkx(G,pos=pos,width=edge_width,node_size=[v * node_size for v in d.values()],node_color=[(v-2)**0.5  for v in d.values()])

def sim(list1,list2):
    res = len(set(list1) & set(list2)) / float(len(set(list1) | set(list2))) 
    return float(res)

class LargeSystem:

    def __init__(self, barcodes, cells):
        self.barcodes = barcodes
        self.cells = cells
        self.X = np.zeros([cells, barcodes], dtype=int)  


    def generate_barcoded_cells(self, p, Poisson):
        XX = np.zeros([cells, barcodes], dtype=int)  
        if Poisson==True:
            for bar in range(barcodes): 
                for i in range(cells):
                    if np.random.rand() <=p:
                        XX[i][bar] = 1
        else:
            for bar in range(barcodes): 
                for i in range(cells):  
                    if np.random.rand() <=(XX[i,:].sum()/barcodes)**(1/1)+1/(2*barcodes):
                        XX[i][bar] = 1
        self.X=XX
        return self.X

    def dropouts(X, p):
        barcodes=X.shape[1]
        cells=X.shape[0]
        for bar in range(barcodes): 
            for i in range(cells):
                if np.random.rand() <=p:   #(X[i,:].sum()/barcodes)**(1/1)+1/(2*barcodes):
                    X[i][bar] = 0
        return X

    def SimilarityMatrix(X):
        A=np.zeros([len(X),len(X)])
        for i in range(len(X)):
            for j in range(len(X)):
                A[i,j]=sim(([l for l in np.flatnonzero(X[j,:])]),([l for l in np.flatnonzero(X[i,:])]))
        return A

    def DistanceMatrix_dropout_conditional(X, dropout_prob, p_i):
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
                A[i, j] = ((1 - dropout_prob) ** Pij_sim_len * dropout_prob ** (Union_len - Pij_sim_len) * (
                            1 - dropout_prob) ** Pji_sim_len * dropout_prob ** (Union_len - Pji_sim_len))
                S1 = len(set([l for l in np.flatnonzero(X[i, :])]))
                S2 = len(set([l for l in np.flatnonzero(X[j, :])]))
                B = (p_i) ** Overlap_len * (1 - p_i) ** (len(X) - Overlap_len) * (
                            scipy.special.binom(100, S1) * scipy.special.binom(len(X), S2) / (
                        scipy.special.binom(100, Union_len)))
                A[j, i] = A[i, j]
                if j == i:
                    A[i, j] = 1
        return (A / B)

