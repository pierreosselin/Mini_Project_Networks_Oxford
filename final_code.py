#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Networks Project : Non Backtracking Random Walks for clustering in networks
"""
#Importation of the libraries used
####################################################
import numpy as np
import matplotlib
from math import *
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
import scipy
import scipy.linalg as la
import seaborn as sns
from random import randint


# In[3]:


#Return values of the Wigner Law for a given x
####################################################
def WignerLaw(x, c):
    return ((sqrt(4*c - x**2))/(2*pi*c))

#Intersection of two lists
####################################################
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

#Intermediary function for attributing labels
####################################################
def arrayToSign(arr):
    lResult = []
    for el in arr:
        if el > 0:
            lResult.append(1)
        elif el < 0:
            lResult.append(-1)
        else :
            lResult.append(2*randint(0,1) - 1)
    return(np.array(lResult))

#Compute the density plot of a given array of real values
####################################################
def plotEigenvalues(arr):
    sns.distplot(arr, hist=True, kde=False, 
             bins=20, color = 'blue',
             hist_kws={'edgecolor':'black'})

#Plot the spectrum with a circle of radius c
####################################################
def plotEigCom(arr,c):
    lReal = []
    lIm = []
    for elem in arr:
        lReal.append(elem.real)
        lIm.append(elem.imag)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.plot(lReal, lIm, 'r.', markersize=2)
    lx = np.linspace(-sqrt(c)+0.0001, sqrt(c)-0.0001, 1000)
    ly1 = np.array([sqrt(c - el*el) for el in lx])
    ly2 = np.array([-sqrt(c - el*el) for el in lx])
    plt.plot(lx, ly1, 'k')
    plt.plot(lx,ly2, 'k')
    plt.savefig("name", dpi = 900)
    
#Plot two spectrum with different colors with a circle of radius c
####################################################
def plotEigComTwo(arr1,arr2,c):
    lReal1 = []
    lIm1 = []
    for elem in arr1:
        lReal1.append(elem.real)
        lIm1.append(elem.imag)
    lReal2 = []
    lIm2 = []
    for elem in arr2:
        lReal2.append(elem.real)
        lIm2.append(elem.imag)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.plot(lReal1, lIm1, 'r.', markersize=6)
    plt.plot(lReal2, lIm2, 'b+', markersize=6)
    lx = np.linspace(-sqrt(c)+0.0001, sqrt(c)-0.0001, 1000)
    ly1 = np.array([sqrt(c - el*el) for el in lx])
    ly2 = np.array([-sqrt(c - el*el) for el in lx])
    plt.xlabel("Real")
    plt.ylabel("Img")
    plt.plot(lx, ly1, 'k')
    plt.plot(lx,ly2, 'k')
    plt.savefig("name", dpi = 900)
    
#Draw a graph and save it
####################################################
def plotGraphWithClusters(A, attribution):
    n = len(A)
    G = igraph.Graph.Adjacency(A, mode = ADJ_UNDIRECTED)
    G.vs["color"] = ["blue" if attribution[i] == 1 else "red" for i in range(n)]
    out = igraph.plot(G)
    out.save("name.png")

#Compute the overlap in the case of the balanced SBM
####################################################
def overlapSBM(listLabel):
    n = len(listLabel)
    trueLabel1 = np.array([1 for i in range(int(n/2))] + [-1 for i in range(int(n/2))])
    trueLabel2 = np.array([-1 for i in range(int(n/2))] + [1 for i in range(int(n/2))])
    l1 = (np.multiply(listLabel, trueLabel1) + 1)/2
    l2 = (np.multiply(listLabel, trueLabel2) + 1)/2
    return((max(np.sum(l1), np.sum(l2)))/n - 0.5)

#Compute the overlap between an attribution and the real partition of the graph
####################################################
def overlapGiven(listLabel, attribution):
    n = len(listLabel)
    trueLabel1 = np.array([-el for el in listLabel])
    trueLabel2 = np.array([el for el in listLabel])
    l1 = (np.multiply(attribution, trueLabel1) + 1)/2
    l2 = (np.multiply(attribution, trueLabel2) + 1)/2
    return((max(np.sum(l1), np.sum(l2)))/n - max(np.mean((trueLabel2 + 1)/2), np.mean((trueLabel1 + 1)/2)))


#Label vector in the space of edges for B and F (in-going edges)
####################################################
def labelVectBF(vec,G):
    n = len(G)
    labels = np.zeros(n)
    lEdges = list(G.edges())
    for index, edge in enumerate(lEdges):
        labels[edge[1]] += vec[2*index]
        labels[edge[0]] += vec[2*index + 1]
    labels = arrayToSign(labels)
    return(labels)

#Label vector in the space of edges for R and P (out-going edges)
####################################################
def labelVectRP(vec,G):
    n = len(G)
    labels = np.zeros(n)
    lEdges = list(G.edges())
    for index, edge in enumerate(lEdges):
        labels[edge[0]] += vec[2*index]
        labels[edge[1]] += vec[2*index + 1]
    labels = arrayToSign(labels)
    return(labels)


# In[4]:


#Explicit title, G is a networkX graph
####################################################
def computeNonBackTracking(G):
    lEdges = list(G.edges())
    m = len(lEdges)
    B = np.zeros((2*m,2*m))
    for i in range(m):
        for j in range(i+1, m):
            couplei = lEdges[i]
            couplej = lEdges[j]
            if len(intersection(list(couplei), list(couplej))) == 1:
                if couplei[1] == couplej[0]:
                    B[2*i,2*j] = 1
                    B[2*j + 1,2*i + 1] = 1
                elif couplei[1] == couplej[1]:
                    B[2*i,2*j+1] = 1
                    B[2*j, 2*i + 1] = 1
                elif couplei[0] == couplej[0]:
                    B[2*i+1,2*j] = 1
                    B[2*j + 1, 2*i] = 1
                else :
                    B[2*i+1,2*j+1] = 1
                    B[2*j, 2*i] = 1
    return(scipy.sparse.csr_matrix(B))


#Explicit title, G is a networkX graph
####################################################
def computeBPrime(G):
    n = len(G)
    z = np.zeros((n,n))
    o = np.eye(n)
    D = np.diag([el[1] for el in G.degree])
    A = scipy.sparse.csr_matrix.toarray(nx.adjacency_matrix(G))
    mat = np.concatenate((np.concatenate((z,D - o),axis=1), np.concatenate((-o,A), axis = 1)))
    return(mat)

#Explicit title, G is a networkX graph
####################################################
def computeFlow(G):
    ## Work Out the Flow matrix
    lEdges = list(G.edges())
    m = len(lEdges)
    degree = dict(G.degree())
    B = scipy.sparse.lil_matrix((2*m,2*m))
    D = scipy.sparse.lil_matrix((2*m,2*m))
    for i in range(m):
        couplei = lEdges[i]
        if degree[couplei[1]] > 1:
            D[2*i, 2*i] = 1/(degree[couplei[1]] - 1)
        if degree[couplei[0]] > 1:
            D[2*i + 1, 2*i + 1] = 1/(degree[couplei[0]] - 1)
        for j in range(i+1, m):
            couplej = lEdges[j]
            if len(intersection(list(couplei), list(couplej))) == 1:
                if couplei[1] == couplej[0]:
                    B[2*i,2*j] = 1
                    B[2*j + 1,2*i + 1] = 1
                elif couplei[1] == couplej[1]:
                    B[2*i,2*j+1] = 1
                    B[2*j, 2*i + 1] = 1
                elif couplei[0] == couplej[0]:
                    B[2*i+1,2*j] = 1
                    B[2*j + 1, 2*i] = 1
                else :
                    B[2*i+1,2*j+1] = 1
                    B[2*j, 2*i] = 1
    return(scipy.sparse.csr_matrix(D) * scipy.sparse.csr_matrix(B))

#Explicit title, G is a networkX graph
####################################################
def reluctantAndNormalized(G):
    lEdges = list(G.edges())
    m = len(lEdges)
    degree = dict(G.degree())
    R = scipy.sparse.lil_matrix((2*m,2*m))
    P = scipy.sparse.lil_matrix((2*m,2*m))
    for i in range(m):
        couplei = lEdges[i]
        R[2*i, 2*i + 1] = 1/degree[couplei[0]]
        R[2*i + 1, 2*i] = 1/degree[couplei[1]]
        P[2*i, 2*i + 1] = R[2*i, 2*i + 1]/(degree[couplei[1]] - 1 + (1/degree[couplei[0]]))
        P[2*i + 1, 2*i] = R[2*i + 1, 2*i]/(degree[couplei[0]] - 1 + (1/degree[couplei[1]]))
        for j in range(i+1, m):
            couplej = lEdges[j]
            if len(intersection(list(couplei), list(couplej))) == 1:
                if couplei[1] == couplej[0]:
                    R[2*i,2*j] = 1
                    R[2*j + 1,2*i + 1] = 1
                    P[2*i,2*j] = 1/(degree[couplei[1]] - 1 + (1/degree[couplei[0]]))
                    P[2*j + 1,2*i + 1] = 1/(degree[couplej[0]] - 1 + (1/degree[couplej[1]]))
                elif couplei[1] == couplej[1]:
                    R[2*i,2*j+1] = 1
                    R[2*j, 2*i + 1] = 1
                    P[2*i,2*j + 1] = 1/(degree[couplei[1]] - 1 + (1/degree[couplei[0]]))
                    P[2*j + 1,2*i + 1] = 1/(degree[couplej[1]] - 1 + (1/degree[couplej[0]]))
                elif couplei[0] == couplej[0]:
                    R[2*i+1,2*j] = 1
                    R[2*j + 1, 2*i] = 1
                    P[2*i + 1,2*j] = 1/(degree[couplei[0]] - 1 + (1/degree[couplei[1]]))
                    P[2*j + 1,2*i] = 1/(degree[couplej[0]] - 1 + (1/degree[couplej[1]]))
                else :
                    R[2*i+1,2*j+1] = 1
                    R[2*j, 2*i] = 1
                    P[2*i+1,2*j+1] = 1/(degree[couplei[0]] - 1 + (1/degree[couplei[1]]))
                    P[2*j, 2*i] = 1/(degree[couplej[1]] - 1 + (1/degree[couplej[0]]))
    return([scipy.sparse.csr_matrix(R),scipy.sparse.csr_matrix(P)])

#Build the concatenation of two trees of depth i respectively
####################################################
def buildTree(i):
    def nextTree(tree):
        tree = scipy.linalg.block_diag(tree,tree)
        n = tree.shape[0]
        u = np.zeros((1,int(n/2)))
        u[0,0] = 1
        u = np.concatenate((u,u), axis = 1)

        tree = np.concatenate((u,tree))
        u = np.transpose(np.concatenate((np.array([[0]]), u), axis = 1))
        tree = np.concatenate((u, tree), axis = 1)
        return(tree)
    A = np.array([[0,1,1],[1,0,0],[1,0,0]])
    for j in range(i):
        A = nextTree(A)
    tree = scipy.linalg.block_diag(A,A)
    n = tree.shape[0]
    tree[0,int(n/2)] = 1
    tree[int(n/2),0] = 1
    return(tree)

#Compute the bi-partition given G and the name of a matrix of interest
####################################################
def bipartition(G,name):
    switcher = {
        "B": computeNonBackTracking,
        "F": computeFlow,
        "R": lambda x : reluctantAndNormalized(x)[0],
        "P": lambda x : reluctantAndNormalized(x)[0],
        "L" : lambda x : scipy.sparse.csr_matrix.toarray(scipy.sparse.csr_matrix(nx.adjacency_matrix(x), dtype = 'd'))
    }
    # Matrix of interest
    B = switcher[name](G)
    #Spectrum
    w = scipy.sparse.linalg.eigs(B, k = 2, which = 'LR')
    eigenval = w[0]
    eigenvec = w[1]
    indexSortedList = sorted([i for i in range(len(eigenval))], key = lambda x : eigenval[x].real, reverse = True)
    # Get second eigenvector
    comEigenvect = eigenvec[:,indexSortedList[1]]

    # Attribute community to the sign
    if name == "R" or name == "P":
        attribution = labelVectRel(comEigenvect, G)
    elif name == "L":
        attribution = arrayToSign(comEigenvect)
    else:
        attribution = labelVect(comEigenvect, G)
    return(attribution)

#Run one analysis with a given matrix, return the overlap with the SBM in two communities
####################################################
def overlapAnalysis(name, n, c, cdif):
    switcher = {
        "B": computeNonBackTracking,
        "F": computeFlow,
        "R": lambda x : reluctantAndNormalized(x)[0],
        "P": lambda x : reluctantAndNormalized(x)[0],
        "A" : lambda x : scipy.sparse.csr_matrix.toarray(nx.adjacency_matrix(x))
    }
    #In and Out degree
    cin = c + cdif
    cout = c - cdif
    #Graph
    G = nx.stochastic_block_model([int(n/2),int(n/2)], [[cin/n, cout/n], [cout/n, cin/n]])
    #Matrix
    B = switcher[name](G)
    #Spectrum
    if name == "A":
        w = scipy.sparse.linalg.eigs(B, k = 2, which = 'LR')
    else:
        w = scipy.sparse.linalg.eigs(B, k = 2, which = 'LR')
    eigenval = w[0]
    eigenvec = w[1]
    # Sort by eigenval
    indexSortedList = sorted([i for i in range(len(eigenval))], key = lambda x : eigenval[x].real, reverse = True)
    # Get second eigenvector
    comEigenvect = eigenvec[:,indexSortedList[1]]
    # Attribute community to the sign,
    if name == "R" or name == "P":
        attribution = labelVectRel(comEigenvect, G)
    elif name == "A":
        attribution = attribution = arrayToSign(comEigenvect)
    else:
        attribution = labelVect(comEigenvect, G)
    #Compute Overlap
    overlap = overlap2(attribution)
    return(overlap)


# In[ ]:


#Plot partition of the tree model
####################################################
A = buildTree(4)
tree = nx.from_numpy_matrix(buildTree(i))
attribution1 = bipartition(tree,"F")
attribution2 = bipartition(tree,"P")
Adj = A.tolist()
n = len(Adj)
G = igraph.Graph.Adjacency(Adj)
G.to_undirected()
G.vs["color"] = ["blue" if attribution1[i] == 1 else "red" for i in range(n)]
out = igraph.plot(G)
out.save("treeB")
out


# In[ ]:


#Compute the superposition of the spectrum of B and R for the Karate Club Network
####################################################
GKarate = nx.karate_club_graph()
B = computeBPrime(GKarate)
F = computeFlow(GKarate)
[R,P] = reluctantAndNormalized(GKarate)
A = scipy.sparse.csr_matrix(nx.adjacency_matrix(GKarate), dtype = 'd')
w1, v1 = LA.eig(B)
w2, v2 = LA.eig(np.array(scipy.sparse.csr_matrix.todense(R)))
plotEigComTwo(w1,w2, 4.58824)


# In[ ]:





# In[ ]:





# In[ ]:




