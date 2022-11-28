"""
@File   :   utils.py
@Author  :   chend
@Contact :   chend@hust.edu.cn
"""
import numpy as np
import networkx as nx
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', family='Times New Roman')


def relabel_nodes_labels(G):
    mapping = dict(zip(G, range(len(G.nodes()))))
    G = nx.relabel_nodes(G, mapping)

    return G

def delete_zero_degree_nodes(G):
    null_nodes = [i for i in G.nodes() if G.degree(i)==0]
    G.remove_nodes_from(null_nodes)
    G = relabel_nodes_labels(G)

    return G

# degree_thresholding_renormalization(DTR)
def DTR(G, KT):
    remove_nodes = [i for i in G.nodes() if G.degree(i) <= KT]
    G.remove_nodes_from(remove_nodes)
    G = delete_zero_degree_nodes(G)

    return G


# The Chung-Lu random graph model with expected degree sequence w
"""
Fasino D, Tonetto A, Tudisco F. Generating large scale‐free networks with the Chung–Lu random graph model[J]. Networks, 2021, 78(2): 174-187.
"""
def make_sparse_adj_matrix(w):
    n = np.size(w)
    s = np.sum(w)
    m = ( np.dot(w,w)/s )**2 + s
    m = int(m/2)
    wsum = np.cumsum(w)
    wsum = np.insert(wsum,0,0)
    wsum = wsum / wsum[-1]
    I = np.digitize(np.random.rand(m,1),wsum)
    J = np.digitize(np.random.rand(m,1),wsum)
    row_ind = np.append(I.reshape(m,)-1,J.reshape(m,)-1)
    col_ind = np.append(J.reshape(m,1)-1,I.reshape(m,)-1)
    ones = [1 for i in range(2*m)]    
    A = csr_matrix((ones, (row_ind,col_ind)), shape=(n,n))
    A.data.fill(1)
    return A


def make_nx_graph(w):
    n = np.size(w)
    s = np.sum(w)
    m = ( np.dot(w,w)/s )**2 + s
    m = int(m/2)
    wsum = np.cumsum(w)
    wsum = np.insert(wsum,0,0)
    wsum = wsum / wsum[-1]
    I = np.digitize(np.random.rand(m,1),wsum)
    J = np.digitize(np.random.rand(m,1),wsum)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    G.add_edges_from(tuple(zip(I.reshape(m,),J.reshape(m,))))
    return G


# complementary cumulative degree distribution
def get_ccdf(deg, data, avks, nums):
    bins = np.round(np.logspace(np.log10(min(data)), np.log10(max(data)), nums))
    cnt = Counter(np.digitize(np.array(list(data)), bins))
    sorted_cnt = dict(sorted(dict(cnt).items(), key=lambda x: x[0], reverse=False))
    x, y = list(zip(*[(bins[k-1], v/len(data)) for k, v in sorted_cnt.items()]))
    Pck = [sum(y[i:]) for i in range(len(y))]
    rescaled_x = [xi/avks for xi in x]

    return rescaled_x, Pck


# degree-dependent clustering coefficient
def clustering_vs_degree(clu, data, deg, avks, nums):
    hist, bins = np.histogram(data, bins=nums, density=True)
    x = []
    C_k = []
    for i in range(len(bins)-1):
        ct = 0
        sc = 0
        for j in deg.keys():
            if deg[j] >= bins[i] and deg[j] < bins[i+1]:
                ct = ct + 1
                sc = sc + clu[j]
        if sc>0 and ct>0:
            x.append((bins[i]+bins[i+1])/2/avks)
            C_k.append(sc/ct)
        else:
            x.append((bins[i]+bins[i+1])/2/avks)
            C_k.append(np.inf)  
    return x, C_k


# degree-degree correlations
def k_nn_vs_degree(av_nei_deg, data, deg, avks, nums):
    hist, bins = np.histogram(data, bins=nums, density=True)
    x = []
    nom_fac = np.mean(data)/np.mean(np.array(data)**2)
    nom_k_nn_k = []
    for i in range(len(bins)-1):
        ct = 0
        sc = 0
        for j in deg.keys():
            if deg[j] >= bins[i] and deg[j] < bins[i+1]:
                ct = ct + 1
                sc = sc + av_nei_deg[j]

        if sc>0 and ct>0:
            x.append((bins[i]+bins[i+1])/2/avks)
            nom_k_nn_k.append(nom_fac*sc/ct)
        else:
            x.append((bins[i]+bins[i+1])/2/avks)
            nom_k_nn_k.append(np.inf)

    return x, nom_k_nn_k



def cal_three_characteristics(G):
    deg = dict(G.degree())
    all_k = sorted(list(set(deg.values())))
    clu = nx.clustering(G)
    av_nei_deg = nx.average_neighbor_degree(G)
    data = np.array(list(deg.values()))
    avks = np.mean(data)

    x1, y1 = get_ccdf(deg, data, avks, 41)
    x2, y2 = clustering_vs_degree(clu, data, deg, avks, 40)
    x3, y3 = k_nn_vs_degree(av_nei_deg, data, deg, avks, 40)

    return x1, x2, x3, y1, y2, y3


def get_kmax_avk_avc(G0, KT, nums):
    NKT = np.zeros(nums)
    kmax = np.zeros(nums)
    av_k = np.zeros(nums)
    av_clus = np.zeros(nums)
    for i in range(nums):
        G = G0.copy()
        G_KT = DTR(G, KT[i])
        NKT[i] = nx.number_of_nodes(G_KT)

        if NKT[i]==0: 
            break

        deg = [G_KT.degree(i) for i in G_KT.nodes()]
        kmax[i] = max(deg)
        av_k[i] = sum(deg)/NKT[i]
        av_clus[i] = nx.average_clustering(G_KT)

    return NKT, kmax, av_k, av_clus