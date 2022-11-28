import numpy as np
import networkx as nx


# degree_thresholding_renormalization
def degree_thresholding_renormalization(G, n, KT):
    # degree thresholding: KT
    remove_nodes = [i for i in G.nodes() if G.degree(i)<=KT]
    G.remove_nodes_from(remove_nodes)

    n0 = n-len(remove_nodes)
    mapping = dict(zip(G, range(1, n0+1)))
    G = nx.relabel_nodes(G, mapping)
    NT = len(G.nodes())
            
    return G, NT


# moment ratio
def ratio_moments(G, i):
    n = len(G.nodes())
    k_i = [G.degree(j)**i for j in G.nodes()]
    k_i_1 = [G.degree(j)**(i-1) for j in G.nodes()]
    av_ki = sum(k_i)/n
    av_ki_1 = sum(k_i_1)/n

    return av_ki/av_ki_1


