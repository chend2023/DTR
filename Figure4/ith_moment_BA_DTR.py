import networkx as nx
import numpy as np
import functions as fus


if __name__=='__main__':
    n = 100000
    nums = 15
    m = 10

    # KT = [0,3,4,5,10,15,20,30,40,50,60,70,80,100,120] # m = 5
    KT = [0,10,15,17,20,30,35,40,50,60,70,80,100,120,150] # m = 10
    ith = [3.5,4.0,5.5,8.5]


    #===============================
    x0 = np.zeros(nums)

    y0 = np.zeros(nums)
    y1 = np.zeros(nums)
    y2 = np.zeros(nums)
    y3 = np.zeros(nums)
    #===============================

    #===============================
    x_0 = np.zeros(nums)

    y_0 = np.zeros(nums)
    y_1 = np.zeros(nums)
    y_2 = np.zeros(nums)
    y_3 = np.zeros(nums)
    #===============================

    samples = 100
    for i in range(samples):
        print("===========")
        G0 = nx.barabasi_albert_graph(n, m)
        y0 = np.zeros(nums)
        y1 = np.zeros(nums)
        y2 = np.zeros(nums)
        y3 = np.zeros(nums)
        for j in range(nums):
            G = G0.copy()
            GKT, NKT = fus.degree_thresholding_renormalization(G, n, KT[j])
            x0[j] = NKT

            y0[j] = fus.ratio_moments(GKT, ith[0])
            y1[j] = fus.ratio_moments(GKT, ith[1])
            y2[j] = fus.ratio_moments(GKT, ith[2])
            y3[j] = fus.ratio_moments(GKT, ith[3])

        x_0 = x_0 + x0

        y_0 = y_0 + y0
        y_1 = y_1 + y1
        y_2 = y_2 + y2
        y_3 = y_3 + y3

    outf = open("./m10/ith_moment_BA_DTR.dat", "w")
    for i in range(nums):
        outf.write(str(x_0[i]/samples)+" "+str(y_0[i]/samples)+" "+str(y_1[i]/samples)+" "+str(y_2[i]/samples)+" "+str(y_3[i]/samples)+"\n")
    outf.close()

