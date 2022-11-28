import networkx as nx
import numpy as np
import functions as fus


if __name__=='__main__':
    n = [500,700,1000,1400,2000,3100,5000,7000,10000,14000,20000,30000,50000,70000,100000]

    ith = [3.5,4.0,5.5,8.5]
    nums = 15
    m = 10

    #===============================
    y_0 = np.zeros(nums)
    y_1 = np.zeros(nums)
    y_2 = np.zeros(nums)
    y_3 = np.zeros(nums)
    #===============================
    samples = 100
    for sa in range(samples):
        print("===========")
        y0 = np.zeros(nums)
        y1 = np.zeros(nums)
        y2 = np.zeros(nums)
        y3 = np.zeros(nums)
        for j in range(nums):
            G = nx.barabasi_albert_graph(n[j], m)
            y0[j] = fus.ratio_moments(G, ith[0])
            y1[j] = fus.ratio_moments(G, ith[1])
            y2[j] = fus.ratio_moments(G, ith[2])
            y3[j] = fus.ratio_moments(G, ith[3])

        y_0 = y_0 + y0
        y_1 = y_1 + y1
        y_2 = y_2 + y2
        y_3 = y_3 + y3



    outf = open("./m10/ith_moment_BA.dat", "w")
    for i in range(nums):
        outf.write(str(n[i])+" "+str(y_0[i]/samples)+" "+str(y_1[i]/samples)+" "+str(y_2[i]/samples)+" "+str(y_3[i]/samples)+"\n")
    outf.close()
