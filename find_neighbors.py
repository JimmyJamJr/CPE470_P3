import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg
import numpy.matlib


def find_neighbors(nodes,r,n, delta_t_update):
    num_nodes = nodes.shape[0]
    dif = np.zeros(shape=(num_nodes, num_nodes, n))
    # print(dif.shape)
    # print(dif)
    distance_alpha = np.zeros(shape=(num_nodes,num_nodes))
    Nei_agent = np.zeros(shape=(num_nodes, num_nodes, 1))

    for i in range(num_nodes):
        dif[i] = np.tile(np.array(nodes[i,:]), (num_nodes, 1)) - nodes
        tmp = dif[i]
        d_tmp = np.zeros(shape=(num_nodes, 1))
        for j in range(num_nodes):
            d_tmp[j,:] = np.linalg.norm(tmp[j,:])
        # print(d_tmp.shape)
        # print(distance_alpha.shape)
        # print(distance_alpha[i,:].shape)
        # print(d_tmp)
        distance_alpha[i,:] = d_tmp.ravel()

    for k in range(num_nodes):
        # print(distance_alpha[:,k])
        # print(np.logical_and(distance_alpha[:,k] < r, distance_alpha[:,k] != 0))
        Nei_agent[k] = np.logical_and(distance_alpha[:,k] < r, distance_alpha[:,k] != 0).reshape(10, 1)

    
