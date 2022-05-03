# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

from find_neighbors import find_neighbors

r = 1. # Communication rage
num_nodes = 10 # Number of nodes
n=2 # Number of dimensions
delta_t_update = 0.008

rng = default_rng()
nodes = rng.uniform(low=0, high=1, size=(num_nodes, n))
# Add measurement for each node: yi= theta + v_i, or : mi= theta + v_i
nodes_va = 50. * np.ones(shape=(num_nodes, 1)) + 1 * rng.normal(size=(num_nodes, n))
nodes_va0 = nodes_va

nodes_va_old = nodes_va
Nei_agent, A = find_neighbors(nodes, r, n, delta_t_update)

print(Nei_agent)

plt.plot(nodes[:,0], nodes[:,1])
plt.show()

iterations = 80
Con = []
for i in range(iterations):
    # Compute Metropolis weights
    # Compute vertex and edge weights
    break


def metropolis_weights():
    return 0