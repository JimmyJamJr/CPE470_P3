# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

from find_neighbors import find_neighbors

# Parameters
r = 1.5
rs = 4
cv = 0.02
num_nodes = 50
F = 50

# Setup RNG
rng = default_rng()
nodes = rng.uniform(low=0, high=4, size=(num_nodes, 2))

# Avg position of nodes
q = np.sum(nodes, axis=0) / 10.

# Get measurement vector
V = ((np.linalg.norm(nodes - q, axis=1)**2) + cv) / (rs**2)
noise = rng.normal(0, V)
m = F + noise

# Max degree weights
def get_max_degree_weights(i, j, N):
    if j in N[i]:
        return 1. / num_nodes
    elif i == j:
        return 1 - len(N[i]) / num_nodes
    else:
        return 0


# Metropolis weights
def get_metropolis_weight(i, j, N):
    if j in N[i]:
        return 1 / (1 + np.max( [len(N[i]), len(N[j])] ))
    elif i == j:
        sum = 0
        for n in N[i]:
            sum += get_metropolis_weight(i, n, N)
        return 1 - sum
    else:
        return 0


# Go through iterations, store results of iteration using filter 2 formula
iterations = 150
x = np.zeros(shape=(iterations, num_nodes))
for i in range(iterations):
    neighbors = []
    Nei_agent, A = find_neighbors(nodes, r, 2)
    for n in Nei_agent:
        neighbors.append(n[~np.isnan(n)].T.astype(int).tolist())
    for n in range(num_nodes):
        if i == 0:
            x[i][n] = m[n]
        else:
            sum = 0
            for n1 in neighbors[n]:
                sum += get_max_degree_weights(n, n1, neighbors) * x[i - 1][n1]
            x[i][n] = get_max_degree_weights(n, n, neighbors) * x[i-1][n] + sum

mean = np.mean(x[0])
y = [[x[i][n] - mean for i in range(iterations)] for n in range(num_nodes)]
for n in range(num_nodes):
    plt.plot(range(iterations), y[n], label="N" + str(n))
plt.xlabel("Iterations")
plt.subplots_adjust(right=0.7)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.plot(range(num_nodes), [x[0][n] for n in range(num_nodes)], label="Initial")
# plt.plot(range(num_nodes), [x[iterations-1][n] for n in range(num_nodes)], label="Final")
# plt.legend()
# plt.xlabel("Nodes")

plt.show()