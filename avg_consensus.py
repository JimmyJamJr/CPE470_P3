# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

from find_neighbors import find_neighbors

r = 2
rs = 1.6
cv = 0.01
num_nodes = 10
F = 50

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
    if j in N:
        return 1. / num_nodes
    elif i == j:
        return 1 - len(N) / num_nodes
    else:
        return 0


# Metropolis weights
def get_metropolis_weight(i, j, N):
    if j in N[i]:
        print(np.max( [len(N[i]), len(N[j])] ))
        return 1 / (1 + np.max( [len(N[i]), len(N[j])] ))
    elif i == j:
        sum = 0
        for n in N[i]:
            sum += get_metropolis_weight(i, n, N)
        return 1 - sum
    else:
        return 0

iterations = 80
x = np.zeros(shape=(iterations, num_nodes))
for i in range(iterations):
    neighbors = []
    Nei_agent, A = find_neighbors(nodes, r, 2)
    for n in Nei_agent:
        neighbors.append(n[~np.isnan(n)].T.astype(int).tolist())

    for n in range(num_nodes):
        if i == 0:
            x[i][n] = get_metropolis_weight(n, n, neighbors)
        else:
            sum = 0
            for n1 in neighbors[n]:
                sum += get_metropolis_weight(n, n1, neighbors) * x[i - 1][n1]
            x[i][n] = get_metropolis_weight(n, n, neighbors) * x[i-1][n] + sum

mean = np.mean(x[0])
y = [[x[i][n] - mean for i in range(iterations)] for n in range(num_nodes)]
for n in range(num_nodes):
    plt.plot(range(iterations), y[n])

# plt.plot(range(num_nodes), [x[0][n] for n in range(num_nodes)], label="Initial")
# plt.plot(range(num_nodes), [x[iterations-1][n] for n in range(num_nodes)], label="Final")
# plt.legend()

plt.show()