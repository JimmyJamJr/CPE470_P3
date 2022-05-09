# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

from find_neighbors import find_neighbors

# Parameters
r = 17
rs = 4
cv = 0.02
num_nodes = 10
F = 50

rng = default_rng()
nodes = rng.uniform(low=0, high=4, size=(num_nodes, 2))

# Show network
# plt.plot(nodes[:,0], nodes[:,1], marker="o", linestyle="None")
# plt.title("Network with 10 nodes")
# plt.show()

# Avg position of nodes
q = np.sum(nodes, axis=0) / 10.

# Get measurement vector
V = ((np.linalg.norm(nodes - q, axis=1)**2) + cv) / (rs**2)
noise = rng.normal(0, V)
m = F + noise

# Weight design 1
c_w1 = (2 * cv) / (rs**2 * (num_nodes - 1))

def get_weight_one(i, j, N):
    if i != j and j in N:
        return c_w1 / (V[i] + V[j])
    elif i == j:
        sum = 0
        for n in N:
            sum += get_weight_one(i, n, N)
        return 1 - sum
    else:
        return 0


# Weight design 2
c_w2 = cv / (rs ** 2)

def get_weight_two(i, j, N):
    if i != j and j in N:
        return (1 - get_weight_two(i, i, N)) / len(N)
    elif i == j:
        return c_w2 / V[i]
    else:
        return 0


# Go through iterations, store results of iteration using filter 1 formula
iterations = 1500
x = np.zeros(shape=(iterations, num_nodes))
x[0] = m
for i in range(1, iterations):
    neighbors = []
    Nei_agent, A = find_neighbors(nodes, r, 2)
    for n in Nei_agent:
        neighbors.append(n[~np.isnan(n)].T.astype(int).tolist())

    for n in range(num_nodes):
        sum = 0
        for n1 in neighbors[n]:
            sum += get_weight_one(n, n1, neighbors[n]) * x[i - 1][n1]
        x[i][n] = get_weight_one(n, n, neighbors[n]) * x[i-1][n] + sum

mean = np.mean(x[0])
y = [[x[i][n] - mean for i in range(iterations)] for n in range(num_nodes)]
# for n in range(num_nodes):
#     plt.plot(range(iterations), y[n])
# plt.xlabel("Iterations")

plt.plot(range(num_nodes), [x[0][n] for n in range(num_nodes)], label="Initial")
plt.plot(range(num_nodes), [x[iterations-1][n] for n in range(num_nodes)], label="Final")
plt.legend()
plt.xlabel("Nodes")
plt.show()



