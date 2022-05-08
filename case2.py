# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

from find_neighbors import find_neighbors

# Generate scalar field from the data file
lines = []
with open("Scalar_Field_data.txt", "r") as file:
    lines = file.readlines()

lines = [l.strip() for l in lines if not l.isspace() and not l.strip().startswith(('>', 'F', "Columns"))]

field = np.zeros(shape=(25, 25, 3))

for i in range(len(lines)):
    col_start = int(i / 25) * 5
    row = i % 25

    values = lines[i].split()
    for n in range(len(values)):
        field[row][col_start + n] = [ -5.75 + (col_start + n) * .5, -5.75 + row * .5, float(values[n])]

# Show the field
# plt.pcolormesh(field[:, :, 0], field[:, :, 1], field[:, :, 2] )
# plt.show()

rc = 5
rs = 5
cv = 0.01
num_nodes = 30
F = 50

rng = default_rng()
nodes = rng.uniform(low=-6, high=6, size=(num_nodes, 2))


# Get observability of node i for cell kr, kc
def get_observability(i, kr, kc):
    if np.linalg.norm(nodes[i] - np.array([field[kr, kc, 0], field[kr, kc, 1]])) <= rs:
        return 1
    else:
        return 0


# Get noise variance of node i for cell kr, kc
def get_v(i, kr, kc):
    return (np.linalg.norm(nodes[i] - [field[kr, kc, 0], field[kr, kc, 1]]) ** 2 + cv) / (rs ** 2)


# Get measured value of node i for cell kr, kc
def get_m(i, kr, kc):
    return get_observability(i, kr, kc) * (field[kr, kc, 2] + rng.normal(0, get_v(i, kr, kc)))


# Weight design 1
c_w1 = (2 * cv) / (rs**2 * (num_nodes - 1))

def get_weight_one(i, j, N, kr, kc):
    if i != j and j in N:
        return c_w1 / (get_v(i, kr, kc) + get_v(j, kr, kc))
    elif i == j:
        sum = 0
        for n in N:
            sum += get_weight_one(i, n, N, kr, kc)
        return 1 - sum
    else:
        return 0


iterations = 50
x = np.zeros(shape=(iterations, num_nodes, 25, 25))
for n in range(num_nodes):
    for r in range(25):
        for c in range(25):
            x[0][n][r][c] = get_m(n, r, c)

for i in range(1, iterations):
    neighbors = []
    Nei_agent, A = find_neighbors(nodes, rc, 2)
    for n in Nei_agent:
        neighbors.append(n[~np.isnan(n)].T.astype(int).tolist())

    for r in range(25):
        for c in range(25):
            for n in range(num_nodes):
                sum = 0
                for n1 in neighbors[n]:
                    sum += get_weight_one(n, n1, neighbors[n], r, c) * x[i - 1][n1][r][c]
                x[i][n][r][c] = get_weight_one(n, n, neighbors[n], r, c) * x[i-1][n][r][c] + sum


# Average out the observed value between all the nodes
x_avg = np.zeros(shape=(25, 25))
for n in range(num_nodes):
    x_avg += x[iterations-1][n]
x_avg /= num_nodes

print(x_avg)

plt.pcolormesh(field[:, :, 0], field[:, :, 1], x_avg, vmin=x_avg.min(), vmax=x_avg.max() )
plt.show()