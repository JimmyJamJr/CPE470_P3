# Author: Jimson Huang
# CPE 470 Project 3
# Spring 2022

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy.linalg

# Generate scalar field from the data file
lines = []
with open("Scalar_Field_data.txt", "r") as file:
    lines = file.readlines()

lines = [l.strip() for l in lines if not l.isspace() and not l.strip().startswith(('>', 'F', "Columns"))]

field = np.zeros(shape=(25, 25))

for i in range(len(lines)):
    col_start = int(i / 25) * 5
    row = i % 25

    values = lines[i].split()
    for n in range(len(values)):
        field[row][col_start + n] = float(values[n])

x = np.arange(-6, 6, 12. / 24.)
y = np.arange(-6, 6, 0.5)
# Show the field
plt.pcolormesh(np.arange(-5.75, 5.75, .5), np.arange(-5.75, 5.75, .5), field, )
plt.show()

r = 17
rs = 5
cv = 0.01
num_nodes = 30
F = 50

rng = default_rng()
nodes = rng.uniform(low=0, high=4, size=(num_nodes, 2))

