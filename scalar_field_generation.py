import numpy as np

x_neg = -6
x_pos = 6
y_neg = -6
y_pos = 6

mu_x1 = 3
mu_y1 = 2
mu1 = [mu_x1, mu_y1]
variance_x1 = 2.25
variance_y1 = 2.25
rho1 = 0.1333
covariance_xy1 = rho1 * np.sqrt(variance_x1 * variance_y1)
