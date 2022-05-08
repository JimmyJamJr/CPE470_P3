import numpy as np
import numpy.linalg


def find_neighbors(nodes, r, n):
    num_nodes = nodes.shape[0]
    dif = np.zeros(shape=(num_nodes, num_nodes, n))
    distance_alpha = np.zeros(shape=(num_nodes,num_nodes))
    Nei_agent = np.zeros(shape=(num_nodes, num_nodes - 1, 1))

    for i in range(num_nodes):
        dif[i] = np.tile(np.array(nodes[i,:]), (num_nodes, 1)) - nodes
        tmp = dif[i]
        d_tmp = np.zeros(shape=(num_nodes, 1))
        for j in range(num_nodes):
            d_tmp[j,:] = np.linalg.norm(tmp[j,:])
        distance_alpha[i,:] = d_tmp.ravel()

    for k in range(num_nodes):
        out = np.full((num_nodes - 1, 1), np.nan)
        indices = np.nonzero(np.logical_and(distance_alpha[:,k] < r, distance_alpha[:,k] != 0.))[0]
        out.ravel()[:len(indices)] = indices
        Nei_agent[k] = out

    A = np.zeros(shape=(num_nodes, num_nodes))
    for i in range(num_nodes):
        Nei_nodes = nodes[Nei_agent[i][~np.isnan(Nei_agent[i])].ravel().astype(int), :]
        for j in range(num_nodes):
            dist_2nodes = np.linalg.norm(nodes[j,:] - nodes[i,:])

            if dist_2nodes == 0:
                A[i,j] = 0
            elif dist_2nodes < r and dist_2nodes != 0:
                A[i,j] = 1
            elif dist_2nodes > r:
                A[i,j] = 0

    return Nei_agent, A
    
