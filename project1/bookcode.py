# An attempt to recreate the code from the book
# Kevin Moy, 10/5/2020

import pandas as pd
import networkx as nx
import numpy as np
# from sklearn.preprocessing import normalize
from scipy.special import loggamma


def sub2ind(size, x):
    k = np.hstack((1, size[0:-1]))
    return np.dot(k, x)


# Algorithm 4.1: A function for extracting the statistics, or counts,
# from a discrete dataset D assuming a Bayesian network with variables vars and structure G
def statistics(vars, G, D):
    n = D.shape[0]
    r = np.array(vars)
    q = [int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    M = [np.zeros([q[i], r[i]], dtype=int) for i in range(n)]
    for column in D.T:
        for i in range(n):
            k = column[i]
            parents = list(G.predecessors(i))
            j = 0
            if not (not list(G.predecessors(i))):
                j = sub2ind(r[parents], column[parents])
            M[i][j,k] += 1
    return M


# Algorithm 4.2: Forming a uniform prior
def uniform_prior(vars, G):
    r = np.array(vars)
    n = r.size
    q = [int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    return [np.ones([q[i], r[i]], dtype=int) for i in range(n)]


# Algorithm 5.1: An algorithm for computing the Bayesian score for a list of variables vars and a graph G given data D,
# using the uniform prior from Algorithm 4.2.
def bayesian_score_component(m,a):
    p = np.sum(loggamma(a + m))
    p -= np.sum(loggamma(a)) # TODO: Remove this later: loggamma(a) = 0 for uniform prior
    p += np.sum(loggamma(np.sum(a)))
    p -= np.sum(loggamma(np.sum(a) + np.sum(m)))
    return p


def bayesian_score(vars, G, D):
    r = np.array(vars)
    n = r.size
    alpha = uniform_prior(vars, G)
    # p = sum([sum(sum(loggamma(sum(z)))) for z in zip(alpha, M)]) # if we want to be real fancy and do it all in one line!
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


# Example 4.1
G_41 = nx.DiGraph()
G_41.add_nodes_from([0, 1, 2]) # list of nodes
G_41.add_edges_from([(0,1), (2,1)])
vars_41 = [2,2,2] # number of values each variable can take on from [1,2,3]
# print(list(G.predecessors(2))) # Get list of predecessors (1st parents) of a node 2
D_41 = np.array([[1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]) - 1

M = statistics(vars_41, G_41, D_41)

priors = uniform_prior(vars_41, G_41)

# theta = [M[i]/np.amax(M[i],axis=1,keepdims=True) for i in range(D_41.shape[0])] # throws a warning for dem NaNs

score = bayesian_score(vars_41, G_41, D_41)

# print(M)
#
# print(priors)
#
# print(theta)

print(score)

