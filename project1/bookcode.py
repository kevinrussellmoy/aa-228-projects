# An attempt to recreate the code from the book
# Kevin Moy, 10/5/2020

import pandas as pd
import networkx as nx
import numpy as np
from scipy.special import loggamma
import matplotlib.pyplot as plt
import os

currentDirectory = os.getcwd().replace('\\', '/')
exampleDirectory = currentDirectory + '/example'
dataDirectory = currentDirectory + '/data'


def sub2ind(size, x):
    k = np.hstack((1, size[0:-1]))
    return np.dot(k, x)


# Algorithm 4.1: A function for extracting the statistics, or counts,
# from a discrete dataset D assuming a Bayesian network with variables vars and structure G
def statistics(vars, G, D):
    n = D.shape[1]
    r = np.array(vars)
    q = [int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    M = [np.zeros([q[i], r[i]], dtype=int) for i in range(n)]
    for column in D:
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
    p -= np.sum(loggamma(a))  # TODO: Remove this later: loggamma(a) = 0 for uniform prior
    p += np.sum(loggamma(np.sum(a, axis=1)))
    p -= np.sum(loggamma(np.sum(a, axis=1) + np.sum(m, axis=1)))
    return p


def bayesian_score(vars, G, D):
    r = np.array(vars)
    n = r.size
    alpha = uniform_prior(vars, G)
    M = statistics(vars, G, D)
    # p = sum([sum(sum(loggamma(sum(z)))) for z in zip(alpha, M)]) # if we want to be real fancy and do it all in one line!
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))

# TODO: Figure out write_to_gph from this example 4.1
# # Example 4.1
# G_41 = nx.DiGraph()
# G_41.add_nodes_from([0, 1, 2]) # list of nodes
# G_41.add_edges_from([(0,1), (2,1)])
# nx.draw_networkx(G_41, with_labels = True)
# vars_41 = [2,2,2] # number of values each variable can take on from [1,2,3]
# # print(list(G.predecessors(2))) # Get list of predecessors (1st parents) of a node 2
# D_41 = np.array([[1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]) - 1
#
# M = statistics(vars_41, G_41, D_41)
# priors = uniform_prior(vars_41, G_41)
# # theta = [M[i]/np.amax(M[i],axis=1,keepdims=True) for i in range(D_41.shape[0])] # throws a warning for dem NaNs
# score = bayesian_score(vars_41, G_41, D_41)
#
# # print(M)
# # print(priors)
# # print(theta)
# print(score)

# # Example from example
# ex = nx.DiGraph()
# edges = nx.read_edgelist(exampleDirectory + '/example.gph', delimiter=',')
# ex.add_edges_from(edges.edges())
# # nx.draw_networkx(ex, with_labels=True)
# G_ex = nx.convert_node_labels_to_integers(ex)
#
# ex_data = pd.read_csv(exampleDirectory + '/example.csv')
# colnames = np.array(ex_data.columns)
# D_ex = ex_data.values - 1
# vars_ex = np.array(ex_data.max())
#
# score_ex = bayesian_score(vars_ex, G_ex, D_ex)
# print(score_ex)

# small CSV test
data = pd.read_csv(dataDirectory + '/small.csv')
colnames = np.array(data.columns)
n = colnames.size
D = data.values - 1
nodels = [(i,{'name':colnames[i]}) for i in range(n)]
G = nx.DiGraph()
G.add_nodes_from(nodels)
vars = np.array(data.max())
plt.figure(figsize =(9, 9))
nx.draw_networkx(G, with_labels=True)
plt.show()

# TODO: Create separate file for plotting from .gph once completed
# plt.figure()
# pos_nodes = nx.circular_layout(G)
# nx.draw(G, pos_nodes, with_labels=True)
#
# pos_attrs = {}
# for node, coords in pos_nodes.items():
#     pos_attrs[node] = (coords[0], coords[1] + 0.08)
#
# node_attrs = nx.get_node_attributes(G, 'name')
# custom_node_attrs = {}
# for node, attr in node_attrs.items():
#     custom_node_attrs[node] = attr
#
# nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs)
# axes = plt.gca()
# axes.set_xlim([-2,2])
# axes.set_ylim([-2,2])
# plt.savefig('small.png')
# plt.show()
# score_ex = bayesian_score(vars_ex, G_ex, D_ex)
# print(score_ex)

