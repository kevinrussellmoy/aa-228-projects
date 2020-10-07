import sys
import pandas as pd
import networkx as nx
import numpy as np
from scipy.special import loggamma
import os

currentDirectory = os.getcwd().replace('\\', '/')
dataDirectory = currentDirectory + '/data'


def sub2ind(size, x):
    k = np.hstack((1, size[0:-1]))
    return np.dot(k, x)


def statistics(vars, G, D):
    # Algorithm 4.1: A function for extracting the statistics, or counts,
    # from a discrete dataset D assuming a Bayesian network with variables vars and structure G
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


def uniform_prior(vars, G):
    # Algorithm 4.2: Forming a uniform prior
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
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


def graph_from_df(df):
    # Produce graph from dataframe, with headers as variable names and rows of instantiations
    vnames = np.array(df.columns)
    n = vnames.size
    D = df.values - 1
    nodels = [(i, {'name': vnames[i]}) for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(nodels)
    vars = np.array(df.max())
    return vars, G, D


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def k2fit(vars, D):
    # K2 search with ordering taken randomly.
    pass

def compute(infile, outfile):
    # TODO: Create idx2names = dictionary from node numbers (idx) to filenames
    df = pd.read_csv(dataDirectory + '/' + infile)
    vars, G, D = graph_from_df(df)
    score = bayesian_score(vars, G, D)
    print('Score of ' + infile + ' is: ' + str(score))


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
