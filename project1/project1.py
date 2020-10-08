import sys
import pandas as pd
import networkx as nx
import numpy as np
from scipy.special import loggamma
import os
from timeit import default_timer as timer
from numpy.random import randint, shuffle
import matplotlib.pyplot as plt


currentDirectory = os.getcwd().replace('\\', '/')
dataDirectory = currentDirectory + '/data'

MAX_ITER = 1000


def sub2ind(size, x):
    # Supporting function for Algorithm 4.1
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


def bayesian_score_component(m,a):
    # Supporting function for Algorithm 5.1
    p = np.sum(loggamma(a + m))
    p -= np.sum(loggamma(a))  # TODO: Remove this later: loggamma(a) = 0 for uniform prior
    p += np.sum(loggamma(np.sum(a, axis=1)))
    p -= np.sum(loggamma(np.sum(a, axis=1) + np.sum(m, axis=1)))
    return p


def bayesian_score(vars, G, D):
    # Algorithm 5.1: An algorithm for computing the Bayesian score for a list of variables vars and a graph G given data D,
    # using the uniform prior from Algorithm 4.2.
    r = np.array(vars)
    n = r.size
    alpha = uniform_prior(vars, G)
    M = statistics(vars, G, D)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


def graph_from_df(df):
    # Produce graph from dataframe, with headers as variable names and rows of instantiations
    # Return:
    # vars - vector of each variable's total possible number of instantiations
    # G - completely unconnected graph with n nodes
    # D - NumPy array of dataset
    # n - number of nodes
    # vnames - variable names in order
    vnames = np.array(df.columns)
    n = vnames.size
    D = df.values - 1
    nodels = [(i, {'name': vnames[i]}) for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(nodels)
    vars = np.array(df.max())
    return vars, G, D, n, vnames


def rand_g_gen(G, n):
    # Generate a new random graph, G2, in the neighborhood of G with number of nodes n
    i = randint(0, high=n)
    j = (i + randint(2, high=n) - 1)//n
    G2 = G.copy()
    if G2.has_edge(i,j):
        G2.remove_edge(i,j)
    else:
        G2.add_edge(i,j)
    return G2


def localfit(vars, D, n):
    # local directed graph search
    # Start with random graph
    Gl = nx.fast_gnp_random_graph(n,0.05,directed=True)
    Gl.remove_edges_from([(u,v) for (u, v) in Gl.edges() if u < v])
    y = bayesian_score(vars, Gl, D)
    for k in range(MAX_ITER):
        G_rand = rand_g_gen(Gl,n)
        # if len(list(nx.simple_cycles(G2))) > 0:  #if cycles are detected
        if nx.is_directed_acyclic_graph(G_rand):
            y2 = bayesian_score(vars, G_rand, D)
            print('new score: ' + str(y2))
        else:
            y2 = -np.inf
        if y2 > y:
            y = y2
            Gl = G_rand
    return Gl


def k2fit(vars, D, G, n):
    # K2 fit using random ordering given known number of nodes n for completely unconnected graph M
    # with nodes [0, 1, 2, ..., n-1]
    ordering = list(range(n))
    shuffle(ordering)
    print(ordering)
    for k in range(1,n):
        i = ordering[k]
        y = bayesian_score(vars, G, D)
        while True:
            y_best = -np.inf
            j_best = -1
            for m in range(k):
                j = ordering[m]
                if not G.has_edge(j, i):
                    G.add_edge(j, i)
                    y2 = bayesian_score(vars, G, D)
                    if y2 > y_best:
                        y_best = y2
                        j_best = j
                    G.remove_edge(j, i)
            if y_best > y:
                y = y_best
                G.add_edge(j_best, i)
                print("adding edge: " + str(j_best) + " to " + str(i))
            else:
                break
    return G


def write_gph(dag, idx2names, n, filename):
    edges = list(dag.edges())
    num_edges = np.shape(edges)[0]
    with open(filename, 'w') as f:
        for i in range(num_edges):
            f.write("{}, {}\n".format(idx2names[edges[i][0]], idx2names[edges[i][1]]))


def print_gph(dag, idx2names, n):
    edges = list(dag.edges())
    num_edges = np.shape(edges)[0]
    for i in range(num_edges):
        print("{}, {}\n".format(idx2names[edges[i][0]], idx2names[edges[i][1]]))


def compute(infile, outfile):
    start = timer()
    fn_ext = os.path.splitext(infile)
    df = pd.read_csv(dataDirectory + '/' + infile)
    vars, G, D, n, vnames = graph_from_df(df)
    isdag = False
    while not isdag:
        G_fit = k2fit(vars, D, G, n)
        # G_fit = localfit(vars, D, n)
        print(list(G_fit.edges))
        isdag = nx.is_directed_acyclic_graph(G_fit)
        print("is it a dag? " + str(isdag) + "!")
    score = bayesian_score(vars, G_fit, D)
    end = timer()
    elapsed = end - start
    print('Final score of ' + infile + ' is: ' + str(score))
    print('Elapsed time of ' + infile + ' is: ' + str(elapsed))
    write_gph(G_fit, vnames, n, outfile)
    plt.figure()
    pos = nx.circular_layout(G_fit)
    nx.draw_networkx(G_fit, pos=pos, with_labels=True, arrows=True, node_color='bisque')
    plt.savefig((fn_ext[0] + '.png'))
    plt.show()


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
