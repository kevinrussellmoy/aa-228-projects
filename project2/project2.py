import sys
import pandas as pd
import networkx as nx
import numpy as np
import os
from timeit import default_timer as timer
from scipy.sparse import lil_matrix


currentDirectory = os.getcwd().replace('\\', '/')
dataDirectory = currentDirectory + '/data'

MAX_ITER = 1000

GAMMA = 0.95
ALPHA = 0.75
SMALL_ITER = 5
MEDIUM_ITER = 10


def compute(infile, outfile):
    # df = pd.read_csv(currentDirectory + '/project2/data/' + infile)
    start = timer()
    df = pd.read_csv(dataDirectory + '/' + infile)
    if infile == 'small.csv':
        Q = lil_matrix((df.s.max() - df.s.min() + 2, df.a.max() - df.a.min() + 2))
        for j in range(SMALL_ITER):
            for i in range(len(df.index)):
                s = df.s[i]
                a = df.a[i]
                r = df.r[i]
                sp = df.sp[i]
                Q[s,a] += (1/(j+1)) * (r + GAMMA * lil_matrix.getrow(Q, sp).tocsr().max() - Q[s, a])
            print('Iteration complete.')
        policy = pd.DataFrame(Q.tocsr().argmax(axis=1)[1:])
        policy.to_csv(outfile, index=False, header=False)
    if infile == 'medium.csv':
        Q = lil_matrix((50000 + 1, 7 + 1))
        # NOTE: Originally tried decrementing the learning rate and trying this MEDIUM_ITER times but saw no
        # appreciable change in reward vs. a set learning rate
        for i in range(len(df.index)):
            s = df.s[i]
            a = df.a[i]
            r = df.r[i]
            sp = df.sp[i]
            Q[s,a] += 0.75 * (r + lil_matrix.getrow(Q, sp).tocsr().max() - Q[s, a])
        policy = pd.DataFrame(Q[1:,1:].tocsr().argmax(axis=1), columns=['action'])
        policy[policy.action == 0] = 4
        policy.to_csv(outfile, index=False, header=False)
    if infile == 'large.csv':
        # TODO: Construct T, R matrices:
        # Get list of unique s
        # For T, initialize a 9-long list with a 312020 x 312020 sparse matrix
        # For R, initialize a 312020 x 9 sparse matrix
        # Iterate for each a for each sp for T(sp | s,a) = N(s,a,sp)/N(s,a) and
        # Iterate for each s for R(s,a) = rho(s,a)/N(s,a)
        # Find the patterns in T and R?
        Q = lil_matrix((312020 + 1, 9 + 1))
        for i in range(len(df.index)):
            s = df.s[i]
            a = df.a[i]
            r = df.r[i]
            sp = df.sp[i]
            Q[s, a] += 0.75 * (r + GAMMA * lil_matrix.getrow(Q, sp).tocsr().max() - Q[s, a])
        # policy = pd.DataFrame(Q[1:, 1:].tocsr().argmax(axis=1), columns=['action'])
        # policy[policy.action == 0] = 4
        # policy.to_csv(outfile, index=False, header=False)
    end = timer()
    elapsed = end - start
    print('Elapsed time of ' + infile + ' is: ' + str(elapsed))


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()