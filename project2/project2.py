import sys
import pandas as pd
import numpy as np
import os
import csv
from timeit import default_timer as timer
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
import collections

currentDirectory = os.getcwd().replace('\\', '/')
dataDirectory = currentDirectory + '/data'

GAMMA = 0.95
ALPHA = 0.75
LAMBDA = 0.75
SMALL_ITER = 5
MEDIUM_ITER = 10


class NDSparseMatrix:
  def __init__(self):
    self.elements = {}

  def addValue(self, tuple, value):
    self.elements[tuple] = value

  def readValue(self, tuple):
    try:
      value = self.elements[tuple]
    except KeyError:
      # could also be 0.0 if using floats...
      value = 0.0
    return value


def transitionm(df):
    # Maximum likelihood estimate of transition matrix, SxS matrix for each A.
    T = collections.defaultdict(int)
    for sp in df.sp.unique():
        for s in df.s.unique():
            for a in df.a.unique():
                num_sas = len(df[(df.s == s) & (df.a == a) & (df.sp == sp)])
                num_sa = len(df[(df.s == s) & (df.a == a)].a)
                if num_sa == 0:
                    T[(s,a)] = 0
                else:
                    T[(s,a)] = num_sas/num_sa
    return T


def tr(df):
    # Maximum likelihood estimate of reward matrix, SxA matrix, and transition matrix, SxS matrix for each A.
    R = NDSparseMatrix()
    T = NDSparseMatrix()
    # R = collections.defaultdict(int)
    # T = collections.defaultdict(int)
    df_t = df[df.s != df.sp]
    df_r = df[df.r != 0]
    for s in df.s.unique():
        for a in df.a.unique():
            for sp in df.sp.unique():
                rho = sum(df[(df.s == s) & (df.a == a)].r)
                num_sa = len(df[(df.s == s) & (df.a == a)].a)
                num_sas = len(df[(df.s == s) & (df.a == a) & (df.sp == sp)])
                if num_sa != 0:
                    R.addValue((s,a), rho/num_sa)
                    if num_sas != 0:
                        T.addValue((sp, s, a), num_sas/num_sa)
    return T, R


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
        # TODO: Try eligibility traces!
        # Tried Q-learning and it is NOT GOOD LOL
        # Probably due to all of the zero-reward states and that many states are not visited!

        Q = np.zeros((312020,9))
        N = np.zeros((312020,9))
        for i in range(len(df.index)):
            s = df.s[i] - 1
            a = df.a[i] - 1
            r = df.r[i]
            sp = df.sp[i] - 1
            # print([s, a, r, sp])
            # e_tp = r + GAMMA * lil_matrix.getrow(Q, sp).tocsr().max() - Q[s,a]
            # e_t = r + GAMMA * lil_matrix.getrow(Q, sp).tocsr().max() - lil_matrix.getrow(Q, s).tocsr().max()
            # for s in range(312020):
            #     for a in range(9):
            #         N[s,a] = LAMBDA * GAMMA * N[s,a]
            #         Q[s,a] = Q[s,a] + ALPHA * N * e_t
            e_tp = r + GAMMA * np.max(Q[sp,:]) - Q[s,a]
            e_t = r + GAMMA * np.max(Q[sp,:]) - np.max(Q[s,:])
            N = LAMBDA * GAMMA * N
            Q = Q + ALPHA * N * e_t
            # for s in range(312020):
            #     for a in range(9):
            #         N[s,a] = LAMBDA * GAMMA * N[s,a]
            #         Q[s,a] = Q[s,a] + ALPHA * N[s,a] * e_t
            Q[s,a] += ALPHA * N[s,a] * e_tp
            N[s,a] += 1
        # policy = pd.DataFrame(Q[1:, 1:].tocsr().argmax(axis=1), columns=['action'])
        policy = pd.DataFrame((np.max(Q, axis=1) + 1).astype(int))
        # policy[policy.action == 0] = 4
        policy.to_csv(outfile, index=False, header=False)
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