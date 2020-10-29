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


def compute(infile, outfile):
    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()