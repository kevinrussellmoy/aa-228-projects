import sys
import pandas as pd
import networkx as nx
import numpy as np
from scipy.special import loggamma
import os

currentDirectory = os.getcwd().replace('\\', '/')
exampleDirectory = currentDirectory + '/data'

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    # TODO: Port code over from bookcode.py
    # TODO: Handle CSV (--> DataFrame --> NumPy array + list of names
    # TODO: test on example
    # TODO: Create idx2names = dictionary from node numbers (idx) to filenames
    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
