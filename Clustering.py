#!env/bin/python

import sys, argparse
import numpy as np


def main(args):
    pass

def knn(vectors, k, positions=None):
    # If there are no vectors to cluster, we can exit here.
    if len(vectors) == 0: return None

    # Get the dimensionality, which should be uniform.
    dimensionality = vectors[0].size
    if positions == None:
        # Find the min/max of each column, so we get a good range in which to
        # put the random initial points. This will reduce the total number of
        # iterations.
        minvec = np.array([vectors[:,i].min() for i in range(dimensionality)])
        maxvec = np.array([vectors[:,i].max() for i in range(dimensionality)])
        spread = maxvec - minvec

        # Choose the initial positions inside the box of extreme points.
        positions = minvec + np.random.rand(k, dimensionality) * spread

def read_vectors(f, vector_type=float, sep=" ", include_empty=False):
    """Read vectors from an open file stream. Vectors must be listed one per
    line, and be separated with the `sep` character. If `include_empty` is set,
    then empty 

    Returns an iterable, not necessarily a list.
    """
    for line in f:
        fields = [vector_type(entry) for entry in line.split(sep)]

        # Skip empty lines if directed to.
        if len(fields) == 0 and not include_empty:
            continue

        # Yield the vector as a numpy array.
        yield np.array(fields)

# A convenience function for validating clusters argument from argparse.
def natural_number(s):
    """Convert s to a natural number (integer > 0) if possible."""
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("{:r} is not a natural number" \
                .format(n))
    return n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clusters", type=natural_number,
            help="number of clusters (>0)")
    parser.add_argument("data_file", type=argparse.FileType('r'),
            help="file from which to read data points")
    sys.exit(main(parser.parse_args()))
