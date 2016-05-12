# k-means

This is a toy implementation of k-means for CMSC 471, AI.

Invocation is
```
python Clustering.py [num_clusters] [points_file]
```
where `num_clusters` is `k`, and `points_file` is the path to a file containing a list of
n-dimeinsional vectors as follows.

```
9 9 9
11 11 11
10 11 9
10 9 11

-9 -9 -9
-11 -11 -11
-10 -11 -9
-10 -9 -11

-9 9 9
-11 11 11
-10 11 9
-10 9 11
```

Blank linkes are ignored.

Default output is a `numpy`-formatted matrix of cluster centers in arbitrary order.

## Plotting

Plotting is automatic if the vectors are 2D, but may be disabled with `--noplot`.

## Packages

Requirements are `numpy` and `matplotlib`. It is Python2 and Python3 compatible.
