from __future__ import division
from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class KMeans:
    """performs k-means clustering"""

    def __init__(self, k):
        self.k = k  # number of clusters
        self.means = None  # number of clusters

    def classify(self, input):
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        # choose k random points as the initial mean
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # find new assignments
            new_assignments = map(self.classify, inputs)

            # if no assignments have changed, we are done
            if assignments == new_assignments:
                return

            # otherwise keep new assignments
            assignments = new_assignments

            # and compute new means based on the new assignement
            for i in range(self.k):
                # find all points assigned to cluster i
                i_points = [p for p, a in zip(inputs, assignments) if a == i]

                # make sure i_points is not empty so don't divide by 0
                if i_points:
                    self.means[i] = vector_mean(i_points)