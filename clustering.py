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


def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))





# test data copied from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py
inputs = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13], [-46, 5], [-34, -1],
          [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19], [-41, 8], [-11, -6], [-25, -9], [-18, -3]]

random.seed(0)  # so that we can compare results
clusterer = KMeans(3)
clusterer.train(inputs)
print "Error with k = 3: ", clusterer.means

# now plot from 1 up to len(inputs) clusters
ks = range(1, len(inputs) + 1)
errors = [squared_clustering_errors(inputs, k) for k in ks]

plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel("k")
plt.ylabel("total squared error")
plt.title("Total Error vs. # of Clusters")
plt.show()