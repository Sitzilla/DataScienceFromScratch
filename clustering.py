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


def recolor_image(file_path, k = 5):
    def recolor(pixel):
        cluster = clusterer.classify(pixel)  # index of the closest cluster
        return clusterer.means[cluster]  # mean of the closest cluster

    img = mpimg.imread(file_path)
    top_row = img[0]
    top_left_pixel = top_row[0]
    red, green, blue = top_left_pixel

    pixels = [pixel for row in img for pixel in row]  # flattened list of all pixels

    clusterer = KMeans(5)  # reduce to only 5 pixels
    clusterer.train(pixels)  # may take awhile

    new_img = [[recolor(pixel) for pixel in row]  # recolor this row of pixels
               for row in img]  # for each row in image

    plt.imshow(new_img)
    plt.axes('off')
    plt.show()


def is_leaf(cluster):
    """cluster is a leaf if length == 1"""
    return len(cluster) == 1


def get_children(cluster):
    """returns the two children of this cluster if its a merged cluster;
    raises exception if leaf cluster"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]


def get_values(cluster):
    """returns the value in this cluster (if its a leaf cluster)
    or all the values in the leaf cluster below it (if it is not)"""
    if is_leaf(cluster):
        return cluster
    else:
        return [value for child in get_children(cluster)
        for value in get_values(child)]


def cluster_distance(cluster1, cluster2, distance_agg=min):
    """compute all the pairwise distances between cluster1 and cluster2
    and apply _distance_agg_ to the resulting list"""
    return distance_agg([distance(input1, input2)
                         for input1 in get_values(cluster1)
                         for input2 in get_values(cluster2)])


def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0] #merge_order is first element in 2-tuple


def bottom_up_cluster(inputs, distance_agg=min):
    # start with every leaf input a leaf cluster / tuple
    clusters = [(input,) for input in inputs]

    # as long as we have more than one cluster left
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min([(cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2  in clusters[:i]],
                     key=lambda (x, y): cluster_distance(x, y, distance_agg))

        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them, using merge_order = # of clusters left
        merged_cluster = (len(clusters), [c1, c2])

        # and add their merge
        clusters.append(merged_cluster)

    return clusters[0]




# test data copied from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py
inputs = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13], [-46, 5], [-34, -1],
          [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19], [-41, 8], [-11, -6], [-25, -9], [-18, -3]]


# K-Means clustering
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


# Bottom-up Hierarchical Clustering
base_cluster = bottom_up_cluster(inputs)
