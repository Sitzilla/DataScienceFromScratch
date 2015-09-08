from __future__ import division
from collections import Counter, defaultdict
from functools import partial
import math


def entropy(class_probabilities):
    """given a list of class probabilities, compute entropy"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p)  # ignore zero probabilities


def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets.
        subsets is a list of lists of labeled data"""

    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


def partition_by(inputs, attribute):
    """each input is a parit (attribute_dict, labels).
        returns a dict : attribute_values -> inputs"""
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups


def partition_entropy_by(inputs, attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def classify(tree, input):
    """classify the input using the given decision tree"""

    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree

    # otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of a subtrees to consider next
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)  # none if input is missing attribute

    if subtree_key not in subtree_dict:  # if no subtree for key
        subtree_key = None  # use the default 'none' subtree

    subtree = subtree_dict[subtree_key]  # selects appropriate subtree
    return classify(subtree, input)


def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass
    # all keys of the first inputs are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0: return False  # no Trues? return a "False" leaf
    if num_falses == 0: return True  # no Falses? return a "True" leaf

    if not split_candidates:  # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf

    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]

    # recursively build subtree
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.iteritems()}

    subtrees[None] = num_trues > num_falses  # default case

    return best_attribute, subtrees


inputs = [
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)
]

for key in ['level', 'lang', 'tweets', 'phd']:
    print key, "%.2f" % partition_entropy_by(inputs, key)

print "Building tree"
tree = build_tree_id3(inputs)
print tree

print "Junior / Java / tweets / no phd", classify(tree, {"level": "Junior",
                                                         "lang": "Java",
                                                         "tweets": "yes",
                                                         "phd": "no"})  # True

print "Junior / Java / tweets / phd", classify(tree, {"level": "Junior",
                                                      "lang": "Java",
                                                      "tweets": "yes",
                                                      "phd": "yes"})  # False

print "Intern", classify(tree, {"level": "Intern"})  # True
print "Senior", classify(tree, {"level": "Senior"})  # False
