from __future__ import division
import math, random, re, datetime
from collections import defaultdict, Counter
from functools import partial
from naive_bayes import tokenize


def wc_mapper(document):
    """for each word in the document, emit (word, 1)"""
    for word in tokenize(document):
        yield (word, 1)


def wc_reducer(word, counts):
    """sum up the counts for a word"""
    yield (word, sum(counts))


def word_count(documents):
    """count the words in the input documents using MapReduce"""

    # place to store grouped values
    collector = defaultdict(list)

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.iteritems()
            for output in wc_reducer(word, counts)]


documents = ["data science", "big data", "science fiction"]

print word_count(documents)
