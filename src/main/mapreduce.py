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


def map_reduce(inputs, mapper, reducer):
    """runs MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.iteritems()
            for output in reducer(key, values)]


def reduce_values_using(aggregation_fn, key, values):
    """reduces a key-values pair by applying aggregation_fn to the values"""
    yield (key, aggregation_fn(values))


def values_reducer(aggregation_fn):
    """turns a function (values -> output) into a reducer
    that maps (key, values) -> (key, output)"""
    return partial(reduce_values_using, aggregation_fn)


def data_science_day_mapper(status_update):
    """yields (day_of_week, 1) if status_update contains "data science" """
    if "data science" in status_update["text"].lower():
        day_of_week = status_update["created_at"].weekday()
        yield (day_of_week, 1)


def words_per_user_mapper(status_updates):
    user = status_updates["username"]
    for word in tokenize(status_updates["text"]):
        yield (user, (word, 1))


def most_popular_word_reducer(user, words_and_counts):
    """given a sequence of (word, count) pairs,
    return the word with the highest total count"""

    word_counts = Counter();
    for word, count in words_and_counts:
        word_counts[word] += count

    word, count = word_counts.most_common(1)[0]

    yield (user, (word, count))


def liker_mapper(status_update):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)


if __name__ == "__main__":

    documents = ["data science", "big data", "science fiction"]

    word_counts = map_reduce(documents, wc_mapper, wc_reducer)

    print word_counts

    sum_reducer = values_reducer(sum)
    max_reducer = values_reducer(max)
    min_reducer = values_reducer(min)
    count_distinct_reducer = values_reducer(lambda values: len(set(values)))

    status_updates = [
    {"id": 1,
     "username" : "joelgrus",
     "text" : "Is anyone interested in a data science book?",
     "created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
     "liked_by" : ["data_guy", "data_gal", "bill"] },
    # add your own
]

    data_science_days = map_reduce(status_updates, data_science_day_mapper, sum_reducer)
    print data_science_days


    user_words = map_reduce(status_updates, words_per_user_mapper, most_popular_word_reducer)
    print user_words
    
    distinct_likers_per_users = map_reduce(status_updates, liker_mapper, count_distinct_reducer)
    print distinct_likers_per_users