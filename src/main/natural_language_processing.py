from __future__ import division
import random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt


def plot_resumes():
    # data pasted from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py
    data = [("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
            ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
            ("data science", 60, 70), ("analytics", 90, 3),
            ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
            ("actionable insights", 40, 30), ("think out of the box", 45, 10),
            ("self-starter", 30, 50), ("customer focus", 65, 15),
            ("thought leadership", 35, 35)]

    def text_size(total):
        """equals 8 if total is 0, 28 if total is 200"""
        return 8 + total / 200 * 20

    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))

    plt.xlabel("Popularity on Job Postings")
    plt.ylabel("Popularity on Resumes")
    plt.axis([0, 100, 0, 100])
    plt.xticks([])
    plt.yticks([])
    plt.show()


def n_gram_model():
    url = "https://beta.oreilly.com/ideas/what-is-data-science"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')

    content = soup.find("div", "article-body")
    regex = r"[\w']+|[\.]"  # matches a word or a period

    document = []

    def fix_unicode(text):
        return text.replace(u"\u2019", "'")

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    bigrams = zip(document, document[1:])
    transitions = defaultdict(list)
    for prev, current in bigrams:
        transitions[prev].append(current)

    def generate_using_bigrams():
        current = "."  # the next word that will start a sentence
        result = []
        while True:
            next_word_candidates = transitions[current]  # bigrams (current, _)
            current = random.choice(next_word_candidates)  # choose one at random
            result.append(current)  # append to results
            if current == ".": return " ".join(result)  # if "." we are done

    trigrams = zip(document, document[1:], document[2:])
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in trigrams:

        if prev == ".":  # if previous word was a period
            starts.append(current)  # then this is a start word

        trigram_transitions[(prev, current)].append(next)

    def generate_using_trigrams():
        current = random.choice(starts)  # chooses a random starting word
        prev = "."
        result = [current]

        while True:
            next_word_candidates = trigram_transitions[(prev, current)]
            next_word = random.choice(next_word_candidates)

            prev, current = current, next_word
            result.append(current)

            if current == ".":
                return " ".join(result)

    print generate_using_trigrams()


def grammar_rules():
    # data pasted from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py
    grammar = {
        "_S": ["_NP _VP"],
        "_NP": ["_N",
                "_A _NP _P _A _N"],
        "_VP": ["_V",
                "_V _NP"],
        "_N": ["data science", "Python", "regression"],
        "_A": ["big", "linear", "logistic"],
        "_P": ["about", "near"],
        "_V": ["learns", "trains", "tests", "is"]
    }

    def is_terminal(token):
        return token[0] != "_"

    def expand(grammar, tokens):
        for i, token in enumerate(tokens):

            # skip over terminals
            if is_terminal(token): continue

            # if we get here, we found a non-terminal token
            # so we need to choose a replacement at random
            replacement = random.choice(grammar[token])

            if is_terminal(replacement):
                tokens[i] = replacement
            else:
                tokens = tokens[:i] + replacement.split() + tokens[(i + 1):]

            # now call expand on new list of tokens
            return expand(grammar, tokens)

        # if we get here we had all terminals and are done
        return tokens

    def generate_sentence(grammar):
        return expand(grammar, ["_S"])

    print generate_sentence(grammar)


def gibbs_sampling():
    def roll_a_dice():
        return random.choice([1, 2, 3, 4, 5, 6])

    def direct_sample():
        d1 = roll_a_dice();
        d2 = roll_a_dice();
        return d1, d1 + d2

    def random_y_given_x(x):
        return x + roll_a_dice()

    def random_x_given_y(y):
        if y <= 7:
            # first die equally likely to be anything up to total - 1
            return random.randrange(1, y)
        else:
            # first die equally likely to be anything up to 6
            return random.randrange(y - 6, 7)

    def gibbs_sample(num_iters=500):
        x, y = 1, 2
        for _ in range(num_iters):
            x = random_x_given_y(y)
            y = random_y_given_x(x)
        return x, y

    def compare_distributions(num_samples=1000):
        counts = defaultdict(lambda: [0, 0])
        for _ in range(num_samples):
            counts[gibbs_sample()][0] += 1
            counts[direct_sample()][1] += 1
        return counts

    print compare_distributions()


def topic_modeling():
    def sample_from(weights):
        """return i with probability weights[i] / sum(weights)"""
        total = sum(weights)
        rnd = total * random.random()  # uniform between 1 and total
        for i, w in enumerate(weights):
            rnd -= w  # return the smallest i such that
            if rnd <= 0: return i  # weights[0] + ... + weights[i] >= rnd

    # data pasted from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py
    documents = [
        ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
        ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
        ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
        ["R", "Python", "statistics", "regression", "probability"],
        ["machine learning", "regression", "decision trees", "libsvm"],
        ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
        ["statistics", "probability", "mathematics", "theory"],
        ["machine learning", "scikit-learn", "Mahout", "neural networks"],
        ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
        ["Hadoop", "Java", "MapReduce", "Big Data"],
        ["statistics", "R", "statsmodels"],
        ["C++", "deep learning", "artificial intelligence", "probability"],
        ["pandas", "R", "Python"],
        ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
        ["libsvm", "regression", "support vector machines"]
    ]

    # four topics
    K = 4

    # list of counters, one for each document
    document_topic_counts = [Counter() for _ in documents]

    # list of counters, one for each topic
    topic_word_counts = [Counter() for _ in range(K)]

    # list of numbers, one for each topic
    topic_counts = [0 for _ in range(K)]

    # list of numbers, one for each document
    document_lengths = map(len, documents)

    distinct_words = set(word for document in documents for word in document)
    W = len(distinct_words)
    D = len(documents)

    def p_topic_given_document(topic, d, alpha=0.1):
        """fraction of words in document d
        that are assigned to a topic"""
        return ((document_topic_counts[d][topic] + alpha) /
                (document_lengths[d] + K * alpha))

    def p_word_given_topic(word, topic, beta=0.1):
        """fraction of words in topic
        that equal word"""

        return ((topic_word_counts[topic][word] + beta) /
                (topic_counts[topic] + W * beta))

    def topic_weight(d, word, k):
        """given a document and a word in that document
        return the weight for the kth topic"""

        return p_word_given_topic(word, k) * p_topic_given_document(k, d)

    def choose_new_topic(d, word):
        return sample_from([topic_weight(d, word, k)
                            for k in range(K)])

    random.seed(0)
    document_topics = [[random.randrange(K) for word in document]
                       for document in documents]

    for d in range(D):
        for word, topic in zip(documents[d], document_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1

    for iter in range(1000):
        for d in range(D):
            for i, (word, topic) in enumerate(zip(documents[d],
                                                  document_topics[d])):
                # remove this word / topic from the counts
                # so that it doesn't influence the weights
                document_topic_counts[d][topic] -= 1
                topic_word_counts[topic][word] -= 1
                topic_counts[topic] -= 1
                document_lengths[d] -= 1

                # choose a new topic based on weights
                new_topic = choose_new_topic(d, word)
                document_topics[d][i] = new_topic

                # and add back to the counts
                document_topic_counts[d][new_topic] += 1
                topic_word_counts[new_topic][word] += 1
                topic_counts[new_topic] += 1
                document_lengths[d] += 1

    for k, word_counts in enumerate(topic_word_counts):
        for word, count in word_counts.most_common():
            if count > 0: print k, word, count

    topic_names = ["Big Data and programming languages",
                   "Python and statistics",
                   "databases",
                   "machine learning"]

    for document, topic_counts in zip(documents, document_topic_counts):
        print document
        for topic, count in topic_counts.most_common():
            if count > 0:
                print topic_names[topic], count,
        print
