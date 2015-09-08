from __future__ import division
from collections import Counter, defaultdict
import math, random, re, glob


def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z09']+", message)  # extract all words
    return set(all_words)


def count_words(training_set):
    """training set consists of pairs(message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
        w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.iteritems()]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    # iterate through each word in vocabulary
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        # if *word* appears in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # if *word* doesnt appear in the message
        # then add the log prob of not seeing it
        # which is log(1 - prob of seeing it)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        # count spam and non-spam msgs
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training set through pipeline
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def import_data(path):
    data = []

    # returns every filepath that matches the wildcard path
    for fn in glob.glob(path):
        is_spam = "spms" not in fn

        with open(fn, 'r') as file:
            for line in file:
                if line.startswith("Subject:"):
                    # remove the leading "Subject:" and keep the remainder of the line
                    subject = re.sub(r"^Subject: ", "", line).strip()
                    data.append((subject, is_spam))

    return data


def test_and_train_data(data):
    random.shuffle(data)

    train_data = data[:75]
    test_data = data[25:]

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    # triplets (subject, actual is_spam, predicted spam_probability)
    classified = [(subject, is_spam, classifier.classify(subject))
                  for subject, is_spam in test_data]

    # assume that spam prob is > 0.5 corresponds to spam prediction
    # and count the combinations of (actual is_spam, predicted_spam)
    counts = Counter((is_spam, spam_probability > 0.5)
                     for _, is_spam, spam_probability in classified)

    print("Stats:")
    print (counts)

    # sort by spam prob from smallest to largest
    classified.sort(key=lambda row: row[2])

    return classifier, classified


def p_spam_given_words(word_prob):
    """uses bayes theorem to compute spammiest words"""

    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model(path):
    data = import_data(path)
    classifier, classified = test_and_train_data(data)

    # highest predicted spam among non-spams
    spammiest_non_spam = filter(lambda row: not row[1], classified)[-5:]

    # lowest predicted spam probs among actual spams
    most_not_spammy_spams = filter(lambda row: row[1], classified)[:5]

    print("Most 'like spam' non-spam: ")
    print(spammiest_non_spam)
    print("Most 'like non-spam' spam: ")
    print(most_not_spammy_spams)

    words = sorted(classifier.word_probs, key=p_spam_given_words)

    spammiest_words = words[-5:]
    non_spammiest_words = words[:5]

    print("Spammiest words: ")
    print(spammiest_words)
    print("Non-spammiest words: ")
    print(non_spammiest_words)

# path for corpus of data
path = r"/Users/evan/PycharmProjects/DataScienceFromScratch/resources/lingspam_public/lemm/*/*"
train_and_test_model(path)