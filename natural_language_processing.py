from __future__ import division
import math, random, re
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
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
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
                tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

            # now call expand on new list of tokens
            return expand(grammar, tokens)

        # if we get here we had all terminals and are done
        return tokens

    def generate_sentence(grammar):
        return expand(grammar, ["_S"])

    print generate_sentence(grammar)

grammar_rules()

















