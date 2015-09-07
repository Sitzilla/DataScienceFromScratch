from __future__ import division
import math, random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt



def plot_resumes():
    # data pasted from https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py
    data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
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




