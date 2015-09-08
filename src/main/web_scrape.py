from __future__ import division
from collections import Counter
import math, random, csv, json
from bs4 import BeautifulSoup
import re
import requests
from time import sleep

base_url = "http://shop.oreilly.com/category/browse-subjects/apple-mac.do?sortby=publicationDate&page="

books = []


def is_video(td):
    """it's a video if it has exactly one pricelabel, and if
    the stripped text inside that pricelabel starts with 'Video'"""
    pricelabels = td('span', 'pricelabel')
    return (len(pricelabels) == 1 and
            pricelabels[0].text.strip().startswith("Video"))


def book_info(td):
    """given a BeautifulSoup <td> Tag representing a book,
    extract the book's details and return a dict"""

    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()

    return {
        "title": title,
        "authors": authors,
        "isbn": isbn,
        "date": date
    }


def print_books_found(books):
    for book in books:
        print book.get("title")


def web_scrape(NUM_PAGES=20):
    for page_num in range(1, NUM_PAGES + 1):
        print "souping page", page_num, ",", len(books), " found so far"
        url = base_url + str(page_num)
        soup = BeautifulSoup(requests.get(url).text, 'html5lib')

        for td in soup('td', 'thumbtext'):
            if not is_video(td):
                books.append(book_info(td))

        sleep(30)

    print_books_found(books)
    print "Total number of books found: ", len(books)
    print "Total number of pages crawled: ", NUM_PAGES


web_scrape(20)
