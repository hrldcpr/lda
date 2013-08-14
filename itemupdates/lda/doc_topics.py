import collections
import json
import operator
import random
import re
import sys

import numpy
import scipy.special

fst = operator.itemgetter(0)
snd = operator.itemgetter(1)
def sorted_by_value(d):
    if isinstance(d, dict): d = d.items()
    return sorted(d, key=snd, reverse=True)

def factorial(n):
    x = 1
    for i in range(2, int(n) + 1):
        x *= i
    return x

def prep(doc, topics):
    # TODO tokenize doc
    # throw out irrelevant topics:
    words = set(doc)
    words = {w for topic_words in topics.values() for w in topic_words
             if w in words}
    topics = {k: topic_words for k,topic_words in topics.items()
              if any(w in words for w in topic_words)}
    doc = [w for w in doc
           if w in words]
    return doc, topics

def infer_topics(doc, topics, alpha=0.0001, iterations=10000, burn=1000, thin=100):
    doc, topics = prep(doc, topics)

    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    if K == 0: return {}

    n_thetas = 0
    sum_thetas = numpy.zeros(K)
    theta = numpy.random.dirichlet(alpha + numpy.zeros(K))
    zs = [None for _ in doc]
    for j in range(iterations):
        for i,w in enumerate(doc):
            ps = numpy.array([theta[k] * topics[k].get(w, 0) for k in range(K)])
            zs[i] = numpy.random.multinomial(1, ps / ps.sum()).argmax()

        topic_counts = numpy.zeros(K)
        for z in zs: topic_counts[z] += 1
        theta = numpy.random.dirichlet(alpha + topic_counts)

        if j > burn and j % thin == 0:
            sum_thetas += theta
            n_thetas += 1

    theta = sum_thetas / n_thetas
    return {topic_ids[i]: weight for i,weight in enumerate(theta)}

def infer_topics_collapsed(doc, topics, alpha=0.0001, iterations=10000, burn=1000, thin=100):
    doc, topics = prep(doc, topics)

    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    if K == 0: return {}

    n_thetas = 0
    sum_thetas = numpy.zeros(K)
    zs = [None for _ in doc]
    topic_counts = numpy.zeros(K)
    for j in range(iterations):
        for i,w in enumerate(doc):
            if zs[i] is not None: topic_counts[zs[i]] -= 1
            ps = numpy.array([topics[k].get(w, 0) * factorial(topic_counts[k])
                              for k in range(K)])
            zs[i] = numpy.random.multinomial(1, ps / ps.sum()).argmax()
            topic_counts[zs[i]] += 1

        if j > burn and j % thin == 0:
            sum_thetas += numpy.random.dirichlet(alpha + topic_counts)
            n_thetas += 1

    theta = sum_thetas / n_thetas
    return {topic_ids[k]: weight for k,weight in enumerate(theta)}

def generate_doc(topics, alpha=0.0001):
    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    N = 6

    theta = numpy.random.dirichlet(alpha + numpy.zeros(K))
    print(sorted_by_value({topic_ids[i]: weight for i,weight in enumerate(theta) if weight > 0.05}))

    doc = []
    for _ in range(N):
        z = numpy.random.multinomial(1, theta).argmax()
        topic_words = sorted_by_value(topics[z])
        p = random.random()
        for word,weight in topic_words:
            p -= weight
            if p < 0: break
        doc.append(word)
    return doc

def test(topics, alpha=0.0001):
    doc = generate_doc(topics, alpha=alpha)
    print(doc)
    doc_topics = infer_topics(doc, topics, alpha=alpha)
    print(sorted_by_value({k:v for k,v in doc_topics.items() if v > 0.01}))
    doc_topics = infer_topics_collapsed(doc, topics, alpha=alpha)
    print(sorted_by_value({k:v for k,v in doc_topics.items() if v > 0.01}))

topic_words = {}
word_topics = collections.defaultdict(dict)
# with open('lda-2500') as f:
with open('lda.topToWor.txt') as f:
    for line in f:
        # topic, words = line.strip().split('\t')
        topic, words = line.strip().split(':', 1)
        # keys have commas and colons so we can't do nice split comprehensions :'(
        words = {word: float(weight)
                 # for word,weight,_ in re.findall(r'(.*?):([\d\.E\-]*?)(,|$)', words.strip('{}'))}
                 for word,weight in re.findall(r'\((.*?),([\d\.E\-]*?)\)', words.strip('{}'))}
        topic_words[topic] = words
        for word,weight in words.items():
            word_topics[word][topic] = weight


doc = [w.lower() for w in sys.argv[1:]]

if doc:
    import time
    t = time.time()
    doc_topics = infer_topics(doc, topic_words)
    print(time.time() - t)

    print(sorted_by_value(doc_topics))

else:
    test(topic_words)
