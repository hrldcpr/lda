import collections
import json
import random
import re
import sys

import numpy


def infer_topics(doc, topics, alpha=0.0001, iterations=1000):
    # TODO tokenize doc
    # throw out irrelevant topics:
    words = set(doc)
    words = {w for topic_words in topics.values() for w in topic_words
             if w in words}
    topics = {k: topic_words for k,topic_words in topics.items()
              if any(w in words for w in topic_words)}
    doc = [w for w in doc
           if w in words]

    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    if K == 0: return {}

    zs = [random.randrange(K) for _ in doc]
    for _ in range(iterations):
        topic_counts = numpy.zeros(K)
        for z in zs: topic_counts[z] += 1
        theta = numpy.random.dirichlet(alpha + topic_counts)

        for i,w in enumerate(doc):
            ps = numpy.array([theta[k] * topics[k].get(w, 0) for k in range(K)])
            zs[i] = numpy.random.multinomial(1, ps / ps.sum()).argmax()

    return {topic_ids[i]: weight for i,weight in enumerate(theta)}


topic_words = {}
word_topics = collections.defaultdict(dict)
with open('lda-2500') as f:
    for line in f:
        topic, words = line.strip().split('\t')
        # keys have commas and colons so we can't do nice split comprehensions :'(
        words = {word: float(weight)
                 for word,weight,_ in re.findall(r'(.*?):([\d\.E\-]*?)(,|$)', words.strip('{}'))}
        topic_words[topic] = words
        for word,weight in words.items():
            word_topics[word][topic] = weight

print(infer_topics(sys.argv[2:], topic_words))
