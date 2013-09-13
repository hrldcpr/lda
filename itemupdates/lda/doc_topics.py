import argparse
import codecs
import collections
import operator
import random
import re
import sys
import time

#import numpy as np

import clda

fst = operator.itemgetter(0)
snd = operator.itemgetter(1)
def sorted_by_value(d):
    if isinstance(d, dict): d = d.items()
    return sorted(d, key=snd, reverse=True)

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

def infer_topics_gibbs(doc, topics, alpha=0.0001, iterations=10000, burn=1000, thin=100):
    doc, topics = prep(doc, topics)

    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    words = list(set(doc))
    doc = np.array([words.index(w) for w in doc], dtype=int)
    topics = np.array([[topic.get(w, 0) for w in words]
                       for topic in topics])

    if K == 0: return {}

    n_thetas = 0
    sum_thetas = np.zeros(K)
    theta = np.random.dirichlet(alpha + np.zeros(K))
    zs = np.empty(len(doc), dtype=int)
    topic_counts = np.empty(K, dtype=int)
    for j in range(iterations):
        for i,w in enumerate(doc):
            ps = theta * topics[:, w]
            zs[i] = np.random.multinomial(1, ps / ps.sum()).argmax()

        topic_counts.fill(0)
        for z in zs: topic_counts[z] += 1
        theta = np.random.dirichlet(alpha + topic_counts)

        if j > burn and j % thin == 0:
            sum_thetas += theta
            n_thetas += 1

    theta = sum_thetas / n_thetas
    return {topic_ids[i]: weight for i,weight in enumerate(theta)}

def infer_topics_collapsed_gibbs(doc, topics, alpha=0.0001, iterations=10000, burn=1000, thin=100):
    doc, topics = prep(doc, topics)

    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    words = list(set(doc))
    doc = np.array([words.index(w) for w in doc], dtype=int)
    topics = np.array([[topic.get(w, 0) for w in words]
                       for topic in topics])

    if K == 0: return {}

    n_thetas = 0
    sum_thetas = np.zeros(K)
    zs = np.empty(len(doc), dtype=int)
    topic_counts = np.empty(K, dtype=int)
    for j in range(iterations):
        for i,w in enumerate(doc):
            ps = gamma(topic_counts) * topics[:, w]
            zs[i] = np.random.multinomial(1, ps / ps.sum()).argmax()

        topic_counts.fill(0)
        for z in zs: topic_counts[z] += 1

        if j > burn and j % thin == 0:
            theta = np.random.dirichlet(alpha + topic_counts)
            sum_thetas += theta
            n_thetas += 1

    theta = sum_thetas / n_thetas
    return {topic_ids[i]: weight for i,weight in enumerate(theta)}

def infer_topics_c(doc, topics):
    doc, topics = prep(doc, topics)
    if not doc: return None # no known words

    topic_ids = list(topics.keys())
    topics = list(topics.values())

    words = list(set(doc))
    doc = [words.index(w) for w in doc]
    topics = [[topic.get(w, 0) for w in words]
              for topic in topics]

    theta = clda.infer_topics_gibbs(doc, topics)

    return {topic_ids[i]: weight for i,weight in enumerate(theta)}

def generate_doc(topics, alpha=0.0001):
    topic_ids = list(topics.keys())
    topics = list(topics.values())
    K = len(topics)

    N = 6

    theta = np.random.dirichlet(alpha + np.zeros(K))
    print(sorted_by_value({topic_ids[i]: weight for i,weight in enumerate(theta) if weight > 0.05}))

    doc = []
    for _ in range(N):
        z = np.random.multinomial(1, theta).argmax()
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

    t = time.time()
    doc_topics = infer_topics_gibbs(doc, topics, alpha=alpha)
    print(time.time() - t)
    print(sorted_by_value({k:v for k,v in doc_topics.items() if v > 0.01}))

    t = time.time()
    doc_topics = infer_topics_c(doc, topics)
    print(time.time() - t)
    print(sorted_by_value({k:v for k,v in doc_topics.items() if v > 0.01}))

    # doc_topics = infer_topics_collapsed_gibbs(doc, topics, alpha=alpha)
    # print(sorted_by_value({k:v for k,v in doc_topics.items() if v > 0.01}))

def parse_yahoo(f):
    topics = {}
    for line in f:
        # topic, words = line.strip().split('\t')
        topic, words = line.strip().split(':', 1)
        # keys have commas and colons so we can't do nice split comprehensions :'(
        topics[topic] = {word: float(weight)
                         for word,weight in re.findall(r'\((.*?),([\d\.E\-]*?)\)', words.strip('{}'))}
    return topics

def parse_mahout(f):
    topics = {}
    for line in f:
        topic, words = line.strip().split('\t')
        # keys have commas and colons so we can't do nice split comprehensions :'(
        topics[topic] = {word: float(weight)
                         for word,weight,_ in re.findall(r'(.*?):([\d\.E\-]*?)(,|$)', words.strip('{}'))}
    return topics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topics')
    parser.add_argument('infile', nargs='?')
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    with codecs.open(args.topics, encoding='utf8') as f:
        if args.topics.endswith('.tsv'):
            sys.stderr.write('parsing Mahout topics\n')
            topics = parse_mahout(f)
        else:
            sys.stderr.write('parsing Yahoo topics\n')
            topics = parse_yahoo(f)

    args.infile = codecs.open(args.infile, encoding='utf8') if args.infile else sys.stdin

    t = time.time()
    for i,line in enumerate(args.infile):
        if i > 0: sys.stderr.write('\r%d [%dms]' % (i, int(1000 * (time.time() - t) / i))); sys.stderr.flush()
        id,doc = line.strip().split('\t', 1)
        doc = doc.lower().split()

        doc_topics = infer_topics_c(doc, topics)
        if doc_topics:
            args.outfile.write('%s\t%s\n' % (id, repr({k: v for k,v in doc_topics.items() if v >= 0.01})))


if __name__ == '__main__':
    main()
