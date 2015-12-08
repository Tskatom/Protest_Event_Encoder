#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess the document as long sequence of words
"""

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import os
import json
import sys
from collections import defaultdict
import cPickle
import numpy as np
import cPickle

def generate_rupen_docs(gsr_file, clean_str=False):
    docs = []
    vocab = defaultdict(float)
    type2id = {}
    pop2id = {}

    tid = 0
    pid = 0

    with open(gsr_file) as gf:
        for line in gf:
            event = json.loads(line)
            # check the data, remove those data without downloaded articles
            if len(event["downloaded_articles"]) == 0:
                continue
            articles = []
            for url, value in event["downloaded_articles"].items():
                if not isinstance(value, dict):
                    continue
                tokens = value["original_text_basis_enrichment"]["tokens"]
                if len(tokens) > 0:
                    # compare the similarity of current articles with pervious
                    content = u' '.join([t['value'] for t in tokens])
                    dup = False
                    for article in articles:
                        if content[:100] == article[:100]:
                            dup = True
                    if not dup:
                        articles.append(content)
            # we construct each event for each individual articles
            for article in articles:
                doc = {}
                eventType = event["eventType"]
                eventDate = event["eventDate"]
                population = event["population"]
                location = event["location"]
                if eventType not in type2id:
                    type2id[eventType] = tid
                    tid += 1
                if population not in pop2id:
                    pop2id[population] = pid
                    pid += 1
                
                doc["etype"] = type2id[eventType]
                doc["pop"] = pop2id[population]
                doc["location"] = location
                doc["eventDate"] = eventDate
                doc["content"] = article

                if clean_str:
                    content = clean_content(content)
                tokens = content.split()
                words = set(tokens)
                for w in words:
                    vocab[w] += 1
                    
                doc["tokens"] = tokens
                doc["length"] = len(tokens)
                doc["cv"] = np.random.randint(0, 10)
                docs.append(doc)
    return docs, vocab, type2id, pop2id


def generate_docs(gsr_file, clean_str=False):
    # load the docs
    docs = []
    vocab = defaultdict(float)
    type2id = {}
    pop2id = {}
    
    tid = 0
    pid = 0
    with open(gsr_file) as gf:
        for line in gf:
            doc = {}
            event = json.loads(line)
            content = event["content"]
            eventType = event["eventType"]
            eventDate = event["eventDate"]
            population = event["population"]
            location = event["location"]
            
            if eventType not in type2id:
                type2id[eventType] = tid
                tid += 1
            if population not in pop2id:
                pop2id[population] = pid
                pid += 1
            
            doc["etype"] = type2id[eventType]
            doc["pop"] = pop2id[population]
            doc["location"] = location
            doc["eventDate"] = eventDate
            
            if clean_str:
                content = clean_content(content)
            tokens = content.split()
            words = set(tokens)
            for w in words:
                vocab[w] += 1
                
            doc["tokens"] = tokens
            doc["length"] = len(tokens)
            doc["cv"] = np.random.randint(0, 10)
            docs.append(doc)
    return docs, vocab, type2id, pop2id

def nstr(s, lower=False):
    """ normalize the non-english string to ascii string
    :type s: unicode or string
    :param s: string to be transformed
    """
    if isinstance(s, str):
        s = s.decode('utf-8')
    s = unicodedata.normalize('NFKD', s)
    if lower:
        return s.encode('ASCII', 'ignore').strip().lower()
    else:
        return s.encode('ASCII', 'ignore').strip()
    
def clean_content(content):
    """Clean up the content"""
    # normalize the accent to ascii 
    content = nstr(content)
    return content

def load_wikiword2vec(wiki_word2vec):
    wikifile= open(wiki_word2vec)
    wiki = cPickle.load(wikifile)
    wikifile.close()
    vocab = wiki[0]
    embedding = wiki[1]
    return vocab, embedding

def get_word2vec(wiki_cab, wiki_embedding, vocab):
    wiki_cab_dict = {v:i  for i,v in enumerate(wiki_cab)}
    word2vec = {}
    for v in vocab:
        if v in wiki_cab_dict:
            word2vec[v] = wiki_embedding[wiki_cab_dict[v]]
    return word2vec

def add_unknowwords(word2vec, vocab, k=64, min_df=1):
    "Randomize a vec for unknow words"
    for word in vocab:
        if word not in word2vec:
            word2vec[word] = np.random.uniform(-0.25, 0.25, k)

def get_embedding(word2vec, k=64):
    """
    Get the embedding matrix
    """
    word2id = {}
    vocab_size = len(word2vec)
    embedding = np.zeros((vocab_size + 2, k)) # add 0 for paddings
    i = 2
    for word in word2vec:
        embedding[i] = word2vec[word]
        word2id[word] = i
        i += 1
    embedding[0]= np.zeros(k)
    embedding[1] = np.zeros(k) # unknow words
    return embedding, word2id
    

if __name__ == "__main__":
    # compute the summary of the document
    import pandas as pds
    from collections import Counter
    import matplotlib.pyplot as plt
    gsr_file = "../data/gsr_spanish.txt"
    wiki_file = "../data/polyglot-es.pkl"
    #docs, vocab, type2id, pip2id = generate_docs(gsr_file, clean_str=False)
    rupen_gsr_file = "../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json"
    docs, vocab, type2id, pip2id = generate_rupen_docs(rupen_gsr_file, clean_str=False)
    print "Total Articles %d " % (len(docs))
    wiki_vocab, wiki_embedding = load_wikiword2vec(wiki_file)
    
    word2vec = get_word2vec(wiki_vocab, wiki_embedding, vocab)
    print "Total %d words in wiki_vocab, %d words in vocab and %d word from vocab in wiki_vocab" % (len(wiki_vocab), len(vocab), len(word2vec))
    
    max_doc_len = np.max([d["length"] for d in docs])
    min_doc_len = np.min([d["length"] for d in docs])
    print "doc_max_len=%d, doc_min_len=%d" % (max_doc_len, min_doc_len)
    """
    lens = [d["length"] for d in docs]
    lens_count = Counter(lens)
    keys = sorted(lens_count.keys())
    counts = [lens_count[k] for k in keys]
    total = 1.0 * sum(counts)
    s = pds.Series(counts)
    s = s.cumsum()
    
    plt.plot(keys, s/total)
    plt.show()
    """
    # add unknow words
    add_unknowwords(word2vec, vocab)
    embedding, word2id = get_embedding(word2vec)

    # randomlize the word vector
    randvec = {}
    add_unknowwords(randvec, vocab)
    rand_embedding, _ = get_embedding(randvec)
    # dump the data
    data = [docs, type2id, pip2id, word2id, embedding, rand_embedding]

    with open("../data/experiment_dataset2", "wb") as ed:
        cPickle.dump(data, ed)