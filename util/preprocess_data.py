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
from functools import partial
import codecs
from multiprocessing import Pool
import re

def generate_rupen_important_docs(gsr_file, clean_str=True, key_file='../data/protest_keywords.txt'):
    docs = []
    vocab = defaultdict(float)
    type2id = {}
    pop2id = {}

    tid = 0
    pid = 0
    keywords_rule = generate_rule(key_file)

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
                        # extract the important sentence
                        key_sens = keyword_filter(value, "original_text_basis_enrichment", keywords_rule)
                        if len(key_sens) == 0:
                            continue
                        else:
                            key_content = u""
                            for k_sen in key_sens:
                                key_content += u" ".join(k_sen)
                            articles.append(key_content.encode("utf-8"))
            # we construct each event for each individual articles
            for article in articles:
                doc = {}
                eventType = event["eventType"][:3]
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

                tokens = article.split()
                words = set(tokens)
                for w in words:
                    vocab[w] += 1
                    
                doc["tokens"] = tokens
                doc["length"] = len(tokens)
                doc["cv"] = np.random.randint(0, 10)
                docs.append(doc)
    return docs, vocab, type2id, pop2id

def generate_rupen_docs(gsr_file, clean_str=True):
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
                    content = u' '.join([t['value'].lower() for t in tokens])
                    dup = False
                    for article in articles:
                        if content[:100] == article[:100]:
                            dup = True
                    if not dup:
                        articles.append(content)
            # we construct each event for each individual articles
            for article in articles:
                doc = {}
                eventType = event["eventType"][:3]
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

                tokens = article.split()
                words = set(tokens)
                for w in words:
                    vocab[w] += 1
                    
                doc["tokens"] = tokens
                doc["length"] = len(tokens)
                doc["cv"] = np.random.randint(0, 10)
                docs.append(doc)
    return docs, vocab, type2id, pop2id

def extract_sentence(tokens):
    """
    extract sentences from BasisEnrichment tokens
    """
    sens = []
    sen = []
    for token in tokens:
        sen.append(token['value'].lower())
        if token['POS'] == 'SENT':
            sens.append(sen)
            sen = []
    return sens 

def keyword_match(sen, key_rule):
    """
    sen: list of word
    keywords: list of keyword
    return sen if match anykeywords else return None
    """
    sen_str = u' '.join(sen)
    pattern = re.compile(key_rule, re.U)
    matched = pattern.findall(sen_str)
    if len(matched) > 0:
        return sen
    else:
        return None

def generate_rule(rule_file):
    with codecs.open(rule_file, encoding='utf-8') as rf:
        words = [l.strip() for l in rf]
    rule = u"|".join(words)
    return rule

def keyword_filter(article, enrichment_key="BasisEnrichment", key_rule=None):
    """filter the sentences containing the protest keywords"""
    tokens = article[enrichment_key]["tokens"]
    sens = extract_sentence(tokens)
    partial_keyword_match = partial(keyword_match, key_rule=key_rule)
    if len(sens) == 0:
        return []
    pool = Pool(processes=len(sens))
    result = pool.map(partial_keyword_match, sens)
    pool.close()
    pool.join()
    sentences = [s for s in result if s]
    return sentences


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

def nstr(s, lower=True):
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

def load_rssworkd2vec(rss_word2vec_file):
    rss_file = open(rss_word2vec_file)
    emb_dict = {l.split()[0]:np.asarray(l.strip().split()[1:],dtype=float) for l in rss_file}
    vocab = emb_dict.keys()
    embedding = np.asarray([emb_dict[v] for v in vocab])
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
    from collections import Counter
    gsr_file = "../data/gsr_spanish.txt"
    wiki_file = "../data/polyglot-es.pkl"
    rss_file = "../../glove_data/100d_vectors.txt"
    emb_dm = 100
    #docs, vocab, type2id, pip2id = generate_docs(gsr_file, clean_str=False)
    rupen_gsr_file = "../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json"
    docs, vocab, type2id, pip2id = generate_rupen_docs(rupen_gsr_file, clean_str=False)
    #docs, vocab, type2id, pip2id = generate_rupen_important_docs(rupen_gsr_file)
    print "Total Articles %d " % (len(docs))
    #wiki_vocab, wiki_embedding = load_wikiword2vec(wiki_file)
    rss_vocab, rss_embedding = load_rssworkd2vec(rss_file) 
    #word2vec = get_word2vec(wiki_vocab, wiki_embedding, vocab)
    word2vec = get_word2vec(rss_vocab, rss_embedding, vocab)
    print "Total %d words in wiki_vocab, %d words in vocab and %d word from vocab in wiki_vocab" % (len(rss_vocab), len(vocab), len(word2vec))
    
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
    add_unknowwords(word2vec, vocab, k=emb_dm)
    embedding, word2id = get_embedding(word2vec,k=emb_dm)

    # randomlize the word vector
    randvec = {}
    add_unknowwords(randvec, vocab, k=emb_dm)
    rand_embedding, _ = get_embedding(randvec, k=emb_dm)
    # dump the data
    data = [docs, type2id, pip2id, word2id, embedding, rand_embedding]

    with open("../data/full_doc_dataset_3", "wb") as ed:
        cPickle.dump(data, ed)
