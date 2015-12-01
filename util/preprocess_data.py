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


if __name__ == "__main__":
    # compute the summary of the document
    gsr_file = "../data/gsr_spanish.txt"
    wiki_file = "../data/polyglot-es.pkl"
    docs, vocab, type2id, pip2id = generate_docs(gsr_file, clean_str=False)
    wiki_vocab, wiki_embedding = load_wikiword2vec(wiki_file)
    
    max_doc_len = np.max([d["length"] for d in docs])
    min_doc_len = np.min([d["length"] for d in docs])
    
    print "doc_max_len=%d, doc_min_len=%d" % (max_doc_len, min_doc_len)


    
            
            
            
            
            
            
            
    