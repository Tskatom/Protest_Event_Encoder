#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import cPickle
from textblob import TextBlob

def load_ngrams(f='./data/top_ngrams.pkl'):
    f = open(f)
    n3gram = cPickle.load(f)
    n4gram = cPickle.load(f)
    n5gram = cPickle.load(f)
    return n3gram, n4gram, n5gram

def gene_csv(f='./data/top_ngrams.pkl'):
    ngrams = load_ngrams(f)
    names = ["Top_n3", "Top_n4", "Top_n5"]
    for i, ngram in enumerate(ngrams):
        file = open(names[i], 'w')
        num_units = len(ngram)
        head = '\t'.join(["Feature %d" % d for d in range(num_units)])
        file.write(head + "\n")
        for k in range(len(ngram[0])):
            strs = [ngram[d][k] for d in range(num_units)]
            trans_strs = []
            for s in strs:
                try:
                    blob = TextBlob(s)
                    tb = blob.translate(to="en")
                    trans_strs.append(tb.raw)
                except:
                    trans_strs.append(s)
            line = '\t'.join(trans_strs)
            file.write(line.encode('utf-8') + "\n")
        file.flush()
        file.close()
    

if __name__ == "__main__":
    gene_csv()

