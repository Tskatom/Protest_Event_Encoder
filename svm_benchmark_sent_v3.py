#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from sklearn import svm
import timeit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import argparse

def svm_experiment(train_set, test_set): 
    train_set_x, train_set_y = train_set 
    test_set_x, test_set_y = test_set
    
    model = svm.LinearSVC()
    model.fit(train_set_x, train_set_y)

    test_pred = model.predict(test_set_x)
    test_score = 1 - np.mean(np.not_equal(test_set_y, test_pred))
    return test_score


def svm_tfidf(prefix, sufix, dic_fn):
    """
    prefix example: ./data/single_label_sen/sen_spanish_protest
    sufix example: pop_cat
    """

    train_file = prefix + "_train.txt.tok"
    test_file = prefix + "_test.txt.tok"

    train_y_file = prefix + "_train." + sufix
    test_y_file = prefix + "_test." + sufix
    
    dic_cn = {k.strip(): i for i, k in enumerate(open(dic_fn))}


    word_train_set = [l.strip().lower() for l in open(train_file)]
    word_test_set = [l.strip().lower() for l in open(test_file)]

    train_y = [dic_cn[l.strip()] for l in open(train_y_file)]
    test_y = [dic_cn[l.strip()] for l in open(test_y_file)]

    # construct the word count matrix
    count_vect = CountVectorizer()
    train_set_count = count_vect.fit_transform(word_train_set)
    test_set_count = count_vect.transform(word_test_set)

    # construct tfidf matrix
    tfidf_transformer = TfidfTransformer()
    train_set_x = tfidf_transformer.fit_transform(train_set_count)
    test_set_x = tfidf_transformer.transform(test_set_count)

    print "start the model"
    test_score = svm_experiment([train_set_x, train_y], [test_set_x, test_y])
    return test_score

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="single_label")
    return ap.parse_args()

def main():
    
    sent_scores = []

    args = parse_args()
    exp = args.exp

    for i in range(5):
        prefix = "./data/%s/%d/Imdb_review" % (exp, i)
        sufix = "sent_cat"
        dic_fn = "./data/sent_cat.dic"

        start = timeit.default_timer()
        sent_score = svm_tfidf(prefix, sufix, dic_fn)
        sent_scores.append(sent_score)
        end = timeit.default_timer()
        print "Fold %d: Using time %f m get the sent performance %f" % (i, (end - start)/60., sent_score)
    
    print "Average Sent score %f " % (np.mean(sent_scores))

if __name__ == "__main__":
    main()
