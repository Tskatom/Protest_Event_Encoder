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

def svm_experiment(train_set, valid_set, test_set): 
    train_set_x, train_set_y = train_set 
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    c_range = np.logspace(-2, 1, 4)
    best_score = 0.0
    best_model = None

    for c in c_range:
        model = svm.LinearSVC(C=c)
        model.fit(train_set_x, train_set_y)
        valid_pred = model.predict(valid_set_x) 
        score = 1 - np.mean(np.not_equal(valid_pred, valid_set_y))
        if score > best_score:
            best_score = score
            best_param = {"kernel": 'linear', "C": c}
            best_model = model

    test_pred = best_model.predict(test_set_x)
    test_score = 1 - np.mean(np.not_equal(test_set_y, test_pred))
    print "Test Performance %f under Best valid Score: %f" % (test_score, best_score), " Best Params:", best_param
    return test_score, best_param


def svm_tfidf(prefix, sufix, dic_fn):
    """
    prefix example: ./data/single_label_sen/sen_spanish_protest
    sufix example: pop_cat
    """

    train_file = prefix + "_train.txt.tok"
    valid_file = prefix + "_valid.txt.tok"
    test_file = prefix + "_test.txt.tok"

    train_y_file = prefix + "_train." + sufix
    valid_y_file = prefix + "_valid." + sufix
    test_y_file = prefix + "_test." + sufix
    
    dic_cn = {k.strip(): i for i, k in enumerate(open(dic_fn))}


    word_train_set = [l.strip().lower() for l in open(train_file)]
    word_valid_set = [l.strip().lower() for l in open(valid_file)]
    word_test_set = [l.strip().lower() for l in open(test_file)]

    train_y = [dic_cn[l.strip()] for l in open(train_y_file)]
    valid_y = [dic_cn[l.strip()] for l in open(valid_y_file)]
    test_y = [dic_cn[l.strip()] for l in open(test_y_file)]

    # construct the word count matrix
    count_vect = CountVectorizer()
    train_set_count = count_vect.fit_transform(word_train_set)
    valid_set_count = count_vect.transform(word_valid_set)
    test_set_count = count_vect.transform(word_test_set)

    # construct tfidf matrix
    tfidf_transformer = TfidfTransformer()
    train_set_x = tfidf_transformer.fit_transform(train_set_count)
    valid_set_x = tfidf_transformer.transform(valid_set_count)
    test_set_x = tfidf_transformer.transform(test_set_count)

    print "start the model"
    test_score, best_param = svm_experiment([train_set_x, train_y], [valid_set_x, valid_y], 
            [test_set_x, test_y])
    return test_score, best_param

def main():
    print 'start tfidf sentence experiment using sentence for population'
    prefix = "./data/single_label_sen/sen_spanish_protest"
    sufix = "pop_cat"
    dic_fn = "./data/pop_cat.dic"

    start = timeit.default_timer()
    pop_score, pop_bset_param = svm_tfidf(prefix, sufix, dic_fn)
    end = timeit.default_timer()
    print "Using time %f m get the pop performance %f" % ((end - start)/60., pop_score)
    
    start = timeit.default_timer()
    sufix = "type_cat"
    dic_fn = "./data/type_cat.dic"
    type_score, type_bset_param = svm_tfidf(prefix, sufix, dic_fn)
    end = timeit.default_timer()
    print "Using time %f m get the type performance %f" % ((end - start)/60., type_score)

    print 'start tfidf sentence experiment using full text for population'
    prefix = "./data/single_label/spanish_protest"
    sufix = "pop_cat"
    dic_fn = "./data/pop_cat.dic"

    start = timeit.default_timer()
    pop_score, pop_bset_param = svm_tfidf(prefix, sufix, dic_fn)
    end = timeit.default_timer()
    print "Using time %f m get the pop performance %f" % ((end - start)/60., pop_score)


    start = timeit.default_timer()
    sufix = "type_cat"
    dic_fn = "./data/type_cat.dic"
    type_score, type_bset_param = svm_tfidf(prefix, sufix, dic_fn)
    end = timeit.default_timer()
    print "Using time %f m get the type performance %f" % ((end - start)/60., type_score)

if __name__ == "__main__":
    main()
