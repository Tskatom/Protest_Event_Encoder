#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from util import dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import svm

def sen_dig2word(doc, id2word):
    new_doc = []
    for sen in doc:
        new_sen = [id2word[w] for w in sen if w!=-2]
        new_doc.append(' '.join(new_sen))
    return ' '.join(new_doc)

def bayes_bench():
    data_file = "./data/dataset.pkl"
    train_set, valid_set, test_set, word2id, pop2id, type2id = dataset.load_data(data_file)

    train_set_x, train_set_y = train_set
    train_set_pop_y, train_set_type_y, train_set_loc_y = train_set_y

    valid_set_x, valid_set_y = valid_set
    valid_set_pop_y, valid_set_type_y, valid_set_loc_y = valid_set_y
    
    test_set_x, test_set_y = test_set
    test_set_pop_y, test_set_type_y, test_set_loc_y = test_set_y
    
    id2word = {v:k for k,v in word2id.items()}
    word_train_set_x = [sen_dig2word(doc, id2word) for doc in train_set_x]
    word_valid_set_x = [sen_dig2word(doc, id2word) for doc in valid_set_x]
    word_test_set_x = [sen_dig2word(doc, id2word) for doc in test_set_x]
    
    # construct the word count matrix
    
    # construct the word count matrix
    count_vect = CountVectorizer()
    x_train_count = count_vect.fit_transform(word_train_set_x)
    x_valid_count = count_vect.transform(word_valid_set_x)
    x_test_count = count_vect.transform(word_test_set_x)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    x_valid_tfidf = tfidf_transformer.transform(x_valid_count)
    x_test_tfidf = tfidf_transformer.transform(x_test_count)

    # train the pop model
    pop_clf = MultinomialNB().fit(x_train_tfidf, train_set_pop_y)
    pop_pred = pop_clf.predict(x_valid_tfidf)
    pop_pred_test = pop_clf.predict(x_test_tfidf)

    # compute the performance
    pop_errors = np.mean(np.not_equal(pop_pred, valid_set_pop_y))
    pop_errors_test = np.mean(np.not_equal(pop_pred_test, test_set_pop_y))

    # train the event type model
    type_clf = MultinomialNB().fit(x_train_tfidf, train_set_type_y)
    type_pred = type_clf.predict(x_valid_tfidf)
    type_pred_test = type_clf.predict(x_test_tfidf)

    # compute the performance
    type_errors = np.mean(np.not_equal(type_pred, valid_set_type_y))
    type_errors_test = np.mean(np.not_equal(type_pred_test, test_set_type_y))

    print "MB--> Type error: %0.2f, Popuation error: %0.2f" % (type_errors, pop_errors)
    print "MB--> Type error: %0.2f, Popuation error: %0.2f" % (type_errors_test, pop_errors_test)


def svm_bench():
    data_file = "./data/dataset.pkl"
    train_set, valid_set, test_set, word2id, pop2id, type2id = dataset.load_data(data_file)

    train_set_x, train_set_y = train_set
    train_set_pop_y, train_set_type_y, train_set_loc_y = train_set_y

    valid_set_x, valid_set_y = valid_set
    valid_set_pop_y, valid_set_type_y, valid_set_loc_y = valid_set_y
    
    test_set_x, test_set_y = test_set
    test_set_pop_y, test_set_type_y, test_set_loc_y = test_set_y
    
    id2word = {v:k for k,v in word2id.items()}
    word_train_set_x = [sen_dig2word(doc, id2word) for doc in train_set_x]
    word_valid_set_x = [sen_dig2word(doc, id2word) for doc in valid_set_x]
    word_test_set_x = [sen_dig2word(doc, id2word) for doc in test_set_x]
    
    # construct the word count matrix
    
    # construct the word count matrix
    count_vect = CountVectorizer()
    x_train_count = count_vect.fit_transform(word_train_set_x)
    x_valid_count = count_vect.transform(word_valid_set_x)
    x_test_count = count_vect.transform(word_test_set_x)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    x_valid_tfidf = tfidf_transformer.transform(x_valid_count)
    x_test_tfidf = tfidf_transformer.transform(x_test_count)

    # train the pop model
    pop_clf = svm.LinearSVC().fit(x_train_tfidf, train_set_pop_y)
    pop_pred = pop_clf.predict(x_valid_tfidf)
    pop_pred_test = pop_clf.predict(x_test_tfidf)

    # compute the performance
    pop_errors = np.mean(np.not_equal(pop_pred, valid_set_pop_y))
    pop_errors_test = np.mean(np.not_equal(pop_pred_test, test_set_pop_y))

    # train the event type model
    type_clf = svm.LinearSVC().fit(x_train_tfidf, train_set_type_y)
    type_pred = type_clf.predict(x_valid_tfidf)
    type_pred_test = type_clf.predict(x_test_tfidf)

    # compute the performance
    type_errors = np.mean(np.not_equal(type_pred, valid_set_type_y))
    type_errors_test = np.mean(np.not_equal(type_pred_test, test_set_type_y))

    print "SVM Valid--> Type error: %0.2f, Popuation error: %0.2f" % (type_errors, pop_errors)
    print "SVM Tes--> Type error: %0.2f, Popuation error: %0.2f" % (type_errors_test, pop_errors_test)

if __name__ == "__main__":
    bayes_bench()
    svm_bench()

