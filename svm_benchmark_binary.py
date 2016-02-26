#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Construct the Two Step SVM classifier to extract Event Population and Type
1. Generate the TFIDF feature for train/test data
2. construct event detection classifier
3. construct event attribute extraction classifier
4. evaluate the event detection classifier 
5. evaluate the overall 
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from sklearn import svm
import timeit
import argparse
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
import json
from sklearn.metrics import classification_report
import numpy as np
import cPickle

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', type=str, help="experiment name")
    return ap.parse_args()


def run(folder):
    train_protest_file = os.path.join(folder, "spanish_protest_train.txt.tok")
    train_non_protest_file = os.path.join(folder, "spanish_nonprotest_train.txt.tok")

    test_protest_file = os.path.join(folder, "spanish_protest_test.txt.tok")
    test_non_protest_file = os.path.join(folder, "spanish_nonprotest_test.txt.tok")

    train_protest_doc = [' '.join(json.loads(doc)) for doc in open(train_protest_file)]
    train_non_protest_doc = [' '.join(json.loads(doc)) for doc in open(train_non_protest_file)]
    
    test_protest_doc = [' '.join(json.loads(doc)) for doc in open(test_protest_file)]
    test_non_protest_doc = [' '.join(json.loads(doc)) for doc in open(test_non_protest_file)]

    tfidf_vec = TfidfVectorizer()
    train_event_x = tfidf_vec.fit_transform(train_protest_doc + train_non_protest_doc)
    train_event_y = [1] * len(train_protest_doc) + [0] * len(train_non_protest_doc)

    test_event_x = tfidf_vec.transform(test_protest_doc + test_non_protest_doc)
    test_event_y = [1] * len(test_protest_doc) + [0] * len(test_non_protest_doc)

    test_protest_x = tfidf_vec.transform(test_protest_doc)
    test_non_protest_x = tfidf_vec.transform(test_non_protest_doc)

    # construct event detection classifier
    event_classifier = svm.LinearSVC()
    event_classifier.fit(train_event_x, train_event_y)

    test_event_pred = event_classifier.predict(test_event_x)
    precision, recall, beta, support = precision_recall_fscore_support(test_event_y, test_event_pred, pos_label=1)
    print folder, '----', precision, recall, beta, support



def svm_experiment():
    args = parse_args()
    exp = args.prefix
    for i in range(5):
        folder_name = os.path.join(exp, "%d" % i)
        run(folder_name)

if __name__ == "__main__":
    svm_experiment()

