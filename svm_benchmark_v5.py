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
    train_pop_file = os.path.join(folder, "spanish_protest_train.pop_cat")
    train_type_file = os.path.join(folder, "spanish_protest_train.type_cat")

    test_protest_file = os.path.join(folder, "spanish_protest_test.txt.tok")
    test_non_protest_file = os.path.join(folder, "spanish_nonprotest_test.txt.tok")
    test_pop_file = os.path.join(folder, "spanish_protest_test.pop_cat")
    test_type_file = os.path.join(folder, "spanish_protest_test.type_cat")

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
    # construct the population classifier
    pop_cls = svm.LinearSVC()
    pop2id = {l.strip():i for i, l in enumerate(open('./data/pop_cat.dic'))}
    type2id = {l.strip():i for i, l in enumerate(open('./data/type_cat.dic'))}
    train_pop_y = [pop2id[l.strip()] for l in open(train_pop_file)]
    pop_cls.fit(tfidf_vec.transform(train_protest_doc), train_pop_y)
    test_pop_y = [pop2id[l.strip()] for l in open(test_pop_file)] + [11] * len(test_non_protest_doc)
    
    # construct the event type classifier
    train_type_y = [type2id[l.strip()] for l in open(train_type_file)]
    type_cls = svm.LinearSVC()
    type_cls.fit(tfidf_vec.transform(train_protest_doc), train_type_y)
    test_type_y = [type2id[l.strip()] for l in open(test_type_file)] + [5] * len(test_non_protest_doc)  

    # filter out the prediction with events
    pred_event_ids = test_event_pred == 1
    test_pop_preds = [11] * np.ones(len(test_type_y))
    test_pop_x = test_event_x[pred_event_ids]
    test_pop_preds[pred_event_ids] = pop_cls.predict(test_pop_x)

    test_type_preds = [5] * np.ones(len(test_type_y))
    test_type_preds[pred_event_ids] = type_cls.predict(test_pop_x)

    pop_class_names = [l.strip() for l in open('./data/pop_cat.dic')] + ['None']
    type_class_names = [l.strip() for l in open('./data/type_cat.dic')] + ['None']


    print classification_report(test_pop_y, test_pop_preds, target_names=pop_class_names)
    """
    f = open('./tt', 'w')
    cPickle.dump(test_type_preds, f)
    cPickle.dump(test_type_y, f)
    f.close()
    """
    print classification_report(test_type_y, test_type_preds, target_names=type_class_names)



def svm_experiment():
    args = parse_args()
    exp = args.prefix
    for i in range(5):
        folder_name = os.path.join(exp, "%d" % i)
        run(folder_name)

if __name__ == "__main__":
    svm_experiment()

