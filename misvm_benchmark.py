#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import misvm
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import precision_recall_fscore_support


def misvm_run(data_folder):
    train_txt_file = os.path.join(data_folder, "event_train.txt.tok")
    test_txt_file = os.path.join(data_folder, "event_test.txt.tok")

    train_label_file = os.path.join(data_folder, "event_train.event_cat")
    test_label_file = os.path.join(data_folder, "event_test.event_cat")

    label2id = {"protest":1, "non_protest":-1}

    sentences = []
    with open(train_txt_file) as ttf:
        for line in ttf:
            doc = json.loads(line)
            for sen in doc:
                sentences.append(sen.lower())

    vect = TfidfVectorizer()
    vect.fit(sentences)

    # generate bags features
    train_bags = []
    with open(train_txt_file) as ttf:
        for line in ttf:
            doc = [sen.lower() for sen in json.loads(line)]
            train_bags.append(vect.transform(doc))
    test_bags = []
    with open(test_txt_file) as ttf:
        for line in ttf:
            doc = [sen.lower() for sen in json.loads(line)]
            test_bags.append(vect.transform(doc))

    train_labels = []
    with open(train_label_file) as tlf:
        train_labels = [label2id[l.strip()] for l in tlf]
    test_labels = []
    with open(test_label_file) as tlf:
        test_labels = [label2id[l.strip()] for l in tlf]

    svm = misvm.MISVM()
    svm.fit(train_bags, train_labels)

    preds = svm.predict(test_bags)
    precision, recall, beta, support = precision_recall_fscore_support(test_lables, preds, pos_label=1)

    print folder, '----', precision, recall, beta, support

if __name__ == "__main__":
    pass

