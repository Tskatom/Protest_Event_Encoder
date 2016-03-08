#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re
import numpy as np
import json

def generate_k_num(docs, k_port=0.2):
    sent_flags = [len(doc) if len(doc) < 15 else 15 for doc in docs]
    sent_k = np.maximum(1, np.floor(np.asarray(sent_flags) * 0.2)).astype("int32")
    return sent_k

def random_sentences(docs):
    sent_k = generate_k_num(docs)
    out_docs = []
    for i, doc in enumerate(docs):
        doc_size = len(doc)
        k = sent_k[i]
        chosen = np.random.randint(0,doc_size,k)
        out_docs.append([doc[j] for j in chosen])
    return out_docs

def choose_first_sentence(docs):
    sent_k = generate_k_num(docs)
    out_docs = []
    for i, doc in enumerate(docs):
        out_docs.append(doc[:sent_k[i]])
    return out_docs

def protest_key_sentence(docs, keyword_file):
    sent_k = generate_k_num(docs)
    keyword = u'|'.join([w.strip().decode('utf-8') for w in open(keyword_file)]) 
    pattern = re.compile(keyword, re.U)
    out_docs = []
    for i, doc in enumerate(docs):
        matched_sens = []
        for sen in doc:
            matched = pattern.search(sen.lower())
            if matched:
                matched_sens.append(sen)
        out_docs.append(matched_sens[:sent_k[i]])
    return out_docs

def output_folder(out_folder, train_docs, test_docs, train_labels, test_labels):
    out_train_file = os.path.join(out_folder, "event_train.txt.tok")
    out_test_file = os.path.join(out_folder, "event_test.txt.tok")
    out_train_label_file = os.path.join(out_folder, "event_train.event_cat")
    out_test_label_file = os.path.join(out_folder, "event_test.event_cat")

    with open(out_train_file, 'w') as otf, open(out_test_file, 'w') as otef, open(out_train_label_file, 'w') as otlf, open(out_test_label_file, 'w') as otelf:
        for doc in train_docs:
            otf.write(json.dumps(doc, ensure_ascii=False, encoding='utf-8').encode('utf-8') + "\n")
        for doc in test_docs:
            otef.write(json.dumps(doc, ensure_ascii=False, encoding='utf-8').encode('utf-8') + "\n")
        for label in train_labels:
            otlf.write(label.strip() + "\n")
        for label in test_labels:
            otelf.write(label.strip() + "\n")

def run():
    keyword_file = "../data/search_keywords.txt"
    for fold in range(5):
        data_folder = "../data/new_multi_label/%d" % fold
        train_file = os.path.join(data_folder, "event_train.txt.tok")
        test_file = os.path.join(data_folder, "event_test.txt.tok")
        train_label_file = os.path.join(data_folder, "event_train.event_cat")
        test_label_file = os.path.join(data_folder, "event_test.event_cat")

        train_docs = [json.loads(l) for l in open(train_file)]
        test_docs = [json.loads(l) for l in open(test_file)]
        train_labels =[l.strip() for l in open(train_label_file)]
        test_labels = [l.strip() for l in open(test_label_file)]

        # construct random set
        random_train_docs = random_sentences(train_docs)
        random_test_docs = random_sentences(test_docs)
        out_folder = "../data/new_multi_label_random/%d" % fold
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        output_folder(out_folder, random_train_docs, random_test_docs, train_labels, test_labels)

        # construct first k set
        first_train_docs = choose_first_sentence(train_docs)
        first_test_docs = choose_first_sentence(test_docs)
        out_folder = "../data/new_multi_label_first/%d" % fold
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        output_folder(out_folder, first_train_docs, first_test_docs, train_labels, test_labels)

        # construct keywords
        key_train_docs = protest_key_sentence(train_docs, keyword_file)
        key_test_docs = protest_key_sentence(test_docs, keyword_file)
        out_folder = "../data/new_multi_label_protest/%d" % fold
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        output_folder(out_folder, key_train_docs, key_test_docs, train_labels, test_labels)


if __name__ == "__main__":
    run()
