#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import random

def split_dataset(dataset, ratio=[.8,.1,.1], rand=False):
    """split the input dataset into train, valid, test
    
    >>> split_dataset([range(10)])
    [[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]]]
    """

    data_size = len(dataset[0])
    if rand:
        random.seed(1234)
        idx = range(data_size)
        random.shuffle(idx)
    else:
        idx = range(data_size)
   
    train_end = int(ratio[0] * data_size)
    test_end = int(sum(ratio[:2]) * data_size)
    train_ids = idx[:train_end]
    valid_ids = idx[train_end:test_end]
    test_ids = idx[test_end:]

    new_dataset = []
    for d in dataset:
        train = [d[i] for i in train_ids]
        valid = [d[i] for i in valid_ids]
        test = [d[i] for i in test_ids]

        new_dataset.append([train, valid, test])
    return new_dataset


def split_text_set():
    es_file = "../data/spanish_protest.txt.tok"
    en_file = "../data/english_protest.txt.tok"
    pop_label_file = "../data/pop_label.txt"
    eventType_file = "../data/eventType_label.txt"

    es = [l for l in open(es_file)]
    en = [l for l in open(en_file)]
    pops = [l for l in open(pop_label_file)]
    types = [l for l in open(eventType_file)]

    dataset = split_dataset([es, en, pops, types])

    train_es, valid_es, test_es = dataset[0]
    train_en, valid_en, test_en = dataset[1]
    train_pop, valid_pop, test_pop = dataset[2]
    train_type, valid_tupe, test_type = dataset[3]

    names = ["spanish_protest_%s.txt.tok", "english_protest_%s.txt.tok", "pop_%s_label.txt", "type_%s_label.txt"]
    phases = ["train", "valid", "test"]
    for i in range(len(names)):
        data = dataset[i]
        for j in range(len(phases)):
            print names[i], phases[j]
            file_name = os.path.join("../data/", names[i] % phases[j])
            with open(file_name, 'w') as df:
                for line in data[j]:
                    df.write(line)
            

if __name__ == "__main__":
    split_text_set()
