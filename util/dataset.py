#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import cPickle
from glob import glob
from collections import Counter
import unicodedata
import json
import re
import numpy as np
import random
from sklearn.cross_validation import train_test_split


def merge_docs(data_folder, outfile):
    """ Merge gsr articles into a big file
    :type data_folder: string
    :param data_folder: input data folder, ~/workspace/gsr_article/spanish_gsr

    :type outfile: string
    :param outfile: output file

    The original folder is
    ~/workspace/gsr_article/spanish_gsr
    """
    countries = os.listdir(data_folder)
    with open(outfile, 'w') as otf:
        for country in countries:
            folder = os.path.join(data_folder, 
                    "%s/eventType" % country)
            country_files = glob(os.path.join(folder, "*"))
            for f in country_files:
                with open(f) as event_file:
                    for line in event_file:
                        otf.write(line)

def nstr(s, lower=True):
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

def generate_docs(gsr_file):
    """ Generate doc and gsr list from gsr articles
    divide the whole set into train/valid/test set
    :type gsr_file: string
    :param gsr_file: path to gsr file
    """
    docs = []
    gsrs = []
    with open(gsr_file) as gsf:
        for line in gsf:
            event = json.loads(line)
            eventType = event["eventType"]
            population = event["population"]
            eventDate = event["eventDate"][:10]
            location = event["location"]
            
            tokens = event["BasisEnrichment"]['tokens']
            doc = []
            sen = []
            sen_offsets = []
            pre_idx = 0
            for idx, token in enumerate(tokens):
                if token['POS'] == 'SENT':
                    offset = "%d:%d" % (pre_idx, idx+1)
                    sen.append(token["value"].lower())
                    doc.append(sen)
                    sen_offsets.append(offset)
                    sen = []
                    pre_idx = idx
                else:
                    sen.append(token["value"].lower())
            if len(sen) > 0:
                doc.append(sen)
                offset = "%d:%d" % (pre_idx, idx+1)
                sen_offsets.append(offset)

            # construct the location matching sentence
            location_entities = [entity for entity in event["BasisEnrichment"]["entities"] if entity["neType"] == "LOCATION"]
            country, state, city = map(nstr, location)

            if city != '-': # city level warning
                match_target = city
            elif state != '-': # state level warning
                match_target = state
            else:
                match_target = country
            
            sen_loc_match = {}
            for loc_ent in location_entities:
                offset = loc_ent["offset"]
                loc_start, loc_end = map(int, offset.split(":"))
                loc_sen_id = -1
                # searching sentence id
                for sid, s_offset in enumerate(sen_offsets):
                    s_start, s_end = map(int, s_offset.split(":"))
                    if loc_start >= s_start and loc_end <= s_end:
                        loc_sen_id = sid

                # check matching
                loc_ent_str = nstr(' '.join([nstr(t['value']) for t in tokens[loc_start:loc_end]]))

                matched = False
                if re.match(match_target, loc_ent_str):
                    matched = True

                if matched:
                    sen_loc_match[loc_sen_id] = matched
                else:
                    if loc_sen_id not in sen_loc_match:
                        sen_loc_match[loc_sen_id] = matched
            gsr = {}
            gsr["eventType"] = eventType
            gsr["population"] = population
            gsr['eventDate'] = eventDate
            gsr["location"] = location
            gsr["loc_sen_labs"] = sen_loc_match
            gsrs.append(gsr)
            docs.append(doc)
    return docs, gsrs

def dump_data_2_pickle(gsr_file, pickleFile):
    """
    dump the txt gsr file data into picke
    :type gsr_file: string
    :param gsr_file: path to gsr file, default: gsr_article/gsr_spanish.txt

    :type pickleFile: string
    :param pickleFile: path to pickle file, default: ../data/dataset.pkl
    """
    # generate docs and gsrs
    docs, gsrs = generate_docs(gsr_file)
    # shuffle the data
    dataset = zip(docs, gsrs)
    train_set, test_set = train_test_split(dataset, 
            test_size=0.3, 
            random_state=10)
    valid_set, test_set = train_test_split(test_set, 
            test_size=0.5, 
            random_state=11)

    # construct the vocab list and transfer the data into word num
    word2id = {}
    # set UNKNOW word as UUKK
    word2id["UUKK"] = -1
    pop2id = {}
    type2id = {}

    wid = 0
    pid = 0
    tid = 0
    for doc, gsr in train_set:
        for sen in doc:
            for token in sen:
                if token not in word2id:
                    word2id[token] = wid
                    wid += 1
        pop = gsr["population"]
        eType = gsr["eventType"]
        if pop not in pop2id:
            pop2id[pop] = pid
            pid += 1

        if eType not in type2id:
            type2id[eType] = tid
            tid += 1

    train_set = transform_set(train_set, word2id, pop2id, type2id)
    valid_set = transform_set(valid_set, word2id, pop2id, type2id)
    test_set = transform_set(test_set, word2id, pop2id, type2id)

    with open(pickleFile, 'w') as pf:
        cPickle.dump(train_set, pf)
        cPickle.dump(valid_set, pf)
        cPickle.dump(test_set, pf)
        cPickle.dump(word2id, pf)
        cPickle.dump(pop2id, pf)
        cPickle.dump(type2id, pf)


def transform_set(dataset, word2id, pop2id, type2id):
    """transform the tokens into id
    :type dataset: list of list
    :param dataset: dataset consits of docs and gsrs

    :type word2id: dict
    :param word2id: dictionary of vocab

    :type pop2id: dict
    :param pop2id: dictionary of population

    :type type2id: dict
    :param tyep2id: dictionary of eventType
    """
    
    if len(dataset[0]) == 1: # only transform tokens
        new_set = []
        for doc in dataset:
            new_doc = []
            for sen in doc:
                new_sen = []
                for token in sen:
                    wid = word2id.get(token, '-1')
                    new_sen.append(wid)
                new_doc.append(new_sen)
            new_set.append(doc)
    elif len(dataset[0]) == 2:
        new_set = []
        for doc, gsr in dataset:
            new_doc = []
            new_gsr = {}
            for sen in doc:
                new_sen = []
                for token in sen:
                    new_sen.append(word2id.get(token, '-1'))
                new_doc.append(new_sen)
            
            new_gsr["population"] = pop2id[gsr["population"]]
            new_gsr['eventType'] = type2id[gsr["eventType"]]
            new_gsr["loc_sen_labs"] = gsr["loc_sen_labs"]
            new_set.append([new_doc, new_gsr])

    return new_set


def load_data(pklfile):
    """ load the experiment data from pickle
    :type pklfile: string
    :param pklfile: path to the experiment pkl file, default: data/dataset.pkl
    """
    f = open(pklfile)
    train_set = cPickle.load(f)
    valid_set = cPickle.load(f)
    test_set = cPickle.load(f)
    word2id = cPickle.load(f)
    pop2id = cPickle.load(f)
    type2id = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set, word2id, pop2id, type2id

def summary_dataset(pklfile):
    train_set, valid_set, test_set, word2id, pop2id, type2id = load_data(pklfile)
    print "Number Training Set: ", len(train_set)
    print "Number Valid Set: ", len(valid_set)
    print "Number Test Set: ", len(test_set)
    print "---------------------"
    print "Number of vocab: ", len(word2id)
    sen_nums = [len(doc) for doc, gsr in train_set]
    ave_sen_per_doc = int(np.mean(sen_nums))
    print "Average num of sentence per doc: %d" % ave_sen_per_doc
    sen_lens = []
    for doc, _ in train_set:
        for sen in doc:
            sen_lens.append(len(sen))
    ave_sen_len = int(np.mean(sen_lens))
    print "Average len of each sentence: %d " % ave_sen_len

if __name__ == "__main__":
    pass

