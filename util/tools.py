#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import random
import argparse
import numpy as np
import cPickle
import re

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, help="task name")
    ap.add_argument("--vocab_fn", type=str, help="vocab file")
    ap.add_argument("--vec_random_fn", type=str, help="output random vocab embedding file")
    ap.add_argument("--vec_trained_fn", type=str, help="output pretrained vocab embedding file")
    ap.add_argument("--pretrained_fn", type=str, help="pretrained word2vec")
    ap.add_argument("--emb_dm", type=int, help="the dimention of the word embedding")
    ap.add_argument("--es_file", type=str, help="spanish article file",
            default="../data/single_label/spanish_protest.txt.tok")
    ap.add_argument("--en_file", type=str, help="english article file",
            default="../data/single_label/english_protest.txt.tok")
    ap.add_argument("--pop_label_file", type=str, help="population label file",
            default="../data/single_label/pop_label.txt")
    ap.add_argument("--eventType_file", type=str, help="event type label file",
            default="../data/single_label/eventType_label.txt")
    ap.add_argument("--outfolder", type=str, help="output folder")
    ap.add_argument("--keywords_file", type=str, help="keywords file")
    return ap.parse_args()

def filter_sen(es_file, key_file, outfolder):
    words = [w.strip() for w in open(key_file)]
    rule = '|'.join(words)
    pattern = re.compile(rule, re.I)
    
    basename = "sen_" + os.path.basename(es_file)
    outfile = os.path.join(outfolder, basename)

    with open(es_file) as esf, open(outfile, 'w') as otf:
        not_found = 0
        for line in esf:
            sens = line.split(".")
            matched_sens = ''
            for sen in sens:
                if pattern.search(sen):
                    matched_sens += sen.strip() + " . "

            if len(matched_sens) == 0:
                not_found += 1
                print '---------- Not Found', not_found
                matched_sens = sens[0] + " . "
            matched_sens = matched_sens.strip()
            otf.write(matched_sens + "\n")


def generate_vocab_embedding(vocab_fn, pretrained_fn, vec_fn, emb_dm=100, seed=1234):
    # generate pretrained vocab embedding
    vocab = [l.encode('utf-8').strip().split('\t')[0] if isinstance(l, unicode) else l.strip().split("\t")[0] 
            for l in open(vocab_fn)]
    np.random.RandomState(seed)
    embedding = []
    embedding.append(np.zeros(emb_dm)) # for paddings
    embedding.append(np.zeros(emb_dm)) # for unknow words
    
    # load pretrained word embdding
    pre_embedding = {}
    if pretrained_fn is not None:
        with open(pretrained_fn) as pfn:
            for line in pfn:
                info = line.strip().split()
                word = info[0]
                vec = np.asarray(info[1:], dtype=np.float32)
                pre_embedding[word] = vec
    else:
        print "No pretrained vector provided... Will Generate vector randomly"
    
    # fit vocab
    word2id = {'<P>': 0, 'UNK': 1}
    id = 2
    matched = 0
    for w in vocab:
        if w not in pre_embedding:
            vec = np.random.uniform(-1, 1, emb_dm)
        else:
            vec = pre_embedding[w]
            matched += 1
        embedding.append(vec)
        word2id[w] = id
        id += 1
    
    print "[%d] of [%d] words appeared in [%d] pretrained words" % (matched, len(vocab), len(pre_embedding))
    # dump the word embedding
    embedding = np.asarray(embedding, dtype=np.float32)
    with open(vec_fn, 'wb') as vtf:
        cPickle.dump(embedding, vtf)
        cPickle.dump(word2id, vtf)


def split_dataset(dataset, ratio=[.7,.1,.2], rand=False):
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


def split_text_set(es_file, en_file, pop_label_file, eventType_file):
    
    es = [l for l in open(es_file)]
    en = [l for l in open(en_file)]
    pops = [l for l in open(pop_label_file)]
    types = [l for l in open(eventType_file)]

    folder = os.path.dirname(es_file)

    dataset = split_dataset([es, en, pops, types])

    train_es, valid_es, test_es = dataset[0]
    train_en, valid_en, test_en = dataset[1]
    train_pop, valid_pop, test_pop = dataset[2]
    train_type, valid_tupe, test_type = dataset[3]

    names = ["spanish_protest_%s.txt.tok", "english_protest_%s.txt.tok", "spanish_protest_%s.pop_cat", "spanish_protest_%s.type_cat"]
    phases = ["train", "valid", "test"]
    for i in range(len(names)):
        data = dataset[i]
        for j in range(len(phases)):
            file_name = os.path.join(folder, names[i] % phases[j])
            with open(file_name, 'w') as df:
                for line in data[j]:
                    df.write(line)
            

def main():
    args = parse_args()
    task = args.task
    seed = 1234
    if task == "gen_emb":
        emb_dm = args.emb_dm
        vocab_fn = args.vocab_fn
        if args.vec_trained_fn and args.pretrained_fn:
            # using pretrained vector
            pre_fn = args.pretrained_fn
            vec_fn = args.vec_trained_fn
            generate_vocab_embedding(vocab_fn, pre_fn, vec_fn, emb_dm, seed)
        
        if args.vec_random_fn:
            # using random vector
            vec_fn = args.vec_random_fn
            generate_vocab_embedding(vocab_fn, None, vec_fn, emb_dm, seed)
    elif task == "split_data":
        es_file = args.es_file
        en_file = args.en_file
        pop_label_file = args.pop_label_file
        eventType_file = args.eventType_file
        split_text_set(es_file, en_file, pop_label_file, eventType_file)
    elif task == "filter_sen":
        es_file = args.es_file
        keyword_file = args.keywords_file
        outfolder = args.outfolder
        filter_sen(es_file, keyword_file, outfolder)


if __name__ == "__main__":
    main()
