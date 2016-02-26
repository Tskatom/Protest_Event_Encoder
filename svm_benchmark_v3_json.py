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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json

def svm_experiment(train_set, test_set, class_names): 
    train_set_x, train_set_y = train_set 
    test_set_x, test_set_y = test_set
    
    model = svm.LinearSVC()
    model.fit(train_set_x, train_set_y)

    test_pred = model.predict(test_set_x)
    test_score = 1 - np.mean(np.not_equal(test_set_y, test_pred))

    #print confusion_matrix(test_set_y, test_pred)
    print classification_report(test_set_y, test_pred, target_names=class_names)

    return test_score, test_pred, test_set_y


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
    id2class = {v:k for k,v in dic_cn.items()}
    class_names = [k.strip() for k in open(dic_fn)]
    print class_names


    word_train_set = [' '.join(json.loads(l)).lower() for l in open(train_file)]
    word_test_set = [' '.join(json.loads(l)).lower() for l in open(test_file)]

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
    test_score, test_pred, test_true = svm_experiment([train_set_x, train_y], [test_set_x, test_y], class_names)
    
    outfile = open(os.path.join(os.path.dirname(train_y_file), "%s_error.json" % sufix), 'w')
    for i in range(len(test_pred)):
        if test_pred[i] != test_true[i]:
            pred_label = id2class[test_pred[i]]
            true_label = id2class[test_true[i]]
            text = word_test_set[i]
            data = {"true": true_label, "pred": pred_label, "text": text}
            dump = json.dumps(data, ensure_ascii=False, encoding='utf-8')
            if isinstance(dump, unicode):
                dump = dump.encode('utf-8')
            outfile.write(dump + "\n")
    outfile.flush()
    outfile.close()


    return test_score

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="single_label")
    return ap.parse_args()

def main():
    
    pop_scores = []
    type_scores = []

    args = parse_args()
    exp = args.exp

    for i in range(5):
        #prefix = "./data/%s/%d/spanish_protest" % (exp, i)
        prefix = "./data/%s/%d/spanish_protest" % (exp, i)
        sufix = "pop_cat"
        dic_fn = "./data/pop_cat.dic"

        start = timeit.default_timer()
        pop_score = svm_tfidf(prefix, sufix, dic_fn)
        pop_scores.append(pop_score)
        end = timeit.default_timer()
        print "Fold %d: Using time %f m get the pop performance %f" % (i, (end - start)/60., pop_score)

        start = timeit.default_timer()
        sufix = "type_cat"
        dic_fn = "./data/type_cat.dic"
        type_score = svm_tfidf(prefix, sufix, dic_fn)
        type_scores.append(type_score)
        end = timeit.default_timer()
        print "Fold %d: Using time %f m get the type performance %f" % (i, (end - start)/60., type_score)
    
    print "Average Pop score %f, Type Score %f " % (np.mean(pop_scores), np.mean(type_scores))

if __name__ == "__main__":
    main()