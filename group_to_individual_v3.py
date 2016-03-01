#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
    We change the P_i to be a sum of P_ij over j.
    In this case, a bag is decides as positive when most of its instances are positive.
"""

__author__ = "Yue Ning"
__email__ = "yning@vt.edu"
__version__ = "0.0.1"
import ast
import sklearn
import sklearn.svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import ClassifierMixin
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats
from copy import deepcopy
import numpy as np
from numpy import linalg as la
import pdb
from scipy.sparse import issparse, vstack as parse_vstack
import random
import math
import json

def sigmoid(x):
  return 1. / (1. + math.exp(-x))

def kernel(xi, xj):
    diff = np.array(xi) - np.array(xj)
    n2sum = la.norm(diff)
    return math.exp(-n2sum)

def cost_func(X, Y, w):
    beta = 1.5
    K = len(X)
    N = 0
    fisrt_sum = 0. 
    allX = []
    for k in xrange(K):
        N += len(X[k])
        allX.extend(X[k])
        estimate_yk = np.mean([sigmoid(np.dot(w.T, xi)) for xi in X[k]])
        # print estimate_yk - Y[k]
        fisrt_sum += ((estimate_yk - Y[k]) ** 2)

    allX = np.array(allX)
    second_sum = 0.
    for i in xrange(N):
        for j in xrange(N):
            second_sum += kernel(allX[i], allX[j]) * ((sigmoid(np.dot(w.T, allX[i])) - sigmoid(np.dot(w.T, allX[j]))) ** 2)
    return (beta / float(K)) * fisrt_sum + (1./ float(N * N)) * second_sum 

def grad_func(X, Y, w):
    allx = []
    p_kj_list = []
    N = 0
    beta = 1.5
    K = len(X)
    for k in range(K):
        pk_ij = []
        allx.extend(X[k])
        N += len(X[k])
        for j, x in enumerate(X[k]):
            dotproduct = np.dot(w.T, x)
            p_kj = sigmoid(dotproduct)
            pk_ij.append(p_kj)
        p_kj_list.append(pk_ij)

    allx = np.array(allx)
    first_sum  = []#np.array([0.] * len(allx[0]))
    for k in range(len(X)):
        first_derivative = 2. * (np.mean([sigmoid(np.dot(w.T, xi)) for xi in X[k]], axis=0) - Y[k])
        second_part = [np.array(X[k][i]) * p_kj_list[k][i] * (1. - p_kj_list[k][i]) for i in xrange(len(X[k]))]
        second_derivative = np.mean(second_part, axis=0)
        all_together =  first_derivative * second_derivative
        first_sum.append(all_together)
    # print first_sum.shape
    first_sum = np.array(first_sum)
    second_sum = []#np.array([0.] * len(allx[0]))
    for i in xrange(N):
        for j in xrange(N):
            derivative_xi = sigmoid(np.dot(w.T, allx[i])) * (1-sigmoid(np.dot(w.T, allx[i]))) * allx[i]
            derivative_xj = sigmoid(np.dot(w.T, allx[j])) * (1-sigmoid(np.dot(w.T, allx[j]))) * allx[j]
            tmp = 2.* kernel(allx[i], allx[j])\
                    * (derivative_xi - derivative_xj)\
                    * (sigmoid(np.dot(w.T, allx[i])) - sigmoid(np.dot(w.T, allx[j])))
            second_sum.append(tmp)            
    second_sum = np.array(second_sum)
    deltaw = (beta / float(K)) * np.sum(first_sum, axis=0) + (1. / float(N * N)) * np.sum(second_sum, axis=0)
    # print deltaw.shape
    return deltaw

def check_grad(X, Y, w): 
    epison = 1e-6
    tau = 1e-4
    neww1 = deepcopy(w)
    neww1[0] += epison
    neww2 = deepcopy(w)
    neww2[0] -= epison

    # print (grad_func(X, Y, w)[0]), abs(cost_func(X, Y, neww1) - cost_func(X, Y, neww2)) / (2. * epison)
    if abs(abs(cost_func(X, Y, neww1) - cost_func(X, Y, w)) / epison  - (grad_func(X, Y, w)[0])) <= tau:
        return True
    else: return False

def _vstack(m):
    """
    Return vstack
    """
    if issparse(m[0]) or issparse(m[-1]):
        return parse_vstack(m)
    else:
        return np.vstack(m)

class GICF:

    def SGD(self, train_x, train_y, test_x, test_y):

        train_X = np.array(train_x)
        train_Y = np.array(train_y)
        test_X = np.array(test_x)
        test_Y = np.array(test_y)

    
        """
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)
        train_idx, test_idx = [sp for sp in sklearn.cross_validation.StratifiedShuffleSplit(data_Y, 3)][0]

        train_X = data_X[train_idx]
        train_Y = data_Y[train_idx]

        test_X = data_X[test_idx]
        test_Y = data_Y[test_idx]
        """

        lambd = 0.05; beta = 1.0;
        iteration = 2000 
        x_dimension = 150
        w = np.random.rand(x_dimension)
        for t in range(iteration):
            eta = 1./ ((t + 1) * lambd)#lambd#
            kset = random.sample(range(0, len(train_X) - 1), 3)

            X = [train_X[k] for k in kset]
            Y = [train_Y[k] for k in kset]
            delta_w = grad_func(X, Y, w)
            # print check_grad(X, Y, w)
            new_w = w - eta * delta_w

            rate = (1./np.sqrt(lambd)) * (1./la.norm(new_w))
            if  rate < 1.:
                w = np.dot(rate, new_w)
            else:
                w = new_w

            if (t % 500) == 0:
                pred_testY = []
                for idx, testx in enumerate(test_X):
                    p_ij_list = []
                    for j, doc in enumerate(testx):
                        p_ij = sigmoid(np.dot(w.T, doc))
                        p_ij_list.append(p_ij)
                    P_i = np.mean(p_ij_list)
                    if P_i > 0.5:
                        pred_testY.append(1)
                    else:
                        pred_testY.append(0)
                print pred_testY
                print test_Y
                test_score = sklearn.metrics.f1_score(test_Y, pred_testY)

                pred_trainY = []
                for idx, tx in enumerate(train_X):
                    p_ij_list = []
                    for j, doc in enumerate(tx):
                        p_ij = sigmoid(np.dot(w.T, doc))
                        p_ij_list.append(p_ij)
                    P_i = np.mean(p_ij_list)
                    if P_i > 0.5:
                        pred_trainY.append(1)
                    else:
                        pred_trainY.append(0)
                # print pred_trainY
                # print train_Y
                train_score = sklearn.metrics.f1_score(train_Y, pred_trainY)
                print "Test f1-score: {}".format(test_score)
                print "Train f1-score: {}".format(train_score)
        return self, test_score, train_score


def compute_similarity_mat(dataX, idmap_f, k_matf):

    saveGlobalMap = {}
    group2GlobalMap = {}
    count = 0
    for idx, bag_i in enumerate(dataX):
        for idj, instance in enumerate(bag_i):
            saveGlobalMap[str(idx) + ',' + str(idj)] = count
            group2GlobalMap[(idx, idj)] = count
            count += 1

    with open(idmap_f, 'wb') as w:
        w.write(json.dumps(saveGlobalMap))
    # K_mat = np.random.rand(count, count)
    K_mat_dict = {}
    for idx, bag_i in enumerate(dataX):
        for idj, instance in enumerate(bag_i):
            globalid = group2GlobalMap[(idx, idj)]
            K_mat_dict[globalid] = {}
            for bagidx in range(idx, len(dataX)):
                for insidx in range(idj, len(dataX[bagidx])):
                    nextglobal =  group2GlobalMap[(bagidx, insidx)]
                    distance = kernel(instance, dataX[bagidx][insidx])
                    K_mat_dict[globalid][nextglobal] = distance
                    # K_mat[globalid, nextglobal] = K_mat[nextglobal, globalid] = distance

    with open(k_matf, 'wb') as w:
        w.write(json.dumps(K_mat_dict))

def load_event_dataset(fold=0):
    groups = ["train", "test"]
    dataset = []
    for group in groups:
        x_file = "./final_experiments/GICF/results/gicf_%d_%s_sent.vec" % (fold, group)
        y_file = "./data/new_multi_label/%d/event_%s.event_cat" % (fold, group)
        xs = [json.loads(l) for l in open(x_file)]
        ys = [0 if l.strip()=="non_protest" else 1 for l in open(y_file)]
        dataset.append((xs, ys))
    return dataset

def main(args):
    import json
    import os
    """
    dataMap = {}
    # count = 0
    dataX, dataY = [], []
    # data = np.loadtxt("../data/gsr_sen_matrix.txt")
    f = open(args.inputfile, 'rb')
   
    for line in f:
        dlist = json.loads(line.strip())
        doc = dlist['doc_mat']
        label = dlist['label']
        dataX.append(doc)
        dataY.append(label)
    print len(dataX)
    # id_mapf = 'idMapping.json'
    # k_matf = "my_k_matrix.json"
    # compute_similarity_mat(dataX, id_mapf, k_matf)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    with open('group2ind_sent_results_v1.txt', 'a') as writer:
        # direction = 'backword' if args.direction == 1 else 'forward'
        model = GICF()
        model, perf1, perf2 = model.SGD(dataX, dataY)
        writer.write('F1-test:%f\tF1-Train:%f\n' % (perf1, perf2))
    """
    # code for event protest
    dataset = load_event_dataset(args.fold)
    train_set, test_set = dataset
    train_x, train_y = train_set
    test_x, test_y = test_set
    with open("group2ind_result_%d.txt" % args.fold, 'w') as writer:
        model = GICF()
        model, perf1, perf2 = model.SGD(train_x, train_y, test_x, test_y)
        writer.write("F1-Test:%f \t F1-Train: %f\n" % (perf1, perf2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--inputfile", help="input file for training model")
    ap.add_argument("-fold", type=int, help="fold name")
    args = ap.parse_args()
    main(args)
