#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Decode the convolution neural network to get the most significant features
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import theano
import theano.tensor as T
import cPickle
from SIG_Cnn_encoder import make_data_cv, ReLU
import numpy as np
from theano import function, shared
import nn_layers as nn
import json
from theano.tensor.nnet import conv

def load_model(model_file, non_static=True):
    params = {}
    with open(model_file) as f:
        # load classifier params
        clf_W = cPickle.load(f)
        clf_b = cPickle.load(f)
        params["clf"] = [clf_W, clf_b]
        # conv layer params
        conv_params = []
        for i in range(3):
            conv_W = cPickle.load(f)
            conv_b = cPickle.load(f)
            conv_params.append([conv_W, conv_b])
        params["convs"] = conv_params

        if non_static:
            params["embedding"] = cPickle.load(f)
    return params

def find_ngrams(docs, ns=[3,4,5]):
    n_grams = []
    for n in ns:
        n_gram = set()
        for doc in docs:
            doc = doc[:-2]
            for i in xrange(len(doc) - n + 1):
                gram = tuple(doc[i:i+n])
                n_gram.add(gram)
        n_grams.append(n_gram)
    return n_grams


def get_top_features(n_gram, params, word2id, n=0, k=50):
    words = shared(params["embedding"], borrow=True, name='ebmedding')
    x = T.matrix()
    x_input = words[T.cast(x.flatten(), dtype="int32")].reshape(
            (x.shape[0], 1, x.shape[1], words.shape[1])
            )
    
    n_gram_matrix = np.asarray([g for g in n_gram], dtype=theano.config.floatX)

    # get filter
    conv_W, conv_b = params["convs"][n]
    W = shared(value=conv_W.astype(theano.config.floatX), borrow=True, name="conv_W")
    b = shared(value=conv_b.astype(theano.config.floatX), borrow=True, name="conv_b")

    conv_out = conv.conv2d(input=x_input, 
            filters=W,
            filter_shape=conv_W.shape)

    out = conv_out.flatten(2)
    ind = T.argsort(out, axis=0)[-k:,:][::-1].T # REVERSE THE ORDER

    get_out = function(inputs=[x], outputs=ind)
    
    top_conv_ind = get_out(n_gram_matrix)
    
    print 'top_conv_ind.shape: ', top_conv_ind.shape
    top_word_ids = n_gram_matrix[top_conv_ind]
    #convert id to words
    id2words = {v:k for k,v in word2id.items()}
    n_gram_texts = {}
    for h_id, tops in enumerate(top_word_ids):
        strs = []
        for word_ids in tops:
            word_str = ' '.join([id2words.get(wid, '') for wid in word_ids])
            word_str = word_str.strip()
            strs.append(word_str)
        n_gram_texts[h_id] = strs

    return n_gram_texts


def main():
    print '...load the expriment data set'
    data = cPickle.load(open('./data/experiment_dataset2'))
    docs, type2id, pop2id, word2id, embedding, rand_embedding = data
    
    print '... construct the train/valid/test set'
    test_docs = docs[:10000]
    datasets = make_data_cv(test_docs, 0, word2id, max_l=1000, filter_h=5)

    print '... construct ngrams'
    ns = [3, 4, 5]
    n_grams = find_ngrams(datasets[0])
    print '....Load model parameters'
    # load the trained parameters
    params= load_model('./data/pop_model.pkl')

    print '....start test the model'
    # construct the model
    # dump the result
    with open("./data/top_ngrams.pkl", 'wb') as tn:
        for n, n_gram in enumerate(n_grams):
            n_gram_texts = get_top_features(n_gram, params, word2id, n)
            cPickle.dump(n_gram_texts, tn)


if __name__ == "__main__":
    main()
