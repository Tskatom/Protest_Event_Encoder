#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convolution Neural Network for Event Encoding
"""

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import theano
from theano import function, shared
import theano.tensor as T
import numpy as np
import cPickle
import json
import argparse
import nn_layers as nn
import logging
import timeit
from collections import OrderedDict
import re
from theano.tensor.signal.downsample import DownsampleFactorMax
from theano.tensor.signal.downsample import DownsampleFactorMaxGrad

#theano.config.profile = True
#theano.config.profile_memory = True
#theano.config.optimizer = 'fast_run'

def ReLU(x):
    return T.maximum(0.0, x)

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', type=str,
            help="the prefix for input data such as spanish_protest")
    ap.add_argument('--word2vec', type=str,
            help="word vector pickle file")
    ap.add_argument('--sufix', type=str,
            help="the sufix for the target file")
    ap.add_argument('--dict_fn', type=str,
            help='class dictionary')
    ap.add_argument('--max_len', type=int,
            help='the max length for doc used for mini-batch')
    ap.add_argument('--max_sens', type=int, default=40,
            help='the max number of sentences for each doc')
    ap.add_argument('--max_words', type=int, default=80,
            help='the max number of words for each sentence')
    ap.add_argument("--padding", type=int,
            help="the number of padding used to add begin and end doc")
    ap.add_argument("--exp_name", type=str,
            help="experiment name")
    ap.add_argument("--static", action="store_true",
            help="whether update word2vec")
    ap.add_argument("--max_iter", type=int,
            help="max iterations")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--log_fn", type=str,
            help="log filename")
    ap.add_argument("--perf_fn", type=str,
            help="folder to store predictions")
    ap.add_argument("--param_fn", type=str,
            help="sepcific local params")
    ap.add_argument("--top_k", type=int, default=0,
            help="the maximum of sentence to choose")
    ap.add_argument("--print_freq", type=int, default=5,
            help="the frequency of print frequency")
    ap.add_argument("--model_file", type=str,
            help="the model file")
    return ap.parse_args()

def load_model(model_file):
    params = {}
    with open(model_file, 'rb') as mf:
        params["softmax_U"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
        params["softmax_b"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
        params["conv_W"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
        params["conv_b"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
        params["theta"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
        params["embedding"] = theano.shared(cPickle.load(mf).astype(theano.config.floatX))
    return params

def load_dataset(prefix, sufix):
    """Load the train/valid/test set
        prefix eg: ../data/spanish_protest
        sufix eg: pop_cat
    """
    dataset = []
    for group in ["train", "valid", "test"]:
        x_fn = "%s_%s.txt.tok" % (prefix, group)
        y_fn = "%s_%s.%s" % (prefix, group, sufix)
        xs = [l.strip() for l in open(x_fn)]
        ys = [l.strip() for l in open(y_fn)]
        dataset.append((xs, ys))
        print "Load %d %s records" % (len(ys), group)
    return dataset

def split_doc2sen(doc, word2id, max_sens=40, max_words=80, padding=5):
    """
        doc is the raw text where words are seperated by space
    """
    sens = re.split("\.|\?|\|", doc.lower())
    # reduce those sens which has length less than 5
    sens = [sen for sen in sens if len(sen.strip().split(" ")) > 5]
    pad = padding - 1
    sens_pad = []
    for sen in sens[:max_sens]:
        tokens = sen.strip().split(" ")
        sen_ids = [0] * pad
        for w in tokens[:max_words]:
            sen_ids.append(word2id.get(w, 1))
        num_suff = max(0, max_words - len(tokens)) + pad
        sen_ids += [0] * num_suff
        sens_pad.append(sen_ids)
    
    # add more padding sentence
    num_suff = max(0, max_sens - len(sens))
    for i in range(0, num_suff):
        sen_ids = [0] * len(sens_pad[0])
        sens_pad.append(sen_ids)

    return sens_pad


def get_doc_length(dataset, max_sens=40):
    train_set, valid_set, test_set = dataset
    train_doc, train_class = train_set
    valid_doc, valid_class = valid_set
    test_doc, test_class = test_set
    
    def compute_len(doc):
        sens = re.split("\.|\?|\|", doc.lower())
        sens = [sen for sen in sens if len(sen.strip().split(" ")) > 5]
        return len(sens) if len(sens) <= max_sens else max_sens
    train_doc_lens = [compute_len(doc) for doc in train_doc]
    valid_doc_lens = [compute_len(doc) for doc in valid_doc]
    test_doc_lens = [compute_len(doc) for doc in test_doc]
    
    return [train_doc_lens, valid_doc_lens, test_doc_lens]


def transform_dataset(dataset, word2id, class2id, max_sens=40, max_words=80, padding=5):
    """Transform the dataset into digits
    the final doc is a list of list(list of sentence which is a list of word)
    """
    train_set, valid_set, test_set = dataset
    train_doc, train_class = train_set
    valid_doc, valid_class = valid_set
    test_doc, test_class = test_set
    
    train_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in train_doc]
    valid_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in valid_doc]
    test_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in test_doc]

    train_y = [class2id[c] for c in train_class]
    valid_y = [class2id[c] for c in valid_class]
    test_y = [class2id[c] for c in test_class]

    return [(train_doc_ids, train_y), (valid_doc_ids, valid_y), (test_doc_ids, test_y)]


def run_cnn(exp_name,
        dataset, doc_lens,
        model_file,
        log_fn, perf_fn,
        k=0,
        emb_dm=100,
        batch_size=100,
        filter_hs=[1, 2, 3],
        hidden_units=[200, 100, 11],
        dropout_rate=0.5,
        shuffle_batch=True,
        n_epochs=300,
        lr_decay=0.95,
        activation=ReLU,
        sqr_norm_lim=9,
        non_static=True,
        print_freq=5):
    """
    Train and Evaluate CNN event encoder model
    :dataset: list containing three elements[(train_x, train_y), 
            (valid_x, valid_y), (test_x, test_y)]
    :embedding: word embedding with shape (|V| * emb_dm)
    :filter_hs: filter height for each paralle cnn layer
    :dropout_rate: dropout rate for full connected layers
    :n_epochs: the max number of iterations
    
    """
    start_time = timeit.default_timer()
    rng = np.random.RandomState(1234)
   
    input_height = len(dataset[0][0][0][0]) # number of words in the sentences
    num_sens = len(dataset[0][0][0]) # number of sentences
    print "--input height ", input_height 
    input_width = emb_dm
    num_maps = hidden_units[0]

    ###################
    # start snippet 1 #
    ###################
    print "start to construct the model ...."
    x = T.tensor3("x")
    y = T.ivector("y")

    model_params = load_model(model_file)

    words = model_params["embedding"]

    # define function to keep padding vector as zero
    zero_vector_tensor = T.vector()
    zero_vec = np.zeros(input_width, dtype=theano.config.floatX)
    set_zero = function([zero_vector_tensor],
            updates=[(words, T.set_subtensor(words[0,:], zero_vector_tensor))])

    # the input for the sentence level conv layers
    layer0_input = words[T.cast(x.flatten(), dtype="int32")].reshape((
        x.shape[0] * x.shape[1], 1, x.shape[2], emb_dm
        ))

    conv_layers = []
    layer1_inputs = []
    

    for i in xrange(len(filter_hs)):
        filter_shape = (num_maps, 1, filter_hs[i], emb_dm)
        pool_size = (input_height - filter_hs[i] + 1, 1)
        conv_layer = nn.ConvPoolLayer(rng, input=layer0_input, 
                input_shape=None,
                filter_shape=filter_shape,
                pool_size=pool_size, activation=activation,
                W=model_params["conv_W"],
                b=model_params["conv_b"])
        
        sen_vecs = conv_layer.output.reshape((x.shape[0], x.shape[1], num_maps))

        layer1_inputs.append(sen_vecs)
    
    layer1_input = T.concatenate(layer1_inputs, 1)
    
    train_x, train_y = shared_dataset(dataset[0])
    valid_x, valid_y = shared_dataset(dataset[1])
    test_x, test_y = shared_dataset(dataset[2])

    n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
    n_valid_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))
    n_test_batches = int(np.ceil(1.0 * len(dataset[2][0]) / batch_size))

    #####################
    # Train model func #
    #####################
    index = T.iscalar()
    train_sen = function([index], layer1_input, 
            givens={
                x: train_x[index*batch_size:(index+1)*batch_size]
                })

    valid_sen = function([index], layer1_input, 
            givens={
                x: valid_x[index*batch_size:(index+1)*batch_size]
                })
    
    test_sen = function([index], layer1_input, 
            givens={
                x: test_x[index*batch_size:(index+1)*batch_size]
                })
    
    
    # apply early stop strategy
    patience = 100
    patience_increase = 2
    improvement_threshold = 1.005
    
    n_valid = len(dataset[1][0])
    n_test = len(dataset[2][0])

    epoch = 0
    best_params = None
    best_validation_score = 0.
    test_perf = 0

    done_loop = False
    
    print "Start to train the model....."
    cpu_trn_y = np.asarray(dataset[0][1])
    cpu_val_y = np.asarray(dataset[1][1])
    cpu_tst_y = np.asarray(dataset[2][1])

    def compute_score(true_list, pred_list):
        mat = np.equal(true_list, pred_list)
        score = np.mean(mat)
        return score
    
    best_test_score = 0.
    
    while (epoch < n_epochs) and not done_loop:
        done_loop = True
        start_time = timeit.default_timer()
        epoch += 1
        train_sen_vecs = [train_sen(i) for i in xrange(n_train_batches)]
        trains_sens = np.concatenate(train_sen_vecs, axis=0)
        valid_sen_vecs = [valid_sen(i) for i in xrange(n_valid_batches)]
        valid_sens = np.concatenate(valid_sen_vecs, axis=0)
        test_sen_vecs = [test_sen(i) for i in xrange(n_test_batches)]
        test_sens = np.concatenate(test_sen_vecs, axis=0)

        # write sentence vector into files
        groups = ["train", "valid", "test"]
        sens = [train_sens, valid_sens, test_sens]

        for g, ss, lens in zip(groups, sens, doc_lens):
            with open("./data/%s_sens_vec.txt" % g, 'w') as wf:
                for i, s_mat in enumerate(ss):
                    rs_mat = s_mat[:lens[i]]
                    rs_mat_list = json.dumps(rs_mat.tolist())
                    wf.write(rs_mat_list + "\n")


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
        dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def main():
    args = parse_args()
    prefix = args.prefix
    word2vec_file = args.word2vec
    sufix = args.sufix
    expe_name = args.exp_name
    batch_size = args.batch_size
    log_fn = args.log_fn
    perf_fn = args.perf_fn
    top_k = args.top_k # the limit of sentences to choose
    print_freq = args.print_freq
    model_file = args.model_file

    # load the dataset
    print 'Start loading the dataset ...'
    dataset = load_dataset(prefix, sufix)
    wf = open(word2vec_file)
    embedding = cPickle.load(wf)
    word2id = cPickle.load(wf)

    dict_file = args.dict_fn
    class2id = {k.strip(): i for i, k in enumerate(open(dict_file))}
    
    # transform doc to dig list and padding docs
    print 'Start to transform doc to digits'
    max_sens = args.max_sens
    max_words = args.max_words
    padding = args.padding
    digit_dataset = transform_dataset(dataset, word2id, class2id, max_sens, max_words, padding)
    doc_lens = get_doc_length(dataset, max_sens)

    non_static = not args.static
    exp_name = args.exp_name
    n_epochs = args.max_iter

    # load local parameters
    loc_params = json.load(open(args.param_fn))
    filter_hs = loc_params["filter_hs"]
    hidden_units = loc_params["hidden_units"]

    run_cnn(exp_name, digit_dataset, doc_lens, model_file, 
            log_fn, perf_fn,
            k=top_k,
            emb_dm=embedding.shape[1],
            batch_size=batch_size,
            filter_hs=filter_hs,
            hidden_units=hidden_units,
            dropout_rate=0.5,
            shuffle_batch=True,
            n_epochs=n_epochs,
            lr_decay=0.95,
            activation=ReLU,
            sqr_norm_lim=9,
            non_static=True,
            print_freq=print_freq)
     
    
if __name__ == "__main__":
    main()
