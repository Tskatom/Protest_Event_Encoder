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
from CNN_Sen import split_doc2sen

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
    ap.add_argument('--sufix_pop', type=str,
            help="the sufix for the target file")
    ap.add_argument('--sufix_type', type=str,
            help="the sufix for the target file")
    ap.add_argument('--dict_pop_fn', type=str,
            help='population class dictionary')
    ap.add_argument('--dict_type_fn', type=str,
            help='event type class dictionary')
    ap.add_argument('--max_len', type=int,
            help='the max length for doc used for mini-batch')
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
    ap.add_argument("--max_sens", type=int, default=40,
            help="the max number of sens in a document")
    ap.add_argument("--max_words", type=int, default=80,
            help="the max number of sentences for each sentence")
    return ap.parse_args()

def load_dataset(prefix, sufix_1, sufix_2):
    """Load the train/valid/test set
        prefix eg: ../data/spanish_protest
        sufix eg: pop_cat
    """
    dataset = []
    for group in ["train", "test"]:
        x_fn = "%s_%s.txt.tok" % (prefix, group)
        y1_fn = "%s_%s.%s" % (prefix, group, sufix_1)
        y2_fn = "%s_%s.%s" % (prefix, group, sufix_2)
        xs = [l.strip() for l in open(x_fn)]
        y1s = [l.strip() for l in open(y1_fn)]
        y2s = [l.strip() for l in open(y2_fn)]
        dataset.append((xs, y1s, y2s))
        print "Load %d %s records" % (len(y1s), group)
    return dataset

def construct_joint_y(pop_y, type_y, num_pop=11, num_type=5):
    # construct 3D tensor for joint Y label
    num_samples = len(pop_y)
    joint_y = np.zeros((num_samples, num_pop, num_type))
    # set Y_ij to 1 for extract labels
    for i, entry in enumerate(joint_y):
        entry[pop_y[i], type_y[i]] = 1
    return joint_y.reshape((num_samples, num_pop*num_type))


def transform_dataset(dataset, word2id, class2id, max_sens=40, max_words=80, padding=5):
    """Transform the dataset into digits"""
    train_set, test_set = dataset
    train_doc, train_pop_class, train_type_class = train_set
    test_doc, test_pop_class, test_type_class = test_set
    
    train_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in train_doc]
    test_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in test_doc]

    train_pop_y = [class2id["pop"][c] for c in train_pop_class]
    test_pop_y = [class2id["pop"][c] for c in test_pop_class]
    train_y = construct_joint_y(train_pop_y, train_type_y)
    
    train_type_y = [class2id["type"][c] for c in train_type_class]
    test_type_y = [class2id["type"][c] for c in test_type_class]
    test_y = construct_joint_y(test_pop_y, test_type_y)

    return [(train_doc_ids, train_y), (test_doc_ids, test_y)]


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6,
        norm_lim=9, word_vec_name='embedding'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = [] 
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param] 
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='embedding'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0)) 
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim)) 
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def run_cnn(exp_name,
        dataset, embedding,
        log_fn, perf_fn,
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
        non_static=True):
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
   
    input_height = len(dataset[0][0][0][0])
    num_sens = len(dataset[0][0][0])
    print "--input height ", input_height 
    input_width = emb_dm
    num_maps = hidden_units[0]

    ###################
    # start snippet 1 #
    ###################
    print "start to construct the model ...."
    x = T.tensor3("x")
    y = T.matrix("y")

    words = shared(value=np.asarray(embedding,
        dtype=theano.config.floatX), 
        name="embedding", borrow=True)

    # define function to keep padding vector as zero
    zero_vector_tensor = T.vector()
    zero_vec = np.zeros(input_width, dtype=theano.config.floatX)
    set_zero = function([zero_vector_tensor],
            updates=[(words, T.set_subtensor(words[0,:], zero_vector_tensor))])

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
                pool_size=pool_size, activation=activation)
        sen_vecs = conv_layer.output.reshape((x.shape[0], 1, x.shape[1], num_maps))
        sen_vecs = sen_vecs.dimshuffle(0, 2, 1)
        doc_vec = T.max(sen_vecs, axis=2).flatten(2)
        layer1_inputs.append(doc_vec)
        conv_layers.append(conv_layer)

    
    layer1_input = T.concatenate(layer1_inputs, 1)

    ##############
    # Task pop#
    ##############
    print "Construct classifier ...."
    hidden_units[0] = num_maps * len(filter_hs)
    pop_factor = nn.MLDropout(rng, 
            input=layer1_input,
            layer_sizes=hidden_units,
            dropout_rates=[dropout_rate for i in len(hidden_units)-1],
            activations=[activation for i in len(hidden_units)-1])
    pop_factor_output = pop_factor.output.dimshuffle(0,1,'x')
    pop_factor_dropout_output = pop_factor.dropout_output.dumshuffle(0,1,'x')

    #######################
    # Task Type #####
    #######################
    type_hidden_units = [num for num in hidden_units]
    type_hidden_units[-1] = 5
    type_factor = nn.MLDropout(rng,
            input=layer1_input,
            layer_sizes=type_hidden_units,
            dropout_rates=[dropout_rate for i in len(type_hidden_units)-1],
            activations=[activation for i in len(type_hidden_units)-1])
    type_factor_output = type_factor.output.dimshuffle(0,'x',1)
    type_factor_dropout_output = type_factor.dropout_output.dimshuffle(0,'x',1)

    ######################
    ## Joint Y matrix ###
    #####################
    # construct V matrix to model pop type dependency
    V_value = np.random.random((hidden_units[-1], type_hidden_units[-1]))
    V = theano.shared(value=np.asarray(V_value,
        dtype=theano.config.floatX), name="V", borrow=True)

    # compute the Joint propability
    joint_act = T.batched_dot(pop_factor_output, type_factor_output) + V
    joint_act_dropout = T.batched_dot(pop_factor_dropout_output, type_factor_dropout_output) + V

    joint_probs = T.nnet.softmax(joint_act.flatten(2))
    joint_probs_dropout = T.nnet.softmax(joint_act_dropout.flatten(2))
    
    neg_likelihood = -T.mean(T.sum(T.log(T.joint_probs * y), axsi=1))
    neg_likelihood_dropout = -T.mean(T.sum(T.log(T.joint_probs_dropout * y), axsi=1))

    joint_preds = T.argmax(joint_probs, axis=1)
    pop_preds = joint_preds // type_hidden_units[-1]
    type_preds = joint_preds % type_hidden_units[-1]

    y_index = T.argmax(y, axis=1)
    pop_y = y_index // type_hidden_units[-1]
    type_y = y_index % type_hidden_units[-1]

    pop_error = T.mean(T.neq(pop_preds, pop_y))
    type_error = T.mean(T.neq(type_preds, type_y))

    params = pop_factor.params
    params += type_factor.params
    params.append(V)

    for conv_layer in conv_layers:
        params += conv_layer.params

    if non_static:
        params.append(words)

    grad_updates = sgd_updates_adadelta(params, 
            neg_likelihood_dropout, 
            lr_decay, 
            1e-6, 
            sqr_norm_lim)

    #####################
    # Construct Dataset #
    #####################
    print "Copy data to GPU and constrct train/valid/test func"
    np.random.seed(1234)
    
    train_x, train_y = shared_dataset(dataset[0])
    test_x, test_y = shared_dataset(dataset[1])

    n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
    n_test_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))

    #####################
    # Train model func #
    #####################
    index = T.iscalar()
    train_func = function([index], cost, updates=grad_updates,
            givens={
                x: train_x[index*batch_size:(index+1)*batch_size],
                y: train_y[index*batch_size:(index+1)*batch_size]
                })
    
    test_pred = function([index], [pop_error, type_error],
            givens={
                x:test_x[index*batch_size:(index+1)*batch_size],
                y:test_y[index*batch_size:(index+1)*batch_size]
                })


    # apply early stop strategy
    patience = 100
    patience_increase = 2
    improvement_threshold = 1.005
    
    n_test = len(dataset[1][0])

    epoch = 0
    best_params = None
    best_validation_score = 0.
    test_perf = 0

    done_loop = False
    
    log_file = open(log_fn, 'a')

    while (epoch < n_epochs) and not done_loop:
        start_time = timeit.default_timer()
        epoch += 1
        costs = []
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            cost_epoch = train_func(minibatch_index)
            costs.append(cost_epoch)
            set_zero(zero_vec)

        if epoch % 5 == 0:
            # do test
            test_pop_errors = []
            test_type_errors = []
            for i in xrange(n_test_batches):
                test_pop_error, test_type_error = test_pred(i)
                test_pop_errors.append(test_pop_error)
                test_type_errors.append(test_type_error)
            
            test_pop_score = 1 - np.mean(test_pop_errors)
            test_type_score = 1 - np.mean(test_type_errors)
            
            message = "Epoch %d test pop perf %f, type perf %f" % (epoch, test_pop_score, test_type_score)
            print message
            log_file.write(message + "\n")
            log_file.flush()

        end_time = timeit.default_timer()
        print "Finish one iteration using %f m" % ((end_time - start_time)/60.)

    log_file.flush()
    log_file.close()


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
        dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)
    
    return shared_x, shared_y


def main():
    args = parse_args()
    prefix = args.prefix
    word2vec_file = args.word2vec
    sufix_pop = args.sufix_pop
    sufix_type = args.sufix_type
    expe_name = args.exp_name
    batch_size = args.batch_size
    log_fn = args.log_fn
    perf_fn = args.perf_fn

    # load the dataset
    print 'Start loading the dataset ...'
    dataset = load_dataset(prefix, sufix_pop, sufix_type)
    wf = open(word2vec_file)
    embedding = cPickle.load(wf)
    word2id = cPickle.load(wf)

    class2id = {}
    dict_pop_file = args.dict_pop_fn
    class2id["pop"] = {k.strip(): i for i, k in enumerate(open(dict_pop_file))}
    
    dict_type_file = args.dict_type_fn
    class2id["type"] = {k.strip(): i for i, k in enumerate(open(dict_type_file))}
    
    # transform doc to dig list and padding docs
    print 'Start to transform doc to digits'
    max_sens = args.max_sens
    max_words = args.max_words
    padding = args.padding
    digit_dataset = transform_dataset(dataset, word2id, class2id, max_sens, max_words, padding)

    non_static = not args.static
    exp_name = args.exp_name
    n_epochs = args.max_iter

    # load local parameters
    loc_params = json.load(open(args.param_fn))
    filter_hs = loc_params["filter_hs"]
    hidden_units = loc_params["hidden_units"]

    run_cnn(exp_name, digit_dataset, embedding,
            log_fn, perf_fn,
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
            non_static=non_static)
     
    
if __name__ == "__main__":
    main()
