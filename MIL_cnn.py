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
#theano.config.optimizer = 'None'
#theano.config.exception_verbosity='high'

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
    ap.add_argument("--print_freq", type=int, default=5,
            help="the frequency of print frequency") 
    return ap.parse_args()

def load_dataset(prefix, sufix_1, sufix_2):
    """Load the train/valid/test set
        prefix eg: ../data/spanish_protest
        sufix eg: pop_cat
    """
    dataset = []
    for group in ["train", "valid", "test"]:
        x_fn = "%s_%s.txt.tok" % (prefix, group)
        y1_fn = "%s_%s.%s" % (prefix, group, sufix_1)
        y2_fn = "%s_%s.%s" % (prefix, group, sufix_2)
        xs = [l.strip() for l in open(x_fn)]
        y1s = [l.strip() for l in open(y1_fn)]
        y2s = [l.strip() for l in open(y2_fn)]
        dataset.append((xs, y1s, y2s))
        print "Load %d %s records" % (len(y1s), group)
    return dataset


def transform_dataset(dataset, word2id, class2id, max_sens=40, max_words=80, padding=5):
    """Transform the dataset into digits"""
    train_set, valid_set, test_set = dataset
    train_doc, train_pop_class, train_type_class = train_set
    valid_doc, valid_pop_class, valid_type_class = valid_set
    test_doc, test_pop_class, test_type_class = test_set
    
    train_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in train_doc]
    valid_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in valid_doc]
    test_doc_ids = [split_doc2sen(doc, word2id, max_sens, max_words, padding) for doc in test_doc]

    train_pop_y = [class2id["pop"][c] for c in train_pop_class]
    valid_pop_y = [class2id["pop"][c] for c in valid_pop_class]
    test_pop_y = [class2id["pop"][c] for c in test_pop_class]
    
    train_type_y = [class2id["type"][c] for c in train_type_class]
    valid_type_y = [class2id["type"][c] for c in valid_type_class]
    test_type_y = [class2id["type"][c] for c in test_type_class]

    return [(train_doc_ids, train_pop_y, train_type_y), (valid_doc_ids, valid_pop_y, valid_type_y), (test_doc_ids, test_pop_y, test_type_y)]


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


def keep_max(input, theta, k):
    sig_input = T.nnet.sigmoid(T.dot(input, theta))
    if k == 0:
        result = input * T.addbroadcast(sig_input, 3)
        return result, sig_input

    # get the sorted idx
    sort_idx = T.argsort(sig_input, axis=2)
    k_max_ids = sort_idx[:,:,-k:,:]
    dim0, dim1, dim2, dim3 = k_max_ids.shape
    batchids = T.repeat(T.arange(dim0), dim1*dim2*dim3)
    mapids = T.repeat(T.arange(dim1), dim2*dim3).reshape((1, dim2*dim3))
    mapids = T.repeat(mapids, dim0, axis=0).flatten()
    rowids = k_max_ids.flatten()
    colids = T.arange(dim3).reshape((1, dim3))
    colids = T.repeat(colids, dim0*dim1*dim2, axis=0).flatten()
    sig_mask = T.zeros_like(sig_input)
    choosed = sig_input[batchids, mapids, rowids, colids]
    sig_mask = T.set_subtensor(sig_mask[batchids, mapids, rowids, colids], 1)
    input_mask = sig_mask * sig_input
    result = input * T.addbroadcast(input_mask, 3)
    return result, sig_input


def run_cnn(exp_name,
        dataset, embedding,
        log_fn, perf_fn,
        emb_dm=100,
        batch_size=100,
        filter_hs=[1, 2, 3],
        hidden_units=[200, 100, 11],
        type_hidden_units=[200, 100, 6],
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
    type_y = T.ivector("y_type")
    pop_y = T.ivector("y_pop")

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

    #########################
    # Construct Sen Vec #####
    #########################
    conv_layers = []
    filter_shape = (num_maps, 1, filter_hs[0], emb_dm)
    pool_size = (input_height - filter_hs[0] + 1, 1)
    conv_layer = nn.ConvPoolLayer(rng, input=layer0_input,
            input_shape=None, filter_shape=filter_shape,
            pool_size=pool_size, activation=activation)
    sen_vecs = conv_layer.output.reshape((x.shape[0], x.shape[1], num_maps))
    conv_layers.append(conv_layer)
    
    ########################
    ## Task 1: populaiton###
    ######################## 
    pop_layer_sizes = zip(hidden_units, hidden_units[1:])
    pop_layer_input = sen_vecs
    pop_drop_input = sen_vecs
    pop_hidden_outs = []
    pop_drop_outs = []
    pop_hidden_layers = []
    pop_drop_layers = []
    droprate = 0.5
    for layer_size in pop_layer_sizes[:-1]:
        U_value = np.random.random(layer_size).astype(theano.config.floatX)
        b_value = np.zeros((layer_size[-1],), dtype=theano.config.floatX)

        U = theano.shared(U_value, borrow=True, name="U")
        b = theano.shared(b_value, borrow=True, name="b")
        
        pop_hidden_layer = nn.HiddenLayer(rng, pop_layer_input, 
                layer_size[0], layer_size[1], ReLU, 
                U * (1 - droprate), b)
        pop_drop_hidden_layer = nn.DropoutHiddenLayer(rng, pop_drop_input,
                layer_size[0], layer_size[1], ReLU,
                droprate, U, b)

        pop_hidden_layers.append(pop_hidden_layer)
        pop_drop_layers.append(pop_drop_hidden_layer)

        pop_hidden_out = pop_hidden_layer.output
        pop_drop_out = pop_drop_hidden_layer.output

        pop_layer_input = pop_hidden_out
        pop_drop_input = pop_drop_out

        pop_hidden_outs.append(pop_hidden_out)
        pop_drop_outs.append(pop_drop_out)

    # construct pop classifier 
    n_in, n_out = pop_layer_sizes[-1]
    W_value = np.random.random((n_in, n_out)).astype(theano.config.floatX)
    b_value = np.zeros((n_out,), dtype=theano.config.floatX)

    pop_W = theano.shared(W_value, borrow=True, name="pop_W")
    pop_b = theano.shared(b_value, borrow=True, name="pop_b")

    pop_act = T.dot(pop_hidden_outs[-1], pop_W * (1 - droprate)) + pop_b
    pop_drop_act = T.dot(pop_drop_outs[-1], pop_W) + pop_b

    pop_max_act = T.max(pop_act, axis=1).flatten(2)
    pop_drop_max_act = T.max(pop_drop_act, axis=1).flatten(2)

    pop_sen_max = T.argmax(T.max(pop_act, axis=2).flatten(2), axis=1)
    pop_drop_sen_max = T.argmax(T.max(pop_drop_act, axis=2).flatten(2), axis=1)
    
    pop_probs = T.nnet.softmax(pop_max_act)
    pop_drop_probs = T.nnet.softmax(pop_drop_max_act)

    pop_y_pred = T.argmax(pop_probs, axis=1)
    pop_drop_y_pred = T.argmax(pop_drop_probs, axis=1)

    pop_neg_loglikelihood = -T.mean(T.log(pop_probs)[T.arange(pop_y.shape[0]), pop_y])
    pop_drop_neg_loglikelihood = -T.mean(T.log(pop_drop_probs)[T.arange(pop_y.shape[0]), pop_y])

    pop_errors = T.mean(T.neq(pop_y_pred, pop_y))
    pop_errors_detail = T.neq(pop_y_pred, pop_y)

    pop_cost = pop_neg_loglikelihood
    pop_drop_cost = pop_drop_neg_loglikelihood


    
    ########################
    ## Task 1: event type###
    ######################## 
    type_layer_sizes = zip(type_hidden_units, type_hidden_units[1:])
    type_layer_input = sen_vecs
    type_drop_input = sen_vecs
    type_hidden_outs = []
    type_drop_outs = []
    type_hidden_layers = []
    type_drop_layers = []
    droprate = 0.5
    for layer_size in type_layer_sizes[:-1]:
        U_value = np.random.random(layer_size).astype(theano.config.floatX)
        b_value = np.zeros((layer_size[-1],), dtype=theano.config.floatX)

        U = theano.shared(U_value, borrow=True, name="U")
        b = theano.shared(b_value, borrow=True, name="b")
        
        type_hidden_layer = nn.HiddenLayer(rng, type_layer_input, 
                layer_size[0], layer_size[1], ReLU, 
                U * (1 - droprate), b)
        type_drop_hidden_layer = nn.DropoutHiddenLayer(rng, type_drop_input,
                layer_size[0], layer_size[1], ReLU,
                droprate, U, b)

        type_hidden_layers.append(type_hidden_layer)
        type_drop_layers.append(type_drop_hidden_layer)

        type_hidden_out = type_hidden_layer.output
        type_drop_out = type_drop_hidden_layer.output

        type_layer_input = type_hidden_out
        type_drop_input = type_drop_out

        type_hidden_outs.append(type_hidden_out)
        type_drop_outs.append(type_drop_out)

    # construct pop classifier 
    n_in, n_out = type_layer_sizes[-1]
    W_value = np.random.random((n_in, n_out)).astype(theano.config.floatX)
    b_value = np.zeros((n_out,), dtype=theano.config.floatX)

    type_W = theano.shared(W_value, borrow=True, name="pop_W")
    type_b = theano.shared(b_value, borrow=True, name="pop_b")

    type_act = T.dot(type_hidden_outs[-1], type_W * (1 - droprate)) + type_b
    type_drop_act = T.dot(type_drop_outs[-1], type_W) + type_b

    type_max_act = T.max(type_act, axis=1).flatten(2)
    type_drop_max_act = T.max(type_drop_act, axis=1).flatten(2)
    
    type_sen_max = T.argmax(T.max(type_act, axis=2).flatten(2), axis=1)
    type_drop_sen_max = T.argmax(T.max(type_drop_act, axis=2).flatten(2), axis=1)
    
    type_probs = T.nnet.softmax(type_max_act)
    type_drop_probs = T.nnet.softmax(type_drop_max_act)

    type_y_pred = T.argmax(type_probs, axis=1)
    type_drop_y_pred = T.argmax(type_drop_probs, axis=1)

    type_neg_loglikelihood = -T.mean(T.log(type_probs)[T.arange(type_y.shape[0]), type_y])
    type_drop_neg_loglikelihood = -T.mean(T.log(type_drop_probs)[T.arange(type_y.shape[0]), type_y])

    type_errors = T.mean(T.neq(type_y_pred, type_y))
    type_errors_detail = T.neq(type_y_pred, type_y)

    type_cost = type_neg_loglikelihood
    type_drop_cost = type_drop_neg_loglikelihood


    ###################################
    ## Choose the max sens in two task#
    ###################################
    pop_drop_choosed_sens = sen_vecs[T.arange(sen_vecs.shape[0]), pop_drop_sen_max]
    type_drop_choosed_sens = sen_vecs[T.arange(sen_vecs.shape[0]), type_drop_sen_max]
    simi_drop_cost = T.mean(T.exp(T.sum((pop_drop_choosed_sens - type_drop_choosed_sens) ** 2, axis=1)))
    
    pop_choosed_sens = sen_vecs[T.arange(sen_vecs.shape[0]), pop_sen_max]
    type_choosed_sens = sen_vecs[T.arange(sen_vecs.shape[0]), type_sen_max]
    simi_cost = T.mean(T.exp(T.sum((pop_choosed_sens - type_choosed_sens) ** 2, axis=1)))


    ##################################
    # Collect all the parameters #####
    ##################################
    params = []
    # convolution layer params
    for conv_layer in conv_layers:
        params += conv_layer.params

    # params for population task
    for layer in pop_drop_layers:
        params += layer.params

    params.append(pop_W)
    params.append(pop_b)

    # params for event type task
    for layer in type_drop_layers:
        params += layer.params

    params.append(type_W)
    params.append(type_b)

    if non_static:
        params.append(words)

    total_cost = pop_cost + type_cost + simi_cost
    total_drop_cost = pop_drop_cost + type_drop_cost + simi_drop_cost

    total_grad_updates = sgd_updates_adadelta(params, 
            total_drop_cost,
            lr_decay,
            1e-6,
            sqr_norm_lim)

    total_preds = [pop_y_pred, type_y_pred]
    total_errors_details = [pop_errors_detail, type_errors_detail]
    total_choosed_sens = [pop_sen_max, type_sen_max]
    total_out = total_preds + total_errors_details + total_choosed_sens

    #####################
    # Construct Dataset #
    #####################
    print "Copy data to GPU and constrct train/valid/test func"
    np.random.seed(1234)
    
    train_x, train_pop_y, train_type_y = shared_dataset(dataset[0])
    valid_x, valid_pop_y, valid_type_y = shared_dataset(dataset[1])
    test_x, test_pop_y, test_type_y = shared_dataset(dataset[2])

    n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
    n_valid_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))
    n_test_batches = int(np.ceil(1.0 * len(dataset[2][0]) / batch_size))

    #####################
    # Train model func #
    #####################
    index = T.iscalar()
    train_func = function([index], total_drop_cost, updates=total_grad_updates,
            givens={
                x: train_x[index*batch_size:(index+1)*batch_size],
                pop_y: train_pop_y[index*batch_size:(index+1)*batch_size],
                type_y:train_type_y[index*batch_size:(index+1)*batch_size]
                })
    
    valid_train_func = function([index], total_drop_cost, updates=total_grad_updates,
            givens={
                x: valid_x[index*batch_size:(index+1)*batch_size],
                pop_y: valid_pop_y[index*batch_size:(index+1)*batch_size],
                type_y:valid_type_y[index*batch_size:(index+1)*batch_size]
                })
    

    test_pred_detail = function([index], total_out,
            givens={
                x:test_x[index*batch_size:(index+1)*batch_size],
                pop_y:test_pop_y[index*batch_size:(index+1)*batch_size],
                type_y:test_type_y[index*batch_size:(index+1)*batch_size]
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
    
    log_file = open(log_fn, 'w')

    print "Start to train the model....."
    
    total_score = 0.0
    while (epoch < n_epochs) and not done_loop:
        start_time = timeit.default_timer()
        epoch += 1
        costs = []
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            cost_epoch = train_func(minibatch_index)
            costs.append(cost_epoch)
            set_zero(zero_vec)
        
        # do validatiovalidn
        valid_cost = [valid_train_func(i) for i in np.random.permutation(xrange(n_valid_batches))]

        if epoch % print_freq == 0:
            # do test
            pop_preds = []
            type_preds = []
            pop_errors = []
            type_errors = []
            pop_sens = []
            type_sens = []

            for i in xrange(n_test_batches):
                test_pop_pred, test_type_pred, test_pop_error, test_type_error, test_pop_sen, test_type_sen = test_pred_detail(i)

                pop_preds.append(test_pop_pred)
                type_preds.append(test_type_pred)
                pop_errors.append(test_pop_error)
                type_errors.append(test_type_error)
                pop_sens.append(test_pop_sen)
                type_sens.append(test_type_sen)

            pop_preds = np.concatenate(pop_preds)
            type_preds = np.concatenate(type_preds)
            pop_errors = np.concatenate(pop_errors)
            type_errors = np.concatenate(type_errors)
            pop_sens = np.concatenate(pop_sens)
            type_sens = np.concatenate(type_sens)

            pop_perf = 1 - np.mean(pop_errors)
            type_perf = 1 - np.mean(type_errors)

            # dumps the predictions and the choosed sentences
            with open(os.path.join(perf_fn, "%s_%d.pop_pred" % (exp_name, epoch)), 'w') as epf:
                for p in pop_preds:
                    epf.write("%d\n" % int(p))

            with open(os.path.join(perf_fn, "%s_%d.type_pred" % (exp_name, epoch)), 'w') as epf:
                for p in type_preds:
                    epf.write("%d\n" % int(p))
            print pop_sens
            with open(os.path.join(perf_fn, "%s_%d.pop_sens" % (exp_name, epoch)), 'w') as epf:
                for s in pop_sens:
                    epf.write("%d\n" % int(s))

            with open(os.path.join(perf_fn, "%s_%d.type_sens" % (exp_name, epoch)), 'w') as epf:
                for s in type_sens:
                    epf.write("%d\n" % int(s))
            
            message = "Epoch %d test pop perf %f, type perf %f, training_cost %f" % (epoch, pop_perf, type_perf, np.mean(costs))
            print message
            log_file.write(message + "\n")
            log_file.flush()

            if (pop_perf + type_perf) > total_score:
                total_score = pop_perf + type_perf
                # save the model
                model_name = os.path.join(perf_fn, "%s_%d.best_model" % (exp_name, epoch))
                with open(model_name, 'wb') as mn:
                    for param in params:
                        cPickle.dump(param.get_value(), mn)


        end_time = timeit.default_timer()
        print "Finish one iteration using %f m" % ((end_time - start_time)/60.)

    # output the final model params
    print "Output the final model"
    model_name = os.path.join(perf_fn, "%s_%d.final_model" % (exp_name, epoch))
    with open(model_name, 'wb') as mn:
        for param in params:
            cPickle.dump(param.get_value(), mn)


    log_file.flush()
    log_file.close()


def shared_dataset(data_xyz, borrow=True):
    data_x, data_y, data_z = data_xyz
    shared_x = theano.shared(np.asarray(data_x,
        dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)
    
    shared_z = theano.shared(np.asarray(data_z,
        dtype=theano.config.floatX), borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_z, 'int32')


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
    print_freq = args.print_freq


    log_file.flush()
    log_file.close()


def shared_dataset(data_xyz, borrow=True):
    data_x, data_y, data_z = data_xyz
    shared_x = theano.shared(np.asarray(data_x,
        dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)
    
    shared_z = theano.shared(np.asarray(data_z,
        dtype=theano.config.floatX), borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_z, 'int32')


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
    print_freq = args.print_freq

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
    type_hidden_units = loc_params["type_hidden_units"]

    run_cnn(exp_name, digit_dataset, embedding,
            log_fn, perf_fn,
            emb_dm=embedding.shape[1],
            batch_size=batch_size,
            filter_hs=filter_hs,
            hidden_units=hidden_units,
            type_hidden_units=type_hidden_units,
            dropout_rate=0.5,
            shuffle_batch=True,
            n_epochs=n_epochs,
            lr_decay=0.95,
            activation=ReLU,
            sqr_norm_lim=9,
            non_static=non_static,
            print_freq=print_freq)
     
    
if __name__ == "__main__":
    main()
