#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convolution Neural Network for Event Encoding
Encoding token position, and frequency information

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
from collections import OrderedDict, Counter
import re
from nltk import word_tokenize

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
    ap.add_argument("--data_type", type=str, default="str",
            help="the input data formart: String or Json")
    return ap.parse_args()

def load_dataset(prefix, sufix):
    """Load the train/valid/test set
        prefix eg: ../data/spanish_protest
        sufix eg: pop_cat
    """
    dataset = []
    for group in ["train", "test"]:
        x_fn = "%s_%s.txt.tok" % (prefix, group)
        y_fn = "%s_%s.%s" % (prefix, group, sufix)
        xs = [l.strip().lower() for l in open(x_fn)]
        ys = [l.strip() for l in open(y_fn)]
        dataset.append((xs, ys))
        print "Load %d %s records" % (len(ys), group)
    return dataset

def doc_to_id(doc, word2id, data_type, max_len=700, padding=5):
    if data_type == "str":
        # clean the doc first, remove those sentence with less than 5
        sens = re.split("\.|\?|\|", doc.lower())
        sens = [sen.strip() for sen in sens if len(sen.strip().split(" ")) > 5]
    elif data_type == "json":
        sens = json.loads(doc)
        sens = [sen.lower() for sen in sens]
    # construct the word sentence mapping
    wid = 0
    word2sid = []
    for sid, sen in enumerate(sens):
        tokens = word_tokenize(sen)
        for token in tokens:
            if wid >= max_len:
                break
            word2sid.append((token, min(sid+1, 30)))
            wid += 1
    
    pad = padding - 1
    doc_ids = [0] * pad
    doc_sids = [0] * pad
    for w, sid in word2sid[:max_len]:
        doc_ids.append(word2id.get(w, 0))
        doc_sids.append(sid)
        
    num_suff = max([0, max_len - len(word2sid)]) + pad
    doc_ids += [0] * num_suff
    doc_sids += [0] * num_suff

    # compute the word frequency for each words
    word_count = Counter(doc_ids)
    doc_freqs = []
    for id in doc_ids:
        if id == 0:
            doc_freqs.append(0) # padding
        else:
            doc_freqs.append(word_count[id] if word_count[id] <= 20 else 20)


    return doc_ids, doc_freqs, doc_sids

def transform_dataset(dataset, word2id, class2id, data_type="str",max_len=700, padding=5):
    """Transform the dataset into digits"""
    train_set, test_set = dataset
    train_doc, train_class = train_set
    test_doc, test_class = test_set
    
    train_doc_ids, train_doc_freqs, train_doc_sids = zip(*[doc_to_id(doc, word2id, data_type, max_len, padding) for doc in train_doc])
    test_doc_ids, test_doc_freqs, test_doc_sids = zip(*[doc_to_id(doc, word2id, data_type, max_len, padding) for doc in test_doc])

    train_y = [class2id[c] for c in train_class]
    test_y = [class2id[c] for c in test_class]

    return [(train_doc_ids, train_doc_freqs, train_doc_sids, train_y), (test_doc_ids, test_doc_freqs, test_doc_sids, test_y)]

def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6,
        norm_lim=3, word_vec_name='embedding'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = [] 
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name, borrow=True)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name, borrow=True)
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
        non_static=True,
        alpha=0.0001):
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
   
    input_height = len(dataset[0][0][0]) 
    print "--input height ", input_height 
    input_width = emb_dm
    num_maps = hidden_units[0]

    ###################
    # start snippet 1 #
    ###################
    print "start to construct the model ...."
    word_x = T.matrix("word_x")
    freq_x = T.matrix("freq_x")
    pos_x = T.matrix("pos_x")

    y = T.ivector("y")

    words = shared(value=np.asarray(embedding,
        dtype=theano.config.floatX), 
        name="embedding", borrow=True)
    
    sym_dim = 20
    # the frequency embedding is 21 * 50 matrix
    freq_val = np.random.random((21, sym_dim))
    freqs = shared(value=np.asarray(freq_val, dtype=theano.config.floatX), borrow=True, name="freqs")

    # the position embedding is 31 * 50 matrix
    poss_val = np.random.random((31, sym_dim))
    poss = shared(value=np.asarray(poss_val, dtype=theano.config.floatX), borrow=True, name="poss")
    

    # define function to keep padding vector as zero
    zero_vector_tensor = T.vector()
    zero_vec = np.zeros(input_width, dtype=theano.config.floatX)
    set_zero = function([zero_vector_tensor], updates=[(words, T.set_subtensor(words[0,:], zero_vector_tensor))])

    freq_zero_tensor = T.vector()
    freq_zero_vec = np.zeros(sym_dim, dtype=theano.config.floatX)
    freq_set_zero = function([freq_zero_tensor], updates=[(freqs, T.set_subtensor(freqs[0,:], freq_zero_tensor))])

    pos_zero_tensor = T.vector()
    pos_zero_vec = np.zeros(sym_dim, dtype=theano.config.floatX)
    pos_set_zero = function([pos_zero_tensor], updates=[(poss, T.set_subtensor(poss[0,:], pos_zero_tensor))])


    word_x_emb = words[T.cast(word_x.flatten(), dtype="int32")].reshape((word_x.shape[0], 1, word_x.shape[1], emb_dm))
    freq_x_emb = freqs[T.cast(freq_x.flatten(), dtype="int32")].reshape((freq_x.shape[0], 1, freq_x.shape[1], sym_dim))
    pos_x_emb = poss[T.cast(pos_x.flatten(), dtype="int32")].reshape((pos_x.shape[0], 1, pos_x.shape[1], sym_dim))

    layer0_input = T.concatenate([word_x_emb, freq_x_emb, pos_x_emb], axis=3)

    conv_layers = []
    layer1_inputs = []
    rng = np.random.RandomState() 
    for i in xrange(len(filter_hs)):
        filter_shape = (num_maps, 1, filter_hs[i], emb_dm + sym_dim + sym_dim)
        pool_size = (input_height - filter_hs[i] + 1, 1)
        conv_layer = nn.ConvPoolLayer(rng, input=layer0_input, 
                input_shape=None,
                filter_shape=filter_shape,
                pool_size=pool_size, activation=activation)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    
    layer1_input = T.concatenate(layer1_inputs, 1)

    ##############
    # classifier #
    ##############
    print "Construct classifier ...."
    hidden_units[0] = num_maps * len(filter_hs)
    model = nn.MLPDropout(rng,
            input=layer1_input,
            layer_sizes=hidden_units,
            dropout_rates=[dropout_rate],
            activations=[activation])

    params = model.params
    for conv_layer in conv_layers:
        params += conv_layer.params

    params.append(words)
    params.append(freqs)
    params.append(poss)

    cost = model.negative_log_likelihood(y) + alpha * model.L2
    dropout_cost = model.dropout_negative_log_likelihood(y) + alpha * model.L2

    grad_updates = sgd_updates_adadelta(params, 
            dropout_cost, 
            lr_decay, 
            1e-6, 
            sqr_norm_lim)

    #####################
    # Construct Dataset #
    #####################
    print "Copy data to GPU and constrct train/valid/test func"
    
    train_word_x, train_freq_x, train_pos_x, train_y = shared_dataset(dataset[0])
    test_word_x, test_freq_x, test_pos_x, test_y = shared_dataset(dataset[1])

    n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
    n_test_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))

    #####################
    # Train model func #
    #####################
    index = T.iscalar()
    train_func = function([index], cost, updates=grad_updates,
            givens={
                word_x: train_word_x[index*batch_size:(index+1)*batch_size],
                freq_x: train_freq_x[index*batch_size:(index+1)*batch_size],
                pos_x: train_pos_x[index*batch_size:(index+1)*batch_size],
                y: train_y[index*batch_size:(index+1)*batch_size]
                })
    
    test_pred = function([index], model.preds,
            givens={
                word_x:test_word_x[index*batch_size:(index+1)*batch_size],
                freq_x:test_freq_x[index*batch_size:(index+1)*batch_size],
                pos_x:test_pos_x[index*batch_size:(index+1)*batch_size]
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

    print "Start to train the model....."
    cpu_trn_y = np.asarray(dataset[0][3])
    cpu_tst_y = np.asarray(dataset[1][3])

    def compute_score(true_list, pred_list):
        mat = np.equal(true_list, pred_list)
        score = np.mean(mat)
        return score

    while (epoch < n_epochs) and not done_loop:
        start_time = timeit.default_timer()
        epoch += 1
        costs = []
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            cost_epoch = train_func(minibatch_index)
            costs.append(cost_epoch)
            set_zero(zero_vec)
            freq_set_zero(freq_zero_vec)
            pos_set_zero(pos_zero_vec)
            

        if epoch % 5 == 0:
            # do test
            test_preds = np.concatenate([test_pred(i) for i in xrange(n_test_batches)])
            test_score = compute_score(cpu_tst_y, test_preds)
            
            with open(os.path.join(perf_fn, "%s_%d.pred" % (exp_name, epoch)), 'w') as epf:
                for p in test_preds:
                    epf.write("%d\n" % int(p))
                message = "Epoch %d test perf %f with train cost %f" % (epoch, test_score, np.mean(costs))
            print message
            log_file.write(message + "\n")
            log_file.flush()

        end_time = timeit.default_timer()
        print "Finish one iteration using %f m" % ((end_time - start_time)/60.)

    log_file.flush()
    log_file.close()


def shared_dataset(data_xy, borrow=True):
    data_word_x, data_freq_x, data_pos_x, data_y = data_xy
    shared_word_x = theano.shared(np.asarray(data_word_x,
        dtype=theano.config.floatX), borrow=borrow)
    shared_freq_x = theano.shared(np.asarray(data_freq_x,
        dtype=theano.config.floatX), borrow=borrow)
    shared_pos_x = theano.shared(np.asarray(data_pos_x, dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)
    return shared_word_x, shared_freq_x, shared_pos_x, T.cast(shared_y, 'int32')


def main():
    args = parse_args()
    prefix = args.prefix
    word2vec_file = args.word2vec
    sufix = args.sufix
    expe_name = args.exp_name
    batch_size = args.batch_size
    log_fn = args.log_fn
    perf_fn = args.perf_fn
    data_type = args.data_type

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
    max_len = args.max_len
    padding = args.padding
    digit_dataset = transform_dataset(dataset, word2id, class2id, data_type, max_len, padding)

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
            sqr_norm_lim=3,
            non_static=non_static)
     
    
if __name__ == "__main__":
    main()
