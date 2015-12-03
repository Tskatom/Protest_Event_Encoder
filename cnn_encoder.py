#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano import function, shared
import nn_layers as nn
from collections import OrderedDict
import cPickle
import argparse

"""
Implemente the CNN classifier for Population and Event Type
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"


def ReLU(x):
    y = T.maximum(0.0, x)
    return y


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return y


def Tanh(x):
    return T.tanh(x)


def Iden(x):
    return x


def train_cnn_encoder(datasets, word_embedding, input_width=64,
                      filter_hs=[3, 4, 5],
                      hidden_units=[100, 2],
                      dropout_rate=[0.5],
                      shuffle_batch=True,
                      n_epochs=100,
                      batch_size=50,
                      lr_decay=0.95,
                      activations=[ReLU],
                      sqr_norm_lim=9,
                      non_static=True):
    rng = np.random.RandomState(1234)
    input_height = len(datasets[0][0]) - 1
    filter_width = input_width
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_width))
        pool_sizes.append((input_height-filter_h+1, input_width-filter_width+1))

    parameters = [("Input Shape", input_height, input_width),
                  ("Filter Shape", filter_shapes),
                  ("Pool Sizes", pool_sizes),
                  ("dropout rate", dropout_rate),
                  ("hidden units", hidden_units),
                  ("shuffle_batch", shuffle_batch),
                  ("n_epochs", n_epochs),
                  ("batch size", batch_size)]
    print parameters

    # construct the model
    index = T.iscalar()
    x = T.matrix("x")
    y = T.ivector("y")
    words = shared(value=word_embedding, name="embedding")

    zero_vector_tensor = T.vector()
    zero_vec = np.zeros(input_width)
    set_zero = function([zero_vector_tensor], updates=[(words, T.set_subtensor(words[0,:], zero_vector_tensor))])

    layer0_input = words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0],1,x.shape[1],words.shape[1]))

    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = nn.ConvPoolLayer(rng, input=layer0_input,
            input_shape=(batch_size, 1, input_height, input_width),
            filter_shape=filter_shape,
            pool_size=pool_size, activation=ReLU)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    layer1_input = T.concatenate(layer1_inputs, 1)

    hidden_units[0] = feature_maps * len(filter_hs)

    classifier = nn.MLPDropout(rng,
        input=layer1_input,
        layer_sizes=hidden_units,
        dropout_rates=dropout_rate,
        activations=activations)

    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params

    if non_static:
        params.append(words)


    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    np.random.seed(1234)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
    else:
        new_data = datasets[0]

    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))

    # divide the train set intp train/val sets
    test_set_x = datasets[1][:,:input_height]
    test_set_y = np.asarray(datasets[1][:,-1], "int32")

    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x, train_set_y = shared_dataset((train_set[:,:input_height],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:input_height],val_set[:,-1]))

    n_val_batches = n_batches - n_train_batches
    val_model = function([index], classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]
        })

    test_model = function([index], classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]
        })

    train_model = function([index], cost, updates=grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        })

    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, input_height, input_width))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = function([x, y], test_error)

    # start to training the model
    print "Start training the model...."
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    cost_epoch = 0
    while(epoch < n_epochs):
        epoch += 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1 - np.mean(val_losses)
        print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))

        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_losses = test_model_all(test_set_x, test_set_y)
            test_perf = 1 - test_losses
            print "Test Performance %f under Current Best Valid perf %f" % (test_perf, val_perf)

    return test_perf


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6,norm_lim=9, word_vec_name="embedding"):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})

    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step

        if (param.get_value(borrow=True).ndim == 2) and (param.name != "embedding"):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def doc_to_id(tokens, word2id, max_l=5687, filter_h=5):
    pad = filter_h - 1
    doc_ids = [0] * pad
    for word in tokens:
        if word in word2id:
            doc_ids.append(word2id[word])
        else:
            doc_ids.append(1)
    # add padding to the end of the doc
    doc_ids += [0] * (filter_h - 1 + max_l - len(tokens))
    return doc_ids


def make_data_cv(docs, i, word2id, max_l=5687, filter_h=5):
    """Construct the train/test set"""
    train, test = [], []
    for doc in docs:
        cv = doc["cv"]
        tokens = doc["tokens"]
        doc_ids = doc_to_id(tokens, word2id, max_l, filter_h)
        doc_ids.append(doc["pop"]) # population type
        if cv == i:
            test.append(doc_ids)
        else:
            train.append(doc_ids)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand", action="store_true", 
        help="Use random word vector")
    ap.add_argument("--word2vec", action="store_true",
        help="Use pretrained word2vec")
    ap.add_argument("--static", action="store_true",
        help="Don't update the word2vec")
    return ap.parse_args()


if __name__ == "__main__":
    print "Start Loading the data"
    data = cPickle.load(open("./data/experiment_dataset1"))
    docs, type2id, pip2id, word2id, embedding, rand_embedding = data

    args = parse_args()
    if args.rand:
        word2vec = rand_embedding
    elif args.word2vec:
        word2vec = embedding

    non_static = True
    if args.static:
        non_static = False

    folders = range(0, 10)
    results = []
    for i in folders:
        datasets = make_data_cv(docs, i, word2id, max_l=5687, filter_h=5)
        performance = train_cnn_encoder(datasets, word2vec, input_width=64,
                      filter_hs=[3, 4, 5],
                      hidden_units=[100, 2],
                      dropout_rate=[0.5],
                      shuffle_batch=True,
                      n_epochs=100,
                      batch_size=50,
                      lr_decay=0.95,
                      activations=[ReLU],
                      sqr_norm_lim=9,
                      non_static=non_static)
        print "CV %d Performance %f" % (i, performance)
        results.append(performance)
    print "Average Performance %f" % np.mean(results)
