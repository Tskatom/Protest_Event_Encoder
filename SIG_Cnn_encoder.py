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
import timeit

#theano.config.exception_verbosity = 'high'
#theano.config.optimizer = 'None'

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
                      non_static=True,
                      pop=True,
                      etype=True):

    start_time = timeit.default_timer()

    rng = np.random.RandomState(1234)
    input_height = len(datasets[0][0]) - 2
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
    zero_vec = np.zeros(input_width, dtype=theano.config.floatX)
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

    ###################
    # Population Task #
    ###################
    hidden_units[0] = feature_maps * len(filter_hs)

    pop_classifier = nn.MLPDropout(rng,
        input=layer1_input,
        layer_sizes=hidden_units,
        dropout_rates=dropout_rate,
        activations=activations)

    pop_params = pop_classifier.params
    for conv_layer in conv_layers:
        pop_params += conv_layer.params

    if non_static:
        pop_params.append(words)


    pop_cost = pop_classifier.negative_log_likelihood(y)
    pop_dropout_cost = pop_classifier.dropout_negative_log_likelihood(y)

    pop_grad_updates = sgd_updates_adadelta(pop_params, pop_dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    ###################
    # EventType Task #
    ###################
    event_type_hidden_units = [feature_maps * len(filter_hs), 12]
    type_classifier = nn.MLPDropout(rng,
        input=layer1_input,
        layer_sizes=event_type_hidden_units,
        dropout_rates=dropout_rate,
        activations=activations)
    type_params = type_classifier.params
    for conv_layer in conv_layers:
        type_params += conv_layer.params

    if non_static:
        type_params.append(words)

    type_cost = type_classifier.negative_log_likelihood(y)
    type_dropout_cost = type_classifier.dropout_negative_log_likelihood(y)
    type_grad_updates = sgd_updates_adadelta(type_params, type_dropout_cost, lr_decay, 1e-6, sqr_norm_lim)


    ######################
    # Construct Data Set #
    ######################

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
    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        test_set = np.random.permutation(datasets[1])
        extra_data = test_set[:extra_data_num]
        new_test_data = np.append(datasets[1], extra_data, axis=0)
    else:
        new_test_data = datasets[1]
    test_set_x = new_test_data[:,:input_height]
    test_set_pop_y = np.asarray(new_test_data[:,-2], "int32")
    test_set_type_y = np.asarray(new_test_data[:,-1], "int32")


    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]

    print train_set[:,-1]
    borrow = True
    train_set_x = theano.shared(np.asarray(train_set[:,:input_height],dtype=theano.config.floatX),borrow=borrow)
    train_set_pop_y = T.cast(theano.shared(np.asarray(train_set[:,-2], dtype=theano.config.floatX), borrow=borrow), 'int32')
    train_set_type_y = T.cast(theano.shared(np.asarray(train_set[:,-1], dtype=theano.config.floatX), borrow=borrow), 'int32')

    val_set_x = theano.shared(np.asarray(val_set[:,:input_height],dtype=theano.config.floatX),borrow=borrow)
    val_set_pop_y = T.cast(theano.shared(np.asarray(val_set[:,-2], dtype=theano.config.floatX), borrow=borrow), 'int32')
    val_set_type_y = T.cast(theano.shared(np.asarray(val_set[:,-1], dtype=theano.config.floatX), borrow=borrow), 'int32')


    n_val_batches = n_batches - n_train_batches
    n_test_batches = test_set_x.shape[0] / batch_size
    print 'n_test_batches: %d' % n_test_batches
    # transform the data into shared varibale for GPU computing
    test_set_x = theano.shared(np.asarray(test_set_x, dtype=theano.config.floatX), borrow=borrow)
    test_set_pop_y = theano.shared(test_set_pop_y, borrow=True)
    test_set_type_y = theano.shared(test_set_type_y, borrow=True)

    ####################
    # Train Model Func #
    ####################
    # population model
    val_pop_model = function([index], pop_classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_pop_y[index * batch_size: (index + 1) * batch_size]
        })

    test_pop_model = function([index], pop_classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_pop_y[index * batch_size: (index + 1) * batch_size]
        })

    real_test_pop_model = function([index], pop_classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_pop_y[index * batch_size: (index + 1) * batch_size]
        })

    train_pop_model = function([index], pop_cost, updates=pop_grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_pop_y[index*batch_size:(index+1)*batch_size]
        })

    # event type model
    val_type_model = function([index], type_classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_type_y[index * batch_size: (index + 1) * batch_size]
        })

    test_type_model = function([index], type_classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_type_y[index * batch_size: (index + 1) * batch_size]
        })

    real_test_type_model = function([index], type_classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_type_y[index * batch_size: (index + 1) * batch_size]
        })

    train_type_model = function([index], type_cost, updates=type_grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_type_y[index*batch_size:(index+1)*batch_size]
        })

    """
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, input_height, input_width))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    test_layer1_input = T.concatenate(test_pred_layers, 1)

    test_pop_y_pred = pop_classifier.predict(test_layer1_input)
    test_pop_error = T.mean(T.neq(test_pop_y_pred, y))
    test_pop_model_all = function([x, y], test_pop_error)

    test_type_y_pred = type_classifier.predict(test_layer1_input)
    test_type_error = T.mean(T.neq(test_type_y_pred, y))
    test_type_model_all = function([x, y], test_type_error)
    """
    # start to training the model
    print "Start training the model...."
    epoch = 0
    best_pop_val_perf = 0
    best_type_val_perf = 0
    test_pop_perf = 0.
    test_type_perf = 0.

    while(epoch < n_epochs):
        epoch += 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                if pop:
                    cost_pop_epoch = train_pop_model(minibatch_index)
                    set_zero(zero_vec)
                if etype:
                    cost_type_epoch = train_type_model(minibatch_index)
                    set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                if pop:
                    cost_pop_epoch = train_pop_model(minibatch_index)
                    set_zero(zero_vec)
                if etype:
                    cost_type_epoch = train_type_model(minibatch_index)
                    set_zero(zero_vec)
        if pop:
            train_pop_losses = [test_pop_model(i) for i in xrange(n_train_batches)]
            train_pop_perf = 1 - np.mean(train_pop_losses)
            val_pop_losses = [val_pop_model(i) for i in xrange(n_val_batches)]
            val_pop_perf = 1 - np.mean(val_pop_losses)
            print('epoch %i, train pop perf %f %%, val pop perf %f' % (epoch, train_pop_perf * 100., val_pop_perf*100.))
            if val_pop_perf >= best_pop_val_perf:
                best_pop_val_perf = val_pop_perf
                #test_pop_losses = test_pop_model_all(test_set_x, test_set_pop_y)
                test_pop_losses = [real_test_pop_model(i) for i in xrange(n_test_batches)]
                test_pop_perf = 1 - np.mean(test_pop_losses)
                print "Test POP Performance %f under Current Best Valid perf %f" % (test_pop_perf, val_pop_perf)

        if etype:
            train_type_losses = [test_type_model(i) for i in xrange(n_train_batches)]
            train_type_perf = 1 - np.mean(train_type_losses)
            val_type_losses = [val_type_model(i) for i in xrange(n_val_batches)]
            val_type_perf = 1 - np.mean(val_type_losses)

            print('epoch %i, train type perf %f %%, val type perf %f' % (epoch, train_type_perf * 100., val_type_perf*100.))
            if val_type_perf >= best_type_val_perf:
                best_type_val_perf = val_type_perf
                #test_type_losses = test_type_model_all(test_set_x, test_set_type_y)
                test_type_losses = [real_test_type_model(i) for i in xrange(n_test_batches)]
                test_type_perf = 1 - np.mean(test_type_losses)
                print "Test Type Performance %f under Current Best Valid perf %f" % (test_type_perf, val_type_perf)

        end_time = timeit.default_timer()
        print "Epoch %d finish take time %fm " % (epoch, (end_time - start_time)/60.)
        start_time = timeit.default_timer()
    # save the model
    if pop:
        with open('./data/pop_model.pkl', 'wb') as pop_f:
            for param in pop_params:
                cPickle.dump(param.get_value(), pop_f)
    if etype:
        with open('./data/type_model.pkl', 'wb') as type_f:
            for param in type_params:
                cPickle.dump(param.get_value(), type_f)
    return test_pop_perf, test_type_perf


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

def doc_to_id_cut(tokens, word2id, max_l=1000, filter_h=5):
    pad = filter_h - 1
    doc_ids = [0] * pad
    for i, word in enumerate(tokens):
        if i < max_l:
            if word in word2id:
                doc_ids.append(word2id[word])
            else:
                doc_ids.append(1)
    num_suff = max([0, max_l - len(tokens)]) + pad
    doc_ids += [0] * num_suff
    return doc_ids

def make_data_cv(docs, i, word2id, max_l=1000, filter_h=5):
    """Construct the train/test set"""
    train, test = [], []
    for doc in docs:
        cv = doc["cv"]
        tokens = doc["tokens"]
        doc_ids = doc_to_id_cut(tokens, word2id, max_l, filter_h)
        #doc_ids = doc_to_id(tokens, word2id, max_l, filter_h)
        doc_ids.append(doc["pop"]) # population type
        doc_ids.append(doc["etype"])
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
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--max_epochs", type=int, help='the max iterlations',
            default=200)
    ap.add_argument("--pop", action="store_true")
    ap.add_argument("--etype", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    print "Start Loading the data"
    data = cPickle.load(open("./data/experiment_dataset2"))
    docs, type2id, pip2id, word2id, embedding, rand_embedding = data

    args = parse_args()
    if args.test:
        docs = docs[:10000]

    if args.rand:
        word2vec = rand_embedding
    elif args.word2vec:
        word2vec = embedding

    word2vec = np.asarray(word2vec, dtype=theano.config.floatX)

    non_static = True
    if args.static:
        non_static = False

    folders = range(0, 10)
    results = []
    for i in folders:
        datasets = make_data_cv(docs, i, word2id, max_l=1000, filter_h=5)
        pop_performance, type_performance = train_cnn_encoder(datasets, word2vec, input_width=64,
                      filter_hs=[3, 4, 5],
                      hidden_units=[100, 13],
                      dropout_rate=[0.5],
                      shuffle_batch=True,
                      n_epochs=args.max_epochs,
                      batch_size=200,
                      lr_decay=0.95,
                      activations=[ReLU],
                      sqr_norm_lim=9,
                      non_static=non_static,
                      pop=args.pop,
                      etype=args.etype)
        print "CV %d Pop Performance %f, Type performance %f" % (i, pop_performance, type_performance)
        results.append((pop_performance, type_performance))
    print "Average pop_Performance %f type_performance %f " % (np.mean([r[0] for r in result]), np.mean([r[1] for r in results]))
