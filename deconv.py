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
from SIG_Cnn_encoder import make_data_cv

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

def construct_model(params, datasets, filter_hs=[3,4,5],batch_size=200):
    rng = np.random.RandomState(1234)
    input_height = len(datasets[0][0])
    input_width = params["embedding"].shape[1]
    filter_shapes = [p[0].shape for p in params["convs"]]
    pool_sizes = [(input_height-s[2] + 1, input_width -s[3] + 1) for s in filter_shapes]

    param_sizes = {
            "input_height": input_height,
            "input_width": input_width,
            "filter_shapes": filter_shapes,
            "pool_sizes": pool_sizes
            }

    print "Param sizes: ", param_sizes
    index = T.iscalar()
    x = T.matrix('x')
    y = T.ivector('y')
   
    print '....Construct model'
    word_embedding = params["embedding"]
    words = shared(word_embedding, name='embedding')
    layer0_input = words[T.cast(x.flatten(), dtype="int32")].reshape(\
            (x.shape[0], 1, x.shape[1], words.shape[1]))
    # construct layers
    conv_layers = []
    conv_params = params["convs"]
    layer1_inputs = []
    for i, filter_h in enumerate(filter_hs):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_W = shared(value=np.asarray(conv_params[i][0], dtype=theano.config.floatX), borrow=True, name='conv_W')
        conv_b = shared(value=np.asarray(conv_params[i][1], dtype=theano.config.floatX), borrow=True, name='conv_b')
        conv_layer = nn.ConvPoolLayer(rng, 
                input=layer0_input,
                input_shape=(batch_size, 1, input_height, input_width),
                filter_shape=filter_shape,
                pool_size=pool_size,
                activation=ReLU,
                conv_W,
                conv_b
                )
        conv_layers.append(conv_layer)
        layer1_input = conv_layer.output.flatten(2)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, 1)

    # population classifier
    pop_hidden_units = [300, 13]
    clf_w, clf_b = params["clf"]
    Ws = [shared(value=np.asarray(clf_w, dtype=theano.config.floatX), borrow=True, name='logis_w')]
    bs = [shared(value=np.asarray(clf_b, dtype=theano.config.floatX), borrow=True, name='logis_b')]
    
    pop_classifier = nn.MLPDropout(rng,
            input=layer1_input,
            layer_sizes=pop_hidden_units,
            dropout_rates=[0.5],
            activations=[ReLU],
            Ws=Ws,
            bs=bs)

    pop_loss = pop_classifier.errors(y)

    # construct data set
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
    else:
        new_data = dataset[0]

    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    train_set = new_data[:n_train_batches*batch_size,:]
    train_set_x = theano.shared(np.asarray(train_set[:,:input_height],dtype=theano.config.floatX),borrow=True)
    train_set_pop_y = theano.shared(np.asarray(train_set[:,-2], dtype=theano.config.floatX),borrow=True)
    
    print '...construct test function'
    test_fn = theano.function(
            inputs=[index],
            outputs=pos_loss,
            givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_pop_y[index*batch_size:(index+1)*batch_size]
                }
            )

    pop_losses = [test_fn(i) for i in xrange(n_train_batches)]
    pop_train_perf = 1 - np.mean(pop_losses)
    print "Population Train Performance %f" % pop_train_perf


def check_tain_error():
    print '...load the expriment data set'
    data = cPickle('./data/experiment_dataset2')
    docs, type2id, pop2id, word2id, embedding, rand_embedding = data
    
    print '... construct the train/valid/test set'
    test_docs = docs[:10000]
    datasets = make_data_cv(test_docs, 0, word2id, max_l=1000, filter_h=5)

    print '....Load model parameters'
    # load the trained parameters
    params= load_model('./pop_model.pkl')

    print '....start test the model'
    # construct the model
    construct_model(params, datasets, filter_hs=[3,4,5], batch_size=200)


if __name__ == "__main__":
    check_train_error()
