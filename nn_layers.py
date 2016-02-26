#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import theano
from theano import shared
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from collections import OrderedDict
import json
from nltk import word_tokenize

"""
Implement the Neural Network Layers
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6,
        norm_lim=9, word_vec_name='embedding'):
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

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = shared(np.asarray(data_x, dtype=theano.config.floatX),borrow=borrow)
    shared_y = shared(np.asarray(data_y, dtype=theano.config.floatX),borrow=borrow)

    return shared_x, T.cast(shared_y, "int32")

def load_event_dataset(prefix, sufix):
    """Load the event dataset for binary classification"""
    dataset = []
    for group in ["train", "test"]:
        x_fn = "%s_%s.txt.tok" % (prefix, group)
        y_fn = "%s_%s.%s" % (prefix, group, sufix)

        xs = [l.strip() for l in open(x_fn)]
        ys = [l.strip() for l in open(y_fn)]

        dataset.append((xs, ys))
    return dataset

def split_doc2sen(doc, word2id, data_type, max_sens, max_words, padding):
    """
        split the document into sentences
        replace the word by id
    """
    if data_type == "json":
        sens = [sen.lower() for sen in json.loads(doc)]
    elif data_type == "str":
        sens = re.split("\.|\?|\|", doc.lower()) 
        sens = [sen for sen in sens if len(sen.strip().split(" ")) > 5]

    pad = padding
    sens_pad = []
    for j, sen in enumerate(sens[:max_sens]):
        sen_ids = [0] * pad
        tokens = word_tokenize(sen)
        for w in tokens[:max_words]:
            sen_ids.append(word2id.get(w.encode('utf-8'), 1))
        num_suff = max(0, max_words - len(tokens)) + pad
        sen_ids += [0] * num_suff
        sens_pad.append(sen_ids)

    # add more padding sentences
    num_suff = max(0, max_sens - len(sens))
    for i in range(0, num_suff):
        sen_ids = [0] * len(sens_pad[0])
        sens_pad.append(sen_ids)

    return sens_pad

def transform_event_dataset(dataset, word2id, class2id, data_type, max_sens, max_words, padding):
    train_set, test_set = dataset
    train_docs, train_labels = train_set
    test_docs, test_labels = test_set

    train_doc_ids = [split_doc2sen(doc, word2id, data_type, max_sens, max_words, padding) for doc in train_docs]

    test_doc_ids = [split_doc2sen(doc, word2id, data_type, max_sens, max_words, padding) for doc in test_docs]
    
    train_y = [class2id[c] for c in train_labels]
    test_y = [class2id[c] for c in test_labels]

    return [(train_doc_ids, train_y), (test_doc_ids, test_y)]


class HiddenLayer(object):
    """ Hidden Layer class"""
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):
        """
            :type rng: numpy.random.randomstate
            :param rng: random number generator

            :type input: theano.tensor.fvector
            :param input: input vector to the hidden layer

            :type n_in: int
            :param n_in: dimention of input

            :type n_out: int
            :param n_out: number of hiddent units

            :type activation: function
            :param activation: non-linear activation function

            :type W: None or theano shared variable
            :param W: the weights for hidden Layer

            :type b: None or theano shared variable
            :param b: the bias for hidden Layer
        """
        self.input = input,
        self.activation = activation

        if W is None:
            if self.activation.func_name == "ReLU":
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in,
                                                                       n_out)),
                                      dtype=theano.config.floatX)
            else:
                w_bound = np.sqrt(6./(n_in + n_out))
                W_values = np.asarray(rng.uniform(-w_bound,
                                                  w_bound,
                                                  size=(n_in, n_out)),
                                      dtype=theano.config.floatX)
            self.W = shared(value=W_values, borrow=True, name="hidden_W")
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = shared(value=b_values, borrow=True, name="hidden_b")
        else:
            self.b = b

        pre_activation = T.dot(input, self.W) + self.b

        self.output = self.activation(pre_activation)
        self.params = [self.W, self.b]
        self.L2 = T.sum(self.W ** 2)
        self.L1 = T.sum(abs(self.W))


class MLP(object):
    """
        Multi-Layer Neural Network
    """
    def __init__(self, rng, input, n_in, n_hidden, activation, n_out):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=activation)

        # construct the LogisticRegression Layer
        self.logisticLayer = LogisticRegressionLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
            )

        # cost function
        self.negative_log_likelihood = self.logisticLayer.negative_log_likelihood
        self.errors = self.logisticLayer.errors

        self.params = self.hiddenLayer.params + self.logisticLayer.params


class LogisticRegressionLayer(object):
    """
        Loginstic Regression Layer for classfication
    """
    def __init__(self, input, n_in, n_out, W=None, b=None):
        """
        :type input: theano.tensor.TensorType
        :param input: the input vector to the classifier

        :type n_in: int
        :param n_in: the dimention of input vector

        :type n_out: int
        :param n_out: the number of output classes

        :type W: theano shared variable or None
        :param W: the weights of LogisticRegression layer

        :type b: theano shared variable
        :param b: the bias vector of LogisticRegression layer
        """
        self.input = input
        if W is None:
            W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            self.W = shared(value=W_values, borrow=True, name="logis_W")
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = shared(value=b_values, borrow=True, name="logis_b")
        else:
            self.b = b

        pre_activation = T.dot(input, self.W) + self.b

        self.p_y_given_x = T.nnet.softmax(pre_activation)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.L2 = T.sum(self.W ** 2)
        self.L1 = T.sum(abs(self.W))

    def negative_log_likelihood(self, y):
        """
        the cost function of LogisticRegression layer
        :type y: theano.tensor.TensorType
        :param y: the vector which contains the correct
                  labels of the input samples
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        The error rate of the LogisticRegressionLayer
        :type y: theano.tensor.TensorType
        :param y: the vector which contains the correct
                  labels of the input samples
        """
        return T.mean(T.neq(self.y_pred, y))

def k_max(x, k):
    sort_idx = T.argsort(x, axis=2)
    k_max_ids = sort_idx[:,:,-k:,:]
    dim0, dim1, dim2, dim3 = k_max_ids.shape
    batchids = T.repeat(T.arange(dim0), dim1*dim2*dim3)
    mapids = T.repeat(T.arange(dim1), dim2*dim3).reshape((1, dim2*dim3))
    mapids = T.repeat(mapids, dim0, axis=0).flatten()
    rowids = k_max_ids.flatten()
    colids = T.arange(dim3).reshape((1, dim3))
    colids = T.repeat(colids, dim0*dim1*dim2, axis=0).flatten()
    sig_mask = T.zeros_like(x)
    sig_mask = T.set_subtensor(sig_mask[batchids, mapids, rowids, colids], 1)
    result = sig_mask * x
    return result


class ConvPoolLayer(object):
    """Convolution and Max Pool Layer"""
    def __init__(self, rng, input, filter_shape, input_shape,
                 pool_size, activation, k=1, W=None, b=None):
        """
        :type rng: numpy.random.randomstate
        :param rng: the random number generator

        :type input: theano.tensor.TensorType
        :param input: input tensor

        :type filter_shape: list of int with length 4
        :param filter_shape: (number of filters, number of input feature maps,
                              filter height, filter width)
        :type input_shape: list of int with length 4
        :param input_shape: (batch_size, number input feature maps,
                             doc height[the word embedding dimention],
                             doc width[length of doc])

        :type pool_size: list of int with length 2
        :param pool_size: the shape of max pool

        :type activation: function
        :param activation: the non-linear activation function
        """
        self.input = input
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.activation = activation

        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (filter_shape[0] *
                   np.prod(filter_shape[2:]))/np.prod(pool_size)

        if self.activation.func_name == "ReLU":
            W_values = np.asarray(rng.uniform(low=-0.01,
                                              high=0.01, size=filter_shape),
                                  dtype=theano.config.floatX)
        else:
            W_bound = np.sqrt(6./(fan_in + fan_out))
            W_values = np.asarray(rng.uniform(low=-W_bound,
                                              high=W_bound,
                                              size=filter_shape),
                                  dtype=theano.config.floatX)
        if W is None:
            self.W = shared(value=W_values, borrow=True, name="conv_W")
        else:
            self.W = W
        
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = shared(value=b_values, borrow=True, name="conv_b")
        else:
            self.b = b

        conv_out = conv.conv2d(input=self.input,
                               filters=self.W,
                               filter_shape=self.filter_shape,
                               image_shape=self.input_shape)
        act_conv_out = self.activation(conv_out +
                                       self.b.dimshuffle('x', 0, 'x', 'x'))
        if k == 1: # normal max pooling
            pool_out = downsample.max_pool_2d(input=act_conv_out,
                                            ds=self.pool_size,
                                            ignore_border=True)
        elif k > 1: # get top k value for each featue map
            pool_out = k_max(act_conv_out, k)
            

        self.output = pool_out
        self.output_index = T.argmax(act_conv_out)
        self.params = [self.W, self.b]
        self.L2 = T.sum(self.W ** 2)
        self.L1 = T.sum(abs(self.W))

    def predict(self, new_data, batch_size):
        image_shape = (batch_size, 1, self.input_shape[2], self.input_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W,
                               filter_shape=self.filter_shape,
                               image_shape=image_shape)
        act_conv_out = self.activation(conv_out +
                                       self.b.dimshuffle('x', 0, 'x', 'x'))
        pool_out = downsample.max_pool_2d(input=act_conv_out,
                                          ds=self.pool_size,
                                          ignore_border=True)
        return pool_out


def dropout_from_layer(rng, layer, p):
    """General frunction for Dropout Layer
    :type rng: numpy.random.randomstate
    :param rng: random number generator

    :type layer: theano.tensor
    :param layer: the output of the Neural Network Layer

    :type p: float
    :param p: the probability to drop out the units in the output
    """

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in,
            n_out=n_out, activation=activation,
            W=W, b=b
            )
        self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(object):
    """A multi Layer Neural Network with dropout"""
    def __init__(self, rng, input, layer_sizes, dropout_rates, activations,
            Ws=None, bs=None):
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        if Ws is None:
            Ws = [None] * len(self.weight_matrix_sizes)
        else:
            assert len(Ws) == len(self.weight_matrix_sizes)

        if bs is None:
            bs = [None] * len(self.weight_matrix_sizes)
        else:
            assert len(bs) == len(self.weight_matrix_sizes)

        next_layer_input = input
        next_dropout_layer_input = dropout_from_layer(rng,
                                                      input,
                                                      p=dropout_rates[0])
        L2s = []
        L1s = []
        layer_count = 0
        for idx, ns in enumerate(self.weight_matrix_sizes[:-1]):
            n_in, n_out = ns
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                input=next_dropout_layer_input,
                activation=self.activations[layer_count],
                n_in=n_in,
                n_out=n_out,
                dropout_rate=dropout_rates[layer_count],
                W=Ws[idx],
                b=bs[idx])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output
            L2s.append(next_dropout_layer.L2)
            L1s.append(next_dropout_layer.L1)

            # resuse the parameters from the dropout layer here
            next_layer = HiddenLayer(rng=rng,
                input=next_layer_input,
                activation=self.activations[layer_count],
                W=next_dropout_layer.W * (1 - dropout_rates[layer_count]),
                b=next_dropout_layer.b,
                n_in=n_in,
                n_out=n_out)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_count += 1

        # construct final classification Layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegressionLayer(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out, W=Ws[-1], b=bs[-1])
        self.dropout_layers.append(dropout_output_layer)
        L2s.append(dropout_output_layer.L2)
        L1s.append(dropout_output_layer.L1)

        # reuse the parameters again
        output_layer = LogisticRegressionLayer(
            input=next_layer_input,
            W=dropout_output_layer.W,
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.preds = self.layers[-1].y_pred
        self.L2 = T.sum(L2s)
        self.L1 = T.sum(L1s)

        # drop out params
        self.params = [param for layer in self.dropout_layers
                       for param in layer.params]

    def predict(self, newdata):
        next_layer_input = newdata
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,
                                                             layer.W) +
                                                       layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) +
                                             layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred

    def predict_p(self, newdata):
        next_layer_input = newdata
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,
                                                             layer.W) +
                                                       layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) +
                                             layer.b)
        return p_y_given_x

def optimizer(cost, params, learning_rate, eps=0.95, rho=1e-6, momentum=0.95, method='sgd'):
    updates = OrderedDict()
    if(method == 'sgd-memotum'):
        print "Using sgd-memotum"
        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            gparam = T.grad(cost, param)
            updates[param] = param - learning_rate*(gparam / T.sqrt(param_update + eps))
            updates[param_update] = param_update + (gparam ** 2)
    elif (method == 'adagrad'):
        print "Using adagrad"

        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            gparam = T.grad(cost, param)
            param_update_u = param_update + (gparam ** 2)
            updates[param] = param - learning_rate*(gparam / T.sqrt(param_update_u + eps))
            updates[param_update] = param_update_u
    elif(method == 'adadelta'):
        print "Using adadelta"
        for param in params:
            param_update_1 = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            param_update_2 = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            gparam = T.grad(cost, param)
            param_update_1_u = rho*param_update_1+(1. - rho)*(gparam ** 2)
            dparam = -T.sqrt((param_update_2 + eps) / (param_update_1_u + eps)) * gparam
            updates[param] = param+dparam
            updates[param_update_1] = param_update_1_u
            updates[param_update_2] = rho*param_update_2+(1. - rho)*(dparam ** 2)
    else:
        print "Using normal method"
        for param in params:
            updates[param] = param - learning_rate*T.grad(cost, param)
    return updates

