#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import theano
from theano import function, shared
import theano.tensor as T
import theano.typed_list
import numpy as np
from theano.printing import Print
from util import dataset

class DocumentLayer(object):
    """
    Layer take the document as input and out the list of sentences representation
    """
    def __init__(self, rng, input, vocab_size, embed_dm, 
            conv_layer_n, n_kerns, filter_widths, ks, activation,
            embedding=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random state generator used to initiate the parameter matrix

        :type input: theano.TypedList
        :param input: document representation, |input| is the number of sentence,
        |input[i]| is the number of words in the sentence

        :type vocab_size: int
        :param vocab_size: the size of the vocab

        :type embed_dm: int
        :param embed_dm: dimentionality of the word embedding

        :type conv_layer_n: int
        :param conv_layer_n: the number of convolution layers

        :type n_kerns: list of int
        :paramm n_kerns: number of feature map in each convolution layer

        :type filter_widths: list of int
        :param filter_widths: width of filter in each layer

        :type ks: list of int
        :param ks: the k valuf of k-max pooling

        :type activation: Theano.function
        :param activation: the name of non-linear activation func

        :type embedding: theano.tensor.TensorType
        :param embedding: pretrained word embedding
        """
        

        if embedding:
            # using the pretrained word embedding
            assert embedding.get_value().shape == (vocab_size, embed_dm), "%r != %r" % (
                    embedding.get_value().shape,
                    (vocab_size, embed_dm)
                    )
            self.embedding = embedding
        else:
            # initialize the word embedding
            embedding_val = np.asarray(
                    rng.normal(0, 0.05, size=(vocab_size, embed_dm)),
                    dtype=theano.config.floatX
                    )
            embedding_val[vocab_size -1, :] = 0.0 # initiate <PADDING> character as 0
            self.embedding = shared(np.asarray(embedding_val, 
                dtype=theano.config.floatX),
                borrow=True
                )

        self.embed_dm = embed_dm
        self.input = input
        

        # construct first layer parameters
        filter_shape0 = (n_kerns[0], 1, 1, filter_widths[0])
        k0 = ks[0]
        
        fan_in = np.prod(filter_shape0[1:])
        fan_out = filter_shape0[0] * np.prod(filter_shape0[2:])
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W0 = shared(
                np.asarray(
                rng.uniform(-w_bound, w_bound, size=filter_shape0),
                dtype=theano.config.floatX),
                borrow=True
                )
        
        b_val = np.zeros((filter_shape0[0],), dtype=theano.config.floatX)
        self.b0 = shared(b_val, borrow=True)

        # construct second layer parameters       
        filter_shape1 = (n_kerns[1], n_kerns[0], 1, filter_widths[1])
        k1 = ks[1]
        fan_in = np.prod(filter_shape1[1:])
        fan_out = filter_shape1[0] * np.prod(filter_shape1[2:])
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W1 = shared(
                np.asarray(
                rng.uniform(-w_bound, w_bound, size=filter_shape1),
                dtype=theano.config.floatX),
                borrow=True
                )
        
        b_val = np.zeros((filter_shape1[0],), dtype=theano.config.floatX)
        self.b1 = shared(b_val, borrow=True)

        
        layer0 = ConvFoldingPoolLayer(rng, filter_shape0, k0, activation, self.W0, self.b0)
        layer1 = ConvFoldingPoolLayer(rng, filter_shape1, k1, activation, self.W1, self.b1)
        def generate_sen_rep(idx, sen_list):
            sen_word_ids = sen_list[idx]
            sentence_matrix = self.embedding[sen_word_ids].reshape((1, 1, 
                    self.embed_dm, sen_word_ids.shape[0]))
            layer0_output = layer0.output(sentence_matrix)
            layer1_output = layer1.output(layer0_output)
            
            return layer1_output.flatten(1)

        doc = self.input
        num_sens = theano.typed_list.length(doc)
        output, _ = theano.scan(fn=generate_sen_rep,
                non_sequences=[doc],
                sequences=[T.arange(num_sens, dtype='int64')])

        # the output is the list of sentence representation
        self.output = output[:]
        print "-----------DocLayer output", self.output
        self.params = [self.embedding, self.W0, self.b0, self.W1, self.b1]
            

class ConvFoldingPoolLayer(object):
    """Convolution Folding Pool Layer"""
    def __init__(self, rng, filter_shape, k, activation, W=None, b=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: random number generator used for initiate the parameter

        :type filter_shape: tuple or list with 4 int
        :param filter_shape: (number of feature maps in current layer, 
        number of feature maps in last layer, filter height, filter width)

        :type k: int
        :param k: k value for k-max pool
        
        :type activation: theano.function
        :param activation: the non-linear activation 

        :type W: theano.tensor.tensor4
        :param W: tensor4, the kernel weight matrix for filter
        (num of filter, num input feature maps, filter height, filter width)

        :type b: theano.tensor.vectir
        :param b: filter bias, dim (filter number)
        """

        if W is not None:
            self.W = W
        else:
            # initiate W using sqrt(6/(fan_in + fan_out))
            fan_in = np.prod(filter_shape[1:])
            fan_out = filter_shape[0] * np.prod(filter_shape[2:])
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = shared(
                    np.asarray(
                    rng.uniform(-w_bound, w_bound, size=filter_shape),
                    dtype=theano.config.floatX),
                    borrow=True
                    )
        
        if b is not None:
            self.b = b
        else:
            b_val = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = shared(b_val, borrow=True)
        
        self.k = k
        self.shapes = filter_shape
        self.params = [self.W, self.b]
        self.activation = activation

    def fold(self, x):
        """fold along dimention
        :type x: theano.tensor.tensor4
        :param x: output of conv function
        """
        return (x[:,:,T.arange(0, x.shape[2], 2)] + 
                x[:, :, T.arange(1, x.shape[2], 2)]) / 2


    def k_max_pool(self, x, k):
        """get the top k value in the pool and keep order
        :type x: theano.tensor.tensor4
        :param x: the output of convolution layer

        :type k: int
        :param k: k value k-max pool
        """

        ind = T.argsort(x, axis=3)
        sorted_ind = T.sort(ind[:,:,:,-k:], axis=3)
        dim0, dim1, dim2, dim3 = sorted_ind.shape
        indices_dim0 = T.arange(dim0).repeat(dim1 * dim2 * dim3)
        indices_dim1 = T.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1).T.flatten()
        indices_dim2 = T.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).T.flatten()
        return x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)
                

    def output(self, input):
        """ generate the output of the convolution layer
        :type input: theano.tensor.tensor4
        :param input: the input to the convolution layer
        """
        conv_out = T.nnet.conv.conv2d(input, self.W, border_mode="full")
        fold_out = self.fold(conv_out)
        
        pre_acti = fold_out + self.b.dimshuffle('x', 0, 'x', 'x')
        acti_out = self.activation(pre_acti)
        
        # k-max-pool
        pool_out = self.k_max_pool(acti_out, self.k)
        return pool_out


class LogisticRegressionLayer(object):
    """Logistic Regression Layer used to classify"""
    def __init__(self, rng, n_in, n_out, W=None, b=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: random number generator

        :type n_in: int
        :param n_in: number of input units 

        :type n_out: int
        :type n_out: numbef of output class

        :type W: theano.tensor.matrix
        :param W: the weighted maxtrix, (n_in, n_out)

        :type b: theano.tensor.vector
        :param b: the bias parameter, (n_out, )
        """
        self.n_in = n_in
        self.n_out = n_out
        
        if W is not None:
            self.W = W
        else:
            w_bound = np.sqrt(6./(n_in + n_out))
            w_val = rng.uniform(-w_bound, w_bound, size=(n_in, n_out))
            self.W = shared(
                    np.asarray(w_val, dtype=theano.config.floatX),
                    borrow=True
                    )

        if b is not None:
            self.b = b
        else:
            b_val = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = shared(b_val,
                    borrow=True)
        self.params = [self.W, self.b]

    def negtive_log_likelihood(self, x, y):
        """ compute the negative log likelihood 
        :type x: theano.tensor.fvector
        :param x: input to the logistic regression layer

        :type y: theano.tensor.vector
        :param y: the true label with one hot representation
        """
        prob_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return T.mean(-1 * T.log(prob_y_given_x[T.arange(y.shape[0]), y]))
    
    def predict(self, x):
        prob_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        pred_y = T.argmax(prob_y_given_x)
        return pred_y

    def errors(self, x, y):
        pred_y = self.predict(x)
        return T.mean(T.neq(pred_y, y))


def test():
    # construct model
    doc = theano.typed_list.TypedListType(theano.tensor.ivector)()
    rng = np.random.RandomState(10)
    vocab_size = 100
    embed_dm = 4 
    conv_layer_n = 2
    n_kerns = [5, 3]
    ks = [3, 3]
    filter_widths = [4, 2]
    activation = T.tanh

    docLayer = DocumentLayer(rng, doc, vocab_size, embed_dm,
            conv_layer_n, n_kerns, filter_widths, ks, activation)

    #input_doc = [[1,2,3,4,5,6,7,8],[1,10,11,20,6], [1, 2, 3]]
    input_doc = [[1, 2, 3]]
    
    test_doc_layer = function(inputs=[doc], outputs=docLayer.output)

    print test_doc_layer(input_doc)


def train_encoder(data_file="./data/dataset.pkl"):
    rng = np.random.RandomState(10)
    
    train_set, valid_set, test_set, word2id, pop2id, type2id = dataset.load_data(data_file)
    vocab_size = len(word2id)
    embed_dm = 80
    doc_conv_layer_n = 2
    doc_n_kerns = [6, 3]
    doc_ks = [3, 3]
    doc_filter_widths = [5, 3]
    doc_activation = T.tanh
    learning_rate = 0.01

    # define input variables
    index = T.lscalar()
    doc = theano.typed_list.TypedListType(theano.tensor.ivector)()
    docLayer = DocumentLayer(rng, doc, vocab_size, embed_dm, 
            doc_conv_layer_n, doc_n_kerns, 
            doc_filter_widths, doc_ks, doc_activation)

    doc_sen = docLayer.output # set doc_sen as (num of sentence * sen_dim)

    #####################
    # TASK 1 Event Type #
    #####################
    t1_conv_layer_n = 1
    t1_n_kerns = [5]
    t1_ks = [3]
    t1_filter_widths = [1]
    t1_activation = T.tanh

    t1_filter_shape = [t1_n_kerns[0], 1, 1, t1_filter_widths[0]]
    t1_conv_layer = ConvFoldingPoolLayer(rng, t1_filter_shape, t1_ks[0], t1_activation)
    t1_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
    t1_conv_output = t1_conv_layer.output(t1_conv_input).flatten(2)

    t1_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t1_n_kerns[-1] * t1_ks[-1] / 8
    t1_n_out = len(type2id)
    t1_logis_layer = LogisticRegressionLayer(rng, t1_n_in, t1_n_out)
    t1_y = T.ivector()
    t1_cost = t1_logis_layer.negtive_log_likelihood(t1_conv_output, t1_y)

    #####################
    # TASK 2 Population #
    #####################
    t2_conv_layer_n = 1
    t2_n_kerns = [5]
    t2_ks = [3]
    t2_filter_widths = [1]
    t2_activation = T.tanh

    t2_filter_shape = [t2_n_kerns[0], 1, 1, t2_filter_widths[0]]
    t2_conv_layer = ConvFoldingPoolLayer(rng, t2_filter_shape, t2_ks[0], t2_activation)
    t2_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
    t2_conv_output = t2_conv_layer.output(t2_conv_input).flatten(2)

    t2_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t2_n_kerns[-1] * t2_ks[-1] / 8
    t2_n_out = len(pop2id)
    t2_logis_layer = LogisticRegressionLayer(rng, t2_n_in, t2_n_out)
    t2_y = T.ivector()
    t2_cost = t2_logis_layer.negtive_log_likelihood(t2_conv_output, t2_y)

    ###################
    # CONSTRUCT MODEL #
    ###################
    cost = t1_cost + t2_cost
    t1_params = [docLayer.W0] + t1_conv_layer.params + t1_logis_layer.params
    t2_params = [docLayer.W0] + t2_conv_layer.params + t2_logis_layer.params

    #params = t1_conv_layer.params + t1_logis_layer.params + t2_conv_layer.params + t2_logis_layer.params
    for param in t1_params:
        print 'Param -> ', param
        print 'grad ->' ,T.grad(t1_cost, param)
    print "Fnish P1 set"
    for param in t2_params:
        print 'Param -> ', param
        print 'grad ->' ,T.grad(t2_cost, param)
    """
    param_grads = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * pgrad) 
            for param, pgrad in zip(params, param_grads)]
    
    train_model = function(
            inputs=[doc, t1_y, t2_y],
            outputs=cost,
            updates=updates
            )
    """



if __name__ == "__main__":
    #test()
    train_encoder()
