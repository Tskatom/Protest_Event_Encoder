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
                borrow=True,
                name='embedding')

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

        num_sens = theano.typed_list.length(self.input)
        output, _ = theano.scan(fn=generate_sen_rep,
                non_sequences=[self.input],
                sequences=[T.arange(num_sens, dtype='int64')])

        # the output is the list of sentence representation 
        self.output = output
            

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

    
if __name__ == "__main__":
    test()
