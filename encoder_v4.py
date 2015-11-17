#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Using only one convolution layer to model the sentence
and also using only one to model the doc vector
"""

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
import logging
import cPickle
import random
from theano.printing import Print as PP

#theano.config.exception_verbosity = 'high'
#theano.config.optimizer='fast_compile'

__process__ = "encoder_v4.log"

class DocumentLayer(object):
    """
    Layer take the document as input and out the list of sentences representation
    """
    def __init__(self, rng, input, vocab_size, embed_dm, 
            conv_layer_n, n_kerns, filter_widths, ks, activation,
            embedding=None, W0=None, b0=None, W1=None, b1=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random state generator used to initiate the parameter matrix

        :type input: theano.tensor.matrix
        :param input: document representation(num sentence * fix sentence length)
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

        if embedding is not None:
            # using the pretrained word embedding
            if isinstance(embedding, np.ndarray):
                embedding = shared(np.asarray(embedding, 
                    dtype=theano.config.floatX),
                    borrow=True)

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
            embedding_val[3, :] = 0.0 # initiate <PADDING> character as 0
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
        if W0 is not None:
            self.W0 = shared(np.asarray(
                W0, dtype=theano.config.floatX
                ),
                borrow=True
                )
        else:
            self.W0 = shared(
                    np.asarray(
                    rng.uniform(-w_bound, w_bound, size=filter_shape0),
                    dtype=theano.config.floatX),
                    borrow=True
                    )
        
        if b0 is not None:
            self.b0 = shared(np.asarray(b0, dtype=theano.config.floatX),
                    borrow=True)
        else:
            b_val = np.ones((filter_shape0[0],), dtype=theano.config.floatX) * 0.1
            self.b0 = shared(b_val, borrow=True)

        """
        # construct second layer parameters       
        filter_shape1 = (n_kerns[1], n_kerns[0], 1, filter_widths[1])
        k1 = ks[1]
        fan_in = np.prod(filter_shape1[1:])
        fan_out = filter_shape1[0] * np.prod(filter_shape1[2:])
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        if W1 is not None:
            self.W1 = shared(
                    np.asarray(W1, dtype=theano.config.floatX),
                    borrow=True
                    )
        else:
            self.W1 = shared(
                    np.asarray(
                    rng.uniform(-w_bound, w_bound, size=filter_shape1),
                    dtype=theano.config.floatX),
                    borrow=True
                    )
       
        if b1 is not None:
            self.b1 = shared(
                    np.asarray(b1, dtype=theano.config.floatX),
                    borrow=True
                    )
        else:
            b_val = np.ones((filter_shape1[0],), dtype=theano.config.floatX) * 0.1
            self.b1 = shared(b_val, borrow=True)
    
        """
            
        layer0 = ConvFoldingPoolLayer(rng, filter_shape0, k0, activation, self.W0, self.b0)
        # layer1 = ConvFoldingPoolLayer(rng, filter_shape1, k1, activation, self.W1, self.b1)
        
        doc_rep = self.embedding[input].dimshuffle(0,'x',2,1)
        layer0_output = layer0.output(doc_rep)
        # layer1_output = layer1.output(layer0_output)

        # the output is the list of sentence representation
        self.output = layer0_output.flatten(2)
        # self.params = [self.embedding, self.W0, self.b0, self.W1, self.b1]
        self.params = [self.embedding, self.W0, self.b0]
            

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
            if isinstance(W, np.ndarray):
                self.W = shared(np.asarray(W, dtype=theano.config.floatX),
                        borrow=True)
            else:
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
            if isinstance(b, np.ndarray):
                self.b = shared(np.asarray(b, dtype=theano.config.floatX))
            else:
                self.b = b
        else:
            b_val = np.ones((filter_shape[0],), dtype=theano.config.floatX) * 0.1
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
            if isinstance(W, np.ndarray):
                self.W = shared(np.asarray(W, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.W = W
        else:
            w_bound = np.sqrt(6./(n_in + n_out))
            w_val = rng.uniform(-w_bound, w_bound, size=(n_in, n_out))
            self.W = shared(
                    np.asarray(w_val, dtype=theano.config.floatX),
                    borrow=True
                    )

        if b is not None:
            if isinstance(b, np.ndarray):
                self.b = shared(np.asarray(b, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.b = b

        else:
            b_val = np.ones((n_out,), dtype=theano.config.floatX) * 0.1
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

    def predict_prob(self, x):
        prob_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return prob_y_given_x



def train_encoder(data_file="./data/dataset.pkl", embedding_file = None):
    
    logging.basicConfig(filename='./log/%s' % __process__, level=logging.INFO)
    
    rng = np.random.RandomState(10)
    print 'Start to Load Data....' 
    train_set, valid_set, test_set, word2id, pop2id, type2id = dataset.load_data(data_file)
    
    vocab_size = len(word2id)
    n_pop_class = 13
    n_type_class = 12
    embed_dm = 80 # default value
    embedding = None # default value
    
    if embedding_file:
        # load the embedding
        wiki = cPickle.load(open(embedding_file))
        embedding = wiki[1]
        embed_dm = embedding.shape[1]

    doc_conv_layer_n = 2
    doc_n_kerns = [6]
    doc_ks = [3]
    doc_filter_widths = [5]
    doc_activation = T.nnet.relu
    learning_rate = 0.01

    # define input variables
    print "Start to define Doc Layer"
    doc = T.lmatrix('doc')
    docLayer = DocumentLayer(rng, doc, vocab_size, embed_dm, 
            doc_conv_layer_n, doc_n_kerns, 
            doc_filter_widths, doc_ks, doc_activation, embedding)

    doc_sen = docLayer.output # set doc_sen as (num of sentence * sen_dim)

    #####################
    # TASK 1 Event Type #
    #####################
    print "Start define Task 1"
    t1_conv_layer_n = 1
    t1_n_kerns = [5]
    t1_ks = [1]
    t1_filter_widths = [1]
    t1_activation = T.nnet.relu

    t1_filter_shape = [t1_n_kerns[0], 1, 1, t1_filter_widths[0]]
    t1_conv_layer = ConvFoldingPoolLayer(rng, t1_filter_shape, t1_ks[0], t1_activation)
    t1_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
    t1_conv_output = t1_conv_layer.output(t1_conv_input).flatten(2)

    t1_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t1_n_kerns[-1] * t1_ks[-1] / 4 
    t1_n_out = n_type_class
    t1_logis_layer = LogisticRegressionLayer(rng, t1_n_in, t1_n_out)
    t1_y = T.ivector()
    t1_cost = t1_logis_layer.negtive_log_likelihood(t1_conv_output, t1_y)
    t1_error = t1_logis_layer.errors(t1_conv_output, t1_y)
    t1_params = docLayer.params + t1_conv_layer.params + t1_logis_layer.params

    #####################
    # TASK 2 Population #
    #####################
    print "Start define Task 2"
    t2_conv_layer_n = 1
    t2_n_kerns = [5]
    t2_ks = [1]
    t2_filter_widths = [1]
    t2_activation = T.nnet.relu

    t2_filter_shape = [t2_n_kerns[0], 1, 1, t2_filter_widths[0]]
    t2_conv_layer = ConvFoldingPoolLayer(rng, t2_filter_shape, t2_ks[0], t2_activation)
    t2_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
    t2_conv_output = t2_conv_layer.output(t2_conv_input).flatten(2)

    t2_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t2_n_kerns[-1] * t2_ks[-1] / 4
    t2_n_out = n_pop_class
    t2_logis_layer = LogisticRegressionLayer(rng, t2_n_in, t2_n_out)
    t2_y = T.ivector()
    t2_cost = t2_logis_layer.negtive_log_likelihood(t2_conv_output, t2_y)
    t2_error = t2_logis_layer.errors(t2_conv_output, t2_y)
    t2_params = docLayer.params + t2_conv_layer.params + t2_logis_layer.params

    ###################
    # CONSTRUCT MODEL #
    ###################
    print "Start construct model..."
    cost = t1_cost + t2_cost
    t1_grads = [T.grad(t1_cost, param) for param in t1_params]
    t2_grads = [T.grad(t2_cost, param) for param in t2_params]

    t1_updates = [(param, param - learning_rate * grad) 
            for param, grad in zip(t1_params, t1_grads)]

    t2_updates = [(param, param - learning_rate * grad) 
            for param, grad in zip(t2_params, t2_grads)]

    # train task
    train_type = function(inputs=[doc, t1_y], outputs=t1_cost, updates=t1_updates)
    train_pop = function(inputs=[doc, t2_y], outputs=t2_cost, updates=t2_updates)

    valid_type = function(inputs=[doc, t1_y], outputs=t1_error)
    valid_pop = function(inputs=[doc, t2_y], outputs=t2_error)

    test_type = function(inputs=[doc, t1_y], outputs=t1_error)
    test_pop = function(inputs=[doc, t2_y], outputs=t2_error)

    ###############
    # TRAIN MODEL #
    ###############
    print "Start train model..., Load Data"
    # load dataset

    train_set_x, train_set_y = train_set
    train_set_pop_y, train_set_type_y, train_set_loc_y = train_set_y
    
    valid_set_x, valid_set_y = valid_set
    valid_set_pop_y, valid_set_type_y, valid_set_loc_y = valid_set_y
    
    test_set_x, test_set_y = test_set
    test_set_pop_y, test_set_type_y, test_set_loc_y = test_set_y
    print "Data Loaded..."
    n_epochs = 200
    epoch = 0
    train_size = len(train_set_x)
    done_looping = False
    random.seed(10)
    train_set_x, train_set_y = train_set
    train_set_pop_y, train_set_type_y, train_set_loc_y = train_set_y
    while epoch < n_epochs and not done_looping:
        # shuffle the train_set for each epoch
        indexs = range(train_size)
        random.shuffle(indexs)
        epoch += 1
        for index in indexs:
            x = np.asarray(train_set_x[index])
            pop_y = train_set_pop_y[index]
            type_y = train_set_type_y[index]
            cost_type = train_type(x, [type_y])
            cost_pop = train_pop(x, [pop_y])
            message = "Epoch %d index %d cost_type %f cost_pop %f" % (epoch, index, cost_type, cost_pop)
            logging.info(message)
        # Save the model parameters
        model_name = "./data/wiki_model_param_epoch_%d.pkl" % epoch
        print "Dump Model"
        with open(model_name, 'w') as model:
            # dump docLayer Param
            for param in docLayer.params:
                cPickle.dump(param.get_value(), model)

            # dump task eventType param
            for param in t1_conv_layer.params + t1_logis_layer.params:
                cPickle.dump(param.get_value(), model)

            # dump task population param
            for param in t2_conv_layer.params + t2_logis_layer.params:
                cPickle.dump(param.get_value(), model)

        # valid set
        valid_error_types = []
        valid_error_pops = []
        for index in xrange(len(valid_set_x)):
            x = np.asarray(valid_set_x[index])
            pop_y = valid_set_pop_y[index]
            type_y = valid_set_type_y[index]

            x = np.asarray(valid_set_x[index])
            valid_error_types.append(valid_type(x, [type_y]))
            valid_error_pops.append(valid_pop(x, [pop_y]))
        
        message = 'Epoch %d with valid error rate eventType[%0.2f] Population[%0.2f]' % (epoch, 
                np.mean(valid_error_types), 
                np.mean(valid_error_pops))
        logging.info(message)



class Extractor(object):
    """ Event Extractor class"""
    def __init__(self, model_file):
        """
        :type model_file: string
        :param model_file: path to the model params dumps
        """
        mf = open(model_file)
        embedding = cPickle.load(mf)
        doc_w0 = cPickle.load(mf)
        doc_b0 = cPickle.load(mf)
        doc_w1 = cPickle.load(mf)
        doc_b1 = cPickle.load(mf)

        # event type task params
        t1_conv_w = cPickle.load(mf)
        t1_conv_b = cPickle.load(mf)
        t1_logis_w = cPickle.load(mf)
        t1_logis_b = cPickle.load(mf)

        # event population task params
        t2_conv_w = cPickle.load(mf)
        t2_conv_b = cPickle.load(mf)
        t2_logis_w = cPickle.load(mf)
        t2_logis_b = cPickle.load(mf)

        mf.close()

        vocab_size, embed_dm = embedding.shape
        n_pop_class = 13
        n_type_class = 12

        doc_conv_layer_n = 2
        doc_n_kerns = [6, 3]
        doc_ks = [3, 3]
        doc_filter_widths = [5, 3]
        doc_activation = T.nnet.relu
        learning_rate = 0.01

        # define input variables
        print "Start to define Doc Layer"
        rng = np.random.RandomState(10)
        doc = T.lmatrix('doc')
        docLayer = DocumentLayer(rng, doc, vocab_size, embed_dm, 
                doc_conv_layer_n, doc_n_kerns, 
                doc_filter_widths, doc_ks, doc_activation, embedding,
                doc_w0, doc_b0, doc_w1, doc_b1)

        doc_sen = docLayer.output # set doc_sen as (num of sentence * sen_dim)

        #####################
        # TASK 1 Event Type #
        #####################
        print "Start define Task 1"
        t1_conv_layer_n = 1
        t1_n_kerns = [5]
        t1_ks = [1]
        t1_filter_widths = [1]
        t1_activation = T.nnet.relu

        t1_filter_shape = [t1_n_kerns[0], 1, 1, t1_filter_widths[0]]
        t1_conv_layer = ConvFoldingPoolLayer(rng, t1_filter_shape, t1_ks[0], t1_activation,
                t1_conv_w, t1_conv_b)
        t1_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
        t1_conv_output = t1_conv_layer.output(t1_conv_input).flatten(2)

        t1_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t1_n_kerns[-1] * t1_ks[-1] / 4
        t1_n_out = n_type_class
        t1_logis_layer = LogisticRegressionLayer(rng, t1_n_in, t1_n_out,
                t1_logis_w, t1_logis_b)
        t1_y = T.ivector()
        t1_cost = t1_logis_layer.negtive_log_likelihood(t1_conv_output, t1_y)
        t1_error = t1_logis_layer.errors(t1_conv_output, t1_y)
        t1_pred = t1_logis_layer.predict(t1_conv_output)

        #####################
        # TASK 2 Population #
        #####################
        print "Start define Task 2"
        t2_conv_layer_n = 1
        t2_n_kerns = [5]
        t2_ks = [1]
        t2_filter_widths = [1]
        t2_activation = T.nnet.relu

        t2_filter_shape = [t2_n_kerns[0], 1, 1, t2_filter_widths[0]]
        t2_conv_layer = ConvFoldingPoolLayer(rng, t2_filter_shape, t2_ks[0], t2_activation,
                t2_conv_w, t2_conv_b)
        t2_conv_input = doc_sen.reshape((1, 1, doc_sen.shape[1], doc_sen.shape[0]))
        t2_conv_output = t2_conv_layer.output(t2_conv_input).flatten(2)

        t2_n_in = embed_dm * doc_n_kerns[-1] * doc_ks[-1] * t2_n_kerns[-1] * t2_ks[-1] / 4
        t2_n_out = n_pop_class
        t2_logis_layer = LogisticRegressionLayer(rng, t2_n_in, t2_n_out,
                t2_logis_w, t2_logis_b)
        t2_y = T.ivector()
        t2_cost = t2_logis_layer.negtive_log_likelihood(t2_conv_output, t2_y)
        t2_error = t2_logis_layer.errors(t2_conv_output, t2_y)
        t2_pred = t2_logis_layer.predict(t2_conv_output)

        ###################
        # CONSTRUCT MODEL #
        ###################
        print "Start construct model..."

        self.predict_type = function(inputs=[doc], outputs=t1_pred)
        self.predict_pop = function(inputs=[doc], outputs=t2_pred)


    def predict(self, doc):
        event_type_pred = self.predict_type(doc)
        event_population_pred = self.predict_pop(doc)
        return event_type_pred, event_population_pred


if __name__ == "__main__":
    #test()
    # train_encoder()
    data_file = "./data/wikibased_dataset.pkl"
    wiki_file = "./data/polyglot-es.pkl"
    train_encoder(data_file, wiki_file)
