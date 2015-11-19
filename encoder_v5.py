#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
construct the model only do event type prediction problem
add unit test for each function
don't use fold and only use normal conv and max pooling layer tech
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
from theano.tensor.signal import downsample


#theano.config.exception_verbosity = 'high'
#theano.config.optimizer='fast_compile'

__process__ = "encoder_v5.log"


class SentenceLayer(object):
    """
    Layer take the document as input and out the list of sentences representation
    """
    def __init__(self, rng, input, vocab_size, embed_dm, n_kerns,
            embedding=None, W0=None, b0=None):
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
        n_gram = 2
        filter_shape0 = (self.embed_dm, 1, self.embed_dm, n_gram)
        
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

        doc_rep = self.embedding[input].dimshuffle(0,'x',2,1)
        conv_out = T.nnet.conv.conv2d(doc_rep, self.W0, border_mode="valid")
        pool_out = T.argmax(conv_out, axis=3, keepdims=True)
        act_pool_out = T.nnet.sigmoid(pool_out + self.b0.dimshuffle('x',0, 'x','x'))

        self.output = act_pool_out.flatten(2)
        self.params = [self.embedding, self.W0, self.b0]


class DocumentLayer(object):
    """
    Construct Document Vector
    """
    def __init__(self, rng, input, doc_dim, W=None, b=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: random number generator

        :type input: theano.tensor.tensor4
        :param input: input sentence matrix,with (1, 1, sen_dim, sen_num)

        :type doc_dim: int
        :param doc_dim: the dimention of the output doc vector

        :type W: theano.tensor.tensor4
        :param W: filter for conv, with shape(doc_dim, 1, sen_dim, 1)

        :type b: theano.tensor.vector
        :param b: bias for conv layer with shape(doc_dim,)

        """
       
        filter_shape = [doc_dim, 1, doc_dim, 1]
        if W is not None:
            if isinstance(W, np.ndarray):
                self.W = shared(np.asarray(W, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.W = W
        else:
            f_in = np.prod(filter_shape[1:])
            f_out = filter_shape[0] * np.prod(filter_shape[2:])
            w_bound = np.sqrt(6./(f_in + f_out))
            w_val = rng.uniform(-w_bound, w_bound, size=filter_shape)
            self.W = shared(np.asarray(w_val, dtype=theano.config.floatX),
                    borrow=True)

        if b is not None:
            if isinstance(b, np.ndarray):
                self.b = shared(np.asarray(b, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.b = b
        else:
            b_val = rng.uniform(0, 0.1, size=(filter_shape[0],))
            self.b = shared(np.asarray(b_val, dtype=theano.config.floatX),
                    borrow=True)


        # do convolution
        conv_out = T.nnet.conv.conv2d(input, self.W, border_mode="valid")
        
        # get max value for each dimention
        pool_out = T.argmax(conv_out, axis=3, keepdims=True)

        self.output = T.nnet.sigmoid(pool_out + self.b.dimshuffle('x',0, 'x','x')).flatten(1)
        self.params = [self.W, self.b]
            


class HiddenLayer(object):

    def __init__(self, rng, input, f_in, f_out, W=None, b=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: random number generator

        :type input: theano.tensor.vector
        :param input: input vector to the hidden layer

        :type f_in: int
        :param f_in: size of input vector

        :type f_out: int
        :param f_Out: number of hidden units

        :type W: numpy.ndarray
        :param W: weight matrix

        :type b: numpy.ndarray
        :param b: bias vector
        """
        if W is not None:
            if isinstance(W, np.ndarray):
                self.W = shared(value=np.asarray(W, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.W = W
        else:
            w_bound = np.sqrt(6./(f_in + f_out))
            w_val = rng.uniform(-w_bound, w_bound, size=(f_in, f_out))
            self.W = shared(value=np.asarray(w_val, dtype=theano.config.floatX),
                    borrow=True)

        if b is not None:
            if isinstance(b, np.ndarray):
                self.b = shared(value=np.asarray(b, dtype=theano.config.floatX),
                        borrow=True)
            else:
                self.b = b
        else:
            b_val = np.ones((f_out,)) * 0.1 
            self.b = shared(value=np.asarray(b_val, dtype=theano.config.floatX),
                    borrow=True)

        pre_act = T.dot(input, self.W) + self.b
        self.output = T.nnet.sigmoid(pre_act)
        self.params = [self.W, self.b]


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
    embed_dm = 80 # default value
    embedding = None # default value
    
    if embedding_file:
        # load the embedding
        wiki = cPickle.load(open(embedding_file))
        embedding = wiki[1]
        embed_dm = embedding.shape[1]

    doc_n_kerns = [64]
    learning_rate = 0.01

    # define input variables
    print "Start to define Doc Layer"
    doc = T.lmatrix('doc')
    sentLayer = SentenceLayer(rng, doc, vocab_size, embed_dm, doc_n_kerns, embedding)

    doc_sen = sentLayer.output # set doc_sen as (num of sentence * sen_dim)
    # construct doc vector
    doc_layer_input = doc_sen.dimshuffle('x', 'x', 1, 0)
    doc_dim = 64
    docLayer = DocumentLayer(rng, doc_layer_input, doc_dim)
    doc_vec = docLayer.output

    #####################
    # TASK 1 Population #
    #####################
    print "Start define Task Population"
    t1_y = T.ivector()
    f_in = doc_dim
    f_out = 32
    t1_hidd_1 = HiddenLayer(rng, doc_vec, f_in, f_out)
    t1_hidd_1_out = t1_hidd_1.output

    t1_n_in = f_out
    t1_n_out = n_pop_class
    t1_logis = LogisticRegressionLayer(rng, t1_n_in, t1_n_out)
    t1_cost = t1_logis.negtive_log_likelihood(t1_hidd_1_out, t1_y)
    t1_error = t1_logis.errors(t1_hidd_1_out, t1_y)
    t1_params = t1_logis.params + t1_hidd_1.params + docLayer.params + sentLayer.params


    ###################
    # CONSTRUCT MODEL #
    ###################
    print "Start construct model..."
    t1_grads = [T.grad(t1_cost, param) for param in t1_params]

    t1_updates = [(param, param - learning_rate * grad) 
            for param, grad in zip(t1_params, t1_grads)]


    # train task
    train_pop = function(inputs=[doc, t1_y], outputs=t1_cost, updates=t1_updates)
    valid_pop = function(inputs=[doc, t1_y], outputs=t1_error)
    test_pop= function(inputs=[doc, t1_y], outputs=t1_error)

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
        for i, index in enumerate(indexs[:5000]):
            x = np.asarray(train_set_x[index])
            pop_y = train_set_pop_y[index]
            cost_pop = train_pop(x, [pop_y])
            message = "Epoch %d index %d cost_pop %f" % (epoch, i, cost_pop)
            logging.info(message)
        # Save the model parameters
        model_name = "./data/wiki_model_param_epoch_%d.pkl" % epoch
        print "Dump Model"
        with open(model_name, 'w') as model:
            # dump sentenceLayer Param
            for param in sentLayer.params:
                cPickle.dump(param.get_value(), model)

            # dump documentLayer params
            for param in docLayer.params:
                cPickle.dump(param.get_value(), model)

            # dump task population params
            for param in t1_hidd_1.params + t1_logis.params:
                cPickle.dump(param.get_value(), model)

        # valid set
        train_error_pops = []
        for index in xrange(len(train_set_x[:2000])):
            x = np.asarray(train_set_x[index])
            pop_y = train_set_pop_y[index]
            train_error_pops.append(valid_pop(x, [pop_y]))
        
        message = 'Epoch %d with valid error rate Population[%0.2f]' % (epoch, 
                np.mean(train_error_pops))
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
        doc_activation = T.nnet.sigmoid
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
        t1_activation = T.nnet.sigmoid

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
        t2_activation = T.nnet.sigmoid

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
