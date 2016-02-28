#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import theano.tensor as T
import theano
import nn_layers as nn
from sklearn.metrics import precision_recall_fscore_support
import timeit
import argparse
import json
import cPickle
import numpy as np


def ReLU(x):
    return T.maximum(0.0, x)

def Tanh(x):
    return T.tanh(x)

class GICF(object):
    def __init__(self, options):
        self.options = options
        self.params = []

    def run_experiment(self, dataset, word_embedding):
        
        # load parameters
        num_maps_word = self.options["num_maps_word"]
        drop_rate_word = self.options["drop_rate_word"]
        word_window = self.options["word_window"]
        word_dim = self.options["word_dim"]
        k_max_word = self.options["k_max_word"]
        num_maps_sentence = self.options["num_maps_sentence"]
        drop_rate_sentence = self.options["drop_rate_sentence"]
        sentence_window = self.options["sentence_window"]
        k_max_sentence = self.options["k_max_sentence"]
        batch_size = self.options["batch_size"]
        rho = self.options["rho"]
        epsilon = self.options["epsilon"]
        norm_lim = self.options["norm_lim"]
        max_iteration = self.options["max_iteration"]

        sentence_len = len(dataset[0][0][0][0])
        sentence_num = len(dataset[0][0][0])
        
        # define the parameters
        x = T.tensor3("x")
        y = T.ivector("y")
        rng = np.random.RandomState(1234)
        
        words = theano.shared(value=np.asarray(word_embedding,
            dtype=theano.config.floatX),
            name="embedding", borrow=True)
        zero_vector_tensor = T.vector() 
        zero_vec = np.zeros(word_dim, dtype=theano.config.floatX)
        set_zero = theano.function([zero_vector_tensor], updates=[(words, T.set_subtensor(words[0,:], zero_vector_tensor))])

        x_emb = words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0]*x.shape[1], 1, x.shape[2], words.shape[1]))

        dropout_x_emb = nn.dropout_from_layer(rng, x_emb, drop_rate_word)

        # compute convolution on words layer
        word_filter_shape = (num_maps_word, 1, word_window, word_dim)
        word_pool_size = (sentence_len - word_window + 1, 1)
        dropout_word_conv = nn.ConvPoolLayer(rng, 
                input=dropout_x_emb,
                input_shape=None,
                filter_shape=word_filter_shape,
                pool_size=word_pool_size,
                activation=Tanh,
                k=k_max_word)
        sent_vec_dim = num_maps_word*k_max_word
        dropout_sent_vec = dropout_word_conv.output.reshape((x.shape[0], 1, x.shape[1], sent_vec_dim))
        dropout_sent_vec = nn.dropout_from_layer(rng, dropout_sent_vec, drop_rate_sentence)

        word_conv = nn.ConvPoolLayer(rng, 
                input=dropout_x_emb*(1 - drop_rate_word),
                input_shape=None,
                filter_shape=word_filter_shape,
                pool_size=word_pool_size,
                activation=Tanh,
                k=k_max_word,
                W=dropout_word_conv.W,
                b=dropout_word_conv.b)
        sent_vec = word_conv.output.reshape((x.shape[0], 1, x.shape[1], sent_vec_dim))

        # construct the convolution layer on sentences
        sent_filter_shape = (num_maps_sentence, 1, sentence_window, sent_vec_dim)
        sent_pool_size = (sentence_num - sentence_window + 1, 1)
        dropout_sent_conv = nn.ConvPoolLayer(rng,
                input=dropout_sent_vec,
                input_shape=None,
                filter_shape=sent_filter_shape,
                pool_size=sent_pool_size,
                activation=Tanh,
                k=k_max_sentence)

        sent_conv = nn.ConvPoolLayer(rng,
                input=sent_vec*(1 - drop_rate_sentence),
                input_shape=None,
                filter_shape=sent_filter_shape,
                pool_size=sent_pool_size,
                activation=Tanh,
                k=k_max_sentence,
                W=dropout_sent_conv.W,
                b=dropout_sent_conv.b)
        
        dropout_doc_vec = dropout_sent_conv.output.flatten(2)
        doc_vec = sent_conv.output.flatten(2)
        doc_vec_dim = num_maps_sentence * k_max_sentence

        # construct classifier
        dropout_logistic_layer = nn.LogisticRegressionLayer(
                input=dropout_doc_vec,
                n_in=doc_vec_dim,
                n_out=2)

        logistic_layer = nn.LogisticRegressionLayer(
                input=doc_vec,
                n_in=doc_vec_dim,
                n_out=2,
                W=dropout_logistic_layer.W,
                b=dropout_logistic_layer.b)

        
        dropout_cost = dropout_logistic_layer.negative_log_likelihood(y)
        cost = logistic_layer.negative_log_likelihood(y)

        preds = logistic_layer.y_pred
        errors = logistic_layer.errors(y)

        # collect parameters
        self.params.append(words)
        self.params += dropout_word_conv.params
        self.params += dropout_sent_conv.params
        self.params += dropout_logistic_layer.params
        
        grad_updates = nn.sgd_updates_adadelta(self.params,
                dropout_cost,
                rho,
                epsilon,
                norm_lim)

        # construct the dataset
        train_x, train_y = nn.shared_dataset(dataset[0])
        test_x, test_y = nn.shared_dataset(dataset[1])
        test_cpu_y = dataset[1][1]

        n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
        n_test_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))

        # construt the model
        index = T.iscalar()
        train_func = theano.function([index], dropout_cost, updates=grad_updates,
                givens={
                    x: train_x[index*batch_size:(index+1)*batch_size],
                    y: train_y[index*batch_size:(index+1)*batch_size]
                    })

        test_func = theano.function([index], preds,
                givens={
                    x:test_x[index*batch_size:(index+1)*batch_size]
                    })

        get_train_sentvec = theano.function([index], sent_vec,
                givens={
                    x:train_x[index*batch_size:(index+1)*batch_size]
                    })

        get_test_sentvec = theano.function([index], sent_vec,
                givens={
                    x:test_x[index*batch_size:(index+1)*batch_size]
                    })

        epoch = 0
        best_score = 0
        raw_train_x = dataset[0][0]
        raw_test_x = dataset[1][0]
        # get the sentence number for each document
        number_train_sens = []
        number_test_sens = []

        for doc in raw_train_x:
            sen_num = 0
            for sen in doc:
                if np.any(sen):
                    sen_num += 1
            number_train_sens.append(sen_num)
        
        for doc in raw_test_x:
            sen_num = 0
            for sen in doc:
                if np.any(sen):
                    sen_num += 1
            number_test_sens.append(sen_num)

        while epoch <= max_iteration:
            start_time = timeit.default_timer()
            epoch += 1
            costs = []

            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_func(minibatch_index)
                costs.append(cost_epoch)
                set_zero(zero_vec)

            if epoch % 1 == 0:
                test_preds = []
                for i in xrange(n_test_batches):
                    test_y_pred = test_func(i)
                    test_preds.append(test_y_pred)
                test_preds = np.concatenate(test_preds)
                test_score = 1 - np.mean(np.not_equal(test_cpu_y, test_preds))

                precision, recall, beta, support = precision_recall_fscore_support(test_cpu_y, test_preds, pos_label=1)

                if test_score > best_score:
                    best_score = test_score
                    # save the sentence vectors
                    train_sens = [get_train_sentvec(i) for i in range(n_train_batches)]
                    test_sens = [get_test_sentvec(i) for i in range(n_test_batches)]

                    train_sens = np.concatenate(train_sens, axis=0)
                    test_sens = np.concatenate(test_sens, axis=0)

                    out_train_sent_file = "./results/train_sent.vec"
                    out_test_sent_file = "./results/test_sent.vec"

                    with open(out_train_sent_file, 'w') as train_f, open(out_test_sent_file, 'w') as test_f:
                        for i in range(len(train_sens)):
                            tr_doc_vect = train_sens[i][0][:number_train_sens[i]]
                            train_f.write(json.dumps(tr_doc_vect.tolist()) + "\n")

                        for i in range(len(test_sens)):
                            te_doc_vect = test_sens[i][0][:number_test_sens[i]]
                            test_f.write(json.dumps(te_doc_vect.tolist()) + "\n")
                    print "Get best performace at %d iteration" % epoch

                end_time = timeit.default_timer()
                print "Iteration %d , precision, recall, support" % epoch, precision, recall, support
                print "Using time %f m" % ((end_time -start_time)/60.)
        
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--option", type=str, help="the configuration file")
    ap.add_argument("--prefix", type=str, help="The prefix for experiment")
    ap.add_argument("--sufix", type=str, 
            default="event_cat", help="the sufix for experiment")
    ap.add_argument("--data_type", type=str, default="json", 
            help="the data type for text file: json or str")
    ap.add_argument("--event_fn", type=str, help="the event category dictionary file")
    ap.add_argument("--word2vec", type=str, help="word2vec file")
    return ap.parse_args()


def main():
    args = parse_args()
    option = json.load(open(args.option))
    prefix = args.prefix
    sufix = args.sufix
    data_type = args.data_type
    event_fn = args.event_fn
    word2vec_file = args.word2vec

    max_sens = option["max_sens"]
    max_words = option["max_words"]
    padding = option["padding"]

    class2id = {k.strip():i for i,k in enumerate(open(event_fn))}
    
    dataset = nn.load_event_dataset(prefix, sufix)
    wf = open(word2vec_file)
    embedding = cPickle.load(wf)
    word2id = cPickle.load(wf)

    digit_dataset = nn.transform_event_dataset(dataset, word2id, class2id, data_type, max_sens, max_words, padding)

    model = GICF(option)
    model.run_experiment(digit_dataset, embedding)


if __name__ == "__main__":
    main()
