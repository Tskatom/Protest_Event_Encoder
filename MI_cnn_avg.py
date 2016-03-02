#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Multi-Instance Deep Learning Model
"""

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

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

def construct_sentence_flag(digit_dataset):
    train_set, test_set = digit_dataset
    train_x, train_y = train_set
    test_x, test_y = test_set

    def get_flags(xdata):
        sent_flag = []
        for doc in xdata:
            flags = []
            for sen in doc:
                flags.append(1 if np.any(sen) else 0)
            sent_flag.append(flags)
        return sent_flag
    train_flags = get_flags(train_x)
    test_flags = get_flags(test_x)

    return train_flags, test_flags



class GICF(object):
    def __init__(self, options):
        self.options = options
        self.params = []

    def run_experiment(self, dataset, word_embedding, exp_name):

        # load parameters
        num_maps_word = self.options["num_maps_word"]
        drop_rate_word = self.options["drop_rate_word"]
        drop_rate_sentence = self.options["drop_rate_sentence"]
        word_window = self.options["word_window"]
        word_dim = self.options["word_dim"]
        k_max_word = self.options["k_max_word"]
        batch_size = self.options["batch_size"]
        rho = self.options["rho"]
        epsilon = self.options["epsilon"]
        norm_lim = self.options["norm_lim"]
        max_iteration = self.options["max_iteration"]

        sentence_len = len(dataset[0][0][0][0])

        # compute the sentence flags
        train_flags, test_flags = construct_sentence_flag(dataset)
        train_flags = theano.shared(value=np.asarray(train_flags, dtype=theano.config.floatX), borrow=True)
        test_flags = theano.shared(value=np.asarray(test_flags, dtype=theano.config.floatX), borrow=True)


        # define the parameters
        x = T.tensor3("x")
        y = T.ivector("y")
        sen_flags = T.matrix("flag")
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
        dropout_sent_vec = dropout_word_conv.output.reshape((x.shape[0] * x.shape[1], sent_vec_dim))
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
        sent_vec = word_conv.output.reshape((x.shape[0] * x.shape[1], sent_vec_dim))

        # construct sentence level classifier
        n_in = sent_vec_dim
        n_out = 1
        sen_W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
        sen_W = theano.shared(value=sen_W_values, borrow=True, name="logis_W")
        sen_b_value = nn.as_floatX(0.0)
        sen_b = theano.shared(value=sen_b_value, borrow=True, name="logis_b")

        drop_sent_prob = T.nnet.sigmoid(T.dot(dropout_sent_vec, sen_W) + sen_b)
        sent_prob = T.nnet.sigmoid(T.dot(sent_vec, sen_W*(1-drop_rate_sentence)) + sen_b)

        # reform the sent vec to doc level
        drop_sent_prob = drop_sent_prob.reshape((x.shape[0], x.shape[1]))
        sent_prob = sent_prob.reshape((x.shape[0], x.shape[1]))
        # the pos probability bag label is the avg of the probs
        drop_doc_prob = T.sum(drop_sent_prob * sen_flags, axis=1) / T.sum(sen_flags, axis=1)
        doc_prob = T.sum(sent_prob * sen_flags, axis=1) / T.sum(sen_flags, axis=1)

        drop_doc_prob = T.clip(drop_doc_prob, nn.as_floatX(1e-7), nn.as_floatX(1 - 1e-7 ))
        doc_prob = T.clip(doc_prob, nn.as_floatX(1e-7), nn.as_floatX(1 - 1e-7 ))
        """
        # the pos probability bag label equals to 1 - all negative
        drop_doc_prob = T.prod(drop_sent_prob, axis=1)
        drop_doc_prob = T.set_subtensor(drop_doc_prob[:,1], 1 - drop_doc_prob[:,0])

        doc_prob = T.prod(sent_prob, axis=1)
        doc_prob = T.set_subtensor(doc_prob[:,1], 1 - doc_prob[:,0])

        # the pos probability bag label is the most positive probability
        drop_doc_prob = T.max(drop_sent_prob, axis=1)
        drop_doc_prob = T.clip(drop_doc_prob, nn.as_floatX(1e-7), nn.as_floatX(1 - 1e-7 ))
        doc_prob = T.max(sent_prob, axis=1)
        doc_prob = T.clip(doc_prob, nn.as_floatX(1e-7), nn.as_floatX(1 - 1e-7 ))
        """

        doc_preds = doc_prob > 0.5

        # instance level cost
        drop_sent_cost = T.mean(T.maximum(0.0, nn.as_floatX(.5) - T.sgn(drop_sent_prob.reshape((x.shape[0]*x.shape[1], n_out)) - nn.as_floatX(0.6)) * T.dot(dropout_sent_vec, sen_W)))

        # we need that the most positive instance at least 0.7 in pos bags
        # and at most 0.1 in neg bags
        most_positive_prob = T.max(drop_sent_prob, axis=1)
        pos_cost = T.maximum(0.0, nn.as_floatX(0.6) - most_positive_prob)
        neg_cost = T.maximum(0.0, most_positive_prob - nn.as_floatX(0.05))
        penal_cost = T.mean(pos_cost * y + neg_cost * (nn.as_floatX(1.0) - y))

        # bag level cost
        drop_bag_cost = T.mean(-y * T.log(drop_doc_prob) * nn.as_floatX(4.) - (1 - y) * T.log(1 - drop_doc_prob))
        #drop_cost = drop_bag_cost * nn.as_floatX(3.0) + drop_sent_cost + nn.as_floatX(2.0) * penal_cost
        drop_cost = drop_bag_cost * nn.as_floatX(3.) + drop_sent_cost + penal_cost


        # collect parameters
        self.params.append(words)
        self.params += dropout_word_conv.params
        self.params.append(sen_W)
        self.params.append(sen_b)

        grad_updates = nn.sgd_updates_adadelta(self.params,
                drop_cost,
                rho,
                epsilon,
                norm_lim)

        # construct the dataset
        # random the
        train_x, train_y = nn.shared_dataset(dataset[0])
        test_x, test_y = nn.shared_dataset(dataset[1])
        test_cpu_y = dataset[1][1]

        n_train_batches = int(np.ceil(1.0 * len(dataset[0][0]) / batch_size))
        n_test_batches = int(np.ceil(1.0 * len(dataset[1][0]) / batch_size))

        # construt the model
        index = T.iscalar()
        train_func = theano.function([index], [drop_cost, drop_bag_cost, drop_sent_cost, penal_cost], updates=grad_updates,
                givens={
                    x: train_x[index*batch_size:(index+1)*batch_size],
                    y: train_y[index*batch_size:(index+1)*batch_size],
                    sen_flags: train_flags[index*batch_size:(index+1)*batch_size]
                    })

        test_func = theano.function([index], doc_preds,
                givens={
                    x:test_x[index*batch_size:(index+1)*batch_size],
                    sen_flags: test_flags[index*batch_size:(index+1)*batch_size]
                    })

        get_train_sent_prob = theano.function([index], sent_prob,
                givens={
                    x:train_x[index*batch_size:(index+1)*batch_size]
                    })

        get_test_sent_prob = theano.function([index], sent_prob,
                givens={
                    x:test_x[index*batch_size:(index+1)*batch_size]
                    })

        epoch = 0
        best_score = 0


        log_file = open("./log/%s.log" % exp_name, 'w')

        while epoch <= max_iteration:
            start_time = timeit.default_timer()
            epoch += 1
            costs = []

            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_func(minibatch_index)
                costs.append(cost_epoch)
                set_zero(zero_vec)

            total_train_cost, train_bag_cost, train_sent_cost, train_penal_cost = zip(*costs)
            print "Iteration %d, total_cost %f bag_cost %f sent_cost %f penal_cost %f\n" %  (epoch, np.mean(total_train_cost), np.mean(train_bag_cost), np.mean(train_sent_cost), np.mean(train_penal_cost))

            if epoch % 1 == 0:
                test_preds = []
                for i in xrange(n_test_batches):
                    test_y_pred = test_func(i)
                    test_preds.append(test_y_pred)
                test_preds = np.concatenate(test_preds)
                test_score = 1 - np.mean(np.not_equal(test_cpu_y, test_preds))

                precision, recall, beta, support = precision_recall_fscore_support(test_cpu_y, test_preds, pos_label=1)

                if beta[1] > best_score:
                    best_score = beta[1]
                    # save the sentence vectors
                    #train_sens = [get_train_sent_prob(i) for i in range(n_train_batches)]
                    test_sens = [get_test_sent_prob(i) for i in range(n_test_batches)]

                    #train_sens = np.concatenate(train_sens, axis=0)
                    test_sens = np.concatenate(test_sens, axis=0)

                    #out_train_sent_file = "./results/%s_train_sent.vec" % exp_name
                    out_test_sent_file = "./results/%s_test_sent.vec" % exp_name

                    with open(out_test_sent_file, 'w') as test_f:
                        #cPickle.dump(train_sens, train_f)
                        cPickle.dump(test_sens, test_f)
                    print "Get best performace at %d iteration %f" % (epoch, test_score)
                    log_file.write("Get best performance at %d iteration %f \n" % (epoch, test_score))

                end_time = timeit.default_timer()
                print "Iteration %d , precision, recall, f1" % epoch, precision, recall, beta
                log_file.write("Iteration %d, neg precision %f, pos precision %f, neg recall %f pos recall %f , neg f1 %f, pos f1 %f, total_cost %f bag_cost %f sent_cost %f penal_cost %f\n" % (epoch, precision[0], precision[1], recall[0], recall[1], beta[0], beta[1], np.mean(total_train_cost), np.mean(train_bag_cost), np.mean(train_sent_cost), np.mean(train_penal_cost)))
                print "Using time %f m" % ((end_time -start_time)/60.)
                log_file.write("Uing time %f m\n" % ((end_time - start_time)/60.))
            end_time = timeit.default_timer()
            print "Iteration %d Using time %f m" % ( epoch, (end_time -start_time)/60.)
            log_file.write("Uing time %f m\n" % ((end_time - start_time)/60.))
            log_file.flush()

        log_file.close()
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
    ap.add_argument("--exp_name", type=str, help="the name of the experiment")
    return ap.parse_args()


def main():
    args = parse_args()
    option = json.load(open(args.option))
    prefix = args.prefix
    sufix = args.sufix
    data_type = args.data_type
    event_fn = args.event_fn
    word2vec_file = args.word2vec
    exp_name = args.exp_name

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
    model.run_experiment(digit_dataset, embedding, exp_name)


if __name__ == "__main__":
    main()
