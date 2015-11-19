#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
bruce implement
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import theano.tensor as T
import numpy as np
import theano

def train_encoder():
    # load data
    

    # sentenceLayer
    doc = T.matrix('doc') # num sentence * num words
    embed_dm = 64
    num_pop_class = 13

    filter_shape = [64, 1, embed_dm, 2]
    sen_layer_input = 


if __name__ == "__main__":
    pass

