#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import numpy as np

pop_y_file = sys.argv[1]
type_y_file = sys.argv[2]
pop_pred_file = sys.argv[3]
type_pred_file = sys.argv[4]

pop_dic_fn = '../data/pop_cat.dic'
type_dic_fn = "../data/type_cat.dic"

pop_dic = {l.strip():i for i, l in enumerate(open(pop_dic_fn))}
type_dic = {l.strip():i for i, l in enumerate(open(type_dic_fn))}

pop_y = np.asarray([pop_dic[l.strip()] for l in open(pop_y_file)])
type_y = np.asarray([type_dic[l.strip()] for l in open(type_y_file)])

pop_pred = np.asarray([float(l.strip()) for l in open(pop_pred_file)])
type_pred = np.asarray([float(l.strip()) for l in open(type_pred_file)])

count = 0.
for py, ty, pp, tp in zip(pop_y, type_y, pop_pred, type_pred):
    if py == pp and ty == tp:
        count += 1.

perf = count / len(pop_y)

print "The performance is %f" % perf
if __name__ == "__main__":
    pass

