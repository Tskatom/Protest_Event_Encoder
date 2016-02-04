#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import cPickle

name_format = "chosen_sens_%d"
for i in range(5):
    name = name_format % i
    sens = cPickle.load(open(name))
    train = sens["train"]
    test = sens['test']
    
    sen = train + test
    count = {}
    for first, second in sen:
        if first not in count:
            count[first] = 0
        if second not in count:
            count[second] = 0

        count[first] += 1
        count[second] += 1
    #print count
    
    total = sum(count.values()) * 1.0
    first_c = count[0]
    second_c = count[1]

    print "%d Fold ---> 0: %0.2f， 1： %0.2f, Others： %0.2f" % (i, first_c/total, second_c/total, (total-first_c-second_c)/total) 
        
    
if __name__ == "__main__":
    pass

