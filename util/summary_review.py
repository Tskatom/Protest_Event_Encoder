#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import glob
import re
import numpy as np

pos_folder = "./withRats_pos"
neg_folder = "./withRats_neg"
folders = [pos_folder, neg_folder]
for pol in folders:
    files = glob.glob(pol + "/*")
    ratios = []
    for f in files:
        with open(f) as df:
            doc = df.read().strip()
            sens = [sen.strip() for sen in doc.split(".")]
            label_sens = [sen for sen in sens if re.search("<NEG>|<POS>", sen)]
            ratios.append((len(sens), len(label_sens), 1.*len(label_sens)/len(sens)))
    
    t, l, r = zip(*ratios)
    print np.mean(t), np.mean(l), np.mean(r)
if __name__ == "__main__":
    pass

