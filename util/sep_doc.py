#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from glob import glob

infolder = "/raid/tskatom/rss_count/Argentina_10_Cities"
outfolder = "/raid/tskatom/rss_count/Argentina_10_cities_sep"

for sub_folder in os.listdir(infolder):
    city_folder = os.path.join(infolder, sub_folder)
    print city_folder
    files = glob(os.path.join(city_folder, "*"))
    target_folder = os.path.join(outfolder, sub_folder)
    print '--->', target_folder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    for f in files:
        basename = os.path.basename(f)
        with open(f) as dataf:
            articles = [l.strip() for l in dataf]
            num = len(articles)
            for i in range(1, num+1):
                outfile = os.path.join(target_folder, "%s.%04d" % (basename, i))
                with open(outfile, 'w') as otf:
                    otf.write(articles[i-1])
