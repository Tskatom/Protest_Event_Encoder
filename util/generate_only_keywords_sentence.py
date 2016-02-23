#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import glob
import re
import json
import shutil

keyword_file = '../data/search_keywords.txt'
keywords = u'|'.join([w.strip().decode('utf-8') for w in open(keyword_file)])
pattern = re.compile(keywords, re.U)

infolder = '../data/new_single_label'
outfolder = '../data/new_single_label_keysen'

for i in range(5):
    ori_folder = os.path.join(infolder, "%d" % i)
    tar_folder = os.path.join(outfolder, "%d" % i)
    if not os.path.exists(tar_folder):
        os.makedirs(tar_folder)
    
    # filter the sentence with keywords
    files = glob.glob(os.path.join(ori_folder, "*.txt.tok"))
    for f in files:
        basename = os.path.basename(f)
        outfile = os.path.join(tar_folder, basename)
        with open(f) as df, open(outfile, 'w') as otf:
            for line in df:
                sens = json.loads(line)
                filtered = [sen for sen in sens if pattern.search(sen.lower())]
                if len(filtered) == 0:
                    print "==========Wrong"
                otf.write(json.dumps(filtered, ensure_ascii=False, encoding='utf-8').encode('utf-8') + "\n")

    # copy cat files
    files = glob.glob(os.path.join(ori_folder, "*_cat"))
    for f in files:
        tar_file = os.path.join(tar_folder, os.path.basename(f))
        shutil.copy2(f, tar_file)

if __name__ == "__main__":
    pass

