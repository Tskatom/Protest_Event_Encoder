#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
import shutil
import numpy as np

balanced_data_folder = "../data/balanced_set/%d"
old_data_folder = "../data/new_multi_label/%d"
gsr_protest_file = "../data/new_single_label/gsr_spanish_protest.txt.tok"

for fold in range(5):
    train_file = os.path.join(old_data_folder % fold, "event_train.txt.tok")
    train_label_file = os.path.join(old_data_folder % fold, "event_train.event_cat")

    train_docs = [json.loads(l) for l in open(train_file)]
    train_labels = [l.strip() for l in open(train_label_file)]

    gsr_docs = [json.loads(l) for l in open(gsr_protest_file)]

    new_docs = train_docs + gsr_docs
    new_labels = train_labels + ["protest"] * len(gsr_docs)

    random_idx = np.random.permutation(range(len(new_docs)))

    outfolder = balanced_data_folder % fold
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    out_txt = os.path.join(outfolder, "event_train.txt.tok")
    out_label_file = os.path.join(outfolder, "event_train.event_cat")

    with open(out_txt, 'w') as ot, open(out_label_file, 'w') as olf:
        for id in random_idx:
            ot.write(json.dumps(new_docs[id], ensure_ascii=False, encoding='utf-8').encode('utf-8') + "\n")
            olf.write(new_labels[id].strip() + "\n")

    # copy the test txt and event to new folder
    ofolder = old_data_folder % fold
    shutil.copy2(os.path.join(ofolder, "event_test.txt.tok"), outfolder)
    shutil.copy2(os.path.join(ofolder, "event_test.event_cat"), outfolder)


if __name__ == "__main__":
    pass

