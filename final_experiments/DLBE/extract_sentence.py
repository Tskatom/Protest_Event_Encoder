#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import cPickle
import shutil
import json
import numpy as np
import re

def DLBE(dlbe_config):
    config = json.load(open(dlbe_config))
    name_format = config["name"]
    batch_size = config["batch_size"]

    for fold, epoch in config["folds"].items():
        file_name = name_format % (fold, epoch)
        parts = ["train", "test"]
        choosen_sens = {"train":[], "test":[]}
        for part in parts:
            score_file = file_name + "_%s.score" % part
            scores = cPickle.load(open(score_file))
            for batch_score in scores:
                for doc_score in batch_score:
                    top_senids = sorted(np.argsort(doc_score)[-2:])
                    choosen_sens[part].append(top_senids)

            # load articles
            data_folder = os.path.join(config["sen_folder"], "%s" % fold)
            data_file = os.path.join(data_folder, "spanish_protest_%s.txt.tok" % part)
            docs = []
            with open(data_file) as daf:
                for line in daf:
                    sens = re.split("\.|\?|\|", line.strip())
                    sens = [sen for sen in sens if len(sen.strip().split(" ")) > 5]
                    docs.append(sens)

            assert len(choosen_sens[part]) == len(docs)

            # output file
            outfolder = os.path.join(config["out_folder"], "%s" % fold)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            outfile = os.path.join(outfolder, os.path.basename(data_file))
            with open(outfile, 'w') as otf:
                for id in range(len(docs)):
                    sids = choosen_sens[part][id]
                    sens = docs[id]

                    for sid in sids:
                        if sid >= len(sens):
                            continue
                        else:
                            otf.write(sens[sid]) + " . "
                    otf.write("\n")

            # copy the class file
            pop_cat = os.path.join(data_folder, "spanish_protest_%s.pop_cat" % part)
            type_cat = os.path.join(data_folder, "spanish_protest_%s.type_cat" % part)
            shutil.copy2(pop_cat, outfolder)
            shutil.copy2(type_cat, outfolder)

def main():
    dlbe_config_file = sys.argv[1]
    print dlbe_config_file
    DLBE(dlbe_config_file)


if __name__ == "__main__":
    main()

