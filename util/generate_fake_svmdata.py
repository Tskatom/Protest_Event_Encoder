#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re

outfolder = "../data/fake_svm_dataset"
infolder = "../data/single_label"
for i in range(5):
    folder = os.path.join(infolder, "%d" % i)
    otfolder = os.path.join(outfolder, "%d" % i)
    parts = ["train", "test"]
    file_name = "spanish_protest_%s.txt.tok"
    for part in parts:
        doc_file = os.path.join(folder, file_name % part)
        out_doc_file = os.path.join(otfolder, file_name % part)
        with open(doc_file) as df, open(out_doc_file, 'w') as otf:
            for doc in df:
                doc = doc.strip()
                sens = re.split("\.|\?|\|", doc.lower())      
                sens = [sen for sen in sens if len(sen.strip().split(" ")) > 5]
                padding = 3
                max_sens = 30
                max_words = 80
                pad = 3 - 1
                sens_pad = []
                for sen in sens[:max_sens]:
                    tokens = sen.strip().split(" ")
                    sen_ids = ["<pad>"] * pad
                    for w in tokens[:max_words]:
                        sen_ids.append(w)
                    num_suff = max(0, max_words - len(tokens)) + pad
                    sen_ids += ["<pad>"] * num_suff
                    sens_pad.append(sen_ids)
                
                num_suff = max(0, max_sens - len(sens))
                for i in range(0, num_suff):
                    sen_ids = ["<pad>"] * len(sens_pad[0])
                    sens_pad.append(sen_ids)

                # write out
                for sen in sens_pad:
                    sen_text = " ".join(sen)
                    otf.write(sen_text + " ")
                otf.write("\n")


if __name__ == "__main__":
    pass

