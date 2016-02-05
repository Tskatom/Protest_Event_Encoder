#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import glob
import sys
import os

def generate_review():
    # get 4:1 for training and test data set
    rats = ["noRats", "withRats"]
    pols = ["pos", "neg"]
    batch_size = 180
    for fold in range(5):
        start = fold*batch_size
        end = (fold+1)*batch_size
        train_set = []
        test_set = []
        for rat in rats:
            train = {}
            test = {}
            for pol in pols:
                folder = "./%s_%s" % (rat, pol)
                files = glob.glob(folder + "/*")
                files = [f for f in files if int(os.path.basename(f).split("_")[1].split(".")[0]) <= 899]
                docs = []
                for f in files:
                    with open(f) as df:
                        docs.append(df.read().strip())
                test_docs = docs[start:end]
                train_docs = docs[:start] + docs[end:]
                train[pol] = train_docs
                test[pol] = test_docs
            #write out the results
            outfolder = "./%s/%d" % (rat, fold)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            train_doc_file = os.path.join(outfolder, "Imdb_review_train.txt.tok")
            train_label_file = os.path.join(outfolder, "Imdb_review_train.sent_cat")
            with open(train_doc_file, 'w') as tdf, open(train_label_file, 'w') as tlf:
                for k, docs in train.items():
                    for doc in docs:
                        tlf.write("%s\n" % k)
                        tdf.write("%s\n" % doc.strip())

            test_doc_file = os.path.join(outfolder, "Imdb_review_test.txt.tok")
            test_label_file = os.path.join(outfolder, "Imdb_review_test.sent_cat")
            with open(test_doc_file, 'w') as tdf, open(test_label_file, 'w') as tlf:
                for k, docs in test.items():
                    for doc in docs:
                        tlf.write("%s\n" % k)
                        tdf.write("%s\n" % doc.strip())


if __name__ == "__main__":
    generate_review()
