#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
import numpy as np

fid = sys.argv[1]

in_file = "../data/new_multi_label_protest/%s/event_test.txt.tok" % fid
event_label_file = "../data/new_multi_label_protest/%s/event_test.event_cat" % fid
event_labels = [l.strip() for l in open(event_label_file)]
full_file = "../data/new_multi_label/autogsr_protest.txt.tok"
full_label_file = "../data/new_multi_label/autogsr_pop_cat.txt"
full_labels = [l.strip() for l in open(full_label_file)]

docs = [json.loads(l) for i, l in enumerate(open(in_file)) if event_labels[i] == "protest"]
print len(docs)
full_docs = [json.loads(l) for l in open(full_file)]

new_docs = []
new_labels = []

# construct sentence 2 lable map
sen2type = {}
for i, doc in enumerate(full_docs):
    for sen in doc:
        sen2type[sen] = full_labels[i]

for doc in docs:
    found = False
    label = None
    for sen in doc:
        if sen in sen2type and sen2type[sen] != "Other":
            found = True
            label = sen2type[sen]
            break
    if found:
        new_docs.append(doc)
        new_labels.append(label)

print len(sen2type), len(docs)
sorted_id = np.argsort(new_labels)
with open("../data/new_multi_label_protest/%s/event2cluster_pop.txt" % fid, 'w') as nlk:
    for id in sorted_id:
        nlk.write(new_labels[id].strip() + "|" + ' '.join(new_docs[id]).encode('utf-8') + "\n")




if __name__ == "__main__":
    pass

