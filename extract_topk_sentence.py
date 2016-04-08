#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
import numpy as np
import cPickle

def extract_sentences(fold):
    best_iters = {0:16, 1:11, 2:11,3:15, 4:16}
    data_folder = "./data/new_multi_label/%d/" % fold
    result_folder = "./final_experiments/MI/results"
    outfolder = "./data/new_multi_label_ksen/%d/" % fold

    groups = ["train", "test"]
    for group in groups:
        txt_file = os.path.join(data_folder, "event_%s.txt.tok" % group)
        label_file = os.path.join(data_folder, "event_%s.event_cat" % group)
        vec_file = os.path.join(result_folder, "mi_k_max_v4_mix%d_%s_sent_%d.vec" % (fold, group, best_iters[fold]))
        docs = [json.loads(l) for l in open(txt_file)]
        labels = [l for l in open(label_file)]
        sent_flags = [len(doc) if len(doc) < 15 else 15 for doc in docs]
        sent_k = np.maximum(1, np.floor(np.asarray(sent_flags) * 0.2)).astype("int32")
        vec = cPickle.load(open(vec_file))
        # construct the sent_mask
        sent_mask = np.zeros_like(vec).astype("int32")
        for i, sen_m in enumerate(sent_mask):
            sen_m[:sent_flags[i]] = 1
        vec = vec * sent_mask
        sort_idx = np.argsort(vec, axis=1)

        out_txt_file = os.path.join(outfolder, "event_%s.txt.tok" % group)
        out_label_file = os.path.join(outfolder, "event_%s.event_cat" % group)
        out_prob_file = os.path.join(outfolder, "event_%s.prob" % group)
        out_score_file = os.path.join(outfolder, "event_%s.prob_score" % group)
        pos = []
        neg = []
        with open(out_txt_file, 'w') as otf, open(out_label_file, 'w') as olf, open(out_prob_file, 'w') as opf, open(out_score_file, 'w') as osf:
            for id in range(len(sort_idx)):
                ids = sort_idx[id, -sent_k[id]:]
                try:
                    doc_sens = [docs[id][i] for i in ids]
                    probs = [vec[id][i] for i in ids]
                except:
                    print id, ids, len(docs[id])
                    sys.exit()
                prob = np.mean(probs)
                doc_scores = zip(doc_sens, probs)
                otf.write(json.dumps(doc_sens, ensure_ascii=False).encode('utf-8') + "\n")
                osf.write(json.dumps(doc_scores, ensure_ascii=False).encode('utf-8') + "\n")
                olf.write(labels[id])
                opf.write("%0.4f|%s\n" % (prob, labels[id].strip()))
                if labels[id].strip() == "protest":
                    pos.append(prob)
                else:
                    neg.append(prob)

        plot_out = os.path.join(outfolder, "event_%s.prob_col" % group)
        print len(pos)
        with open(plot_out, 'w') as po:
            po.write("%s\t%s\n" % ("Protest", "Non_Protest"))
            for i in range(len(neg)):
                if i < len(pos):
                    po.write("%0.4f\t%0.4f\n" % (pos[i], neg[i]))
                else:
                    po.write(" \t%0.4f\n" % neg[i])


if __name__ == "__main__":
    for i in range(5):
        extract_sentences(i)
