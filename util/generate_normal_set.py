#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json

"""
Prepare the normal text data for classifiers
"""

infile = "../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json"
es_file = "../data/spanish_protest.txt.tok"
en_file = "../data/english_protest.txt.tok"
pop_file = "../data/pop_label.txt"
type_file = "../data/eventType_label.txt"

with open(infile) as itf, open(es_file, 'w') as esf, open(en_file, 'w') as enf, open(pop_file, 'w') as pop_f, open(type_file, 'w') as tf:
    for line in itf:
        event = json.loads(line)
        # for each event it may contain multiple articles
        eventType = event["eventType"]
        population = event["population"]
    
        # we use the 100 tokens as dupilciated case
        article_set = {}
        for article in event["downloaded_articles"].values():
            if not isinstance(article, dict):
                continue
            try:
                spanish_tokens = [t["value"] for t in article["original_text_basis_enrichment"]["tokens"]]
                english_tokens = [t['value'] for t in article["translated_text_basis_enrichment"]["tokens"]]
            except:
                print "Error", sys.exc_info()
                continue

            # check duplicated
            article_key = ' '.join(spanish_tokens[:100])
            if article_key not in article_set:
                article_set[article_key] = 1
                # write out the articles to file
                esf.write(u' '.join(spanish_tokens).encode('utf-8') + "\n")
                enf.write(u' '.join(english_tokens).encode('utf-8') + "\n")

                pop_f.write(population + "\n")
                tf.write(eventType + "\n")


if __name__ == "__main__":
    pass

