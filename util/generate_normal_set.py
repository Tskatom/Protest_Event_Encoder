#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
import re

"""
Prepare the normal text data for classifiers
"""

infile = "../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json"
es_file = "../data/multi_label/spanish_protest.txt.tok"
en_file = "../data/multi_label/english_protest.txt.tok"
pop_file = "../data/multi_label/pop_label.txt"
type_file = "../data/multi_label/eventType_label.txt"

s_es_file = "../data/single_label/spanish_protest.txt.tok"
s_en_file = "../data/single_label/english_protest.txt.tok"
s_pop_file = "../data/single_label/pop_label.txt"
s_type_file = "../data/single_label/eventType_label.txt"

key_file = "../data/seed_keywords.txt"
words = [w.strip() for w in open(key_file)]
rule = '|'.join(words)
pattern = re.compile(rule, re.I)


with open(infile) as itf, open(es_file, 'w') as esf, open(en_file, 'w') as enf, open(pop_file, 'w') as pop_f, open(type_file, 'w') as tf, open(s_es_file, 'w') as s_esf, open(s_en_file, 'w') as s_enf, open(s_pop_file, 'w') as s_pop_f, open(s_type_file, 'w') as s_tf:
    article_set = {}
    for line in itf:
        event = json.loads(line)
        # for each event it may contain multiple articles
        eventType = event["eventType"][:3]
        population = event["population"]
    
        # we use the 100 tokens as dupilciated case
        for article in event["downloaded_articles"].values():
            if not isinstance(article, dict):
                continue
            try:
                # if the orignial text is not spanish, skip
                if article["original_text_basis_enrichment"]["language"] != "Spanish":
                    continue
                
                spanish_tokens = [t["value"] for t in article["original_text_basis_enrichment"]["tokens"]]
                # if the document does not contain the protest keywords
                # continue
                text = u' '.join(spanish_tokens).encode('utf-8').lower()
                matched = pattern.search(text)
                if not matched:
                    continue

                english_tokens = [t['value'] for t in article["translated_text_basis_enrichment"]["tokens"]]
            except:
                print "Error", sys.exc_info()
                continue

            # check duplicated
            article_key = ' '.join(spanish_tokens[:100])
            if article_key not in article_set:
                article_set[article_key] = {"spanish": u' '.join(spanish_tokens).encode('utf-8'), 
                        "english": u' '.join(english_tokens).encode('utf-8'),
                        "pop":set(population), "type":set(eventType)}
            else:
                article_set[article_key]["pop"].add(population)
                article_set[article_key]["type"].add(eventType)
    print "Total Articles %d " % len(article_set)
    # write to files
    for article_key, value in article_set.items():
        esf.write(value["spanish"] + "\n")
        enf.write(value["english"] + "\n")
        pop_f.write('|'.join(value['pop']) + "\n")
        tf.write('|'.join(value['type']) + "\n")

        # write to single label
        if len(value['pop']) == 1:
            s_esf.write(value["spanish"] + "\n")
            s_enf.write(value["english"] + "\n")
            s_pop_f.write(value["pop"].pop() + "\n")
            s_tf.write(value["type"].pop() + "\n")

if __name__ == "__main__":
    pass

