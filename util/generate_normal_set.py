#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
import re
import nltk

"""
Prepare the normal text data for classifiers
"""

infile = "../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json"
es_file = "../data/new_multi_label/spanish_protest.txt.tok"
en_file = "../data/new_multi_label/english_protest.txt.tok"
pop_file = "../data/new_multi_label/pop_label.txt"
type_file = "../data/new_multi_label/eventType_label.txt"

s_es_file = "../data/new_single_label/spanish_protest.txt.tok"
s_en_file = "../data/new_single_label/english_protest.txt.tok"
s_pop_file = "../data/new_single_label/pop_label.txt"
s_type_file = "../data/new_single_label/eventType_label.txt"

key_file = "../data/search_keywords.txt"
words = [w.strip().decode('utf-8') for w in open(key_file)]
rule = '|'.join(words)
pattern = re.compile(rule, re.U)
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

non_words = [u'¿', u'¡']

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
                
                #spanish_tokens = [t["value"] for t in article["original_text_basis_enrichment"]["tokens"]]
                # if the document does not contain the protest keywords
                # continue
                #text = u' '.join(spanish_tokens).encode('utf-8').lower()
                spanish_text = article["original_text"]
                spanish_text = ''.join([c for c in spanish_text if c not in non_words])  
                paragraphs = re.split("\n+", spanish_text)
                paragraphs = [p for p in paragraphs if len(p.split(" ")) > 5]# remote those too short paragraphs
                sens = []
                for p in paragraphs:
                    for sen in tokenizer.tokenize(p):
                        sen = re.sub(" +", " ", sen)
                    if len(sen.split(" ")) > 5:
                        sens.append(sen)
                if len(sens) < 4:
                    continue
                # filter by the keywords
                found = False
                for sen in sens:
                    matched = pattern.search(sen.lower())
                    if matched:
                        found = True
                        break
                if not found:
                    continue

                #english_tokens = [t['value'] for t in article["translated_text_basis_enrichment"]["tokens"]]
                english_text = article["translated_text"]
            except:
                print "Error", sys.exc_info()
                continue

            # check duplicated
            #article_key = ' '.join(spanish_tokens[:100])
            article_key = spanish_text[:500]
            if article_key not in article_set:
                article_set[article_key] = {"spanish": sens, 
                        "english": english_text,
                        "pop":set(), "type":set()}
            article_set[article_key]["pop"].add(population)
            article_set[article_key]["type"].add(eventType)
    print "Total Articles %d " % len(article_set)
    # write to files
    for article_key, value in article_set.items():
        #esf.write(json.dumps(value, ensure_ascii=False, encoding='utf-8') + "\n")
        # cleanup the sentences

        spanish_out = json.dumps(value["spanish"], ensure_ascii=False, encoding='utf-8')
        if isinstance(spanish_out, unicode):
            spanish_out = spanish_out.encode('utf-8')
        esf.write(spanish_out + "\n")
        enf.write(value["english"].encode("utf-8") + "\n")
        pop_s = list(value['pop'])
        type_s = list(value['type'])
        pop_f.write('|'.join(pop_s) + "\n")
        tf.write('|'.join(type_s) + "\n")

        # write to single label
        if len(pop_s) == 1 and len(type_s) == 1 and type_s[0] != "016":
            s_esf.write(spanish_out + "\n")
            s_enf.write(value["english"].encode("utf-8") + "\n")
            s_pop_f.write(pop_s[0] + "\n")
            s_tf.write('|'.join(type_s) + "\n")

if __name__ == "__main__":
    pass

