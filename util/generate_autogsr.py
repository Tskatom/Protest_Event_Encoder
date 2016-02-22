#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re
import nltk
import json


infile_2015 = "../data/autogsr_data_oct_dec_2015.json"
infile_2016 = "../data/autogsr_data_jan_2016.json"
keyword_file = "../data/search_keywords.txt"
Countries = [u'Chile', u'Mexico',u'El Salvador',u'Uruguay',u'Colombia',u'Paraguay',u'Argentina',u'Venezuela',u'Ecuador']

keyword = u'|'.join([w.strip().decode('utf-8') for w in open(keyword_file)])
pattern = re.compile(keyword, re.U)

# get all the protest and non_protest articles
protest_articles = {}
non_protest_articles = {}

tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
non_words = [u'¿', u'¡']

def get_sens(article, key_pattern):
    text = article["text"]
    text = ''.join([c for c in text if c not in non_words])
    text = re.sub(" +", " ", text)
    paragraphs = re.split("\n+", text)
    sens = []
    found = False
    for para in paragraphs:
        p_sens = tokenizer.tokenize(para.strip())
        for sen in p_sens:
            if len(sen.split(" ")) > 5:
                sens.append(sen)
    if len(sens) < 4:
        return []

    # do the keywords match
    found = False
    for sen in sens:
        matched = key_pattern.search(sen.lower())
        if matched:
            found = True
            break
    if not found:
        return []
    return sens

with open(infile_2015) as file_15, open(infile_2016) as file_16:
    lines = [line.strip() for line in file_15] + [line.strip() for line in file_16]
    for line in lines:
        article = json.loads(line)
        if article["country"] not in Countries:
            continue # only consider Spanish Countries
        sens = get_sens(article, pattern)
        if len(sens) == 0:
            continue # do not have sentences qualified

        article_key = article["text"][:300]
        if article["type"] == "Protest":
            encodings = article["encodings"]
            if article_key not in protest_articles:
                protest_articles[article_key] = {"sentences": sens, "pops": set(),
                        "types": set(), "label": "Protest"}
            for e in encodings:
                protest_articles[article_key]["pops"].add(e["population"])
                protest_articles[article_key]["types"].add(e["eventType"])
        else:
            if article_key not in non_protest_articles:
                non_protest_articles[article_key] = {"sentences": sens, "label": "Non_protest"}

# write out the result
non_pro_out = "../data/new_single_label/autogsr_non_protest.txt.tok"
pro_out = "../data/new_single_label/autogsr_protest.txt.tok"
pop_out = "../data/new_single_label/autogsr_pop_cat.txt"
type_out = "../data/new_single_label/autogsr_type_cat.txt"

mult_non_pro_out = "../data/new_multi_label/autogsr_non_protest.txt.tok"
mult_pro_out = "../data/new_multi_label/autogsr_protest.txt.tok"
mult_pop_out = "../data/new_multi_label/autogsr_pop_cat.txt"
mult_type_out = "../data/new_multi_label/autogsr_type_cat.txt"


with open(non_pro_out, 'w') as npo, open(pro_out, 'w') as po, open(pop_out, 'w') as pot, open(type_out, 'w') as tot, open(mult_non_pro_out, 'w') as mnpo, open(mult_pro_out, 'w') as mpo, open(mult_pop_out, 'w') as mpop_out, open(mult_type_out, 'w') as mto:
    for key, value in protest_articles.items():
        pops = list(value["pops"])
        types = list(value["types"])

        if len(pops) == 1 and len(types) == 1 and types[0] != "Other":
            out_text = json.dumps(value["sentences"], ensure_ascii=False, encoding='utf-8')
            if isinstance(out_text, unicode):
                out_text = out_text.encode('utf-8')
            po.write(out_text + "\n")

            pot.write(pops[0] + "\n")
            tot.write(types[0] + "\n")

        # write the output to multi label folder
        out_text = json.dumps(value["sentences"], ensure_ascii=False, encoding='utf-8')
        if isinstance(out_text, unicode):
            out_text = out_text.encode('utf-8')
        mpo.write(out_text + "\n")
        mpop_out.write("|".join(pops) + "\n")
        mto.write("|".join(types) + "\n")


    # out for non_protest_docs
    for key, value in non_protest_articles.items():
        out_text = json.dumps(value["sentences"], ensure_ascii=False, encoding='utf-8')
        if isinstance(out_text, unicode):
            out_text = out_text.encode('utf-8')
        npo.write(out_text + "\n")

        mnpo.write(out_text + "\n")

if __name__ == "__main__":
    pass

