#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", type=str, default="../data/seed_keywords.txt")
    ap.add_argument("--es_file", type=str, default="../data/spanish_protest.txt.tok")
    ap.add_argument("--en_file", type=str, default="../data/english_protest.txt.tok")
    ap.add_argument("--pop_file", type=str, default="../data/pop_label.txt")
    ap.add_argument("--type_file", type=str, default="../data/type_label.txt")
    return ap.parse_args()

def clean(esfile, enfile, word_file):
    words = [w.strip() for w in open(word_file)]
    rule = '|'.join(words)
    print rule
    pattern = re.compile(rule, re.I)
    clean_name = os.path.join('../data/', "cleaned_" + os.path.basename(esfile))
    noise_name = os.path.join('../data/', "noise_" + os.path.basename(esfile))
    en_clean_name = os.path.join('../data/', "cleaned_" + os.path.basename(enfile))
    en_noise_name = os.path.join('../data/', "noise_" + os.path.basename(enfile))
    with open(esfile) as f, open(clean_name, 'w') as cn, open(noise_name, 'w') as nn, open(enfile) as ef, open(en_clean_name, 'w') as ecn, open(en_noise_name, 'w') as enn:
        en_texts = [text for text in ef]
        for id, text in enumerate(f):
            if isinstance(text, unicode):
                text = text.encode('utf-8').lower()
            matched = pattern.search(text)
            if matched:
                cn.write(text)
                ecn.write(en_texts[id])
            else:
                nn.write(text)
                enn.write(en_texts[id])


def main():
    args = parse_args()
    esfile = args.es_file
    enfile = args.en_file
    keyword = args.keyword
    clean(esfile, enfile, keyword)

if __name__ == "__main__":
    main()

