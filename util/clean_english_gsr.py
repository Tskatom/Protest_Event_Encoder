#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json


en_gsr = "../data/en_gsr.txt"
out_file = "../data/en_gsr_artricle.txt"

with open(en_gsr) as eg, open(out_file, 'w') as otf:
    for line in eg:
        event = json.loads(line)
        keys = ["eventDate", "location","population", "eventType","en_content"]
        content = [event[k].encode('utf-8') if k!= "location" else json.dumps(event[k], ensure_ascii=False).encode('utf-8') for k in keys]
        content = "  |  ".join(content)
        otf.write(content + "\n")

if __name__ == "__main__":
    pass

