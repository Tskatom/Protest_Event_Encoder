#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Translate the Spanish GSR to English
"""

from textblob import TextBlob
import json
import re
import time
import sys

def translate(gsr_file):
    with open(gsr_file) as gf, open('./data/en_gsr.txt', 'w') as ef:
        for line in gf:
            try:
                event = json.loads(line)
                content = event["content"]
                content = re.sub('\\"', '"', content)
                blob = TextBlob(content)
                en_blob = blob.translate(to="en")
                en_content = en_blob.raw
                event["en_content"] = en_content

                dumps = json.dumps(event, ensure_ascii=False)
                if isinstance(dumps, unicode):
                    dumps = dumps.encode('utf-8')
                ef.write(dumps + "\n")
            except:
                print sys.exc_info()
                time.sleep(1)
                continue

def main():
    gsr_file = "./data/gsr_spanish.txt"
    translate(gsr_file)

if __name__ == "__main__":
    main()
