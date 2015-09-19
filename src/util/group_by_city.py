#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
from multiprocessing import Pool
import codecs
from glob import glob
import unicodedata
import gzip
import re
from dateutil import parser

def nstr(s,lower=True):
    if isinstance(s, str):
        s = s.decode("utf8")
    s = unicodedata.normalize("NFKD", s)
    if lower:
        return s.encode('ASCII', 'ignore').strip().lower()
    else:
        return s.encode('ASCII', 'ignore').strip()


target_cities = ["Barriloche", "Buenos Aires", "Córdoba",
                 "Rosario", "Mendoza", "Tucumán",
                 "La Plata", "Mar del Plata", "Salta",
                 "Santa Fe", "San Juan"]


def main():
    outfolder = "/raid/tskatom/rss_count/Argentina_10_Cities"
    infolder = "/raid/tskatom/rss_count/Argentina_analysis"
    files = glob(os.path.join(infolder, "*"))

    pattern = re.compile('\d{4}-\d{2}-\d{2}')
    for f in files:
        file_handlers = {}
        with open(f) as df:
            for line in df:
                try:
                    a = json.loads(line)
                except:
                    print '----', line
                    print '===', f
                    continue
                loc = a["geoCode"]
                city = nstr(loc["city"], lower=True)
                if city not in file_handlers:
                    city_folder = os.path.join(outfolder, city)
                    if not os.path.exists(city_folder):
                        os.mkdir(city_folder)
                day = a["date"]
                if day is None:
                    day = pattern.search(f).group()
                else:
                    day = parser.parse(day).strftime("%Y-%m-%d")
                file_handlers.setdefault(city, {})
                content = a["content"]
                content = content.replace("\n", " ")
                content = content.replace('\"', '"')
                content = re.sub('<[^<]+?>', '', content)
                if re.search(u'\ufffd', content):
                    continue
                if content[0] == '"':
                    content = content[1:]
                if content[-1] == '"':
                    content = content[:-1]
                #file_handlers[city][day].write(content.encode('utf-8') + "\n")
                if day not in file_handlers[city]:
                    try:
                        file_handlers[city][day] = codecs.open(os.path.join(city_folder, day), 'a', 'utf-8')
                    except:
                        print "city_folder:", city_folder
                        print "day: ", day
                file_handlers[city][day].write(content + u"\n")
        for s, h in file_handlers.items():
            for day, item in h.items():
                item.flush()
                item.close()

if __name__ == "__main__":
    main()
