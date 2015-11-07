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
import goose
import re

def nstr(s,lower=True):
    if isinstance(s, str):
        s = s.decode("utf8")
    s = unicodedata.normalize("NFKD", s)
    if lower:
        return s.encode('ASCII', 'ignore').strip().lower()
    else:
        return s.encode('ASCII', 'ignore').strip()


target_cities = ["Bariloche", "Buenos Aires", "Córdoba",
                 "Rosario", "Mendoza", "Tucumán",
                 "La Plata", "Mar del Plata", "Salta",
                 "Santa Fe", "San Juan"]


def filter_articles(params):
    dayFile = params["dayFile"]
    outFolder = params["outFolder"]
    outfile = os.path.join(outFolder,
                           os.path.basename(dayFile))
    n_targs = map(nstr, target_cities)
    linenum = 0
    g = goose.Goose()
    with codecs.open(dayFile) as dayf, open(outfile, 'w') as outf:
        for line in dayf:
            try:
                linenum += 1
                if len(line.strip()) == 0:
                    continue
                article = json.loads(line)
                lan = article["BasisEnrichment"]["language"]
                if lan != "Spanish":
                    continue
                loc = article["embersGeoCode"]
                city = loc["city"]
                nor_city = nstr(city)
                if nor_city in n_targs:
                    doc = {}
                    content = article['content']
                    if re.search("</html>", content):
                        content = g.extract(raw_html=content).cleaned_text
                        if len(content.strip()) == 0:
                            continue
                    doc["content"] = content
                    doc["geoCode"] = article["embersGeoCode"]
                    doc["date"] = article["date"]
                    doc_str = json.dumps(doc, ensure_ascii=False)
                    if not isinstance(doc_str, str):
                        doc_str = doc_str.encode('utf-8')
                    outf.write(doc_str + '\n')
            except:
                print "File %s Error, linenum: %d" % (dayFile, linenum)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_type, exc_tb.tb_lineno)

def main():
    infolder = "/raid/tskatom/rss_count/protest/Argentina"
    outfolder = "/raid/tskatom/rss_count/Argentina_analysis"
    files = glob(os.path.join(infolder, "*"))
    para_list = []
    for f in files:
        params = {}
        params["dayFile"] = f
        params["outFolder"] = outfolder
        para_list.append(params)
    core = 40
    pol = Pool(processes=core)
    pol.map(filter_articles, para_list)
    pol.close()
    pol.join()

if __name__ == "__main__":
    main()
