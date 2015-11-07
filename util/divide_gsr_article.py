#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json

outfolder = "/raid/tskatom/event_encoder/spanish_gsr"
gsr_article = "/raid/parang/auto-gsr/dev/auto-gsr/data/gsr_articles/enriched_gsr_articles/2011-01_2014-06_enriched_articles"
file_handlers = {}
linenum = 0
with open(gsr_article) as gsr:
    for line in gsr:
        linenum += 1
        try:
            event = json.loads(line)
        except:
            print "Error linenum:", linenum
            continue
        country = event["location"][0]
        population = event["population"]
        population = population.replace("/", "_")
        eventtype = event["eventType"][:3]
        lan = event["BasisEnrichment"]["language"]
        if lan != "Spanish":
            continue

        tokens = event["BasisEnrichment"]["tokens"]
        content = ' '.join([t["value"] for t in tokens])
        event["content"] = content

        country_path = os.path.join(outfolder, country)
        if not os.path.exists(country_path):
            os.mkdir(country_path)

        pop_folder = os.path.join(country_path, "population")
        if not os.path.exists(pop_folder):
            os.mkdir(pop_folder)
        type_folder = os.path.join(country_path, "eventType")
        if not os.path.exists(type_folder):
            os.mkdir(type_folder)

        line = json.dumps(event, ensure_ascii=False)
        if isinstance(line, unicode):
            line = line.encode('utf-8')
        line += "\n"
        file_handlers.setdefault(country, {})
        pop_file = os.path.join(pop_folder, population)
        if population not in file_handlers[country]:
            file_handlers[country][population] = open(pop_file, 'w')
        file_handlers[country][population].write(line)

        type_file = os.path.join(type_folder, eventtype)
        if eventtype not in file_handlers[country]:
            file_handlers[country][eventtype] = open(type_file, 'w')
        file_handlers[country][eventtype].write(line)

# flush the file handlers
for country, files in file_handlers.items():
    for fname, f in files.items():
        f.flush()
        f.close()
