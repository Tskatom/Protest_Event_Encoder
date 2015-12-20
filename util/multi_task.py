#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import numpy as np
import re
import sys
import os
from multiprocessing import Process, Queue, Pool, Manager
import json
import argparse
import glob
from dateutil import parser
from collections import Counter
import codecs
import gzip
from functools import partial

def create_rawtext_pool(args):
    infolder = args.inFolder
    files = glob.glob(os.path.join(infolder, "*"))
    param_list = []
    for f in files:
        params = {"dayFile": f, "outFolder": args.outFolder,}
        param_list.append(params)
    return rawtext_extract, param_list

def rawtext_extract(params):
    dayfile = params["dayFile"]
    outFolder = params["outFolder"]
    if dayfile.endswith(".gz"):
        day_f = gzip.open(dayfile)
    else:
        day_f = open(dayfile)
    basename = os.path.basename(dayfile)
    outfile = os.path.join(outFolder, basename)
    otf = open(outfile, 'w')
    for line in day_f:
        try:
            event = json.loads(line)
            basis = event["BasisEnrichment"]
            if basis.get("language", "") != "Spanish":
                continue
            tokens = basis["tokens"]
            token_str = ' '.join([t['value'].encode("utf-8").lower() for t in tokens])
            otf.write(token_str + "\n")
        except:
            print sys.exc_info()
    otf.flush()
    otf.close()
    day_f.close()


def write_out(infolder, outfile):
    with open(outfile, 'w') as out_f:
        files = glob.glob(os.path.join(infolder,"*"))
        for f in files:
            with open(f) as df:
                for line in df:
                    out_f.write(line)

            # remove the tempfile
            try:
                os.remove(f)
                sys.stdout.write('Remove File %s' % f)
                sys.stdout.flush()
            except:
                print sys.exc_info()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inFolder', type=str)
    ap.add_argument("--outFolder", type=str)
    ap.add_argument("--keywordsFile", type=str,
            default="./CU_Keywords.2013-01-25T15-36-29")
    ap.add_argument('--core', type=int)
    ap.add_argument('--task', type=str)
    ap.add_argument('--outFile', type=str)
    return ap.parse_args()


def main():
    args = parse_args()
    result_queue = None
    if args.outFolder and not os.path.exists(args.outFolder):
        os.mkdir(args.outFolder)

    if args.task == "countRss":
        func, param_list = create_count_length_pool(args)
    elif args.task == "rawtext":
        func, param_list = create_rawtext_pool(args)

    pool = Pool(processes=args.core)
    pool.map(func, param_list)
    pool.close()
    #pool.join()

    if args.task == "rawtext":
        write_out(args.outFolder, args.outFile)

    """
    if args.task == "countRss":
        task_count = create_count_length_tasks(args, task_queue)
    elif args.task == "batchCountRss":
        task_count = create_batch_count_task(args, task_queue)

    for i in range(args.core):
        Process(target=worker, args=(task_queue,result_queue)).start()
        task_queue.put('STOP')

    """
    """
    results ={}
    for i in range(task_count):
        r = result_queue.get()
        for month, countInfo in r.items():
            results.setdefault(month, {})
            for country, count in countInfo.items():
                results[month].setdefault(country, 0)
                results[month][country] += count

    print "------------------I am Done"
    outfolder = args.outFolder
    for year_month, countInfo in results.items():
        outfile = os.path.join(outfolder, year_month)
        with open(outfile, 'w') as otf:
            json.dump(countInfo, otf)
    print "----Finally------"
    """

if __name__ == "__main__":
    main()
