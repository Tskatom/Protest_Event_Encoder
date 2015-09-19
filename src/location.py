#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The module works on extract the location information from the text
"""

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import codecs
import re


class Location:
    def __init__(self):
        self.city_file = None
        self.entity_file = None

    def load_citydata(self, city_file, country_codes):
        """
        Loading the city information from world gazetteer file
        return diction with city_name/ascii as key and city_id as value
        we use city1000 file to filter the cities
        """
        self.city_data = {}
        with codecs.open(city_file, encoding='utf-8') as ctf:
            for l in ctf:
                # cid, ori_cname, asc_cname, gid = l.strip().split('\t')
                infos = l.strip().split('\t')
                coun_code = infos[8]
                gid = infos[0]
                ori_cname = infos[1]
                ori_cname = re.sub("\([^\(]+?\)", '', ori_cname).strip()
                asc_cname = infos[2]
                asc_cname = re.sub("\([^\(]+?\)", '', asc_cname).strip()
                if coun_code in country_codes:
                    self.city_data.setdefault(ori_cname, [])
                    self.city_data.setdefault(asc_cname, [])
                    self.city_data[ori_cname].append((coun_code, gid))
                    if asc_cname != ori_cname:
                        self.city_data[asc_cname].append((coun_code, gid))

        # construct the city_search rule
        self.city_rule = u"|".join(self.city_data.keys())
        self.city_pattern = re.compile(self.city_rule, re.U)

    def extract_city(self, text):
        if isinstance(text, str):
            text = text.decode('utf-8')
        matched = self.city_pattern.findall(text)
        if len(matched) == 0:
            return None
        else:
            return [(list([item[0] for item in self.city_data[m]]), m) for m in matched]


def test():
    loc = Location()
    loc.load_citydata('../data/admin2Codes.txt', ['AR'])

if __name__ == "__main__":
    test()
