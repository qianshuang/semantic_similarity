# -*- coding: utf-8 -*-

import datetime
import re
import string
from zhon.hanzi import punctuation


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    return [line.strip() for line in open(filename).readlines() if line.strip() != ""]


def write_file(filename, content):
    open_file(filename, mode="w").write(content)


def append_file(filename, content):
    open_file(filename, mode="a").write(content)


def write_lines(filename, list_res):
    test_w = open_file(filename, mode="w")
    for j in list_res:
        test_w.write(j + "\n")


def time_cost(start, type_="sec"):
    interval = datetime.datetime.now() - start
    if type_ == "sec":
        return interval.total_seconds()
    elif type_ == "day":
        return interval.days


def pre_process(query):
    # 1. 转小写
    query = query.lower()

    # 2. 去标点
    query = re.sub(r"[%s]+" % punctuation, " ", query)
    for c in string.punctuation:
        query = query.replace(c, " ")

    # 3. 去空格
    query = re.sub(r'\s+', ' ', query)
    return query
