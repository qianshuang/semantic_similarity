# -*- coding: utf-8 -*-

from annoy import AnnoyIndex

dim = 768  # 向量维度
n_trees = 50  # 树越多越准确


def build_index(vecs):
    index = AnnoyIndex(dim, "dot")
    for i in range(len(vecs)):
        index.add_item(i, vecs[i])
    index.build(n_trees)
    index.save('data/index')
    return index


def load_index():
    index = AnnoyIndex(dim, 'dot')
    index.load('data/index')  # super fast, will just mmap the file
    return index


def search(index_, vec, cnt):
    res = index_.get_nns_by_vector(vec, cnt, include_distances=True)
    return res
