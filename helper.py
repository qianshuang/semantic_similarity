# -*- coding: utf-8 -*-
import numpy as np

from config import *
from scipy.spatial.distance import cosine


# from sklearn.metrics.pairwise import cosine_similarity


def cos_simi_search(bot_n, q_v, threshold, size):
    q_vs = bot_vecs[bot_n]

    # 计算相似度矩阵较慢
    # q_vs.insert(0, q_v)
    # cos_simis = cosine_similarity(q_vs)[0][1:]
    cos_simis = np.array([(1 - cosine(qv, q_v)) for qv in q_vs])

    res_idxes = [i for i in range(len(cos_simis)) if cos_simis[i] >= threshold]
    final_intents = bot_intents[bot_n][res_idxes]
    final_scores = cos_simis[res_idxes]
    intent_score_dict = dict(zip(final_intents, final_scores))
    res = sorted(intent_score_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:size]
    return [{i[0]: i[1]} for i in res]


def annoy_search(bot_n, q_v, threshold):
    """
    向量检索：
    1. annoy向量检索在样本量百万量级以上时，才能显示真正威力，几千上万条数据，意义不大。
    2. demo已实现，参见annoy_demo.py & annoy_common.py
    """
    pass
