# -*- coding: utf-8 -*-

from config import *
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def rsync(bot_n):
    bot_version = r.hget("bot_version", bot_n)
    bot_process_version = r.hget("bot_version&" + bot_n, str(os.getpid()))

    if bot_version != bot_process_version or bot_n not in bot_intents or bot_n not in bot_intent_vecs or bot_n not in bot_vecs:
        print("starting rsync...")
        bot_intents[bot_n] = r_get_pickled(r, "bot_intents", bot_n)
        bot_intent_vecs[bot_n] = r_get_pickled(r, "bot_intent_vecs", bot_n)
        bot_vecs[bot_n] = r_get_pickled(r, "bot_vecs", bot_n)

        r.hset("bot_version&" + bot_n, str(os.getpid()), int(bot_version))


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


def integrity(intents):
    cnt = len(intents)

    # 缓存取数
    cache_sent_vecs = {}
    for bn in os.listdir(BOT_SRC_DIR):
        if bn in bot_intent_vecs:
            cache_sent_vecs.update(bot_intent_vecs[bn])

    uncache_sents = [i for i in intents if i not in cache_sent_vecs]
    uncache_vecs = get_bert_sent_vecs(uncache_sents)
    uncache_sent_vecs = dict(zip(uncache_sents, uncache_vecs))

    cache_sent_vecs.update(uncache_sent_vecs)
    q_vs = [cache_sent_vecs[i] for i in intents]

    cos_simi_m = cosine_similarity(q_vs)
    return (np.sum(cos_simi_m) - cnt) / (cnt * (cnt - 1))

# def annoy_search(bot_n, q_v, threshold):
#     """
#     向量检索：
#     1. annoy向量检索在样本量百万量级以上时，才能显示真正威力，几千上万条数据，意义不大。
#     2. demo已实现，参见annoy_demo.py & annoy_common.py
#     """
