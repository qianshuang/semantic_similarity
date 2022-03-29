# -*- coding: utf-8 -*-

import redis
import redis_lock

import os
import numpy as np

from common import *
from bert_common import *

BOT_SRC_DIR = "bot_resources"

bot_intents = {}
bot_intent_vecs = {}
bot_vecs = {}

r = redis.Redis(db=1)
global_lock = redis_lock.Lock(r, "global_lock")


def build_bot_intents_dict(bot_name):
    INTENT_FILE_ = os.path.join(BOT_SRC_DIR, bot_name, "intents.txt")
    intents = []
    for intent_ in read_file(INTENT_FILE_):
        intent_pro_ = pre_process(intent_)
        if intent_pro_ != "":
            intents.append(intent_)
    bot_intents[bot_name] = np.array(intents)
    r_set_pickled(r, "bot_intents", bot_name, np.array(intents))


def build_bot_vecs_dict(bot_name):
    intent_vecs = []
    intent_vec_dict = {}

    for intent in bot_intents[bot_name]:
        intent_ = pre_process(intent)
        q_v = get_bert_sent_vecs(intent_)[0]

        intent_vec_dict[intent_] = q_v
        intent_vecs.append(q_v)

    bot_intent_vecs[bot_name] = intent_vec_dict
    bot_vecs[bot_name] = intent_vecs

    r_set_pickled(r, "bot_intent_vecs", bot_name, intent_vec_dict)
    r_set_pickled(r, "bot_vecs", bot_name, intent_vecs)

    # 序列化
    # open_file(VEC_FILE_, mode='w').write(json.dumps(intent_vec_dict, ensure_ascii=False, cls=NumpyEncoder))


redis_lock.reset_all(r)

if global_lock.acquire(blocking=False):
    for bot_na in os.listdir(BOT_SRC_DIR):
        redis_lock.Lock(r, "lock_" + bot_na)

        # 创建bot version
        r.hdel("bot_version", bot_na)
        r.hdel("bot_version&" + bot_na, str(os.getpid()))
        r_v_ = r.hincrby("bot_version", bot_na)
        r.hset("bot_version&" + bot_na, str(os.getpid()), r_v_)

        # build bot intent dict
        build_bot_intents_dict(bot_na)
        print(bot_na, "intents dict finished building...")

        # build bot vec dict
        build_bot_vecs_dict(bot_na)
        print(bot_na, "vecs dict finished building...")
