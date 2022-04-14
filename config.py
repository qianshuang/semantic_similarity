# -*- coding: utf-8 -*-

import threading
import time

import redis
import redis_lock

import os
import numpy as np
import schedule

from common import *
from bert_common import *

BOT_SRC_DIR = "bot_resources"

bot_intents = {}
bot_intent_vecs = {}
bot_vecs = {}

r = redis.Redis(db=1)
global_lock = redis_lock.Lock(r, "global_lock")

# 设置bot expire time，3天未被调用则自动移除bot，释放内存空间
bot_expire_time = 3 * 24 * 3600


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
    proed_intents = [pre_process(i) for i in bot_intents[bot_name] if pre_process(i) != ""]
    intent_vecs = get_bert_sent_vecs(proed_intents)
    intent_vec_dict = dict(zip(proed_intents, intent_vecs))

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


# 每天回收不常被使用bot内存
def run_recycle():
    for _bot_name_ in os.listdir(BOT_SRC_DIR):
        bot_ex = r.hget("bot_invoke", _bot_name_)
        if bot_ex is not None and time.time() - float(bot_ex) > bot_expire_time:
            print(_bot_name_, 'starting recycle...')
            del_dict_key(bot_intents, _bot_name_)
            del_dict_key(bot_intent_vecs, _bot_name_)
            del_dict_key(bot_vecs, _bot_name_)


schedule.every().day.do(run_recycle)


# 多线程调度
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)


threading.Thread(target=run_schedule).start()
