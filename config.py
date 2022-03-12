# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from hanlp.utils.tf_util import NumpyEncoder

from common import *
from bert_common import *

BOT_SRC_DIR = "bot_resources"

bot_intents = {}
bot_intent_vecs = {}
bot_vecs = {}


def build_bot_intents_dict(bot_name):
    INTENT_FILE_ = os.path.join(BOT_SRC_DIR, bot_name, "intents.txt")
    intents = []
    for intent_ in read_file(INTENT_FILE_):
        intent_pro_ = pre_process(intent_)
        if intent_pro_ != "":
            intents.append(intent_)
    bot_intents[bot_name] = np.array(intents)


def build_bot_vecs_dict(bot_name):
    VEC_FILE_ = os.path.join(BOT_SRC_DIR, bot_name, "intent_vec.json")
    if os.path.exists(VEC_FILE_):
        with open(VEC_FILE_, encoding="utf-8") as f:
            intent_vec_dict = json.load(f)
    else:
        intent_vec_dict = {}

    intent_vecs = []
    for intent in bot_intents[bot_name]:
        intent_ = pre_process(intent)
        if intent_ not in intent_vec_dict:
            q_v = get_bert_sent_vecs(intent_)[0]
            intent_vec_dict[intent_] = q_v
        intent_vecs.append(intent_vec_dict[intent_])

    bot_intent_vecs[bot_name] = intent_vec_dict
    bot_vecs[bot_name] = intent_vecs

    # 序列化
    open_file(VEC_FILE_, mode='w').write(json.dumps(intent_vec_dict, ensure_ascii=False, cls=NumpyEncoder))


for bot_na in os.listdir(BOT_SRC_DIR):
    # build bot intent dict
    build_bot_intents_dict(bot_na)
    print(bot_na, "intents dict finished building...")

    # build bot vec dict
    build_bot_vecs_dict(bot_na)
    print(bot_na, "vecs dict finished building...")
