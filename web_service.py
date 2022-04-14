# -*- coding: utf-8 -*-

import json
import shutil

from flask import Flask, jsonify
from flask import request

from helper import *

app = Flask(__name__)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    input json:
    {
        "bot_name": "xxxxxx",
        "threshold": 0.95,  # 相似度阈值，默认0.8
        "query": "xxxxxx",  # 用户query
        "size": 10          # 最大返回大小，默认5
    }

    return:
    {   'code': 0,
        'msg': 'success',
        'data': []
    }
    """
    ori_rd = request.get_data(as_text=True)
    ori_rd = ori_rd.replace('\\"', ',')
    ori_rd = ori_rd.replace('\\', '')
    resq_data = json.loads(ori_rd)

    query = resq_data["query"].strip()
    bot_n = resq_data["bot_name"].strip()
    size = int(resq_data["size"]) if "size" in resq_data else 5
    threshold = float(resq_data["threshold"]) if "threshold" in resq_data else 0.8

    query = pre_process(query)
    if query == "":
        return {'code': 0, 'msg': 'success', 'data': []}

    # 刷新bot调用时间
    r.hset("bot_invoke", bot_n, time.time())

    # 同步Redis缓存
    rsync(bot_n)

    # 缓存加速
    if query in bot_intent_vecs[bot_n]:
        q_v = bot_intent_vecs[bot_n][query]
    else:
        q_v = get_bert_sent_vecs(query)[0]
    final_res = cos_simi_search(bot_n, q_v, threshold, size)
    return jsonify({'code': 0, 'msg': 'success', 'data': final_res})


@app.route('/intent_integrity', methods=['GET', 'POST'])
def intent_integrity():
    """
    意图语料完整度
    input json:
    {
        "intents": ["xxxxxx", "xxxxxx", "xxxxxx", "xxxxxx"]
    }

    return:
    {   'code': 0,
        'msg': 'success',
        'score': o.95
    }
    """
    ori_rd = request.get_data(as_text=True)
    ori_rd = ori_rd.replace('\\"', ',')
    ori_rd = ori_rd.replace('\\', '')
    resq_data = json.loads(ori_rd)

    intents = resq_data["intents"]
    intents = [pre_process(i) for i in intents if pre_process(i) != ""]
    if len(intents) == 0:
        return {'code': -1, 'msg': 'input is null', 'score': 0}
    elif len(intents) == 1:
        return {'code': 0, 'msg': 'success', 'score': 1}

    final_res = integrity(intents)
    return {'code': 0, 'msg': 'success', 'score': final_res}


@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    """
    更新intents.txt后，需要手动刷新才生效（重新计算语义向量）
    {
        "bot_name": "xxxxxx",  # 要操作的bot name
        "operate": "upsert",  # 操作。upsert：更新或新增；delete：删除
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    bot_n = resq_data["bot_name"].strip()
    operate = resq_data["operate"].strip()

    bot_lock = redis_lock.Lock(r, "lock_" + bot_n)
    if bot_lock.acquire(blocking=False):
        if operate == "upsert":
            build_bot_intents_dict(bot_n)
            build_bot_vecs_dict(bot_n)

            r_v = r.hincrby("bot_version", bot_n)
            r.hset("bot_version&" + bot_n, str(os.getpid()), r_v)

            ret = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
        elif operate == "delete":
            # 删除bot
            bot_path = os.path.join(BOT_SRC_DIR, bot_n)
            if os.path.exists(bot_path):
                shutil.rmtree(bot_path)
            r.delete("bot_version&" + bot_n)
            r.hdel("bot_version", bot_n)
            r.hdel("bot_intents", bot_n)
            r.hdel("bot_intent_vecs", bot_n)
            r.hdel("bot_vecs", bot_n)
            r.hdel("bot_invoke", bot_n)
            del_dict_key(bot_intents, bot_n)
            del_dict_key(bot_intent_vecs, bot_n)
            del_dict_key(bot_vecs, bot_n)
            ret = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
        else:
            ret = {'code': -1, 'msg': 'unsupported operation', 'time_cost': time_cost(start)}

        bot_lock.release()
        return ret
    else:
        return {'code': -2, 'msg': 'someone is refreshing this bot, please wait.', 'time_cost': time_cost(start)}


if __name__ == '__main__':
    app.run()
