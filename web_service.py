# -*- coding: utf-8 -*-

import shutil

from flask import Flask, jsonify
from flask import request
from gevent import pywsgi

from helper import *

app = Flask(__name__)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    input json:
    {
        "bot_name": "xxxxxx",
        "threshold": 0.95,  # 相似度阈值
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
    threshold = float(resq_data["threshold"]) if "threshold" in resq_data else 0.9

    query = pre_process(query)
    if query == "":
        return {'code': 0, 'msg': 'success', 'data': []}

    # 缓存加速
    if query in bot_intent_vecs[bot_n]:
        q_v = bot_intent_vecs[bot_n][query]
    else:
        q_v = get_bert_sent_vecs(query)[0]
    final_res = cos_simi_search(bot_n, q_v, threshold, size)
    return jsonify({'code': 0, 'msg': 'success', 'data': final_res})


@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    """
    更新intents.txt、priority.txt后，需要手动刷新才生效（重新计算语义向量）
    {
        "bot_name": "xxxxxx",  # 要操作的bot name
        "operate": "upsert",  # 操作。upsert：更新或新增；delete：删除
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    bot_n = resq_data["bot_name"].strip()
    operate = resq_data["operate"].strip()
    src_bot_name = resq_data["src_bot_name"].strip() if "src_bot_name" in resq_data else ""

    if operate == "upsert":
        build_bot_intents_dict(bot_n)
        build_bot_vecs_dict(bot_n)
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
    elif operate == "copy":
        # 复制bot。一方面不用从头训练，直接复用原始bot的能力；另一方面避免误删除bot
        if src_bot_name == "":
            return {'code': -1, 'msg': 'parameter error', 'time_cost': time_cost(start)}
        if bot_n in os.listdir(BOT_SRC_DIR):
            return {'code': -1, 'msg': bot_n + ' already exists', 'time_cost': time_cost(start)}
        src_bot_path = os.path.join(BOT_SRC_DIR, src_bot_name)
        bot_path = os.path.join(BOT_SRC_DIR, bot_n)
        shutil.copytree(src_bot_path, bot_path)

        bot_intents[bot_n] = bot_intents[src_bot_name]
        bot_intent_vecs[bot_n] = bot_intent_vecs[src_bot_name]
        bot_vecs[bot_n] = bot_vecs[src_bot_name]
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
    elif operate == "delete":
        # 删除bot
        try:
            shutil.rmtree(os.path.join(BOT_SRC_DIR, bot_n))
            del bot_intents[bot_n]
            del bot_intent_vecs[bot_n]
            del bot_vecs[bot_n]
        except:
            print(bot_n, "deleted already...")
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
    else:
        return {'code': -1, 'msg': 'unsupported operation', 'time_cost': time_cost(start)}


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
    # app.run(debug=False, threaded=True, host='0.0.0.0', port=8088)
