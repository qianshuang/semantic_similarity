# -*- coding: utf-8 -*-

import json

from flask import Flask, jsonify
from flask import request
from gevent import pywsgi

from common import *
from bert_common import *

app = Flask(__name__)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    input json:
    {
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

    query = resq_data["query"]
    size = int(resq_data["size"]) if "size" in resq_data else 5
    threshold = float(resq_data["threshold"]) if "threshold" in resq_data else 0.9

    query = pre_process(query).strip()
    if query == "":
        return {'code': 0, 'msg': 'success', 'data': []}

    return {'code': 0, 'msg': 'success', 'data': final_res[:size]}


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
    # app.run(debug=False, threaded=True, host='0.0.0.0', port=8088)
