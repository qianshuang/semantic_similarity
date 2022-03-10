# -*- coding: utf-8 -*-

import tensorflow as tf

from common import *
from bert_common import *
from annoy_common import *

# 得到bert向量
predict_fn = tf.contrib.predictor.from_saved_model("model")

sents = [line_.split("\t")[1] for line_ in read_file("data/data.txt")]
bert_sent_vecs = get_bert_sent_vecs(predict_fn, sents)
print("get bert vectors finished...")

# 构建faiss索引
index = build_index(bert_sent_vecs)
# index = load_index()
print("build annoy index finished...")

# 查询
query_vec = get_bert_sent_vecs(predict_fn, ["I see a cup of coffee animation"])[0]  # 不支持批量查询
ids, scores = search(index, query_vec, 5)
for i in range(5):
    print(sents[ids[i]], scores[i])
