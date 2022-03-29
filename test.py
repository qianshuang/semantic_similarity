# -*- coding: utf-8 -*-

# from bert_common import *

# from bert import tokenization
#
# vocab_file = "bert/uncased_L-12_H-768_A-12/vocab.txt"
# tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
# tokens_a = tokenizer.tokenize("我是中fdhgfdjhgfj国人")
# print(tokens_a)

# intent_score_dict = {"q": 1, "a": 3, "b": 2}
# print(sorted(intent_score_dict.items(), key=lambda kv: {kv[1], kv[0]}, reverse=True))
# print(sorted(intent_score_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))

# vec = get_bert_sent_vecs(["我是中国人"])
# print(vec)
# print(len(vec))
# print(len(vec[0]))
# print(np.shape(vec))

# last2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print(last2[:, 0, :])
# from hanlp.utils.tf_util import NumpyEncoder
#
# from common import *
# import json
# import os
#
# VEC_FILE_ = os.path.join("bot_resources", "bot1", "intent_vec.json")
# intent_vec_dict = {"我是中国人": np.array([1.0, 2.0, 3.0])}
# open_file(VEC_FILE_, mode='w').write(json.dumps(intent_vec_dict, ensure_ascii=False, cls=NumpyEncoder))
# with open(VEC_FILE_, encoding="utf-8") as f:
#     intent_vec_dict = json.load(f)
# print(intent_vec_dict)

# from scipy.spatial.distance import cosine
#
# cos_simis = [(1 - cosine(qv, [2, 3, 4])) for qv in [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
# print(cos_simis)

import redis
from common import *

r = redis.Redis(db=1)
print(r_get_pickled(r, "111", "111"))
