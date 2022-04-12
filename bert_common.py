# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from bert import tokenization
from bert import modeling

max_seq_length = 128

vocab_file = "bert/uncased_L-12_H-768_A-12/vocab.txt"
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

bert_config_file = "bert/uncased_L-12_H-768_A-12/bert_config.json"
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

bert_ckpt = "bert/uncased_L-12_H-768_A-12/bert_model.ckpt"

# 初始化BERT
input_ids = tf.placeholder(tf.int32, shape=[None, None])
input_mask = tf.placeholder(tf.int32, shape=[None, None])
segment_ids = tf.placeholder(tf.int32, shape=[None, None])
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids
)

# 加载BERT模型
(assignment, _) = modeling.get_assignment_map_from_checkpoint(tf.trainable_variables(), bert_ckpt)
tf.train.init_from_checkpoint(bert_ckpt, assignment)
# 获取最后一层和倒数第二层
# encoder_layer = model.get_sequence_output()  # = model.all_encoder_layers[-1]
encoder_layer = model.all_encoder_layers[-2]
# 创建session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def convert_single_example(text):
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids_ = []
    tokens.append("[CLS]")
    segment_ids_.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids_.append(0)
    tokens.append("[SEP]")
    segment_ids_.append(0)

    input_ids_ = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask_ = [1] * len(input_ids_)
    while len(input_ids_) < max_seq_length:
        input_ids_.append(0)
        input_mask_.append(0)
        segment_ids_.append(0)

    return input_ids_, input_mask_, segment_ids_


def get_bert_sent_vecs(sent_list):
    feed_dict = {input_ids: [], input_mask: [], segment_ids: []}
    for line in sent_list:
        input_ids_, input_mask_, segment_ids_ = convert_single_example(line)
        feed_dict[input_ids].append(input_ids_)
        feed_dict[input_mask].append(input_mask_)
        feed_dict[segment_ids].append(segment_ids_)
    last2 = sess.run(encoder_layer, feed_dict=feed_dict)  # (batch_size, max_len, 768)
    q_v = last2[:, 0, :]  # (batch_size, 768)
    return list(q_v)
