import math

import tensorflow.compat.v1 as tf

import config


def cal_logits(features, bias, ws, vs):
    # first order term
    first_order = tf.sparse.sparse_dense_matmul(features, ws)

    # second order term
    embedding = tf.sparse.sparse_dense_matmul(features, vs)
    embedding_square = tf.sparse.sparse_dense_matmul(tf.square(features), tf.square(vs))
    second_order = tf.reduce_sum(tf.subtract(tf.square(embedding), embedding_square), axis=1, keepdims=True)

    logits = tf.add(bias, first_order + 0.5 * second_order, name="logits")

    return logits


def cal_loss(labels, logits):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")

    return loss_op


class FMModel(object):
    def __init__(self):
        self.feature_dim = config.FEATURE_DIM
        self.factor = config.FACTOR

        self.bias = tf.get_variable("bias", [1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.ws = tf.get_variable("ws", [self.feature_dim, 1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.vs = tf.get_variable("vs", [self.feature_dim, self.factor], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01 / math.sqrt(self.factor)))
