import math

import tensorflow.compat.v1 as tf

import config


# TODO: relu on mlp will make high order scale incompatible with first order?
def cal_logits(features, bias, ws, vs, hls_ws, hls_bias, ol_ws, test=tf.constant(True, dtype=tf.bool)):
    # first order term
    first_order = tf.sparse.sparse_dense_matmul(features, ws)

    # bi-interactions
    embedding = tf.sparse.sparse_dense_matmul(features, vs)
    embedding_square = tf.sparse.sparse_dense_matmul(tf.square(features), tf.square(vs))
    bi_interactions = 0.5 * tf.subtract(tf.square(embedding), embedding_square)

    # high order term
    x = bi_interactions
    for i in range(len(hls_ws)):
        x = tf.cond(test, true_fn=lambda: x, false_fn=lambda: tf.nn.dropout(x, rate=1.0 - config.DROP_PROBS[i]))
        x = tf.nn.relu_layer(x, hls_ws[i], hls_bias[i])

    high_order = tf.matmul(x, ol_ws)

    logits = tf.add(bias, first_order + high_order, name="logits")

    return logits


def cal_loss(labels, logits):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")

    return loss_op


class NFMModel(object):
    def __init__(self):
        self.feature_dim = config.FEATURE_DIM
        self.factor = config.FACTOR

        self.bias = tf.get_variable("bias", [1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.ws = tf.get_variable("ws", [self.feature_dim, 1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.vs = tf.get_variable("vs", [self.feature_dim, self.factor], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01 / math.sqrt(self.factor)))

        self.hls_ws = []
        self.hls_bias = []
        hidden_layers = [config.FACTOR] + config.HIDDEN_LAYERS
        for i in range(len(config.HIDDEN_LAYERS)):
            self.hls_ws.append(tf.get_variable("hls_ws_%d" % i, [hidden_layers[i], hidden_layers[i + 1]], dtype=tf.float32,
                                               initializer=tf.glorot_normal_initializer()))
            self.hls_bias.append(tf.get_variable("hls_bias_%d" % i, [hidden_layers[i + 1]], dtype=tf.float32,
                                                 initializer=tf.glorot_normal_initializer()))

        self.ol_ws = tf.get_variable("ol_ws", [hidden_layers[-1], 1], dtype=tf.float32,
                                     initializer=tf.glorot_normal_initializer())
