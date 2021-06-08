import time
import os

import tensorflow.compat.v1 as tf

import config


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


def load_files(input):
    files = []
    for dir in os.listdir(input):
        path = os.path.join(input, dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if not file.endswith("_SUCCESS"):
                    files.append(os.path.join(path, file))
        else:
            if not path.endswith("_SUCCESS"):
                files.append(path)

    return files


def get_optimizer():
    if config.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    elif config.OPTIMIZER == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=config.LEARNING_RATE, initial_accumulator_value=1e-8)
    else:
        print("Unsupported optimizer: %s" % config.OPTIMIZER)

    return optimizer


def with_prefix(prefix, op):
    #return "%s/%s" % (prefix, op)
    return op
