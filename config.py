# -*- coding: utf-8 -*-
import sys
import datetime

TRAIN_DATE = sys.argv[1]
DATE_FORMAT = '%Y%m%d'
TEST_DATE = (datetime.datetime.strptime(TRAIN_DATE, DATE_FORMAT) + datetime.timedelta(days=1)).strftime(DATE_FORMAT)
print("TRAIN_DATE: %s, TEST_DATE: %s" % (TRAIN_DATE, TEST_DATE))

TRAIN_PATH = 'data/train/%s' % TRAIN_DATE
TEST_PATH = 'data/test/%s' % TEST_DATE

CKPT_DIR = 'ckpt'
MODEL_DIR = 'models'

SHUFFLE_SIZE = 16000000

TEST_BATCH_SIZE = 10000

PREFETCH_BATCH_NUM = 2

OPTIMIZER = "adam"

FEATURE_DIM = int(sys.argv[2])
print("FEATURE_DIM: %d" % FEATURE_DIM)

"""
# fm settings
MODEL_NAME = "fm"

STEPS_PER_CKPT = 5

BATCH_SIZE = 600000

FACTOR = 8

LEARNING_RATE = 1e-2

# adam, l2 regularization
#L2_REGU = False
L2_REGU = True
L2_LAMBDA = 0.0001

NUM_EPOCH = 12
"""

# nfm settings
MODEL_NAME = "nfm"

STEPS_PER_CKPT = 50

BATCH_SIZE = 80000

FACTOR = 64

HIDDEN_LAYERS = [16]
DROP_PROBS = [0.7, 0.7]

LEARNING_RATE = 1e-3

NUM_EPOCH = 8
