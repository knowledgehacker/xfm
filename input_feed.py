import tensorflow.compat.v1 as tf

import config
from utils import current_time, load_files

tf.disable_v2_behavior()


def create_dataset(input):
    print(current_time(), "create_dataset starts...")

    with tf.device('/cpu:0'):
        tfrecords_files = [file for file in list(load_files("%s/tfrecords" % input))]
        dataset = tf.data.Dataset.list_files(tfrecords_files)
        dataset = dataset.flat_map(lambda tfrecords_file: tf.data.TFRecordDataset(tfrecords_file))
        # dataset = tf.data.TFRecordDataset(tfrecords_files)

    print(current_time(), "create_dataset finishes...")

    return dataset


def transform_dataset(dataset, test):
    with tf.device('/cpu:0'):
        if not test:
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)
            dataset = dataset.batch(config.BATCH_SIZE)
        else:
            dataset = dataset.batch(config.TEST_BATCH_SIZE)

        dataset = dataset.map(parse_batch, num_parallel_calls=32)

        dataset = dataset.prefetch(config.PREFETCH_BATCH_NUM)

    return dataset


def parse_batch(records):
    examples = tf.io.parse_example(records,
        features={
            'indices': tf.VarLenFeature(tf.int64),
            'values': tf.VarLenFeature(tf.float32),
            'label': tf.FixedLenFeature([1], tf.float32)
        })

    indices = examples['indices']
    batch_indices = tf.stack([indices.indices[:, 0], indices.values], 1)

    return batch_indices, examples['values'].values, [tf.shape(indices)[0], config.FEATURE_DIM], examples['label']
