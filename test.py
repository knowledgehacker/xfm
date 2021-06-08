# -*- coding: utf-8 -*-
import os

import tensorflow.compat.v1 as tf

import config
from input_feed import create_dataset, transform_dataset
from utils import current_time, with_prefix

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def test():
    print(current_time(), "test starts...")

    g = tf.Graph()
    with tf.Session(graph=g, config=cfg) as sess:
        # load trained model
        model_name = config.MODEL_NAME
        load_ckpt_model(sess, config.CKPT_DIR)

        # create iterator for train dataset
        handle_ph = g.get_tensor_by_name(with_prefix(model_name, "handle_ph:0"))
        test_dataset = create_dataset(config.TEST_PATH)
        test_dataset = transform_dataset(test_dataset, test=True)
        test_iterator = test_dataset.make_initializable_iterator()

        test_handle = sess.run(test_iterator.string_handle())

        fis_ph = g.get_tensor_by_name(with_prefix(model_name, "fis_ph:0"))
        fvs_ph = g.get_tensor_by_name(with_prefix(model_name, "fvs_ph:0"))
        fs_ph = g.get_tensor_by_name(with_prefix(model_name, "fs_ph:0"))
        labels_ph = g.get_tensor_by_name(with_prefix(model_name, "labels_ph:0"))
        if model_name == "fm":
            test_ph = None
        elif model_name == "nfm":
            test_ph = g.get_tensor_by_name(with_prefix(model_name, "test_ph:0"))
        else:
            print("Unsupported model - %s" % model_name)
            exit(-1)

        fis = g.get_tensor_by_name(with_prefix(model_name, "next_batch:0"))
        fvs = g.get_tensor_by_name(with_prefix(model_name, "next_batch:1"))
        fs = g.get_tensor_by_name(with_prefix(model_name, "next_batch:2"))
        labels = g.get_tensor_by_name(with_prefix(model_name, "next_batch:3"))

        logits = g.get_tensor_by_name(with_prefix(model_name, "logits:0"))

        predictions = tf.sigmoid(logits)
        auc_value, update_op = tf.metrics.auc(labels_ph, predictions, num_thresholds=20000, name="auc")
        get_auc(sess, test_handle, handle_ph, test_iterator, update_op, auc_value,
                fis, fvs, fs, labels, fis_ph, fvs_ph, fs_ph, labels_ph, test_ph)

    print(current_time(), "test finished!")


def get_auc(sess, test_handle, handle_ph, test_iterator, update_op, auc_value,
            fis, fvs, fs, labels, fis_ph, fvs_ph, fs_ph, labels_ph, test_ph=None):
    print(current_time(), "get_auc starts...")

    sess.run(test_iterator.initializer)

    tf.local_variables_initializer().run()

    auc = 0.0
    while True:
        try:
            # auc return every iteration is the mean auc up to now
            fis_ts, fvs_ts, fs_ts, labels_ts = sess.run([fis, fvs, fs, labels], feed_dict={handle_ph: test_handle})
            if config.MODEL_NAME == "fm":
                sess.run(update_op, feed_dict={fis_ph: fis_ts, fvs_ph: fvs_ts, fs_ph: fs_ts, labels_ph: labels_ts})
            elif config.MODEL_NAME == "nfm":
                sess.run(update_op, feed_dict={fis_ph: fis_ts, fvs_ph: fvs_ts, fs_ph: fs_ts, labels_ph: labels_ts, test_ph: True})
            else:
                print("Unsupported model - %s" % config.MODEL_NAME)
            auc = sess.run(auc_value)
        except tf.errors.OutOfRangeError:
            break

    print("auc: %.5f" % auc)

    print(current_time(), "get_auc finished!")


def load_ckpt_model(sess, ckpt_dir):
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("ckpt_file: %s" % ckpt_file)
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
    saver.restore(sess, ckpt_file)


def main():
    # test
    test()


if __name__ == "__main__":
    main()
