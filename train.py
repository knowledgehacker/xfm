# -*- coding: utf-8 -*-
import os
import time

import tensorflow.compat.v1 as tf

import config
from input_feed import create_dataset, transform_dataset
import fm
import nfm
from utils import current_time, get_optimizer

CKPT_PATH = '%s/%s' % (config.CKPT_DIR, config.MODEL_NAME)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def build_ops(model_name, features_ph, labels_ph, test_ph=None):
    if model_name == "fm":
        model = fm.FMModel
        logits = fm.cal_logits(features_ph, model.bias, model.ws, model.vs)
        loss_op = fm.cal_loss(labels_ph, logits)
    elif model_name == "nfm":
        model = nfm.NFMModel()
        logits = nfm.cal_logits(features_ph, model.bias, model.ws, model.vs, model.hls_ws, model.hls_bias, model.ol_ws,
                                test_ph)
        loss_op = nfm.cal_loss(labels_ph, logits)
    else:
        print("Unsupported model - %s" % config.MODEL_NAME)
        exit(-1)

    optimizer = get_optimizer()
    train_op = optimizer.minimize(loss_op)

    return loss_op, train_op


def train():
    print(current_time(), "training starts...")

    g = tf.Graph()
    with g.as_default():
        # create iterator for train dataset
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        train_dataset = create_dataset(config.TRAIN_PATH)
        train_dataset = transform_dataset(train_dataset, test=False)
        train_iterator = train_dataset.make_initializable_iterator()

        # create a feedable iterator
        iterator = tf.data.Iterator.from_string_handle(
            handle_ph, train_dataset.output_types, train_dataset.output_shapes, train_dataset.output_classes)
        fis, fvs, fs, labels = iterator.get_next("next_batch")

        # build ops
        fis_ph = tf.placeholder(dtype=tf.int64, name="fis_ph")
        fvs_ph = tf.placeholder(dtype=tf.float32, name="fvs_ph")
        fs_ph = tf.placeholder(dtype=tf.int64, name="fs_ph")
        labels_ph = tf.placeholder(dtype=tf.float32, name="labels_ph")
        if config.MODEL_NAME == "fm":
            test_ph = None
        elif config.MODEL_NAME == "nfm":
            test_ph = tf.placeholder(dtype=tf.bool, name="test_ph")
        else:
            print("Unsupported model - %s" % config.MODEL_NAME)
            exit(-1)
        loss_op, train_op = build_ops(config.MODEL_NAME, tf.SparseTensor(fis_ph, fvs_ph, fs_ph), labels_ph, test_ph)

        # create saver
        saver = tf.train.Saver()

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()

        train_handle = sess.run(train_iterator.string_handle())

        loss = 0.0
        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(train_iterator.initializer)

            start_time = time.time()
            while True:
                try:
                    fis_ts, fvs_ts, fs_ts, labels_ts = sess.run([fis, fvs, fs, labels], feed_dict={handle_ph: train_handle})
                    if config.MODEL_NAME == "fm":
                        loss, _ = sess.run([loss_op, train_op],
                                           feed_dict={fis_ph: fis_ts, fvs_ph: fvs_ts, fs_ph: fs_ts, labels_ph: labels_ts})
                    elif config.MODEL_NAME == "nfm":
                        loss, _ = sess.run([loss_op, train_op],
                                           feed_dict={fis_ph: fis_ts, fvs_ph: fvs_ts, fs_ph: fs_ts, labels_ph: labels_ts, test_ph: False})
                    else:
                        print("Unsupported model - %s" % config.MODEL_NAME)
                        exit(-1)

                    step += 1
                    if step % config.STEPS_PER_CKPT == 0:
                        end_time = time.time()
                        print(current_time(), "step: %d, loss: %.5f, time: %f" % (step, loss, end_time - start_time))
                        saver.save(sess, CKPT_PATH, global_step=step)
                        start_time = end_time
                except tf.errors.OutOfRangeError:
                    print(current_time(), "step: %d, loss: %.5f, time: %f" % (step, loss, time.time() - start_time))
                    saver.save(sess, CKPT_PATH, global_step=step)
                    break
            #saver.export_meta_graph("%s/%s/model.ckpt.meta.json" % (config.MODEL_DIR, config.MODEL_NAME), as_text=True)
            # save model
            save_model(sess, "%s/%s" % (config.MODEL_DIR, config.MODEL_NAME), "model")

    print(current_time(), "training finishes...")


def save_model(sess, model_dir, filename):
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["logits", "loss"])

    model_filepath = "%s/%s.pb" % (model_dir, filename)
    with tf.gfile.GFile(model_filepath, "wb") as fout:
        fout.write(output_graph_def.SerializeToString())


def main():
    # train
    train()


if __name__ == "__main__":
    main()
