#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
import tqdm
import time
from Utils import Toolkit, add_name_scope

config = tf.ConfigProto(allow_soft_placement=True)

class Train(Toolkit):
    def __init__(self, *args, **kwargs):
        super(Train, self).__init__(*args, **kwargs)

    def set_optimizer(self, opt):
        self.opt = opt

    def set_summary(self, Is_summary):
        self.Is_summary = bool

    @add_name_scope('Opt')
    def optimizer(self, loss, gclip=None):
        global_step = tf.get_collection('global_step')[0]
        opt = self.opt(self.FLAGS.learning_rate)
        gvs = opt.compute_gradients(loss)
        if gclip is not None:
            gvs = [(tf.clip_by_value(grad, -gclip, gclip), var) \
                   for grad, var in gvs]
        if self.Is_summary:
            for grad, var in gvs:
                tf.summary.histogram(var.op.name + '/gradients',
                                     grad,
                                     collections=['train'])
        if self.FLAGS.Is_BN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(gvs, global_step=global_step)
        else:
            train_op = opt.apply_gradients(gvs, global_step=global_step)
        return train_op

    @add_name_scope('train')
    def train(self, loss, train_op):
        """it is an example, should rewriten in main function"""
        summary_train_op = tf.summary.merge_all('train')
        with tf.Session(config=config) as sess:
            self.Saver = tf.train.Saver(max_to_keep=10)
            local_step = -1
            global_step = tf.get_collection('global_step')[0]
            self.init_model(sess)
            writer = tf.summary.FileWriter(
                os.path.join(self.FLAGS.log_dir, "train"), sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            pbar = tqdm.tqdm(total=self.FLAGS.train_steps)
            is_first_step = 0
            while not coord.should_stop():
                now = time.time()
                _, step = sess.run([train_op, global_step])
                time_end = time.time()
                if local_step % (self.FLAGS.train_steps //
                                 self.update_log_step) == 0:
                    if is_first_step == 0:
                        pbar.update(0)
                        is_first_step += 1
                    else:
                        pbar.update(
                            self.FLAGS.train_steps //
                            self.update_log_step)

                if local_step % self.update_log_step == 0:  # and local_step != 0:
                    summary_train, loss_train = sess.run(
                        [summary_train_op, loss])
                    summary_test, loss_test= sess.run(
                        [summary_train_op, loss])

                    if self.print_log:
                        print " training step %d done (global step: %d):" \
                              % (local_step, step), loss_train, loss_test, \
                              "%.4f" % (time_end - now)
                    writer.add_summary(summary_train, step)
                if local_step == 0:
                    local_step += 1
                    continue
                if (step + 1) % self.save_per_steps == 0:
                    self.save_model(sess=sess)
                local_step += 1
                if local_step >= self.FLAGS.train_steps:
                    coord.request_stop()
            coord.request_stop()
            coord.join(threads)
            pbar.close()
            writer.close()

