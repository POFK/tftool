#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf
import tqdm
import time
import os
from Network import Model
from Train import Train
from Utils import add_name_scope
from InputPipeline import InputPipeline

config = tf.ConfigProto(allow_soft_placement=True)
print_fn = tf.logging.info  # print
tf.logging.set_verbosity(tf.logging.INFO)

class Main(Train, Model, InputPipeline):
    def __init__(self, *args, **kwargs):
        super(Main, self).__init__(*args, **kwargs)
        print_fn(Main.__mro__)
        #----------------------------------------
        # default setting
        self.opt = tf.train.AdamOptimizer
        self.Is_summary = True
        self.update_log_step = 100
        self.save_per_steps = 1000
        self.print_log = True
        self.valid_size = 1024
        self.config = config
        #----------------------------------------

    def loaddata(self, repeat=1, shuffle=True):
        self.filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
        features, labels = self.input_fn(self.filenames, 
                                         repeat=repeat, 
                                         shuffle=shuffle)
        return features, labels

    def inference(self, *args, **kwargs):
        self.model()
        net = self.Network(self, *args, **kwargs)
        return net

    def eval(self, data):
        BS = self.FLAGS.batch_size
        flux = data['flux']
        label = data['label'].astype(np.int32)
        print flux.dtype, flux.shape
        print label.dtype, label.shape
        if label.shape[0]%BS == 0:
            loop_num = label.shape[0]//BS
        else:
            loop_num = label.shape[0]//BS + 1
        print_fn("loop number: %d"%loop_num)
        #------------------------------------------------------------
        x = tf.placeholder(tf.float32, shape=[None, 4000])
        y_ = tf.placeholder(tf.int32, shape=[None])
        net = self.inference(x, is_training=False)
        loss = self.Loss_fn(self, net=net, y_=y_)
        prob = tf.get_collection('prob')[0]
        LOSS = []
        PROB = []
        with tf.Session(config=self.config) as sess:
            self.Saver = tf.train.Saver(max_to_keep=1)
            self.init_model(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in tqdm.tqdm(xrange(loop_num)):
                el, ep, = sess.run([loss, prob], 
                                   feed_dict={x:flux[i*BS:(i+1)*BS],
                                              y_:label[i*BS:(i+1)*BS]})
                LOSS.append(el)
                PROB.append(ep)
            coord.request_stop()
            coord.join(threads)
        LOSS = np.array(LOSS)
        PROB = np.vstack(PROB)
        return LOSS, PROB

    def train(self, loss, train_op):
        """it is an example, should rewriten in main function"""
        summary_train_op = tf.summary.merge_all('train')
        summary_valid_op = tf.summary.merge_all('validation')
        filenames = self.filenames
    #   iterator_handle = tf.get_collection('iterator')[0].initializer
        iterator_handle = self.iterator.initializer
        with tf.Session(config=self.config) as sess:
            self.Saver = tf.train.Saver(max_to_keep=10)
            local_step = 0
            global_step = tf.get_collection('global_step')[0]
            self.init_model(sess)
            writer_train = tf.summary.FileWriter(
                os.path.join(self.FLAGS.log_dir, "train"), sess.graph)
            writer_valid = tf.summary.FileWriter(
                os.path.join(self.FLAGS.log_dir, "valid"), sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #------------------------------------------------------------
            sess.run(iterator_handle, 
                     feed_dict={filenames: self.train_filenames})
            pbar = tqdm.tqdm(total=self.FLAGS.train_steps)
            is_first_step = 0
            epoch = 1
            while not coord.should_stop():
                try:
                    _, step = sess.run([train_op, global_step], 
                                       feed_dict={self.Is_training: True})
                except tf.errors.OutOfRangeError:
                    sess.run(iterator_handle, 
                             feed_dict={filenames: self.valid_filenames})
                    valid_losses = []
                    loop_num = self.valid_size//self.FLAGS.batch_size
                    for i in xrange(loop_num):
                        summary_valid, l_valid = sess.run(
                            [summary_valid_op, loss],
                            feed_dict={self.Is_training: False})
                        if i == 0:
                            writer_valid.add_summary(summary_valid, step)
                        valid_losses.append(l_valid)
                    valid_loss = np.mean(valid_losses)
                    print "epoch %d: %f"%(epoch, valid_loss)
                    sess.run(iterator_handle, 
                             feed_dict={filenames: self.train_filenames})
                    epoch += 1
                    self.Saver.save(sess, os.path.join(
                        self.FLAGS.model_dir, 
                        self.FLAGS.model_basename+'_epoch'),
                        global_step=epoch)
                    continue
                if local_step % (self.FLAGS.train_steps //
                                 self.update_log_step) == 0:
                    if is_first_step == 0:
                        pbar.update(0)
                        is_first_step += 1
                    else:
                        pbar.update(self.FLAGS.train_steps//self.update_log_step)

                if local_step % self.update_log_step == 0:
                    summary_train, loss_train = sess.run(
                        [summary_train_op, loss], feed_dict={self.Is_training: False})
                    if self.print_log:
                        print " training step %d done (global step: %d):" \
                              % (local_step, step), loss_train
                    writer_train.add_summary(summary_train, step)
            #------------------------------------------------------------
                local_step += 1
                if local_step == 1:
                    continue
                if (step + 1) % self.save_per_steps == 0:
                    self.save_model(sess=sess)
                if local_step > self.FLAGS.train_steps:
                    coord.request_stop()
            coord.request_stop()
            coord.join(threads)
            pbar.close()
            writer_train.close()
            writer_valid.close()

    def __call__(self):
        """an example for constructing network model"""
        self.train_filenames = ["/data/dell5/userdir/maotx/DSC/data/training.tfrecords"]
        self.valid_filenames = ["/data/dell5/userdir/maotx/DSC/data/valid.tfrecords"]
        self.Is_training = tf.placeholder(tf.bool,shape=[],name='is_train')
        x, y_ = self.loaddata()
        net = self.inference(x, is_training=self.Is_training)
        loss = self.Loss_fn(self, net=net, y_=y_)
        train_op = self.optimizer(loss, gclip=None)
        self.train(loss, train_op)
