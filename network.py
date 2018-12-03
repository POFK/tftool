#!/usr/bin/env python
# coding=utf-8
import time
import os
import tqdm
import numpy as np
import tensorflow as tf
from Base import Utils
from toolkit import Toolkit, add_name_scope

#================================================================================
layers = tf.contrib.layers
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
print_fn = tf.logging.info  # print

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config.gpu_options.allow_growth = True
# REGULARIZATION_LOSSES
#================================================================================


def arg_scope(weight_delay=1e-4):
    with slim.arg_scope(
        [layers.conv2d],
        #       weights_initializer=tf.contrib.layers.xavier_initializer(),
    #   activation_fn=tf.nn.relu,
        activation_fn=None,
        #       normalizer_fn=normalizer_fn,
        #       normalizer_params=normalizer_params,
        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_delay),
        biases_initializer=None,
        variables_collections=None,
        data_format='NHWC',
    ) as arg_sc:
        return arg_sc


class Network(Utils, Toolkit):
    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        print_fn(Network.__mro__)
        self.data_format = 'channels_last'
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(-1),
                                      trainable=False)
        tf.add_to_collection('global_step', global_step)
        self.update_log_step = 100
        self.save_per_steps = 1000
        self.print_log = True
        self.checkpoint_basename = 'RI'
        self.InferenceOnTheFly = True

    @add_name_scope('Network/model')
    def model(self, dm):
        if dm.shape.ndims != 4:
            dm = tf.reshape(
                dm, [-1, self.FLAGS.image_size, self.FLAGS.image_size, 1])
        with slim.arg_scope(arg_scope(self.FLAGS.l2w)):
            net = layers.conv2d(
                dm,
                num_outputs=8,
                kernel_size=[9, 9],
                stride=1,
                padding='SAME',
                scope='Network/model/conv1')

            net = layers.conv2d(
                net,
                num_outputs=8,
                kernel_size=[9, 9],
                stride=1,
                padding='SAME',
                scope='Network/model/conv2')

            net = layers.conv2d(
                net,
                num_outputs=1,
                kernel_size=[9, 9],
                stride=1,
                padding='SAME',
                activation_fn=None,
                scope='Network/model/conv3')

        net = tf.squeeze(net, [3], name='SpatialSqueeze')
        return net

    @add_name_scope('Network/Opt')
    def optimizer(self, loss):
        global_step = tf.get_collection('global_step')[0]
        train_op = tf.train.AdamOptimizer(learning_rate=\
                    self.FLAGS.learning_rate).minimize(
                        loss,
                        global_step=global_step)
        return train_op

    @add_name_scope('Network/loss')
    def loss_fn(self, net, gt):
        # gt: ground truth images
        loss = tf.reduce_mean(tf.square(net - gt))
    #   loss = tf.reduce_mean(tf.square(net - gt))/\
    #           tf.reduce_mean(tf.square(gt-tf.reduce_mean(gt)))
        tf.summary.scalar('MSE_loss', loss)
#       l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#       l2_loss = tf.reduce_sum(l2_loss)
        prior = tf.reduce_mean(tf.abs(net))
        l2_loss = prior * self.FLAGS.l2w
        tf.summary.scalar('l2_loss', l2_loss)
        loss = loss + l2_loss
        tf.summary.scalar('loss', loss)
        return loss

    def save_model(self, sess):
        global_step = tf.get_collection('global_step')[0]
        self.Saver.save(sess, os.path.join(
            self.FLAGS.model_dir, self.checkpoint_basename),
            global_step=tf.train.global_step(sess, global_step))

    def init_model(self, sess):
        init_gvar = tf.variables_initializer(
            tf.global_variables(),
            name='global_init')
        init_local = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES),
            name='local_init')
        init_op = [init_gvar, init_local]
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
        if ckpt is not None:
            print_fn('load variable from %s' % ckpt.model_checkpoint_path)
            self.Saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(init_op[1])
        else:
            print_fn('Init variable')
            sess.run(init_op)

    @add_name_scope('Network/train')
    def train(self, loss, train_op, summary_merged):
        if self.InferenceOnTheFly:
            sess_infer, tensors_infer = self.inference()
            writer_infer = tf.summary.FileWriter(
                os.path.join(self.FLAGS.log_dir, "infer"), sess_infer.graph)
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
                        [summary_merged, loss])
                    if self.print_log:
                        print " training step %d done (global step: %d):" \
                              % (local_step, step), loss_train, \
                              "%.4f" % (time_end - now)
                    writer.add_summary(summary_train, step)
                if local_step == 0:
                    local_step += 1
                    continue
                if (step + 1) % self.save_per_steps == 0:
                    self.save_model(sess=sess)
                    if self.InferenceOnTheFly:
                        loss_i, net_i, gt_i, step_i, summary_i =\
                            self.run_infer(sess_infer, tensors_infer)
                        SNR = 20. * \
                            np.log10(np.sum(gt_i**2.)**0.5 /
                                     np.sum((gt_i - net_i)**2)**0.5)
                        writer_infer.add_summary(summary_i, step_i)
                        assert step_i == step, 'error in inference restore!'
                        print_fn("infer loss @ step %d:\t%f(loss), %f(SNR)"
                                 % (step_i, loss_i, SNR))
                local_step += 1
                if local_step >= self.FLAGS.train_steps:
                    coord.request_stop()
            coord.request_stop()
            coord.join(threads)
            pbar.close()
            writer.close()

    @add_name_scope('inference')
    def inference(self):
        gt_M31 = np.load(self.FLAGS.gt_map).astype(np.float32).reshape(1, 128, 128)
        dm_M31 = np.load(self.dminfer_path).astype(np.float32).reshape(1, 128, 128)
        g_infer = tf.Graph()
        with g_infer.as_default():
            global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                          initializer=tf.constant_initializer(
                                              -1),
                                          trainable=False)
            gt = tf.Variable(gt_M31,
                             trainable=False,
                             collections=[tf.GraphKeys.LOCAL_VARIABLES])
            dm = tf.Variable(dm_M31,
                             trainable=False,
                             collections=[tf.GraphKeys.LOCAL_VARIABLES])
            net = self.model(dm)
            loss = self.loss_fn(net, gt)
            tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()
            init_local = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES),
                name='local_init')
            sess_infer = tf.Session(config=config)
            sess_infer.run(init_local)
            #------------------------------------------------------------
            # load model
            variables = tf.contrib.framework.get_variables_to_restore()
            self.saver_infer = tf.train.Saver(variables)
            # load model
            #------------------------------------------------------------
            return sess_infer, [loss, net, gt, global_step, merged]

    def run_infer(self, sess_infer, tensors):
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
        self.saver_infer.restore(sess_infer, ckpt.model_checkpoint_path)
        return sess_infer.run(tensors)

    def __call__(self, dm, gt):
        """
        dm: dirty map tensors
        gt: ground truth images
        """
        net = self.model(dm)
        loss = self.loss_fn(net, gt)
        train_op = self.optimizer(loss)
        merged = tf.summary.merge_all()
        return loss, train_op, merged


if __name__ == '__main__':
    from sim import Pipeline

    def uv_gen(self, path):
        try:
            fp = open(path, 'r')
            uv = np.load(fp)
            fp.close()
        except IOError:
            uv = self.uvcoverage().astype(np.float32)
            np.save(path, uv)
        return uv

    NGpus=4
    parallel_perGPU = 2
    shard_bs = 4
    uvpath = '/nfs/P100/RIimaging/vis/uv.npy'
    #=================  visibility simulation  ==================
    Batch_gt = []
    Batch_dm = []

    for i in xrange(NGpus*parallel_perGPU):
        with tf.name_scope('SimVis_%d'%i):
            with tf.device("/device:GPU:%d"%(i%NGpus)):
                test = Pipeline(n_vis=128 * 128 / 8)
                uv = uv_gen(test,uvpath)
                uv = tf.Variable(uv, 
                                 trainable=False, 
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 name='input/input_uv')
                batch_shard = test(uv, N_uv=256, 
                                   shard_bs=shard_bs, 
                                   noise_add=True)
                Batch_gt.append(batch_shard[0])
                Batch_dm.append(batch_shard[1])
    with tf.name_scope('SimOut'):
        with tf.device("/device:GPU:4"):
#       with tf.device('/CPU:0'):
            batch_gt = tf.concat(Batch_gt,0, name='batch_gt')
            batch_dm = tf.concat(Batch_dm,0, name='batch_dm')
    #============================================================
    with tf.device("/device:GPU:4"):
        model = Network()
        loss, train_op, merged_op = model(batch_dm, batch_gt)
        model.train(loss, train_op, merged_op)
