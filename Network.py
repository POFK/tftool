#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import os
import importlib
from Utils import Toolkit, add_name_scope

print_fn = tf.logging.info  # print

class Model(Toolkit):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(-1),
                                      trainable=False)
        tf.add_to_collection('global_step', global_step)

    def save_model(self, sess):
        global_step = tf.get_collection('global_step')[0]
        self.Saver.save(sess, os.path.join(
            self.FLAGS.model_dir, self.FLAGS.model_basename),
            global_step=tf.train.global_step(sess, global_step))

    def init_model(self, sess):
        init_glovars = tf.variables_initializer(
            tf.global_variables(),
            name='global_init')
        init_locvars = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES),
            name='local_init')
        init_op = [init_glovars, init_locvars]

        ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
        if ckpt is not None:
            print_fn('load variable from %s' % ckpt.model_checkpoint_path)
            self.Saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(init_op[1])
        else:
            print_fn('Init variable')
            sess.run(init_op)

    def _get_model(self, ModelBaseName=''):
        dict = {}
        if ModelBaseName == '':
            raise NotImplementedError("model name error!")
        model = importlib.import_module('tftool.model.%s' % ModelBaseName)
        dict['network'] = model.Network
        dict['loss_fn'] = model.loss_fn
        return dict

    @add_name_scope('Network/model')
    def model(self):
        tf.summary.scalar('Par/learning_rate', 
                          self.FLAGS.learning_rate,
                          collections=['train'])
        tf.summary.scalar('Par/batch_size', 
                          self.FLAGS.batch_size,
                          collections=['train'])
        ModelDict = self._get_model(ModelBaseName=self.FLAGS.model_basename)
        self.Network = ModelDict['network'] # load network model
        self.Loss_fn= ModelDict['loss_fn'] # load loss function

    def _construct(self, x, y_):
        """
        an example for constructing network model, 
        rewrite it in the __call__ function of main.py
        """
        net = self.Network(self, x, is_training=True)
        loss = self.Loss_fn(net=net, y_=y_)
        train_op = self.optimizer(loss, gclip=None)
        self.train(loss, train_op)


