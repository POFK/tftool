#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

#============================ setting ===========================================
tf.logging.set_verbosity(tf.logging.INFO)
#layers = tf.layers
layer = tf.contrib.framework.add_arg_scope(tf.layers.conv1d)
slim = tf.contrib.slim
print_fn = tf.logging.info  # print
#================================================================================

def layer_batch_relu(func, is_training=tf.bool):
    '''wrapper for tf.layers.conv based functions'''
    def wrapper(*args, **kwargs):
        with slim.arg_scope([func], activation=None, use_bias=False):
            net = func(*args, **kwargs)

        net = tf.contrib.layers.batch_norm(net, 
                                       center=True, 
                                       scale=False,
                                       decay=0.999,
                                       epsilon=0.001,
                                       is_training=is_training,
                                       scope=kwargs['name']+'/BN')
        return tf.nn.relu(net, name=kwargs['name']+'/relu')
    return wrapper

def arg_scope(weight_decay=1e-4):
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    # l2_loss = tf.losses.get_regularization_loss()
    with slim.arg_scope(
        [tf.layers.conv1d],
        strides=1,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=kernel_regularizer,
        trainable=True,
    ) as arg_sc:
        return arg_sc



