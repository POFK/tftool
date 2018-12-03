#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

#============================ setting ===========================================
tf.logging.set_verbosity(tf.logging.INFO)
#layers = tf.layers
layer = tf.contrib.framework.add_arg_scope(tf.layers.conv1d)
slim = tf.contrib.slim
print_fn = tf.logging.info  # print
# setting for GPU
#config = tf.ConfigProto(allow_soft_placement=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
#config.gpu_options.allow_growth = True
#================================================================================

def arg_scope(weight_delay=1e-4):
    with slim.arg_scope(
        [tf.layers.conv1d],
        strides=1,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        
#       kernel_regularizer=None,
#       bias_regularizer=None,
#       activity_regularizer=None,
#       kernel_constraint=None,
#       bias_constraint=None,
        trainable=True,
    ) as arg_sc:
        return arg_sc



