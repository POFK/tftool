#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from model_base import slim, layer, print_fn, arg_scope
from tftool.Utils import add_name_scope
tf.logging.set_verbosity(tf.logging.INFO)

#@add_name_scope('Network/model')
def Network(self, x, is_training=tf.bool):
    namebase = 'model/'
    num = 1
    print_fn(x)
    print_fn('=' * 10 + 'test' + '=' * 10)
    x = tf.reshape(x, [-1, 4000, 1], namebase+'reshape')
    with slim.arg_scope(arg_scope(weight_delay=0)):
        net = layer(x, 64, 9, name=namebase+'conv%d'%num)
        num += 1

        net = layer(net, 64, 9, name=namebase+'conv%d'%num)
        net = tf.layers.max_pooling1d(net, 300, 2,name=namebase+'maxpool%d'%num)
        num += 1

#       net = layer(net, 64, 9, name='Network/conv%d'%num)
#       net = tf.layers.max_pooling1d(net, 3, 2,name='Network/maxpool%d'%num)
#       num += 1

#       net = layer(net, 128, 9, name='Network/conv%d'%num)
#       net = tf.layers.max_pooling1d(net, 3, 2,name='Network/maxpool%d'%num)
#       num += 1

#       net = layer(net, 128, 9, name='Network/conv%d'%num)
#       net = tf.layers.max_pooling1d(net, 3, 2,name='Network/maxpool%d'%num)
#       num += 1

        shape = [-1, np.prod(net.get_shape().as_list()[1:])]
        net = tf.reshape(net, shape, name=namebase+'reshape')

#       net = tf.layers.dense(
#           net, 
#           128, 
#           activation = tf.nn.relu,
#           kernel_initializer = tf.contrib.layers.xavier_initializer(),
#           name = namebase+'dense1',
#           )

        net = tf.layers.dense(
            net, 
            2, 
            activation = None,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = namebase+'dense2',
            )
        return net

@add_name_scope('loss')
def loss_fn(self, net, y_):
    """
    net: output of network
    y_: ground truth label, with shape [batch_size, 1]
    """
    # add recall
    y_ = tf.reshape(y_, [-1])
    prob = tf.nn.softmax(net) # probability
    tf.add_to_collection('prob', prob)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_,
            logits=net)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss, collections=['train', 'validation'])
    return loss


