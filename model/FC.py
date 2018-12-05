#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from model_base import slim, print_fn, arg_scope, layer_batch_relu
from model_base import layer as base_layer
from tftool.Utils import add_name_scope
tf.logging.set_verbosity(tf.logging.INFO)


def Network(self, x, is_training=tf.bool):
    namebase = 'model/'
    num = 1
    print_fn(x)
    print_fn('=' * 10 + 'test' + '=' * 10)
    x = tf.reshape(x, [-1, 4000], namebase+'reshape')
    kernel_regularizer = tf.contrib.layers.l2_regularizer(
        scale=self.FLAGS.weight_decay)
    net = tf.layers.dense(x, 256, 
                          activation=tf.nn.relu,
                          kernel_regularizer=kernel_regularizer,
                          name=namebase+'dense%d'%num)
    net = tf.layers.dropout(net, rate=0.5, training=is_training,
                            name=namebase+'dense%d/dropout'%num)
    num += 1

    net = tf.layers.dense(net, 128, 
                          activation=tf.nn.relu,
                          kernel_regularizer=kernel_regularizer,
                          name=namebase+'dense%d'%num)
    net = tf.layers.dropout(net, rate=0.3, training=is_training,
                            name=namebase+'dense%d/dropout'%num)
    num += 1

    net = tf.layers.dense(net, 32, 
                          activation=tf.nn.relu,
                          kernel_regularizer=kernel_regularizer,
                          name=namebase+'dense%d'%num)
    net = tf.layers.dropout(net, rate=0.1, training=is_training,
                            name=namebase+'dense%d/dropout'%num)
    num += 1

    net = tf.layers.dense(
        net, 
        2, 
        activation = None,
        kernel_regularizer=kernel_regularizer,
        name = namebase+'dense',
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
    tf.summary.scalar('MSE', loss, collections=['train', 'validation'])

    l2_loss = tf.reduce_sum(tf.get_collection('regularization_losses'))
    tf.summary.scalar('l2_loss', l2_loss, collections=['train'])

    loss = loss + l2_loss
    tf.summary.scalar('loss', loss, collections=['train', 'validation'])
    return loss


