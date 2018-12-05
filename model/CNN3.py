#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from model_base import slim, print_fn, arg_scope, layer_batch_relu
from model_base import layer as base_layer
from tftool.Utils import add_name_scope
tf.logging.set_verbosity(tf.logging.INFO)


def Network(self, x, is_training=tf.bool):
    if self.FLAGS.Is_BN:
        layer = layer_batch_relu(base_layer, is_training=is_training)
    else:
        layer = base_layer
    namebase = 'model/'
    frac = 2
    num = 1
    print_fn(x)
    print_fn('=' * 10 + 'CNN' + '=' * 10)
    x = tf.reshape(x, [-1, 4000, 1], namebase+'reshape')
    with slim.arg_scope(arg_scope(weight_decay=self.FLAGS.weight_decay)):
        net = layer(x, 8*frac, 7, name=namebase+'conv%d'%num)
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'conv%d/dropout'%num)
        net = tf.layers.max_pooling1d(net, 3, 2,name=namebase+'maxpool%d'%num)
        num += 1

        net = layer(net, 16*frac, 7, name=namebase+'conv%d'%num)
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'conv%d/dropout'%num)
        net = tf.layers.max_pooling1d(net, 3, 2,name=namebase+'maxpool%d'%num)
        num += 1

        net = layer(net, 32*frac, 7, name=namebase+'conv%d'%num)
#       net = tf.layers.max_pooling1d(net, 3, 2,name=namebase+'maxpool%d'%num)
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'conv%d/dropout'%num)
        num += 1

        shape = [-1, np.prod(net.get_shape().as_list()[1:])]
        net = tf.reshape(net, shape, name=namebase+'reshape')

        net = tf.layers.dense(
            net, 
            128, 
            activation = tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = namebase+'dense%d'%num,
            )
        net = tf.layers.dropout(net, rate=0.5, training=is_training,
                                name=namebase+'dense%d/dropout'%num)
        num += 1


        net = tf.layers.dense(
            net, 
            2, 
            activation = None,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
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


