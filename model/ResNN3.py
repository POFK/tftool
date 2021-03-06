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
    print_fn('=' * 10 + 'ResNet_sigmoid' + '=' * 10)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(
        scale=self.FLAGS.weight_decay)
    #------------------------------------------------------------
    x = tf.reshape(x, [-1, 4000, 1], namebase+'reshape')
    shortcut = tf.layers.conv1d(x, 32*frac, 1, 4, padding='same',
                               activation=None,
                               kernel_regularizer=kernel_regularizer,
                               name = namebase+'shortcut')

    with slim.arg_scope(arg_scope(weight_decay=self.FLAGS.weight_decay)):

        net = layer(x, 8*frac, 7, name=namebase+'conv%d'%num, padding='same')
        net = tf.layers.max_pooling1d(net, 3, 2,
                                      padding='same',
                                      name=namebase+'conv%d/maxpool'%num)
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'conv%d/dropout'%num)
        num += 1

        net = layer(net, 16*frac, 7, name=namebase+'conv%d'%num, padding='same')
        net = tf.layers.max_pooling1d(net, 3, 2,
                                      padding='same',
                                      name=namebase+'conv%d/maxpool'%num)
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'conv%d/dropout'%num)
        num += 1

        net = tf.layers.conv1d(net, 32*frac, 7, padding='same',
                              activation=None, 
                              use_bias=False,
                              kernel_regularizer=kernel_regularizer,
                              name=namebase+'conv%d'%num)
        num += 1

        #----------- residual -------------------
        net = tf.add(net, shortcut, name=namebase+'shortcut/add')
        net = tf.contrib.layers.batch_norm(net, 
                                       center=True, 
                                       scale=True,
                                       decay=0.99,
                                       epsilon=0.001,
                                       is_training=is_training,
                                       scope=namebase+'shortcut/BN')
        net = tf.nn.relu(net, name=namebase+'shortcut/relu')
        net = tf.layers.dropout(net, rate=0.1, training=is_training,
                                name=namebase+'shortcut/dropout')
        #----------- residual -------------------
#       net = tf.layers.dropout(net, rate=0.1, training=is_training,
#                               name=namebase+'dense%d/dropout'%num)

        shape = [-1, np.prod(net.get_shape().as_list()[1:])]
        net = tf.reshape(net, shape, name=namebase+'reshape')
        net = tf.layers.dense(
            net, 
            32, 
            activation = None,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=kernel_regularizer,
            name = namebase+'dense1',
            )
        net = tf.contrib.layers.batch_norm(net, 
                                       center=True, 
                                       scale=False,
                                       decay=0.99,
                                       epsilon=0.001,
                                       is_training=is_training,
                                       scope=namebase+'dense1/BN')
        net = tf.nn.relu(net, name=namebase+'dense1/relu')
        net = tf.layers.dropout(net, rate=0.5, training=is_training,
                                name=namebase+'dense1/dropout')

        net = tf.layers.dense(
            net, 
            2, 
            activation = None,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=kernel_regularizer,
            name = namebase+'dens2',
            )
        return net

@add_name_scope('loss')
def loss_fn(self, net, y_):
    """
    net: output of network
    y_: ground truth label, with shape [batch_size, 1]
    """
    y_ = tf.reshape(y_, [-1])
#   prob = tf.nn.softmax(net) # probability
    prob = tf.nn.sigmoid(net) # probability
    tf.add_to_collection('prob', prob)

#----------------------------------------
    indicates = tf.one_hot(y_, depth=2)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=indicates,
                                                   logits=net)
#   loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#           labels=y_,
#           logits=net)
#----------------------------------------
    loss = tf.reduce_mean(loss)
    tf.add_to_collection('MES_l',loss)
    tf.summary.scalar('MSE', loss, collections=['train', 'validation'])

    l2_loss = tf.reduce_sum(tf.get_collection('regularization_losses'))
    tf.summary.scalar('l2_loss', l2_loss, collections=['train'])

    loss = loss + l2_loss
    tf.summary.scalar('loss', loss, collections=['train', 'validation'])
    return loss


