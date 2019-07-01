#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from Utils import Toolkit, add_name_scope

class InputPipeline(Toolkit):
    def __init__(self, *args, **kwargs):
        super(InputPipeline, self).__init__(*args, **kwargs)

    @add_name_scope('InputPipeline/preprocessing')
    def preprocessing(self, feature, label, Len=4000):
        feature = tf.reshape(feature, Len)
    #   feature = feature - tf.reduce_mean(feature) # zero centering
        feature = tf.clip_by_value(feature, -100., 100.) # std is 1. here.
        label = tf.cast(label, dtype=tf.int32)
        label = tf.reshape(label, [])
        print feature, label
        return feature, label

    @add_name_scope('InputPipeline/parse_fn')
    def parse_fn(self, example, Len=4000):
        "Parse TFExample records and perform simple data augmentation."
        example_fmt = {
        #   "index": tf.FixedLenFeature([], tf.int64),
            "feature_raw": tf.FixedLenFeature([np.prod(self.data_shape)], tf.float32),
            "label_raw": tf.FixedLenFeature([], tf.float32)
        }
        parsed = tf.parse_single_example(example, example_fmt)
        feature = parsed['feature_raw']
        label = parsed['label_raw']
        #-------------preprocessing--------------
        feature, label = self.preprocessing(feature, label, Len=self.data_shape)
        #-------------preprocessing--------------
        return feature, label
    
    @add_name_scope('InputPipeline/input_fn')
    def input_fn(self, filenames, repeat=20, shuffle=False):
        """
        filenames: placeholder, []
        return: feature, label
        """
        dataset = tf.data.TFRecordDataset(filenames)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.FLAGS.batch_size*10)
        dataset = dataset.map(self.parse_fn)
        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size=self.FLAGS.batch_size)
        self.iterator = dataset.make_initializable_iterator()
#       tf.add_to_collection('iterator',iterator)
        next_element = self.iterator.get_next()
        return next_element

    def __call__(self):
        """An example"""
        filenames = tf.placeholder(tf.string, shape=[None])
        elements = self.input_fn(filenames, repeat=1)
        train_filenames = ["/data/dell5/userdir/maotx/DSC/data/test.tfrecords"]*3
        valid_filenames = ["/data/dell5/userdir/maotx/DSC/data/test.tfrecords"]*3
        #sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
        #sess.run(iterator.initializer, feed_dict={filenames: valid_filenames})
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(tf.get_collection('iterator')[0].initializer,
                    feed_dict={filenames: train_filenames})
            features_records, labels_records = sess.run(elements)
            print 'training'
            print 'batch size:',self.FLAGS.batch_size
            for i in xrange(1000):
                t1, t2 = sess.run(elements)
                print t2.shape

            print 'validation'
            sess.run(tf.get_collection('iterator')[0].initializer,
                    feed_dict={filenames: valid_filenames})
            features_records, labels_records = sess.run(elements)
            for i in xrange(10):
                t1, t2 = sess.run(elements)
                print t2

if __name__ == '__main__':
    IP = InputPipeline()
    IP()
