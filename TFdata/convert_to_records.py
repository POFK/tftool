import os
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to(data_set, Dir, name):
    """Converts a dataset to tfrecords."""
    features = data_set.features
    labels = data_set.labels
    num_examples = data_set.num_examples

    if features.shape[0] != num_examples:
        raise ValueError('Features size %d does not match label size %d.' %
                         (features.shape[0], num_examples))

    filename = os.path.join(Dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in xrange(num_examples):
        data_set.index = index
        feature_raw = features[index].reshape(-1).tolist()
        label_raw = labels[index].reshape(-1).tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
            'index': _int64_feature(data_set.index),
            'label_raw': _float_feature(label_raw),
            'feature_raw': _float_feature(feature_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    # Get the data.
    class data_set():
        pass

    data = np.load('/home/maotx/git/DSclassify/data/small_flux_resample.npy')
    Len = data.shape[0]
    features = data['flux']
    labels = data['label'].reshape(-1,1)
  
    data_set.features = features
    data_set.labels = labels
    data_set.num_examples = Len

    convert_to(data_set, Dir='/data/dell5/userdir/maotx/DSC/data', name='test')

main()

