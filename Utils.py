#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
from Base import Base
from flags import FLAGS

def catch_exception(func):
    """a catch exception example for class function"""

    def wrapper(self, *args, **kwargs):
        try:
            u = func(self, *args, **kwargs)
            return u
        except Exception:
            #   self.revive() # call method in class
            return 'an Exception raised.'
    return wrapper


def add_name_scope(text):
    """add name scope for a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.name_scope(text):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class Toolkit(Base):
    def __init__(self, *args, **kwargs):
        """
        a Toolkit for Convnet code.
        """
        super(Toolkit, self).__init__(*args, **kwargs)
        self.FLAGS_module()

    def FLAGS_module(self):
        self.import_FLAGS()
        self.get_dirlist()
        for name in self.FLAGS_dirlist:
            self.check_dir(name)
        with open(os.path.join(self.FLAGS.log_dir, 'params.log'), 'w') as f:
            for name, value in self.FLAGS.__dict__.items():
                f.writelines("{0}:\t{1}\n".format(name, value))

    def import_FLAGS(self):
        self.FLAGS = FLAGS

    def get_dirlist(self):
        self.FLAGS_dirlist = []
        for key, value in self.FLAGS.__dict__.items():
            if '_dir' in key:
                self.FLAGS_dirlist.append(value)

    def check_dir(self, path):
        if not os.path.isdir(path):
            print('mkdir: ',path)
            os.makedirs(path)


if __name__ == '__main__':
    tool = Toolkit()
