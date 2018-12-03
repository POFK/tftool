#!/usr/bin/env python
# coding=utf-8
from abc import ABCMeta, abstractmethod

class Base(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

#   @abstractmethod
#   def inference(self, x):
#       '''return net'''
#       pass

#   @abstractmethod
#   def train(self):
#       pass

#   @abstractmethod
#   def eval(self):
#       pass

