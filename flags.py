#!/usr/bin/env python
# coding=utf-8
import argparse
parser = argparse.ArgumentParser(usage="For RI-Imaging 0.1",
                                 description="help info.")

def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):
    """
    Add a boolean argument to an ArgumentParser instance.
    usage:
        add_boolean_argument(parser, 'Is_BN', default=False)
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')

#----------------------------------------
parser.add_argument(
    "--log_dir",
    type=str,
    default='/tmp/test/log',
    help="log directory name")

parser.add_argument(
    "--model_dir",
    type=str,
    default='/tmp/test/model',
    help="model directory")

parser.add_argument(
    "-MB",
    "--model_basename",
    type=str,
#   default='model',
    default='NetworkTest',
    help="model base name")

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="batch size")

parser.add_argument(
    "--train_steps",
    type=int,
    default=5000,
    help="train steps")

parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="learning rate")

parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.,
    help="L2 regularization weight decay.")

add_boolean_argument(parser, 'Is_BN', default=False)
#----------------------------------------
FLAGS = parser.parse_args()
if __name__ == '__main__':
    print FLAGS
    for key, value in FLAGS.__dict__.items():
        print key, value
