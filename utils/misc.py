#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :utils/misc.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :10/10/2020
# @version         :1.0
# @python_version  :3.7
"""
Miscellaneous Utilities
-----------------------

A collection of helper functions.
"""
import numpy as np
import torch

def str_to_ints(str_arg):
    """Helper function to convert a string which is a list of comma separated
    integers into an actual list of integers.

    Args:
        str_arg: String containing list of comma-separated ints. For convenience
            reasons, we allow the user to also pass single integers that a put
            into a list of length 1 by this function.

    Returns:
        (list): List of integers.
    """
    if isinstance(str_arg, int):
        return [str_arg]

    if len(str_arg) > 0:
        return [int(float(s)) for s in str_arg.split(',')]
    else:
        return []

def str_to_act(act_str):
    """Convert the name of an activation function into the actual PyTorch
    activation function.

    Args:
        act_str: Name of activation function (as defined by command-line
            arguments).

    Returns:
        Torch activation function instance or ``None``, if ``linear`` is given.
    """
    if act_str == 'linear':
        act = None
    elif act_str == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif act_str == 'relu':
        act = torch.nn.ReLU()
    elif act_str == 'elu':
        act = torch.nn.ELU()
    elif act_str == 'tanh':
        act = torch.nn.Tanh()
    else:
        raise Exception('Activation function %s unknown.' % act_str)
    return act


def str_round_to_int(string):
    """Convert the string of a floating point to a rounded integer (floored).

    Args:
        string (str): string of a floating point number.

    Returns: 
        (int): The integer number.
    """
    if isinstance(string, int):
        return string
    elif isinstance(string, float):
        return int(string)
    elif isinstance(string, str):
        string = float(string)
        return int(string)
    else:
        raise ValueError('str_round_to_int() cannot handle type {}'.
                         format(type(string)))


def str_to_float_or_list(string, delim=','):
    """ Convert a string of a list or float to a list of floats or a float."""

    if string[0] in ('[', '(') and string[-1] in (']', ')'):
        string = string[1:-1]
        lst = [float(num) for num in string.split(delim)]
        return lst
    else:
        return float(string)


def dict_for_json(d):
    """Make a dictionary JSON serializable.

    Args:
        d (dict): The dictionary.

    Return:
        (dict): The adapted dictionary.
    """
    for key in d.keys():
        if isinstance(d[key], np.ndarray):
            d[key] = list(d[key])
        elif isinstance(d[key], np.int64):
            d[key] = int(d[key])

    return d