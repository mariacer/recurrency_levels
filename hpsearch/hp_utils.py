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
# @title           :hpsearch/hp_utils.py
# @author          :am
# @contact         :ameulema@ethz.ch
# @created         :27/10/2020
# @version         :1.0
# @python_version  :3.7
"""
Hpsearch Utilities
------------------

A collection of helper functions for the hpsearch module.
"""

def dict2csv(dct, file_path):
    """Save a dictionary to a csv file.

    Args:
        dct: dictionary
        file_path: file path of the csv file
    """
    with open(file_path, 'w') as f:
        for key in dct.keys():
            f.write("{}, {} \n".format(key, dct[key]))


def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.

    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.

    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret


def dict2config(dct, file_path, name='config'):
    """Save a dictionary to a config.py file.

    Args:
        dct: dictionary
        file_path: file path of the .py file
        name: name of the dictionary
    """
    with open(file_path, 'w') as f:
        write_dict_to_txt(f, dct, name)


def write_dict_to_txt(file_handler, dct, name):
    """Write the dict to a txt file in python format (so the python code 
    for creating a dictionary with the same content as dct).

    Args:
        file_handler: python file handler of the file where the dct will be
            written to.
        dct: dictionary
        name: name of the dictionary
    """
    file_handler.write('config_' + name + ' = {\n')
    for key, value in dct.items():
        if isinstance(value, str):
            value = "'" + value + "'"
        else:
            value = str(value)
        file_handler.write("'" + key + "': " + value + ",\n")
    file_handler.write('}\n\n')


def search_space_to_grid(grid_dict, fix_dict):
    """Translate a dictionary of parameter values into a list of commands.

    The entire set of possible combinations is generated.

    Args:
        grid_dict: A dictionary of argument names to lists, where each list
            contains possible values for this argument.
        fix_dict: A dictionary of argument names to lists, that should be
            fixed and identical in all configs.

    Returns:
        A list of dictionaries. Each key is an argument name that maps onto a
        single value.
    """
    # We build a list of dictionaries with key value pairs.
    all_dicts = []

    # We need track of the index within each value array.
    gkeys = list(grid_dict.keys())
    fix_keys = list(fix_dict.keys())
    indices = [0] * len(gkeys)

    stopping_criteria = False
    while not stopping_criteria:

        cmd = dict()
        for i, k in enumerate(gkeys):
            v = grid_dict[k][indices[i]]
            cmd[k] = v
        for ii, kk in enumerate(fix_keys):
            cmd[kk] = fix_dict[kk]
        all_dicts.append(cmd)
        
        for i in range(len(indices)-1,-1,-1):
            indices[i] = (indices[i] + 1) % len(grid_dict[gkeys[i]])
            if indices[i] == 0 and i == 0:
                stopping_criteria = True
            elif indices[i] != 0:
                break

    return all_dicts
