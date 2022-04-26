#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
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
# @title          :hpsearch/multiple_hpsearches.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
This is a script to easily run multiple hpsearches with one command.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
from argparse import Namespace
import importlib
import numpy as np
import random
import sys
import os
import copy

from hpsearch import run_hpsearch

_DEFAULT_GRID = 'hpsearch.multiple_hpsearch_config'


def append_fixed_attr(config, module):
    """
    append the fixed attributes of the module to the config directory
     that each hpsearch needs (to be compatible with
    the run_hpsearch)
    """
    config._MODULE_NAME = module._MODULE_NAME
    config._SUMMARY_FILENAME = module._SUMMARY_FILENAME
    config._SUMMARY_KEYWORDS = module._SUMMARY_KEYWORDS
    config._OUT_ARG = module._OUT_ARG


def delete_doubles(dict1, dict2):
    """ Delete the key value pairs from dict1 that have a corresponding key
    in dict2"""
    for key in dict2.keys():
        if key in dict1.keys():
            del dict1[key]


def nonsequential_hpsearch(args, config):

    args.out_dir = args.seq_dir
    delete_doubles(config.fixed_space, config.search_space)
    analysis, best_config = run_hpsearch.run(args, config)

    return best_config


def create_config_file(config_multiple_hpsearch, run_key):
    """
    Create a config file that can be used for the hpsearch
    Args:
        config_multiple_hpsearch: the config module of the multiple hpsearch
        run_key: the key corresponding to for which run this config file is made

    Returns: config file

    """
    config = copy.copy(Namespace(**vars(config_multiple_hpsearch)))
    del config.multiple_runs
    config.fixed_space.update(
        config_multiple_hpsearch.multiple_runs[run_key])
    return config


def write_dict_to_txt(file_handler, dct, name):
    file_handler.write('config_' + name + ' = {\n')
    for key, value in dct.items():
        if isinstance(value, str):
            value = "'" + value + "'"
        else:
            value = str(value)
        file_handler.write("'" + key + "': " + value + ",\n")
    file_handler.write('}\n\n')


def write_dict_to_cmdstyle(file_handler, dct, name):
    file_handler.write('\n\n##### ' + name + ' ####\n')
    for key, value in dct.items():
        if isinstance(value, bool):
            if value:
                file_handler.write("--" + key + " ")
        elif isinstance(value, list):
            lst_str = '['
            for el in value:
                lst_str += str(el) + ','
            lst_str = lst_str[:-1] + "]"
            file_handler.write("--" + key + "=" + lst_str + " ")
        else:
            value = str(value)
            file_handler.write("--" + key + "=" + value + " ")


def create_fig_config_file(filename, best_configs, config_multiple_hpsearch,
                           args):
    file_path = os.path.join(args.mother_dir, filename)
    with open(file_path, 'w') as f:
        for name, dct in best_configs.items():
            dct.update(config_multiple_hpsearch.multiple_runs[name])
            write_dict_to_txt(f, dct, name)

        f.write('config_collection = {\n')
        for name in best_configs.keys():
            f.write("'"+name+"': " + "config_" + name + ",\n")
        f.write("}\n\n")

        f.write('result_keys = [\n')
        f.write("'loss_train',\n'loss_test',\n'bp_activation_angles',\n")
        f.write("'gn_activation_angles',\n'rec_loss'\n]\n\n")

        fixed_space = {}
        fixed_space.update(config_multiple_hpsearch.fixed_space)
        if hasattr(config_multiple_hpsearch, 'variable_fixed_space2'):
            fixed_space.update(config_multiple_hpsearch.variable_fixed_space2)
        fixed_space.update(config_multiple_hpsearch.figure_config)
        write_dict_to_txt(f, fixed_space, 'fixed')

        f.write("if __name__ == '__main__':\n")
        f.write("\t pass\n")


def create_cmd_args(filename, best_configs, config_multiple_hpsearch, args):
    file_path = os.path.join(args.mother_dir, filename)
    with open(file_path, 'w') as f:
        for name in best_configs.keys():
            cmd_dct = {}
            cmd_dct.update(config_multiple_hpsearch.fixed_space)
            if hasattr(config_multiple_hpsearch, 'variable_fixed_space2'):
                cmd_dct.update(config_multiple_hpsearch.variable_fixed_space2)
            cmd_dct.update(config_multiple_hpsearch.multiple_runs[name])
            cmd_dct.update(best_configs[name])
            cmd_dct.update(config_multiple_hpsearch.figure_config)
            write_dict_to_cmdstyle(f, cmd_dct, name)

def run_multiple_hpsearches():
    parser = argparse.ArgumentParser(
        description='hpsearch - Automatic Parameter Search for multiple '
                    'hpsearches -- ' +
                    'Note, that the search values are defined '
                    'in the source code of the ' +
                    'accompanied configuration file!')
    parser.add_argument('--out_dir', type=str,
                        default='./logs/multiple_hyperparam_search',
                        help='Where should all the output files be written ' +
                             'to? Default: %(default)s.')
    parser.add_argument('--grid_module', type=str, default=_DEFAULT_GRID,
                        help='Name of module to import from which to read ' +
                             'the hyperparameter search grid. The module ' +
                             'must define the two variables "grid" and ' +
                             '"conditions". Default: %(default)s.')
    parser.add_argument(
        '--run_cwd',
        type=str,
        default='.',
        help='The working directory in which runs are ' +
             'executed (in case the run script resides at a ' +
             'different folder than this hpsearch script. ' +
             'All outputs of this script will be relative to ' +
             'this working directory (if output folder is ' +
             'defined as relative folder). ' +
             'Default: "%(default)s".')
    parser.add_argument(
        '--force_run',
        type=bool,
        default=False,
        help='If False, checks first if hp_search was run already with the given fixed values.' +
             'If True, runs always.'
    )
    parser.add_argument(
        '--cpu_per_trial',
        type=float,
        default=1.,
        help='this is the number of cpu used per trial. Note that this can be a fraction.')
    parser.add_argument(
        '--gpu_per_trial',
        type=float,
        default=0.5,
        help='this is the number of gpu used per trial. Note that this can be a fraction.')
    parser.add_argument(
        '--num_sample',
        type=int,
        metavar='N',
        default=100,
        help='this is the number of hp search trials performed.')
    parser.add_argument('--num_sample_fb', type=int, default=None,
                        help='number of hp samples for the feedback hpsearch.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    parser.add_argument('--figure_config_filename', type=str,
                        default='config_figures_hpsearch.py',
                        help='the name for the config file that will be created'
                             'for the figure script.')
    parser.add_argument('--nonsequential', action='store_true',
                        help='Flag indicating whether the hpsearches are '
                             'nonsequential (e.g. only train the forward '
                             'parameters, not the feedback ones.')
    # TODO build in "continue" option to finish incomplete commands.

    args = parser.parse_args()

    # process num_sample
    args.num_sample_forward = args.num_sample
    if args.num_sample_fb is None:
        args.num_sample_fb = args.num_sample

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    args.mother_dir = args.out_dir

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    config_multiple_hpsearch = importlib.import_module(args.grid_module)

    best_configs = {}

    for name in config_multiple_hpsearch.multiple_runs.keys():
        args.seq_dir = os.path.join(args.mother_dir, name)
        if not os.path.isdir(args.seq_dir):
            os.mkdir(args.seq_dir)
        current_config = create_config_file(config_multiple_hpsearch,
                                            name)
        best_config = nonsequential_hpsearch(args, current_config)
        best_configs[name] = best_config

    create_fig_config_file(args.figure_config_filename, best_configs,
                           config_multiple_hpsearch, args)
    create_cmd_args('cmd_args.txt', best_configs, config_multiple_hpsearch,
                    args)



if __name__ == '__main__':
    run_multiple_hpsearches()
