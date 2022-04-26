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
# @title           :hpsearch/run_hpsearch.py
# @author          :am
# @contact         :ameulema@ethz.ch
# @created         :27/10/2020
# @version         :1.0
# @python_version  :3.7
"""
Executable script to perform hyperparameter searches
----------------------------------------------------

This scripts runs a hyperparameter search based on configuration file provided
via the command line.
"""
# Do not delete the following import for all executable scripts!
import argparse
import os
import random
import numpy as np
import json
import importlib

from hpsearch import hpsearch
#import hpsearch.hp_utils as utils


# From which module to read the default grid.
_DEFAULT_GRID = 'hpsearch.hpsearch_config'

################################################################
### The following variables will be otherwritten in the main ###
################################################################
# See method `_read_config`.
# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script.
_SCRIPT_NAME = None  # Has to be specified in helper module!
# This file is expected to reside in the output folder of the simulation.
_SUMMARY_KEYWORDS = None  # Has to be specified in helper module!
# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'  # Default value if attribute `_OUT_ARG` does not exist.
# Function handle to parser of performance summary file.
# Default parser `_get_performance_summary` used.
_SUMMARY_PARSER_HANDLE = None
# A function handle, that is used to evaluate whether an output folder should
# be kept.
_PERFORMANCE_EVAL_HANDLE = None  # Has to be set in config file.
# According to which keyword will the CSV be sorted.
_PERFORMANCE_KEY = None  # First key in `_SUMMARY_KEYWORDS` will be used.
# Sort order.
_PERFORMANCE_SORT_ASC = False
# FIXME should be deleted soon.
_ARGPARSE_HANDLE = None
################################################################

# This will be a list of booleans, each representing whether a specific cmd has
# been executed.
_CMD_FINISHED = None


def run():
    """Run the hpsearch."""

    print(os.getcwd())
    parser = argparse.ArgumentParser(
        description='hpsearch - Automatic Parameter Search -- ' +
                    'Note, that the search values are defined in the source ' +
                    'code of the accompanied configuration file!')
    parser.add_argument('--out_dir', type=str,
                        default='hpsearch/out',
                        help='Where should all the output files be written ' +
                             'to? Note that this needs to be a folder within ' +
                             'a folder of the root. Default: %(default)s.')
    parser.add_argument('--grid_module', type=str, default=_DEFAULT_GRID,
                        help='Name of module to import from which to read ' +
                             'the hyperparameter search grid. The module ' +
                             'must define the two variables "grid" and ' +
                             '"conditions". Default: %(default)s.')
    parser.add_argument('--cpu_per_trial', type=float, default=1.,
                        help='This is the number of cpus used per trial. ' +
                             'Note that this can be a fraction.')
    parser.add_argument('--gpu_per_trial', type=float, default=0.5,
                        help='This is the number of gpus used per trial. ' + 
                             'Note that this can be a fraction.')
    parser.add_argument('--num_sample', type=int, metavar='N', default=100,
                        help='The number of hp search trials to perform.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    # TODO build in "continue" option to finish incomplete commands.
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Get hyperparameter search grid from specified module.
    grid_module = importlib.import_module(args.grid_module)
    hpsearch._read_config(grid_module)

    # Run the search.
    hp_search_handler = hpsearch.HPSearchHandler(grid_module, args)
    analysis = hp_search_handler.search_hp()

    # Get the best config.
    best_config = analysis.get_best_config(
        metric=hp_search_handler._performance_key,
        mode=hp_search_handler._max_or_min)
    print("Best config is", best_config)

    print('### Running Hyperparameter Search ... Done')


if __name__ == '__main__':
    run()