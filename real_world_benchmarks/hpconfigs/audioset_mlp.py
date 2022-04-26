#!/usr/bin/env python3
# Copyright 2020 by Alexander Meulemans
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
# @title           :real_world_benchmarks/hpconfigs/audioset_mlp.py
# @author          :am
# @contact         :ameulema@ethz.ch
# @created         :27/10/2020
# @version         :1.0
# @python_version  :3.7
"""
Template for hyperparameter configuration files
-----------------------------------------------

Note, this is just a template for a hyperparameter configuration and not an
actual source file.
"""

import numpy as np
from hyperopt import hp

##########################################
### Please define all parameters below ###
##########################################

# *Full* name of the module corresponding to the script that should be executed
# by the hyperparameter search.
_MODULE_NAME = 'real_world_benchmarks.run_audioset'

### ALL COMMAND-LINE OPTIONS TO BE SEARCHED
# Same as the attribute grid, but for hyperopt hyper parameter tuning.
# HyperOpt requires a distribution to be defined for each hyperparemeter.
search_space = {
    'lr': hp.choice('lr', [1e-4, 1e-3, 1e-2]),
    'n_iter': hp.choice('n_iter', [2000, 5000, 10000, 20000]),
    'weight_decay': hp.choice('weight_decay', [1e-6, 1e-4, 0]),
    'rnn_arch': hp.choice('rnn_arch', ["32", "64", "128"]),
}

### ALL COMMAND-LINE OPTIONS NOT ACCESSIBLE TO THE HPSEARCH
fixed_space = {
    # IMPORTANT: --hpsearch must always be true for doing a hpsearch, to
    # prevent that the dataset is downloaded for each run.
    'hpsearch': True,
    'use_mlp':True,

    # TRAIN ARGS
    'batch_size': 64,
    'dont_use_adam': False,

    # RNN ARGS
    'dont_use_bias': False,
}

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
# This code only has to be adapted if you are setting up this template for a
# new simulation script!

# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

# These are the keywords that are supposed to be in the summary file.
_SUMMARY_KEYWORDS = [
    # 'loss_train',
    # 'loss_test',
    'loss_train_last',
    'loss_test_last',
    'loss_train_best',
    'loss_test_best',
    # 'acc_train',
    # 'acc_test',
    'acc_train_last',
    'acc_test_last',
    'acc_train_best',
    'acc_test_best',
]


# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = 'acc_test_last'
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# IMPORTANT: indicate below whether the performance key must be
# maximized or minimized. use {'min', 'max'}
_MAX_OR_MIN = 'max'
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False


if __name__ == '__main__':
    pass
