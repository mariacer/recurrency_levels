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
# @title           :student_teacher/hpconfigs/simple_vanilla_rnn.py
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
import os

from student_teacher import st_utils

##########################################
### Please define all parameters below ###
##########################################

# *Full* name of the module corresponding to the script that should be executed
# by the hyperparameter search.
_MODULE_NAME = 'student_teacher.run_same_vs_different_topology'

### ALL COMMAND-LINE OPTIONS TO BE FIXED
fixed_space_config = {'batch_size': 64,
                      'use_vanilla_rnn': True,
                      'teacher_use_vanilla_rnn': True,
                      'save_weights': True,
                      'save_logs': True,
                      'use_same_resources':True,
                     }

### ALL COMMAND-LINE OPTIONS TO BE SEARCHED
search_space_config = {
    'lr': [1e-4, 1e-3, 1e-2],
    'n_iter': [20], #100, 1000],
    'teacher_rnn_arch': [6],# 16, 32],
    'teacher_rec_sparsity': [0.2],# 16, 32],
    # 'rec_sparsity': [0.1, 0.5],
    # 'teacher_output_size': [5, 10], 
    # 'teacher_input_size': [5, 10], 
    # 'teacher_n_ts_in': [50, 100], 
    # 'teacher_n_ts_out': [50, 100]
    }

### Create the actual search space for the current hpsearch.
all_config_names = st_utils.get_configs_multiple_student_hpsearch(\
                                    search_space_config, fixed_space_config)
search_space = {
    'same_weight_sign': hp.choice('same_weight_sign', [True, False]),
    'config_module': hp.choice('config_module', all_config_names),
}

### ALL COMMAND-LINE OPTIONS NOT ACCESSIBLE TO THE HPSEARCH
fixed_space = {
    # IMPORTANT: --hpsearch must always be true for doing a hpsearch, to
    # prevent that the dataset is downloaded for each run.
    'hpsearch':True,
    'n_students':2
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
    'nonidentical_feedforwardness',
    'nonidentical_feedforwardness_std',
    'identical_feedforwardness',
    'identical_feedforwardness_std',
    'p_value',
]


# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = 'p_value'
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# IMPORTANT: indicate below whether the performance key must be
# maximized or minimized. use {'min', 'max'}
_MAX_OR_MIN = 'min'
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False


if __name__ == '__main__':
    pass
