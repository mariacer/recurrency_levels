#!/usr/bin/env python3
# Copyright 2020 Alexander Meulemans
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
# @title          :hpsearch/run_config.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :03/11/2020
# @version        :1.0
# python_version  :3.7
"""
Script for running an experiment from a given config file.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import importlib
import argparse
import os
import sys
import numpy as np
import pickle

from real_world_benchmarks import run_audioset, run_pos
from student_teacher import run_student_teacher
from utils.config_utils import _override_cmd_arg

def run():
    """Run the experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str,
                        default='configs.audioset_rnn_microcircuit_random',
                        help='The name of the module containing the config. '
                             'Alternatively, a path to a pickle file can be '
                             'provided.')
    parser.add_argument('--dataset', type=str,
                        choices=['audioset', 'pos', 'student_teacher'], 
                        default='pos',
                        help='The dataset for the experiment.')
    args = parser.parse_args()

    # Extract the config.
    try:
        # If a config module is provided.
        config_module = importlib.import_module(args.config_module)
        config = config_module.config
        _override_cmd_arg(config)
    except:
        # If a path to pickle file is provided.
        with open(os.path.join(args.config_module, 'config.pickle'), 'rb') as f:
            config = pickle.load(f)

    # Run the requested experiment.
    if args.dataset == 'audioset':
        summary = run_audioset.run(config)
    elif args.dataset == 'pos':
        summary = run_pos.run(config)
    elif args.dataset == 'student_teacher':
        summary = run_student_teacher.run(config)

    return summary

if __name__ == '__main__':
    run()
