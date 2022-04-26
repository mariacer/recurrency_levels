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
Hyperparameter search objects
-----------------------------

Implementation of a class to handle hyperparameter searches as well as a
collection of helper functions for the hpsearch module.
"""
import __init__ # pylint: disable=unused-import
import importlib
import os
import json
import sys
import argparse
import warnings
import random
import numpy as np
from datetime import datetime
import re
import torch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
import shutil

#import hpsearch.hp_utils as utils

################################################################
### The following variables will be otherwritten in the main ###
################################################################
_OUT_ARG = 'out_dir'  # Default value if attribute `_OUT_ARG` does not exist.
# Function handle to parser of performance summary file.
# Default parser `_get_performance_summary` used.

class HPSearchHandler:
    """Wrapper class to store the config of the hp-search and pass it to each
    ray workers.
    """

    def __init__(self, module, args):
        assert hasattr(module, 'search_space'), \
            "search_space attribute missing in your template. " \
            "Update your template referring to hpsearch_config.py"
        assert hasattr(module, 'fixed_space'), \
            "fixed_space attribute missing in your template. " \
            "Update your template referring to hpsearch_config.py"
        assert hasattr(module, '_MODULE_NAME'), \
            "_MODULE_NAME attribute missing in your template. " \
            "Update your template referring to hpsearch_config.py"
        if hasattr(module, 'conditions') and (len(module.conditions) > 0):
            # FIXME: Support conditions here, although it is unclear what the 
            # behavior should be.
            warnings.warn('Conditions are not supported in Tune hp search. ' +
                'They will be ignored.')

        self._module_name = args.grid_module
        self._search_space = module.search_space
        self._fixed_space = module.fixed_space
        self._train_module = module._MODULE_NAME
        self._performance_key = module._PERFORMANCE_KEY
        self._performance_sort_asc = module._PERFORMANCE_SORT_ASC
        self._num_sample = args.num_sample
        self._cpu_per_trial = args.cpu_per_trial
        self._gpu_per_trial = args.gpu_per_trial
        self._out_dir = args.out_dir
        self._name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._root_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self._max_or_min = module._MAX_OR_MIN

        print('The mode of optimization is ' + self._max_or_min)
        # print('root dir: {}'.format(self._root_dir))

        # Verify the train module has a run method.
        sys.path.append(self._root_dir)
        train_module = importlib.import_module(self._train_module)
        assert hasattr(train_module, 'run'), \
            "run attribute missing in your train module. " \
            "Wrap your main script in a run() method."
        # assert not args.run_cluster, "run_cluster not enabled yet with tune."

    def search_hp(self):
        """Perform the hyperparameter search."""

        print("Starting Hyper Parameter Search.")
        print("USING {} {} AS OBJECTIVE".format(self._max_or_min,
                                                self._performance_key))
        print('Fixed parameters: {}'.format(self._fixed_space))
        if not os.path.isdir(self._out_dir):
            os.mkdir(self._out_dir)
        if not os.path.isdir(os.path.join(self._out_dir, self._name)):
            os.mkdir(os.path.join(self._out_dir, self._name))

        ### Restrict the total number of CPU/GPU used on this machine.
        # ray.init(num_cpus=<int>, num_gpus=<int>)

        analysis = tune.run(
            lambda config: self.run_training(config),
            name=self._name,
            config=self._search_space,
            num_samples=self._num_sample,
            resources_per_trial={
                "cpu": self._cpu_per_trial,
                "gpu": self._gpu_per_trial if torch.cuda.is_available() else 0},
            search_alg=HyperOptSearch(
                self._search_space,
                max_concurrent=max(
                    3,
                    self._num_sample // 4),
                metric=self._performance_key,
                mode=self._max_or_min),
            local_dir=self._out_dir,
            raise_on_failed_trial=False)

        ### Save results
        # Save the runs to the specified file.
        print("Done. Saving now!")
        self.save_results(analysis)

        return analysis

    def save_results(self, analysis):
        """Save the results of the hpsearch.

        A csv file with the results of all the runs is saved, and a 
        best_config.py file with the config of the best run.

        Args:
            analysis: the analysis file resulting from tune.run()
        """
        print('saving at {}'.format(os.path.join(self._out_dir, self._name)))

        # Store the hpsearch config file.
        hpconfig_name = self._module_name.replace('.', '/') + '.py'
        hpconfig_dest = os.path.join(os.path.join(self._out_dir, self._name), \
                                     'hpconfig.py')
        shutil.copyfile(hpconfig_name, hpconfig_dest)

        # Store the results.
        results_file = os.path.join(
            self._out_dir, self._name, 'search_results.csv')
        analysis.dataframe().to_csv(results_file, sep=',', index=False)
        save_compact_results(analysis.dataframe(),
                             os.path.join(self._out_dir, self._name,
                                          'search_results_compact.csv'),
                             performance_key=self._performance_key,
                             ascending=self._max_or_min == 'min')
        
        # Store the best configuration.
        best_config = analysis.get_best_config(
            metric=self._performance_key,
            mode=self._max_or_min)
        best_config.update(self._fixed_space)
        best_config['hpsearch'] = False
        dict2config(best_config, os.path.join(self._out_dir, self._name,
                                          'best_config.py'))

    def run_training(self, config):
        # Setting up environment.
        sys.path.append(self._root_dir)
        # Set output dir where tmp values will be stored.
        config[_OUT_ARG] = './tmp'
        _override_cmd_arg(config, self._fixed_space)
        # _override_cmd_arg(config)
        train_module = importlib.import_module(self._train_module)
        performance_dict = getattr(train_module, "run")()
        print('TRAINING FINISHED')
        tune.track.log(**performance_dict)

def _override_cmd_arg(config, fixed_space):
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)
    for k, v in fixed_space.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)


def save_compact_results(result_df, file_name, ascending=False,
                         performance_key=None):
    """Save a compact csv file containing only the logged performance
    results and the hyperparam values.

    Args:
        result_df (pd.DataFrame): result dataframe
    """
    unnecessary_keys = ['trial_id',
                        'training_iteration', 'time_this_iter_s', 'done',
                        'timesteps_total',
                        'episodes_total', 'experiment_id', 'date', 'timestamp',
                        'time_total_s',
                        'pid', 'hostname', 'node_ip', 'time_since_restore',
                        'timesteps_since_restore', 'iterations_since_restore',
                        'experiment_tag',
                        'logdir']
    if performance_key is None:
        performance_key = result_df.keys()[0]
    copy_df = result_df.copy()
    for key in unnecessary_keys:
        if key in copy_df.keys():
            del copy_df[key]
    copy_df.sort_values(by=performance_key, axis=0, ascending=ascending,
                        inplace=True)
    copy_df.to_csv(file_name, sep=',', index=False)


def _read_config(config_mod, require_perf_eval_handle=False,
                 require_argparse_handle=False):
    """Parse the configuration module and check whether all attributes are set
    correctly.

    This function will set the corresponding global variables from this script
    appropriately.

    Args:
        config_mod: The implemented configuration template
            :mod:`hpsearch.hpsearch_postprocessing`.
        require_perf_eval_handle: Whether :attr:`_PERFORMANCE_EVAL_HANDLE` has
            to be specified in the config file.
        require_argparse_handle: Whether :attr:`_ARGPARSE_HANDLE` has to be
            specified in the config file.
    """
    assert(hasattr(config_mod, '_MODULE_NAME'))
    assert(hasattr(config_mod, '_SUMMARY_KEYWORDS'))
    globals()['_MODULE_NAME'] = config_mod._MODULE_NAME
    globals()['_SUMMARY_KEYWORDS'] = config_mod._SUMMARY_KEYWORDS

    # Ensure downwards compatibility -- attributes did not exist previously.
    if hasattr(config_mod, '_OUT_ARG'):
        globals()['_OUT_ARG'] = config_mod._OUT_ARG

    if hasattr(config_mod, '_PERFORMANCE_KEY') and \
            config_mod._PERFORMANCE_KEY is not None:
        globals()['_PERFORMANCE_KEY'] = config_mod._PERFORMANCE_KEY
    else:
        globals()['_PERFORMANCE_KEY'] = config_mod._SUMMARY_KEYWORDS[0]

    if hasattr(config_mod, '_PERFORMANCE_SORT_ASC'):
        globals()['_PERFORMANCE_SORT_ASC'] = config_mod._PERFORMANCE_SORT_ASC

    if require_argparse_handle:
        assert(hasattr(config_mod, '_ARGPARSE_HANDLE') and
               config_mod._ARGPARSE_HANDLE is not None)
        globals()['_ARGPARSE_HANDLE'] = config_mod._ARGPARSE_HANDLE

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
    if not (name is None or name == ''):
        name = '_' + name
    file_handler.write('config' + name + ' = {\n')
    for key, value in dct.items():
        if isinstance(value, str):
            value = "'" + value + "'"
        else:
            value = str(value)
        file_handler.write("'" + key + "': " + value + ",\n")
    file_handler.write('}\n\n')

if __name__ == '__main__':
    pass
