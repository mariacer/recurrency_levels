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
"""
- **title**          :hpsearch/postprocess_hpsearch.py
- **author**         :alexander meulemans
- **contact**        :ameulema@ethz.ch
- **created**        :20/02/2020
- **version**        :1.0
- **python_version** :3.6.8

A postprocessing for a hyperparameter search that has been executed via the
script :mod:`hpsearch.py`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
sys.path.append('.')

import hpsearch

def get_hyperparam_keys(result_df):
    """ Get the keys of the result dataframe concerning the hyperparameters."""
    hp_keys = []
    hp_names = []
    for key in result_df.keys():
        if 'config/' in key:
            hp_keys.append(key)
            hp_names.append(key[7:])
    return hp_keys, hp_names


def get_best_config(result_df, performance_key, min_or_max):
    if min_or_max == 'max':
        optimal_idx = result_df[performance_key].idxmax()
    elif min_or_max == 'min':
        optimal_idx = result_df[performance_key].idxmin()
    else:
        raise ValueError('min_or_max argument should be either "min" or "max".'
                         'Given value <{}> not recognized.'.format(min_or_max))

    config_dict = {}
    hp_keys, hp_names = get_hyperparam_keys(result_df)
    for i, key in enumerate(hp_keys):
        config_dict[hp_names[i]] = result_df[key][optimal_idx]
    return config_dict

def is_log_distributed(series):
    """ Check whether the data in series is log distributed or not."""
    if isinstance(series[0], (str, bool, np.bool_)):
        return False
    # print(type(series.max()))
    interval = series.max() - series.min()
    # check how many datapoints are within the first bucket of the data
    first_bucket = series < series.min() + interval/10
    return first_bucket.sum() > len(series)/2

def run():
    parser = argparse.ArgumentParser(
        description='Postprocessing of the Automatic Parameter Search')
    parser.add_argument('--out_dir', type=str,
                        default='../out/hyperparam_search',
                        help='The output directory of the hyperparameter ' +
                             'search. Default: %(default)s.')
    parser.add_argument('--performance_key', type=str, default='acc_test_last',
                        help='Used for displaying the best hyperparameters.')
    parser.add_argument('--grid_module', type=str,
                        default='hpsearch.hpsearch_config',
                        help='Name of module to import from which to read ' +
                             'the hyperparameter search grid. The module ' +
                             'must define the two variables "grid" and ' +
                             '"conditions". Default: %(default)s.')
    parser.add_argument('--mode', type=str, choices=['min', 'max'],
                        default='max',
                        help='Indicates whether the performance values should'
                             'be maximized or minimized. Default: %(default)s.')
    # parser.add_argument('--plot_results', action='store_true',
    #                     help='make plots of the performance value i.f.o. the'
    #                          'hyperparameters'
    # )
    args = parser.parse_args()

    print('### Running Hyperparameter Search Postprocessing ...')

    # grid_module = importlib.import_module(args.grid_module)
    file_path = os.path.join(args.out_dir,
                             'search_results_compact.csv')
    # if not os.path.exists(file_path):
    #     results = pd.read_csv(os.path.join(args.out_dir,
    #                                    'search_results.csv'),
    #                       delimiter=';')
    #     save_compact_results(results, file_path)

    results = pd.read_csv(file_path,
                          delimiter=';')
    if args.performance_key is None:
        args.performance_key = results.keys()[0]

    best_config = get_best_config(results, args.performance_key, args.mode)
    print('### BEST CONFIGURATION ###')
    print(best_config)
    parsing_arguments = ""
    for key, entry in best_config.items():
        parsing_arguments += " --" + key + " " + str(entry)
    print(parsing_arguments)

    best_config_file = os.path.join(args.out_dir, 'best_config.csv')
    # utils.dict2csv(best_config, best_config_file)

    if True:  # args.plot_results:
        performance_values = results[args.performance_key]
        hp_keys, hp_names = get_hyperparam_keys(results)
        for key, name in zip(hp_keys, hp_names):
            # ax = results.plot.scatter(x=key, y=args.performance_key,
            #                           logx=is_log_distributed(results[key]))
            plt.figure()
            ax = plt.gca()
            if is_log_distributed(results[key]):
                ax.set_xscale('log')
            ax.scatter(results[key], performance_values)
            ax.set_ylabel(args.performance_key)
            plt.title(name)
            plt.savefig(os.path.join(args.out_dir, name + ".svg"))
            plt.show()

    # with open(os.path.join(args.out_dir, 'analysis.pickle'), "rb") as pickle_in:
    #     analysis = pickle.load(pickle_in)

    print('### Running Hyperparameter Search Postprocessing ... Done')


if __name__ == '__main__':
    run()
