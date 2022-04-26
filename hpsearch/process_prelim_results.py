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
# @title          :hpsearch/process_prelim_results.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
This is a script to preprocess results form a hyperparameter search.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
import json
import os
import pandas as pd

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str)
    args = parser.parse_args()
    result_dir = args.result_dir

    keys = [
        'acc_train_last',
        'acc_train_best',
        'loss_train_last',
        'loss_train_best',
        'acc_test_last',
        'acc_test_best',
        'loss_test_last',
        'loss_test_best',
    ]

    hp_keys = [
        "adam_beta1",
        "beta1_fb",
        "weight_decay",
        "rnn_arch",
        "n_iter",
        "input_fraction",
        "output_fraction",
        "lr",
    ]

    performance_measure = 'acc_test_last'

    def write_dict_to_txt(file_handler, dct, name):
        file_handler.write('config_' + name + ' = {\n')
        for key, value in dct.items():
            if isinstance(value, str):
                value = "'" + value + "'"
            else:
                value = str(value)
            file_handler.write("'" + key + "': " + value + ",\n")
        file_handler.write('}\n\n')


    hpsearch_results = pd.DataFrame(columns=[key for key in keys+hp_keys])

    idx = 0

    best_performance = 0
    best_hp_run = False
    best_results = None
    best_hp = None
    for subdir, dirs, files in os.walk(result_dir):
        result_file = os.path.join(subdir, 'result.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                try:
                    results = json.load(f)
                    for key in keys:
                        if key in results.keys():
                            hpsearch_results.at[idx, key] = results[key]
                    if results[performance_measure] > best_performance:
                        best_performance = results[performance_measure]
                        best_results = results
                        best_hp_run = True
                except Exception as error:
                    print(error)
                    print('Could not load {}'.format(result_file))

        param_file = os.path.join(subdir, 'params.json')
        if os.path.exists(param_file):
            with open(param_file) as f:
                try:
                    params = json.load(f)
                    for hp in hp_keys:
                        if hp in params.keys():
                            hpsearch_results.at[idx, hp] = params[hp]
                    if best_hp_run:
                        best_hp = params
                        best_hp_run = False
                except Exception as error:
                    print(error)
                    print('Could not load {}'.format(param_file))

        idx += 1

    hpsearch_results.to_csv(os.path.join(result_dir, 'preliminary_results.csv'))

    with open(os.path.join(result_dir, 'best_preliminary_results.py'), 'w') as f:
        # json.dump(best_results, f)
        write_dict_to_txt(f, best_results, 'results')

    with open(os.path.join(result_dir, 'best_preliminary_hp.py'), 'w') as f:
        # json.dump(best_hp, f)
        write_dict_to_txt(f, best_hp, 'hp')

if __name__ == '__main__':
    run()
