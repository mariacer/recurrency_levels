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
# @title          :student_teacher/run_same_vs_different_topology.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :12/05/2021
# @version        :1.0
# python_version  :3.7
"""
This is a script to run multiple student-teacher experiments, where half of
the students will have identical topology to the teacher, and the other half
will have same architecture (number of neurons and connections), but different
topology and thus recurrency levels.

This file is inspired from `run_multiple_student_teacher`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import importlib
import argparse
import os
import sys
import numpy as np
import pandas as pd
import shutil
import select
from warnings import warn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

from student_teacher import run_student_teacher
from student_teacher.run_multiple_student_teacher import post_process, \
    plot_perf_vs_feedforwardness
from utils.config_utils import _override_cmd_arg
from utils.misc import str_to_float_or_list

def run_training(config):
    """Run the mainfile with the given config file and save the results."""
    _override_cmd_arg(config)
    summary = run_student_teacher.run()
    return summary

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='',
                        help='Directory where the results will be saved.')
    parser.add_argument('--filename', type=str, default='results',
                        help='Name of the file in which the results will be '
                             'saved.')
    parser.add_argument('--config_module', type=str,
                        default='configs.simple_vanilla_rnn_for_tests',
                        help='The name of the module containing the configs.')
    parser.add_argument('--hpsearch', action='store_true',
                        help='Whether this is an ongoing hpsearch.')
    parser.add_argument('--n_students', type=int, default=100,
                        help='Number of students to evaluate in total.')
    parser.add_argument('--same_weight_sign', action='store_true',
                        help='Whether the student should have the same weight '
                             'sign as the teacher.')
    args = parser.parse_args()

    filename = args.filename
    if filename[-4:] != '.csv':
        filename += '.csv'

    ### Output folder.
    if os.path.exists(args.out_dir):
        print('The output folder %s already exists. Post-processing it.' % \
            args.out_dir)
        results = pd.read_csv(os.path.join(args.out_dir, filename), index_col=0)
        post_process(results, args.out_dir, filename)
        exit()
    elif args.out_dir == '':
        args.out_dir = './out/multiple_students/run_' +  \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(args.out_dir)
        print("Created output folder %s." % (args.out_dir))
    else:
        os.makedirs(args.out_dir)
        print("Created output folder %s." % (args.out_dir))

    # Get the number of students in each group.
    n_students_per_group = int(args.n_students/2)
    assert n_students_per_group >= 1

    config_module = importlib.import_module(args.config_module)
    random_seeds = np.random.randint(0, 10000, n_students_per_group)

    # Get results for different students (topology is different).
    config_module.config['set_identical_topology'] = False
    for i, seed in enumerate(random_seeds):
        print('Initiating run {} ...'.format(i))
        config_module.config['random_seed'] = seed
        config_module.config['arch_random_seed'] = seed
        config_module.config['out_dir'] = os.path.join(args.out_dir, \
            'seed_%i'%i)
        if args.hpsearch:
            config_module.config['hpsearch'] = True
        summary = run_training(config_module.config)

        if i == 0:
            # initialize result dataframe
            columns = summary.keys()
            index = [j for j in range(n_students_per_group)]
            results = pd.DataFrame(index=index, columns=columns)

        for key in columns:
            results.at[i, key] = summary[key]

        results.to_csv(os.path.join(args.out_dir, filename))

    # Get results for identical students (only weight values are different).
    config_module.config['set_identical_topology'] = True
    if args.same_weight_sign:
        config_module.config['set_same_weight_sign'] = True
    for i, seed in enumerate(random_seeds):
        print('Initiating run {} ...'.format(i))
        config_module.config['random_seed'] = seed
        config_module.config['out_dir'] = os.path.join(args.out_dir, \
            'seed_%i_identical'%i)
        summary = run_training(config_module.config)

        if i == 0:
            # initialize result dataframe
            columns = summary.keys()
            index = [j for j in range(n_students_per_group)]
            results_identical = pd.DataFrame(index=index, columns=columns)

        for key in columns:
            results_identical.at[i, key] = summary[key]

        results_identical.to_csv(os.path.join(args.out_dir, 'identical_' + 
            filename))

    # Post_process.
    msg = ''
    if args.same_weight_sign:
        msg = '_same_weight_sign'
    results = post_process(results, args.out_dir, args.filename, plot=False)
    results_identical = post_process(results_identical, args.out_dir,
        'identical_' + msg + args.filename, plot=False)

    # Build some experiment summary for the hpsearches.
    exp_summary = dict(results.loc['mean'])
    exp_summary['nonidentical_feedforwardness_std'] = \
        results.loc['std']['feedforwardness']
    exp_summary['nonidentical_loss_test_last_std'] = \
        results.loc['std']['loss_test_last']
    exp_summary['nonidentical_loss_train_last_std'] = \
        results.loc['std']['loss_train_last']
    exp_summary['identical_loss_test_last_std'] = \
        results_identical.loc['std']['loss_test_last']
    exp_summary['identical_loss_train_last_std'] = \
        results_identical.loc['std']['loss_train_last']
    exp_summary['students_per_group'] = n_students_per_group
    exp_summary['same_weight_sign'] = args.same_weight_sign

    # Compute t-test to see differences across populations.
    _, p_value = stats.ttest_ind(results_identical['loss_test_last'], \
                              results['loss_test_last'])
    exp_summary['p_value'] = p_value

    # Compute t-test across different training points.
    loss_test_all = []
    loss_train_all = []
    loss_test_all_identical = []
    loss_train_all_identical = []
    for i in range(len(results['loss_test_all'][:-2])):
        loss_test_all.append(str_to_float_or_list(
                results['loss_test_all'][i]))
        loss_train_all.append(str_to_float_or_list(\
                results['loss_train_all'][i]))
        loss_test_all_identical.append(str_to_float_or_list(
                results_identical['loss_test_all'][i]))
        loss_train_all_identical.append(str_to_float_or_list(\
                results_identical['loss_train_all'][i]))
    loss_test_all = np.array(loss_test_all)
    loss_train_all = np.array(loss_train_all)
    loss_test_all_identical = np.array(loss_test_all_identical)
    loss_train_all_identical = np.array(loss_train_all_identical)

    exp_summary['train_p_values'] = []
    exp_summary['test_p_values'] = []
    for i in range(loss_test_all.shape[1]):
        _, train_p_val = stats.ttest_ind(loss_train_all[:, i], \
                                      loss_train_all_identical[:, i])
        exp_summary['train_p_values'].append(train_p_val)
        _, test_p_val = stats.ttest_ind(loss_test_all[:, i], \
                                        loss_test_all_identical[:, i])
        exp_summary['test_p_values'].append(test_p_val)
    exp_summary['min_train_p_values'] = np.min(exp_summary['train_p_values'])
    exp_summary['min_test_p_values'] = np.min(exp_summary['test_p_values'])

    # Plot results.
    plot_combined_results(exp_summary, results, results_identical, args.out_dir,
                          args.filename, test=False, 
                          same_weight_sign=args.same_weight_sign)
    plot_combined_results(exp_summary, results, results_identical, args.out_dir,
                          args.filename, same_weight_sign=args.same_weight_sign)

    return exp_summary

def plot_combined_results(summary, results, results_identical, out_dir,
                          filename, test=True, same_weight_sign=False):
    """Plot results of identical and non-identical students together.
    """

    key = 'loss_test_last'
    mode = 'test'
    p_vals_key = 'test_p_values'
    if not test:
        key = 'loss_train_last'
        mode = 'train'
        p_vals_key = 'train_p_values'

    ### Scatter plot.
    ff_teacher = results['feedforwardness_teacher'][0]
    ff_identical_students = results_identical['feedforwardness']\
        .tolist()[:-2]
    ff_students = results['feedforwardness'].tolist()[:-2]
    plt.figure()
    y_vals = results[key].tolist()[:-2]
    y_vals_identical = results_identical[key].tolist()[:-2]
    # Normalize for visualization ease.
    val_max = np.max(y_vals)
    y_vals /= val_max
    y_vals_identical /= val_max
    print(results[key], results_identical[key])
    plt.scatter(ff_students, y_vals, label='non-identical student')
    plt.scatter(ff_identical_students, y_vals_identical, color='g',
                label='identical student')
    plt.xlabel('student feedforwardness')
    plt.ylabel('%s loss (AU)'%mode)
    plt.plot([ff_teacher, ff_teacher], plt.gca().get_ylim(), '-r',
             label='teacher ff')
    plt.legend()
    msg = ''
    if same_weight_sign:
        msg = '_same_weight_sign'
    plt.savefig(os.path.join(out_dir, filename + msg + '_%s.png' % mode))

    ### Histrogram.
    plt.figure()
    plt.hist(y_vals, alpha=0.5, color='b', 
             label="non-identical student")
    plt.hist(y_vals_identical, alpha=0.5, color='g', label="identical student")
    plt.xlabel('%s loss (AU)'%mode)
    plt.ylabel("number of students")
    plt.legend()
    plt.savefig(os.path.join(out_dir, filename + '_hist' + msg + '_%s.png' % \
        mode))

    ### Evolution of p-value of performance throughout training.
    plt.figure()
    plt.plot(summary[p_vals_key])
    plt.xlabel('training time')
    plt.ylabel('%s performance p-value'%mode)
    plt.savefig(os.path.join(out_dir, filename + '_pvals' + msg + '_%s.png' % \
        mode))

if __name__ == '__main__':
    run()
