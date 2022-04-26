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
# @title          :student_teacher/run_multiple_student_teacher.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :12/04/2021
# @version        :1.0
# python_version  :3.7
"""
This is a script to run a certain hyperparameter setting for various random
seeds, to test random seed robustness.
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
from sklearn.neighbors import KernelDensity

from student_teacher import run_student_teacher
from utils.config_utils import _override_cmd_arg
from utils.args import parse_cmd_arguments

def run_training(config):
    """Run the mainfile with the given config file and save the results."""
    _override_cmd_arg(config)
    summary = run_student_teacher.run()
    return summary

def run(config_module=None, student_key=None, values=None, plot=True,
        return_all_results=False):
    """Run the student-teacher setup with multiple students.

    By default, this function simply runs the experiment multiple times with
    the same config, but a different random seeds. However, this function can
    also be used to vary one particular feature of the students, as provided
    by the inputs.

    Args:
        config_module: The config module to be used. If provided, it overwrites
            the one given by the command line arguments.
        student_key (str, optional): The command line feature to be changed in
            each run.
        values (list, optional): The values to be used for the given feature.
        plot (boolean, optional): Whether to plot some results.
        return_all_results (boolean, optional): Whether the detailed results
            should be returned. If ``False``, only ``summary_dict`` is
            returned.

    Returns:
        (....): Tuple containing:

        -**summary_dict**: The results summary to be used as hpsearch results.
        -**results**: The detailed results containing info on individual runs.
        -**args.out_dir**: The directory where results are stored.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='',
                        help='Directory where the results will be saved.')
    parser.add_argument('--filename', type=str, default='results',
                        help='Name of the file in which the results will be '
                             'saved.')
    parser.add_argument('--hpsearch', action='store_true',
                        help='Whether this is an ongoing hpsearch.')
    parser.add_argument('--config_module', type=str,
                        default='configs.simple_vanilla_rnn',
                        help='The name of the module containing the configs.')
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='Number of random seeds to run the hp config on.')
    args = parser.parse_args()

    # Correct the number of total runs.
    n_runs = args.n_seeds
    if student_key is not None:
        assert values is not None
        n_runs = len(values) * args.n_seeds
    else:
        values = [None]

    # Extract correct config module.
    if config_module is None:
        config_module = args.config_module

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

    config_module = importlib.import_module(config_module)
    random_seeds = np.random.randint(0, 10000, args.n_seeds)

    count = 1
    for val in values:
        for i, seed in enumerate(random_seeds):
            print('\nInitiating run {} ...'.format(count))
            config_module.config['random_seed'] = seed
            config_module.config['arch_random_seed'] = seed
            config_module.config['out_dir'] = os.path.join(args.out_dir, \
                'val%2f_seed%i'% (val, i) if val is not None else 'seed%i' % i)
            if args.hpsearch:
                config_module.config['hpsearch'] = True
            if student_key is not None:
                config_module.config[student_key] = val
            summary = run_training(config_module.config)

            if count == 1:
                # initialize result dataframe
                columns = summary.keys()
                index = [j for j in range(n_runs)]
                results = pd.DataFrame(index=index, columns=columns)

            for key in columns:
                results.at[count-1, key] = summary[key]

            results.to_csv(os.path.join(args.out_dir, filename))
            count += 1

    # Post_process.
    results = post_process(results, args.out_dir, args.filename, plot=plot)

    # Get all config values accessible.
    config = parse_cmd_arguments(experiment='student_teacher')
    for key in config_module.config.keys():
        setattr(config, key, config_module.config[key])

    ### Build some experiment summary for the hpsearches.
    exp_summary = dict()
    exp_summary['lr'] = config.lr
    exp_summary['n_iter'] = config.n_iter

    # Teacher statistics.
    exp_summary['teacher_n_ts_in'] = config.teacher_n_ts_in
    exp_summary['teacher_n_ts_out'] = config.teacher_n_ts_out
    exp_summary['teacher_input_size'] = config.teacher_input_size
    exp_summary['teacher_output_size'] = config.teacher_output_size
    for key in ['num_weights_teacher', 'num_neurons_teacher', \
                            'rnn_arch_teacher', 'rec_sparsity_teacher']:
        exp_summary[key] = results.loc['mean'][key]

    # Student statistics.
    exp_summary['num_students'] = args.n_seeds
    for key in ['feedforwardness', 'cycles_adjacency', \
                            'cycles_recursive', 'ratio_cycles_adjacency', \
                            'cycles_recursive_corr', 'cycles_adjacency_corr']:
        exp_summary[key + '_mean'] = results.loc['mean'][key]
        exp_summary[key + '_std'] = results.loc['std'][key]

    # # Correlation coefficients between performance and feedforwardness.
    corr_feedforwardness = np.corrcoef(\
                        results['loss_test_last'][:-2].tolist(),\
                        results['feedforwardness'][:-2].tolist())[0, 1]
    # corr_cycles_adjacency = np.corrcoef(\
    #                     results['loss_test_last'][:-2].tolist(),\
    #                     results['cycles_adjacency'][:-2].tolist())[0, 1]
    # corr_cycles_recursive = np.corrcoef(\
    #                     results['loss_test_last'][:-2].tolist(),\
    #                     results['cycles_recursive'][:-2].tolist())[0, 1]
    # corr_ratio_cycles = np.corrcoef(\
    #                     results['loss_test_last'][:-2].tolist(),\
    #                     results['ratio_cycles_adjacency'][:-2].tolist())[0, 1]
    # corr_cycles_adjacency_corr = np.corrcoef(\
    #                     results['loss_test_last'][:-2].tolist(),\
    #                     results['cycles_adjacency_corr'][:-2].tolist())[0, 1]
    # corr_cycles_recursive_corr = np.corrcoef(\
    #                     results['loss_test_last'][:-2].tolist(),\
    #                     results['cycles_recursive_corr'][:-2].tolist())[0, 1]

    # # Compute the maximal correlation.
    exp_summary['corr_feedforwardness'] = corr_feedforwardness
    # exp_summary['corr_cycles_adjacency'] = corr_cycles_adjacency
    # exp_summary['corr_cycles_recursive'] = corr_cycles_recursive
    # exp_summary['corr_ratio_cycles'] = corr_ratio_cycles
    # exp_summary['corr_cycles_adjacency_corr'] = corr_cycles_adjacency_corr
    # exp_summary['corr_cycles_recursive_corr'] = corr_cycles_recursive_corr
    # exp_summary['max_correlation'] = np.max([corr_feedforwardness, \
    #     corr_cycles_adjacency, corr_cycles_recursive, corr_ratio_cycles, \
    #     corr_cycles_adjacency_corr, corr_cycles_recursive_corr])

    if return_all_results:
        return exp_summary, results, args.out_dir
    else:
        return exp_summary

def post_process(results, out_dir, filename, plot=True):
    """Post-process the results.

    This function can be called either once all students have been trained,
    or post-hoc in case the original call failed and only some students were
    trained.

    Args:
        results (dict): The results dictionary.
    """
    # Save average results.
    if not 'distance_to_teacher' in results.columns.values:
        results['distance_to_teacher'] = \
            np.abs((results['feedforwardness'] - \
            results['feedforwardness_teacher']).tolist())
    if isinstance(results.index.values, np.ndarray) or \
            not 'mean' in results.index.values:
        results.loc['mean'] = results.mean(axis=0)
        results.loc['std'] = results.std(axis=0)

    results.to_csv(os.path.join(out_dir, filename + '.csv'))

    # Plot the results.
    if plot:
        plot_perf_vs_feedforwardness(results, out_dir, filename)
        plot_perf_vs_feedforwardness(results, out_dir, filename, test=False)

    return results


def plot_perf_vs_feedforwardness(results, out_dir, filename, test=True):
    """Get scatter plot of performance vs. feedforwardness level.

    Args:
        results (dict): The results dictionary.
        out_dir (str): The directory where to save the figures.
        filename (str): The filename.
        test (boolean, optional): Whether test or training accuracies should
            be plotted.
    """

    key = 'loss_test_last'
    mode = 'test'
    if not test:
        key = 'loss_train_last'
        mode = 'train'

    ### Directedness
    fig, ax = plt.subplots(2,3, figsize=(14, 8))
    ax = [ax[0][0], ax[0][1], ax[0][2], ax[1][0], ax[1][1], ax[1][2]]
    for i, metric in enumerate(['feedforwardness', 'cycles_adjacency', \
                'cycles_recursive', 'ratio_cycles_adjacency', \
                'cycles_recursive_corr', 'cycles_adjacency_corr']):

        ff_teacher = results[metric + '_teacher'][0]
        x_vals = results[metric].tolist()[:-2]
        y_vals = results[key].tolist()[:-2]
        y_vals /= np.max(y_vals)

        ax[i].scatter(x_vals, y_vals)
        ax[i].set_xlabel('student %s' % metric)
        ax[i].set_ylabel('%s loss (AU)'%mode)
        ax[i].plot([ff_teacher, ff_teacher], plt.gca().get_ylim(), '-r',
                 label='teacher')
        # Do KDE fit.
        # kde = KernelDensity(kernel='gaussian').fit(y_vals[:, np.newaxis])
        # log_dens = kde.score_samples(np.array(x_vals)[:, np.newaxis])
        # plt.plot(x_vals, np.exp(log_dens), '-b')
    ax[i].legend()
    plt.savefig(os.path.join(out_dir, filename + '_%s.png' % mode))

if __name__ == '__main__':
    run()
