#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
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
# @title          :utils/sim_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.7
"""
General helper functions for simulations
----------------------------------------

The module :mod:`utils.sim_utils` comprises a bunch of functions that are in
general useful for writing simulations in this repository.

The implementation is based on an implementation by Christian Henning in
another project:

    https://github.com/mariacer/cl_in_rnns
"""
import torch
import tensorboardX
from tensorboardX import SummaryWriter
import numpy as np
import random
import os
import select
import shutil
import sys
import pickle
import logging
from time import time
from warnings import warn
import json
import matplotlib.pyplot as plt

import data
from feedforwardness import utils as futils
from utils import misc
from utils.train import train, test
from utils.config_utils import _args_to_cmd_str

def run_experiment(net, config, shared, dhandler, device, logger, writer,
                   loss_func, accuracy_func):
    """Basic code to run an experiment.

    For the purposes of avoiding code duplication, we create this function that
    takes important objects for the experiment and actually runs it.

    Args:
        net: The network object.
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        dhandler: The data handler for the task. Needed to extract the number 
            of inputs/outputs of the main network.
        device: Torch device.
        logger: Console (and file) logger.
        writer: The tensorboard summary writer.
        loss_func (fct_handler): The function to compute task loss.
        accuracy_func (fct_handler): The function to compute the accuracy, only
            necessary for classification tasks.

    Returns:
        (dict): The results summary.
    """

    ### Test before training.
    loss, acc = test(dhandler, net, device, config, logger, writer, shared, 
          loss_func=loss_func, accuracy_func=accuracy_func, pre_training=True,
          plot_samples=config.plot_samples)
    msg = 'Pre training test loss {:.5f}'
    if acc is not None:
        msg += ', and accuracy {:.1f}%'
    logger.info(msg.format(loss, acc))

    ### Train the network.
    summary_dict = train(dhandler, net, device, config, logger, writer, shared,
          loss_func=loss_func, accuracy_func=accuracy_func)

    ### Test.
    loss, acc = test(dhandler, net, device, config, logger, writer, shared, 
          loss_func=loss_func, accuracy_func=accuracy_func,
          plot_samples=config.plot_samples)
    msg = 'Post training test loss {:.5f}'
    if acc is not None:
        msg += ', and accuracy {:.1f}%'
    logger.info(msg.format(loss, acc))

    summary_dict['loss_test_last'] = float(loss.item())
    summary_dict['loss_test_best'] = float(np.min(np.array(loss.cpu().numpy())))

    # Note that acc will be None for non classification tasks.
    if accuracy_func is not None:
        summary_dict['acc_test_last'] = float(acc)
        summary_dict['acc_test_best'] = float(np.max(np.array(acc)))

    if config.prune_fraction > 0:
        print('Pruning {}% of the hidden-to-hidden weights'.
              format(100 * config.prune_fraction))
        net.prune_hh_weights(config.prune_fraction)
        loss, acc = test(dhandler, net, device, config, logger, writer, shared,
                         loss_func=loss_func, accuracy_func=accuracy_func)
        msg = 'Post pruning test loss {:.5f}'
        if acc is not None:
            msg += ', and accuracy {:.1f}%'
        logger.info(msg.format(loss, acc))

        summary_dict['loss_test_pruned'] = float(loss.item())
        if accuracy_func is not None:
            summary_dict['acc_test_pruned'] = float(acc)

    def save_topology_info(net, is_teacher=False):
        """Save important topological information."""

        sfx = '' if not is_teacher else '_teacher'

        def get_config(attr):
            """Small helper function to extra the correct hyperparameter options
            depending on whether a teacher or not is being constructed."""
            prefix = ''
            if is_teacher:
                prefix = 'teacher_'
            return getattr(config, prefix + attr)

        # General architecture information.
        summary_dict['num_weights%s'%sfx] = net.get_num_weights()
        summary_dict['num_neurons%s'%sfx] = net.get_num_neurons()
        summary_dict['rnn_arch%s'%sfx] = \
                            misc.str_to_float_or_list(get_config('rnn_arch'))
        summary_dict['rec_sparsity%s'%sfx] = get_config('rec_sparsity')

        # Feedforwardness statistics.
        if get_config('use_vanilla_rnn'):
            metrics = futils.analyze_feedforwardness(net, config, logger, 
                                                     teacher=is_teacher)
            for key in metrics.keys():
                summary_dict[key + sfx] = metrics[key]

    def plot_cycles_per_length(summary_dict, teacher=False):
        """Plot the number of cycles per cycle length."""
        sfx = '' if not teacher else '_teacher'
        pfx = 'mainnet_' if not teacher else 'teacher_'

        fig, ax = plt.subplots()
        plt.plot(summary_dict['cycles_adjacency_list%s'%sfx])
        ax.set_xlabel('cycle length')
        ax.set_ylabel('number of cycles')
        writer.add_figure('nets/%scycles_adj_per_length'%pfx, fig, close=True)

        # Plot ratio of cycles per cycle length.
        fig, ax = plt.subplots()
        plt.plot(summary_dict['ratio_cycles_adjacency_list%s'%sfx])
        ax.set_xlabel('cycle length')
        ax.set_ylabel('ratio of cycles')
        writer.add_figure('nets/%sratio_cycles_adj_per_length'%pfx, fig, \
                          close=True)

    save_topology_info(net)
    if hasattr(data, 'teacher_rnn'):
        save_topology_info(dhandler.teacher_rnn, is_teacher=True)

    if config.save_logs:
        plot_cycles_per_length(summary_dict)  
        if hasattr(data, 'teacher_rnn'):
            plot_cycles_per_length(summary_dict, teacher=True)

    writer.close()
    logger.info('Program finished successfully.')

    # save summary dict
    file_path = os.path.join(config.out_dir, 'summary_dict.json')
    summary_dict = misc.dict_for_json(summary_dict)
    with open(file_path, 'w') as f:
        json.dump(summary_dict, f)

    if config.save_weights:
        net.save_weights(os.path.join(config.out_dir, 'weight_hh.csv'))

    # Make it .csv compatible.
    for key in summary_dict.keys():
        if isinstance(summary_dict[key], list):
            summary_dict[key] = str(summary_dict[key])

    return summary_dict


def setup_environment(config, logger_name='sim_logger', script_name=None):
    """Setup the general environment for training.

    This function should be called at the beginning of a simulation script
    (right after the command-line arguments have been parsed). The setup will
    incorporate:

        - creating the output folder
        - initializing logger
        - making computation deterministic (depending on config)
        - selecting the torch device
        - creating the Tensorboard writer

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger_name (str): Name of the logger to be created (time stamp will be
            appended to this name).
        script_name (str): Name of the script that is being executed.

    Returns:
        (tuple): Tuple containing:

        - **device**: Torch device to be used.
        - **writer**: Tensorboard writer. Note, you still have to close the
          writer manually!
        - **logger**: Console (and file) logger.
    """
    ### Output folder.
    if os.path.exists(config.out_dir):
        # FIXME We do not want to use python its `input` function, as it blocks
        # the program completely. Therefore, we use `select`, but this might
        # not work on all platforms!
        #response = input('The output folder %s already exists. ' % \
        #                 (config.out_dir) + \
        #                 'Do you want us to delete it? [y/n]')
        print('The output folder %s already exists. ' % (config.out_dir) + \
              'Do you want us to delete it? [y/n]')
        inps, _, _ = select.select([sys.stdin], [], [], 30)
        if len(inps) == 0:
            warn('Timeout occurred. No user input received!')
            response = 'n'
        else:
            response = sys.stdin.readline().strip()
        if response != 'y':
            raise IOError('Could not delete output folder!')
        shutil.rmtree(config.out_dir)

        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    else:
        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    # A JSON file is easier to read for a human.
    with open(os.path.join(config.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    # Get the command line and store in output directory.
    if script_name is not None:
        cmd_str = _args_to_cmd_str(vars(config), script_name)
        with open(os.path.join(config.out_dir, 'command_line.sh'), 'w') as f:
            f.write('#!/bin/sh\n')
            f.write(cmd_str)

    ### Initialize logger.
    logger_name = '%s_%d' % (logger_name, int(time() * 1000))
    logger = config_logger(logger_name,
        os.path.join(config.out_dir, 'logfile.txt'),
        logging.DEBUG, logging.DEBUG)
    # FIXME If we don't disable this, then the multiprocessing from the data
    # loader causes all messages to be logged twice. I could not find the cause
    # of this problem, but this simple switch fixes it.
    logger.propagate = False

    ### Deterministic computation.
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Ensure that runs are reproducible. Note, this slows down training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.deterministic_run:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ### Select torch device.
    assert(hasattr(config, 'no_cuda') or hasattr(config, 'use_cuda'))
    assert(not hasattr(config, 'no_cuda') or not hasattr(config, 'use_cuda'))

    if config.no_cuda:
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info('Using cuda: ' + str(use_cuda))

    ### Initialize summary writer.
    # Flushes every 120 secs by default.
    # DELETEME Ensure downwards compatibility.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(config.out_dir, 'summary'))

    return device, writer, logger


def config_logger(name, log_file, file_level, console_level):
    """Configure the logger that should be used by all modules in this
    package.
    This method sets up a logger, such that all messages are written to console
    and to an extra logging file. Both outputs will be the same, except that
    a message logged to file contains the module name, where the message comes
    from.

    The implementation is based on an earlier implementation of a function I
    used in another project:

        https://git.io/fNDZJ

    Args:
        name: The name of the created logger.
        log_file: Path of the log file. If None, no logfile will be generated.
            If the logfile already exists, it will be overwritten.
        file_level: Log level for logging to log file.
        console_level: Log level for logging to console.

    Returns:
        The configured logger.
    """
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                       + ' - %(module)s - %(message)s', \
                                       datefmt='%m/%d/%Y %I:%M:%S %p')
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                         + ' - %(message)s', \
                                         datefmt='%m/%d/%Y %I:%M:%S %p')

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir != '' and not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if os.path.exists(log_file):
            os.remove(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_file is not None:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
