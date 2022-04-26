#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans
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
# @title          :student_teacher/st_utils.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :08/03/2021
# @version        :1.0
# @python_version :3.7
"""
Utilities for Teacher-Student experiments
-----------------------------------------

All command-line arguments and default values are handled in this module.
"""
import numpy as np
import os
from time import time
import torch

from data.teacher_rnn import RndRecTeacher
from hpsearch.hp_utils import search_space_to_grid
from hpsearch.hpsearch import dict2config
from utils.train_utils import sequential_nll
from utils import misc

def get_dhandler(config):
    """Get a datahandler for the student-teacher regression.

    Args:
        config (argparse.Namespace): Command line args.

    Returns: 
        The data handler.
    """
    return RndRecTeacher(config)

def get_loss_func(config, device, logger):
    """Get a function handle that can be used as MSE task loss function.

    Args:
        config (argparse.Namespace): The command line arguments.
        device: Torch device (cpu or gpu).
        logger: Console (and file) logger.

    Returns:
        (func): A function handler as described by argument ``custom_nll``.

        Note:
            This loss **sums** the NLL across the batch dimension. A proper
            scaling wrt other loss terms during training would require a
            multiplication of the loss with a factor :math:`N/B`, where
            :math:`N` is the training set size and :math:`B` is the mini-batch
            size.
    """
    mse_loss = sequential_nll(loss_type='mse', reduction='sum',
                              return_per_ts=True)

    # Get the timestep factors, in case the MSE shouldn't be evaluated
    # homogeneously over the entire output sequence.
    ts_factors = None
    if config.compute_late_mse:
        # The MSE will be computed for late timesteps from timestep ``t``,
        # where ``t`` is the total number of recurrent neurons.
        n_ts_out = config.teacher_n_ts_out
        if n_ts_out == -1:
            n_ts_out = config.teacher_n_ts_in
        num_recurrent_neurons = np.sum(misc.str_to_ints(config.rnn_arch))
        ts_factors = torch.zeros((n_ts_out, 1), device=device)
        ts_factors[num_recurrent_neurons:] = 1

    # We need to ensure additionally that `batch_ids` can be passed to the loss,
    # even though we don't use them here as all sequences have the same length.
    # Note, `dh`, `ao`, `ef` are also unused by `mse_loss` and are just provided
    # to certify a common interface.
    loss_func = lambda Y, T, dh, ao, ef, _: mse_loss(Y, T, None, None, None,
        ts_factors=ts_factors)

    return loss_func


def get_configs_multiple_student_hpsearch(search_space, fixed_space):
    """Write the config files and their names for the given search space.

    Args:
        search_space (dict): The dictionary of the search space.
        fixed_space (dict): The dictionary of fixed values.

    Return:
        (list): The name of the config files.
    """

    all_dicts = search_space_to_grid(search_space, fixed_space)
    all_config_names = []
    for i, dct in enumerate(all_dicts):
        config_name = 'config_%i_%i' % (int(time() * 1000), i)
        dict2config(dct, os.path.join('student_teacher/configs', \
                    config_name + '.py'), name='')
        all_config_names.append('student_teacher.configs' + '.' + config_name)

    return all_config_names

