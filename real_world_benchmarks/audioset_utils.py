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
# @title          :real_world_benchmarks/audioset_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Helper functions for training RNNs on Audioset
----------------------------------------------

Set of helper functions that will be used when training an RNN on Audioset.
"""
import torch

from data import audioset_data
from data.split_audioset import get_split_audioset_handlers
from utils.train_utils import sequential_nll

def get_dhandler(config, num_classes=100):
    """Get a datahandler for Audioset.

    Note that we use the same Split Audioset wrapper as in the CL in RNNs
    paper, but just so that we are able to directly compare the results,
    even if we only allow for a setting with a single task.

    Args:
        config: Command-line arguments.
        num_classes (int): The number of classes to consider.

    Returns:
        (AudiosetData) The datahandler.
    """
    path = '../datasets'
    if config.hpsearch:
        path = '../../../../datasets'

    dhandler = get_split_audioset_handlers(path,
        use_one_hot=True, num_tasks=1, num_classes_per_task=num_classes,
        rseed=config.data_random_seed, validation_size=config.val_set_size)[0]

    return dhandler

def get_loss_func(config, device, logger):
    """Get a function handle that can be used as task loss function.

    For Audioset, the loss is only computed on the last timestep.

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
    ce_loss = sequential_nll(loss_type='ce', reduction='sum')

    # Build batch specific timestep mask.
    # Note, all samples have the same sequence length.
    seq_length = 10
    ts_factors = torch.zeros(seq_length, 1).to(device)
    ts_factors[-1, :] = 1

    # We need to ensure additionally that `batch_ids` can be passed to the loss,
    # even though we don't use them here as all sequences have the same length.
    # Note, `dh`, `ao`, `ef` are also unused by `ce_loss` and are just provided 
    # to certify a common interface.
    loss_func = lambda Y, T, dh, ao, ef, _: ce_loss(Y, T, None, None, None,
        ts_factors=ts_factors, beta=None)

    return loss_func

def get_accuracy_func(config):
    """Get the accuracy function for an Audioset task.

    Note:
        The accuracy will be computed depending **only on the prediction in
        the last timestep**, where the last timestep refers to the **unpadded
        sequence**.

    Args:
        config (argparse.Namespace): The command line arguments.

    Returns:
        (func): An accuracy function handle.
    """
    def get_accuracy(logit_outputs, targets, data, batch_ids):
        """Get the accuracy for an Audioset task.

        Note that here we expect that, in the multi-head scenario, the correct 
        output head has already been selected, and ``logit_outputs`` corresponds 
        to the outputs in the correct head.

        Args:
            (....) See docstring of function
                :func:`sequential.copy.train_utils_copy.get_accuracy`.

        Returns:
            (float): The accuracy.
        """
        seq_length = targets.shape[0]
        batch_size = targets.shape[1]

        # Pick the last prediction per sample.
        logit_outputs = logit_outputs[seq_length-1, :, :]
        # Get the predicted classes.
        # Note, we don't need to apply the softmax, since it doesn't change the
        # argmax.
        predicted = logit_outputs.argmax(dim=1)
        # Targets are the same for all timesteps.
        targets = targets[0, :, :]
        targets = targets.argmax(dim=1)

        accuracy = 100. * (predicted == targets).sum().cpu().item() / \
            batch_size

        return accuracy

    return get_accuracy
