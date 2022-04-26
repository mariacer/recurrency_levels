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
# @title          :real_world_benchmarks/pos_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Helper functions for training RNNs on the Part-of-Speech tagging dataset.
-------------------------------------------------------------------------

Set of helper functions that will be used when training an RNN on PoS.
"""
import numpy as np
import torch
import torch.nn as nn
import pickle

from data import mud_data
from utils.train_utils import sequential_nll

def get_dhandler(config):
    """Get a datahandler for PoS-tagging.

    Args:
        config: Command-line arguments.

    Returns:
        The datahandler.
    """
    path = '../datasets'
    if config.hpsearch:
        path = '../../../../datasets'
    dhandler = mud_data.get_mud_handlers(path, num_tasks=1)[0]

    return dhandler

def get_loss_func(config, device, logger):
    """Get a function handle that can be used as task loss function.
    Args:
        config (argparse.Namespace): The command line arguments.
        device: Torch device (cpu or gpu).
        logger: Console (and file) logger.

    Returns:
        (func): A function handler as described by argument ``custom_nll``.
    """
    ce_loss = sequential_nll(loss_type='ce', reduction='sum')

    sample_loss_func = lambda Y, T, tsf, beta: ce_loss(Y, T, None, None, None,
        ts_factors=tsf, beta=beta)

    # Unfortunately, we can't just use the above loss function, since we need
    # to respect the different sequence lengths.
    # We therefore create a custom time step weighting mask per sample in a
    # given batch.
    def task_loss_func(Y, T, data, allowed_outputs, empirical_fisher,
                       batch_ids):
        # Build batch specific timestep mask.
        tsf = torch.zeros(T.shape[0], T.shape[1]).to(T.device)

        seq_lengths = data.get_out_seq_lengths(batch_ids)

        for i in range(batch_ids.size):
            sl = int(seq_lengths[i])

            tsf[:sl, i] = 1

        return sample_loss_func(Y, T, tsf, None)

    return task_loss_func

def get_accuracy_func(config):
    """Get the accuracy function for an PoS-tagging task.

    Args:
        config (argparse.Namespace): The command line arguments.

    Returns:
        (func): An accuracy function handle.
    """
    def get_accuracy(logit_outputs, targets, data, batch_ids):
        """Get the accuracy.

        Args:
            (....) See docstring of function
                :func:`utils.audioset_utils.get_accuracy`.

        Returns:
            (float): The accuracy.
        """
        seq_lengths = data.get_out_seq_lengths(batch_ids)
        input_data = data._data['in_data'][batch_ids,:]

        predicted = logit_outputs.argmax(dim=2)
        targets = targets.argmax(dim=2)
        all_compared = predicted == targets

        num_correct = 0
        num_total = 0

        for i in range(batch_ids.size):
            # we exclude tokens for which there is no word embedding from the
            # accuracy computation. these tokens have dict_idx == 0
            comp_idx = np.arange(0, int(seq_lengths[i]))
            exclude_idx = np.where(input_data[i,:] == 0)
            comp_idx = np.setdiff1d(comp_idx, exclude_idx)

            num_correct += all_compared[comp_idx, i].sum().cpu().item()
            num_total += len(comp_idx)

        if num_total != 0:
            accuracy = 100. * num_correct / num_total
        else:
            # FIXME Can this case really appear?
            accuracy = 0

        return accuracy

    return get_accuracy


def generate_emb_lookups(config, filename=None, padding_idx=0, device=None):
    """Generate a list of models that contain embeddings for different tasks.

    Args:
        config: The configuration.
        filename (str, optional): If provided, the embeddings will be loaded
            from the provided location. Else, they will be initialized randomly.
        padding_idx (int, optional): The value of the indices which correspond
            to padded tokens.
        device: PyTorch device.

    Returns:
        (list): The embedding lookup model (:class:`WordEmbLookup`).
    """
    if filename is not None:
        # Load the embeddings.
        embeddings = pickle.load(open(filename, 'rb'), encoding='bytes')
    else:
        raise NotImplementedError # generate randomly

    # Choose the first embeddings only (single task!)
    lookup = WordEmbLookup(embeddings[0], padding_idx=padding_idx).to(device)

    return lookup

class WordEmbLookup(nn.Module):
    """A wrapper class for word embeddings.

    This class will instantiate and initialize a set of word embeddings. In
    addition, it will provide a :meth:`forward` method that can be used to
    translate a batch of vocabulary indices into word embeddings.
    
    Attributes:
        embeddings (nn.Embedding): The embeddings.
    """
    def __init__(self, initial_embeddings, padding_idx=0):
        nn.Module.__init__(self)

        vocab_size, embedding_dim = initial_embeddings.shape
        self._embeddings = nn.Embedding(vocab_size, embedding_dim,
            padding_idx=padding_idx)
        self._embeddings.weight.data = torch.tensor(initial_embeddings)

    @property
    def embeddings(self):
        """Getter for read-only attribute :attr:`embeddings`."""
        return self._embeddings

    def forward(self, x):
        """Translate vocabulary indices into word embeddings.

        Args:
            x (torch.Tensor): Batch of vocabulary indices.
                The tensor is of shape ``[T, B]`` or ``[T, B, 1]`` with ``T``
                denoting the number of timesteps and ``B`` denoting the batch
                size.

        Returns:
            (torch.Tensor): A batch of word embeddings. The output tensor is of
            shape ``[T, B, K]``, where ``K`` is the dimensionality of individual
            word embeddings.
        """
        assert len(x.shape) == 2 or len(x.shape) == 3 and x.shape[2] == 1

        embedded = self.embeddings(x)
        if len(embedded.shape) > 3:
            embedded = torch.squeeze(embedded, dim=2)

        return embedded
