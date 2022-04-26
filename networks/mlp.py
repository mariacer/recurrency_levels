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
# @title          :networks/mlp.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :11/01/2021
# @version        :1.0
# python_version  :3.7
"""
Multilayer Perceptron
---------------------

Implementation of a multilayer perceptron.

"""
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.utils.prune

from utils import misc
from utils.torch_utils import init_params


class MLP(nn.Module):
    """Implementation of a simple MLP.

    This is a simple multilayer perceptron, that is designed to process
    input :math:`\mathbf{x}` and generate outputs :math:`\mathbf{y}`. Note that
    the inputs are sequences, similar to what a recurrent network would process.
    If there is a label per timestep, the timesteps are processed independently,
    otherwise all timesteps are concatenated and provided simultaneously to the
    network to generate a single output for the entire sequence. This is
    indicated by the argument ``output_per_ts``.

    Attributes:
        n_in (int): See constructor argument ``n_in``.
        n_out (int): See constructor argument ``n_out``.
        hidden_sizes (list or tuple): See constructor argument ``hidden_sizes``.
        activation: See constructor argument ``activation``.
        use_bias (bool): See constructor argument ``use_bias``.
        output_per_ts (bool): See constructor argument ``output_per_ts``.
        n_ts (bool): See constructor argument ``n_ts``.

    Args:
        n_in (int): Number of inputs.
        n_out (int): Number of outputs.
        arch (list or tuple): List of integers. Each entry denotes the size of a
            hidden layer. If empty, there is no hidden layer.
        activation (str): The name of the nonlinearity used in fully connected 
            layers, as well as recurrent layers for vanilla RNNs. In LSTMs, 
            tanh is always used.
        use_bias (bool): Whether layers may have bias terms.
        kaiming_init (bool): Whether the layers should be initialized using
            the kaiming initialization.
        output_per_ts (bool): If ``True``, it indicates that one output for
            each timestep should be generated (instead of one for the entire
            sequence).
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
        n_ts (None or int): If ``output_per_ts`` is ``False`` this value gives
            the number of timesteps in the provided sequences, such that the
            entire sequence can be processes simultaneously. Note that this is
            only valid for input sequences of identical length.
    """
    def __init__(self, n_in=1, n_out=1, hidden_sizes=(10), activation='relu',
                 use_bias=True, kaiming_init=True, output_per_ts=False,
                 verbose=True, n_ts=None):       
        super(MLP, self).__init__()

        self._n_in = n_in
        self._n_out = n_out
        self._hidden_sizes = hidden_sizes
        self._use_bias = use_bias
        self._activation = misc.str_to_act(activation)
        self._output_per_ts = output_per_ts
        self._n_ts = n_ts

        ### Initialize the hidden layers.
        self._layers = []
        pre_size = n_in
        if not self._output_per_ts:
            pre_size = n_in * n_ts 
        for l, l_size in enumerate(hidden_sizes):
            weights = nn.Linear(pre_size, l_size, bias=use_bias)
            if kaiming_init:
                init_params(weights)
            self._layers.append(weights)
            pre_size = l_size

        ### Initialize the output layer.
        weights = nn.Linear(pre_size, n_out, bias=use_bias)
        if kaiming_init:
            init_params(weights)
        self._layers.append(weights)

        self._layers = nn.Sequential(*self._layers)

        if verbose:
            num_neurons = n_in + np.sum(hidden_sizes).astype(int) + n_out
            num_weights = self.get_num_weights()
            print('Creating an MLP with %d weights.' % num_weights)

    def forward(self, x):
        """Compute the output :math:`y` of this network given the input 
        :math:`x`.

        Args:
            x: The inputs :math:`x` to the network. It has dimensions
                ``[n_ts, batch_size, n_in]``.

        Returns:
            (torch.Tensor): The output of the network.  It has dimensions
                ``[n_ts, batch_size, n_out]`` or ``[batch_size, n_out]`` if
                ``output_per_ts```is ``False``.
        """
        # If there is a single output per sequence, all timesteps are
        # concatenated and provided to the network simultaneously.
        n_ts = x.shape[0]
        if not self._output_per_ts:
            x = torch.reshape(x, (x.shape[1], x.shape[0]*x.shape[2]))

        h = x

        for l in range(len(self._layers)):
            h = self._layers[l](h)
            # Do not apply nonlinearity in the output layer.
            if l < len(self._layers)-1:
                h = self._activation(h)

        out = h
        if not self._output_per_ts:
            # Hacky, because the loss function expects the outputs sequences to
            # have the same length as the input sequence, we pad with zeros at
            # the beginning, knowing that these will be ignored in the loss
            # computation when only one timestep is used for classification.
            out = torch.zeros(n_ts, *h.shape, device=x.device)
            out[-1, :, :] = h

        return out

    def get_num_weights(self):
        """Compute the total number of weights of the network.

        Returns:
            (int): The number of weights.
        """
        num_weights = 0
        for params in self._layers:
            num_weights += params.weight.numel()
            if self._use_bias:
                num_weights += params.bias.numel()

        return num_weights

    @property
    def use_bias(self):
        """Getter for read-only attribute :attr:`use_bias`."""
        return self._use_bias

    def save_logs(self, writer, curr_iter):
        pass

    def save_weights(self, path):
        """Save the hidden-to-hidden weights of the MLP."""
        raise NotImplementedError

    def has_same_architecture(self, net, return_same_weight_sign=True):
        """NotImplemented"""
        if return_same_weight_sign:
            return False, False, False
        else:
            return False, False