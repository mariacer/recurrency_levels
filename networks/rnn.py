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
# @title          :networks/rnn.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Recurrent Neural Network
------------------------

Implementation of a recurrent neural network with custom connectivity.

"""
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.utils.prune
from warnings import warn

from utils import misc
from utils.torch_utils import init_params


class RNN(nn.Module):
    """Implementation of a simple RNN.

    This is a simple recurrent network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    Custom connectivity can be provided for the input to first hidden layer,
    for all recurrent layers as well as for the last recurrent to output layer
    via the provided masks. For this, we assume there are no pre recurrent
    fc layers.

    Attributes:
        n_in (int): See constructor argument ``n_in``.
        activation: See constructor argument ``activation``.
        use_lstm (bool): See constructor argument ``use_lstm``.
        use_bias (bool): See constructor argument ``use_bias``.
        fc_layers_pre (list): The pre-recurrent fc layers.
        rec_layers (list): The recurrent layers.
        fc_layers (list): The post-recurrent fc layers.
        masks_fc_layers_pre (list): Connectivity masks of the pre-recurrent fc
            layers.
        masks_fc_layers (torch.Tensor): Connectivity masks of the post-recurrent
            fc layers.
        masks_rec (list): The list of masks defining the 
            connectivity between successive recurrent layers.
        masks_rec_hh (list): The list of masks defining the 
            hidden-to-hidden recurrent connectivity.
        full_connectivity (bool): Whether all the layers are fully connected.

    Args:
        n_in (int): Number of inputs.
        rec_layers_size (list or tuple): List of integers. Each entry denotes 
            the size of a recurrent layer. Recurrent layers will simply be 
            stacked as layers of this network.

            If ``fc_layers_pre`` is empty, then the recurrent layers are the
            initial layers.
            If ``fc_layers`` is empty, then the last entry of this list will
            denote the output size.

            Note:
                This list may never be empty.
        fc_layers_pre_size (list or tuple): List of integers. Before the 
            recurrent layers a set of fully-connected layers may be added. 

            If ``fc_layers_pre`` is not empty, its first entry will denote the
            input size of this network.
        fc_layers_size (list or tuple): List of integers. After the recurrent 
            layers, a set of fully-connected layers is added. The entries of 
            this list will denote the sizes of those layers.

            If ``fc_layers`` is not empty, its last entry will denote the output
            size of this network.
        masks_fc_layers_pre (list or None, optional): Connectivity masks of the
            pre-recurrent fc layers. Has to be of length equal to
            ``fc_layers_pre_size``.
        masks_fc_layers (torch.Tensor, optional):  Connectivity masks of the
            post-recurrent fc layers. Has to be of length equal to
            ``fc_layers_pre``.
        masks_rec (list or None, optional): The list of masks defining the 
            connectivity between successive recurrent layers. It has to be of
            same size as ``rec_layers_size``.
        masks_rec_hh (list or None, optional): The list of masks defining the 
            hidden-to-hidden recurrent connectivity. It has to be of
            same size as ``rec_layers_size``, or if it's a vanilla RNN with
            ``fc_rec_output``, then it has to be twice the size, in which case
            it will correspond to alternating hidden-to-hidden masks and
            hidden-to-recurrent output masks.
        fc_rec_output (bool, optional): If ``True``, the RNN layers will consist 
            of a recurrent layer together with a fully-connected output, as done
            in Elman networks. Else, only a recurrent layer will be added. Note
            that this is only relevant when using vanilla RNNs.
        activation (str): The name of the nonlinearity used in fully connected 
            layers, as well as recurrent layers for vanilla RNNs. In LSTMs, 
            tanh is always used.
        use_lstm (bool): If set to `True``, the recurrent layers will be LSTM
            layers.
        use_bias (bool): Whether layers may have bias terms.
        kaiming_init (bool): Whether the layers should be initialized using
            the kaiming initialization.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
    """
    def __init__(self, n_in=1, rec_layers_size=(10,), fc_layers_pre_size=(),
                 fc_layers_size=(1,), activation='relu', use_lstm=False, 
                 use_bias=True, verbose=True, kaiming_init=True,
                 fc_rec_output=True, masks_fc_layers_pre=None,
                 masks_fc_layers=None, masks_rec_hh=None, masks_rec=None):       
        super(RNN, self).__init__()

        ### Sanity checks.
        full_connectivity = True
        non_fully_connected_layers = 0
        if masks_fc_layers_pre is not None:
            assert len(masks_fc_layers_pre) == len(fc_layers_pre_size)
            non_fully_connected_layers += np.sum([m is not None for m in \
                masks_fc_layers_pre])
        if masks_rec_hh is not None:
            if not use_lstm and fc_rec_output:
                assert len(masks_rec_hh) == 2*len(rec_layers_size)
            else:
                assert len(masks_rec_hh) == len(rec_layers_size)
            non_fully_connected_layers += np.sum([m is not None for m in \
                rec_layers_size])
        if masks_rec is not None:
            assert len(masks_rec) == len(rec_layers_size)
            non_fully_connected_layers += np.sum([m is not None for m in \
                rec_layers_size])
        if masks_fc_layers is not None:
            assert len(masks_fc_layers) == len(fc_layers_size)
            non_fully_connected_layers += np.sum([m is not None for m in \
                masks_fc_layers])
        if non_fully_connected_layers != 0:
            full_connectivity = False

        ### Define basic attributes.
        self._n_in = n_in
        self._fc_layers_pre_size = fc_layers_pre_size
        self._rec_layers_size = rec_layers_size
        self._fc_layers_size = fc_layers_size
        self._use_lstm = use_lstm
        self._use_bias = use_bias
        self._kaiming_init = kaiming_init
        self._activation_name = activation
        self._activation = misc.str_to_act(activation)
        self._full_connectivity = full_connectivity
        if not use_lstm:
            self._fc_rec_output = fc_rec_output

        self.initialize_layers(masks_fc_layers_pre, masks_fc_layers, masks_rec, 
                               masks_rec_hh, verbose=verbose)

    def initialize_layers(self, masks_fc_layers_pre, masks_fc_layers, masks_rec, 
                          masks_rec_hh, verbose=False):
        """Initialize the layers.

        Args:
            masks_fc_layers_pre: List of pre-recurrent fc layer masks.
            masks_fc_layers: List of post-recurrent fc layer masks.
            masks_rec: List of recurrent input-to-hidden layer masks.
            masks_rec_hh: List of recurrent hidden-to-hidden layer masks.
        """
        #############################
        ### Initialize the layers ###
        #############################

        ### Initialize the fully connected layers before the recurrent layers.
        _fc_layers_pre = []
        pre_size = self._n_in
        for l, l_size in enumerate(self._fc_layers_pre_size):
            weights = nn.Linear(pre_size, l_size, bias=self._use_bias)
            if self._kaiming_init:
                init_params(weights)
            _fc_layers_pre.append(weights)
            pre_size = l_size
        self._fc_layers_pre = nn.Sequential(*_fc_layers_pre)

        ### Initialize the recurrent layers.
        _rec_layers = []
        for l, l_size in enumerate(self._rec_layers_size):
            if self._use_lstm:
                weights = nn.LSTM(pre_size, l_size, bias=self._use_bias)
            else:
                weights = nn.RNN(pre_size, l_size, bias=self._use_bias,
                        nonlinearity=self._activation_name)
            # Native init is uniformly between [-sqrt(k), +sqrt(k)] where
            # k is 1/l_size.
            if self._kaiming_init:
                init_params(weights)
            _rec_layers.append(weights)
            pre_size = l_size

            # If vanilla recurrent layers include a fully connected output,
            # add it.
            if not self._use_lstm and self._fc_rec_output:
                weights = nn.Linear(pre_size, l_size, bias=self._use_bias)
                if self._kaiming_init:
                    init_params(weights)
                _rec_layers.append(weights)
        self._rec_layers = nn.Sequential(*_rec_layers)

        ### Initialize the fully connected layers after the recurrent layers.
        _fc_layers = []
        for l, l_size in enumerate(self._fc_layers_size):
            weights = nn.Linear(pre_size, l_size, bias=self._use_bias)
            if self._kaiming_init:
                init_params(weights)
            _fc_layers.append(weights)
            pre_size = l_size
        self._fc_layers = nn.Sequential(*_fc_layers)

        #####################################
        ### Define the connectivity masks ###
        #####################################

        masks_fc_layers_pre = self.get_masks(masks_fc_layers_pre,   
            self._fc_layers_pre)
        masks_fc_layers = self.get_masks(masks_fc_layers, 
            self._fc_layers)
        masks_rec = self.get_masks(masks_rec, self._rec_layers,
            conn='weight_ih_l0')
        masks_rec_hh = self.get_masks(masks_rec_hh, self._rec_layers,
            conn='weight_hh_l0')

        # if use_lstm:
        #     # LSTMs have 4 sets of input weights and recurrent weights,
        #     # concatenated along the row dimension. Hence, we must copy
        #     # the input and recurrent mask 4 times along the row dimension
        #     # before applying. However, if the mask was None initially, it will
        #     # now already have the correct shape (because it is set to ones of
        #     # the same dimension as the LSTM weight matrices).
        #     if masks_fc_layers_pre.shape != self._rec_layers[0].weight_ih_l0.shape:
        #         masks_fc_layers_pre = masks_fc_layers_pre.repeat(4, 1)
        #     for i, mask in enumerate(masks_rec_hh):
        #         if mask.shape != self._rec_layers[i].weight_hh_l0.shape:
        #             masks_rec_hh[i] = mask.repeat(4, 1)

        self.masks_fc_layers_pre = masks_fc_layers_pre
        self.masks_rec = masks_rec
        self.masks_rec_hh = masks_rec_hh
        self.masks_fc_layers = masks_fc_layers

        if not self._full_connectivity:
            ### Apply the masks to initialized layers.
            self.apply_mask(masks_fc_layers_pre, self._fc_layers_pre)
            self.apply_mask(masks_rec, self._rec_layers, conn='weight_ih_l0')
            self.apply_mask(masks_rec_hh, self._rec_layers, conn='weight_hh_l0')
            self.apply_mask(masks_fc_layers, self._fc_layers)

            ### Register the backward hooks.
            self.register_hook(masks_fc_layers_pre, self._fc_layers_pre)
            self.register_hook(masks_rec, self._rec_layers, conn='weight_ih_l0')
            self.register_hook(masks_rec_hh, self._rec_layers,
                conn='weight_hh_l0')
            self.register_hook(masks_fc_layers, self._fc_layers)

        if verbose:
            rnn_type = 'LSTM' if self._use_lstm else 'vanilla RNN'
            if not self._use_lstm and self._fc_rec_output:
                rec_neurons = np.sum(self._rec_layers_size)*2
            else:
                rec_neurons = np.sum(self._rec_layers_size)
            num_neurons = np.sum(np.concatenate((self._fc_layers_pre_size, \
                self._fc_layers_size))).astype(int) + self._n_in + rec_neurons
            num_weights = self.get_num_weights()
            msg = '.'
            if self._full_connectivity:
                msg = ' and full connectivity.'
            print('Creating a %s network with ' % rnn_type + \
                '%d weights' % num_weights + msg)

    def forward(self, x):
        """Compute the output :math:`y` of this network given the input 
        :math:`x`.

        Args:
            x: The inputs :math:`x` to the network. It has dimensions
                ``[number_timesteps, batch_size, input_size]``.

        Returns:
            (torch.Tensor): The output of the network.
        """
        h = x

        ### Initial fully-connected layer activities.
        for layer in self._fc_layers_pre:
            h = layer(h)
            h = self._activation(h)

        ### Recurrent layer activities.
        for layer in self._rec_layers:
            if self.is_linear(layer):
                h = layer(h)
            else:
                h, _ = layer(h)

        ### Fully-connected layer activities (last is linear).
        for layer in self._fc_layers[:-1]:
            h = layer(h)
            h = self._activation(h)
        h = self._fc_layers[-1](h)

        return h

    def get_masks(self, masks, layers, conn='weight'):
        """Get masks from user input.

        Default masks are full masks (all ones). The user might have not
        provided any mask for a given set of layers, in which case we
        generate a list of all-one matrices. However, he might also have
        provided a list where some elements are ``None``, in which case we
        replace these ``None`` elements by all-one matrices. Else, we just
        verify that the dimensionality is suitable.

        When using vanilla RNNs with a fully-connected output, the list of
        recurrent layers will consist of RNN and Linear layers interleaved.
        When requesting a connection type of ``weight``, only the masks
        corresponding to the Linear layers will be returned. When requesting
        ``weight_hh_l0`` only the weights corresponding to the RNN layers
        will be returned.

        Args:
            masks (None or list): The user provided masks.
            layers (list): The corresponding layers.
            conn (str, optional): The type of weight to look at in the
                layer. Might be ``weight`` for fully-connected layers,
                ``weight_ih_l0`` for input to recurrent layers or
                ``weight_hh_l0`` for hidden-to-hidden weights.

        Returns:
            (list): The processed list of masks.
        """
        conn_list = self.get_connectivity_type(layers, conn)

        if masks is None:
            # Generate all-one matrices in all layers.
            masks = [torch.ones_like(getattr(layers[l], conn_list[l])) for l in 
                range(len(layers)) if conn_list[l] is not None]
        else:
            # Check dimensions of provided masks and generate all-ones for
            # the ones that are missing.
            l = 0
            for i, layer in enumerate(layers):
                if conn_list[i] is not None:
                    if masks[l] is None:
                        masks[l] = torch.ones_like(getattr(layers[i],
                                                   conn_list[i]))
                    else:
                        assert masks[l].shape == \
                            getattr(layers[i], conn_list[i]).shape
                    l += 1

        return masks

    def get_connectivity_type(self, layers, conn):
        """Get the type of connectivity weights for a list of layers.

        Given a list of layers, and a desired type of connectivity to study,
        this function returns a list of length equal to layers, where each
        element is a string with the name of the weights to extract from the
        layer. For example, if the layers are all linear, the desired
        connectivity type will be `weight` and thus a list with all `weight`
        elements is returned. This function is relevant to handle the case
        where the list of recurrent layers might include fully-connected layers
        (in the case of a vanilla RNN with fully-connected outputs). In this
        case, if the required connectivity type is `weight_ih_l0`, then the
        elements of the returned list will be either `weight_ih_l0` or `None`,
        if the layer is fully-connected. Else, if the required connectivity
        type is `weight_hh_l0`, the elements of the list will be interleaved
        `weight_hh_l0` and `weight`, since the fully-connected outputs of the
        recurrent layer are by design considered to belong to the
        hidden-to-hidden weights.

        Args:
            layers (list): A list of layers.
            conn (str): The desired type of connection to look at.

        Returns:
            (list): The list of connectivity types to extract.
        """

        # Build a list indicating which connection type to check for each mask.
        # If None, the corresponding layer will be ignored.
        conn_list = [None] * len(layers)
        for l, layer in enumerate(layers):
            is_fc_layer = self.is_linear(layer)
            if not self._use_lstm and self._fc_rec_output:
                # For vanilla RNNs with fc outputs, the recurrent layers also
                # contain fully connected layers, and their masks is considered
                # as hh type of mask.
                if is_fc_layer and conn == 'weight_hh_l0':
                    conn_list[l] = 'weight'
                elif is_fc_layer and conn == 'weight_ih_l0':
                    conn_list[l] = None
                else:
                    conn_list[l] = conn
            else:
                conn_list[l] = conn

            assert conn_list[l] is None or hasattr(layer, conn_list[l])

        return conn_list

    def apply_mask(self, masks, layers, conn='weight'):
        """Apply the masks to the layers.

        This function zeroes out weights according to the provided weights.

        Args:
            (....): See docstring of method :meth:`get_masks`.
        """
        conn_list = self.get_connectivity_type(layers, conn)
        l = 0
        for i, layer in enumerate(layers):
            if conn_list[i] is not None:
                setattr(layer, conn_list[i], nn.Parameter(getattr(layer, \
                    conn_list[i]) * masks[l]))
                l += 1

    def register_hook(self, masks, layers, conn='weight', device='cpu'):
        """Register the backward hooks according to the masks.
,
        This function hooks the gradients according to the masks such that
        the masked weights stay zero despite learning.

        Args:
            (....): See docstring of method :meth:`get_masks`.
        """
        conn_list = self.get_connectivity_type(layers, conn)
        l = 0
        for i, layer in enumerate(layers):
            if conn_list[i] is not None:
                maskl = masks[l].to(device)

                # Hook the gradient.
                if conn_list[i] == 'weight':
                    layer.weight.register_hook(
                        lambda x, mask=maskl: self.backward_hook(x, mask))
                if conn_list[i] == 'weight_hh_l0':
                    layer.weight_hh_l0.register_hook(
                        lambda x, mask=maskl: self.backward_hook(x, mask))
                elif conn_list[i] == 'weight_ih_l0':
                    layer.weight_ih_l0.register_hook(
                        lambda x, mask=maskl: self.backward_hook(x, mask))
                l += 1

    @staticmethod
    def backward_hook(grad, mask):
        """Hook to zero out new outgoing gradients with the mask.

        Args: 
            mask (torch.Tensor): The gradient masks.

        Returns:
            (torch.Tensor): The masked gradient.
        """
        out = grad.clone()
        mask = mask.to(grad.device)
        out *= mask
        return out

    def is_linear(self, layer):
        """Determine whether a given layer is linear.

        Args:
            layer: The layer.

        Returns:
            (bool): Whether the layer is linear or not.
        """
        return isinstance(layer, nn.Linear)

    def get_num_weights(self):
        """Compute the total number of weights of the network.

        Returns:
            (int): The number of weights.
        """
        num_weights = 0

        # Pre-recurrent fc weights.
        num_pre_weights = 0
        for l, params in enumerate(self._fc_layers_pre):
            layer_weights = params.weight.numel()
            if not self._full_connectivity:
                layer_weights = int(self.masks_fc_layers_pre[l].sum())
            num_pre_weights += layer_weights
            if self._use_bias:
                num_pre_weights += params.bias.numel()

        # Recurrent weights.
        num_rec_weights = 0
        num_nonlinear_rec = 0
        for l, params in enumerate(self._rec_layers):
            if hasattr(params, 'weight_hh_l0'):
                layer_weights = params.weight_hh_l0.numel()
                input_weights = params.weight_ih_l0.numel()
                if not self._full_connectivity:
                    layer_weights = int(self.masks_rec_hh[l].sum())
                    input_weights = int(self.masks_rec[num_nonlinear_rec].sum())
                num_rec_weights += layer_weights + input_weights      
                if self._use_bias:
                    num_rec_weights += params.bias_hh_l0.numel() + \
                        params.bias_ih_l0.numel()
                num_nonlinear_rec += 1
            else:
                # For the fc output of recurrent layers, if it exists.
                num_rec_weights += params.weight.numel()
                if not self._full_connectivity:
                    num_rec_weights += int(self.masks_rec_hh[l].sum())
                if self._use_bias:
                    num_rec_weights += params.bias.numel()

        # Post-recurrent fc weights.
        num_post_weights = 0
        for l, params in enumerate(self._fc_layers):
            layer_weights = params.weight.numel()
            if not self._full_connectivity:
                layer_weights = int(self.masks_fc_layers[l].sum())
            num_post_weights += layer_weights
            if self._use_bias:
                num_post_weights += params.bias.numel()

        return num_pre_weights + num_rec_weights + num_post_weights

    def get_num_neurons(self):
        """Compute the total number of neurons of the network.

        Returns:
            (int): The number of neurons.
        """
        # Pre-recurrent fc neurons.
        num_neurons = self._n_in + np.sum(self._fc_layers_pre_size)

        # Recurrent neurons.
        num_neurons += np.sum(self._rec_layers_size)

        # Post-recurrent fc neurons.
        num_neurons += np.sum(self._fc_layers_size)

        return num_neurons

    def init_hh_weights_orthogonal(self, n_iter=5000, lr=1e-3):
        """Initialize hidden-to-hidden weights orthogonally.

        This method will overwrite the hidden-to-hidden weights of recurrent
        layers. Whenever the connectivity in the microcircuit is sparse (i.e.
        not full) as described by the masks, PyTorch's native orthogonal
        initialization is not applicable. Instead, we iteratively learn the
        initial weights to make them orthogonal.

        Args:
            n_iter (int, optional): The number of optimizer iterations. Only
                relevant for non full connectivity.
            lr (float, optional): The learning rate. Only relevant for non full
                connectivity.
        """
        if self._full_connectivity:
            # For full connecitivity, the weights can be orthogonally
            # initialized using standard Pytorch's libraries.
            for i, layer in enumerate(self._rec_layers):
                if hasattr(layer, 'weight_hh_l0'):
                    print('Initializing hidden-to-hidden weights of recurrent '+
                          'layer %d orthogonally.' % i)
                    W = layer.weight_hh_l0

                    # LSTM weight matrices are stored such that the
                    # hidden-to-hidden matrices for the 4 gates are
                    # concatenated.
                    if self._use_lstm:
                        out_dim, _ = W.shape
                        assert out_dim % 4 == 0
                        fs = out_dim // 4

                        W1 = W[:fs, :]
                        W2 = W[fs:2*fs, :]
                        W3 = W[2*fs:3*fs, :]
                        W4 = W[3*fs:, :]

                        torch.nn.init.orthogonal_(W1.data)
                        torch.nn.init.orthogonal_(W2.data)
                        torch.nn.init.orthogonal_(W3.data)
                        torch.nn.init.orthogonal_(W4.data)

                        # Sanity check to show that the init on partial matrices
                        # propagates back to the original tensor.
                        assert W[0,0] == W1[0,0]
                    else:
                        torch.nn.init.orthogonal_(W.data)
        else:
            # For non-full connectivity, orthogonal optimization has to be
            # achieved via optimization.
            for i, layer in enumerate(self._rec_layers):
                if hasattr(layer, 'weight_hh_l0'):
                    W = layer.weight_hh_l0
                    device = W.device
                    print('Initializing hidden-to-hidden weights of recurrent '+
                          'layer %d orthogonally.' % i)

                    # LSTM weight matrices are stored such that the
                    # hidden-to-hidden matrices for the 4 gates are
                    # concatenated.
                    if self._use_lstm:
                        optimizer = torch.optim.SGD([W], lr=lr, momentum=0)
                        print('Orthogonally initializing the hidden-to-hidden '
                              'weights ... ')
                        for i in range(n_iter):
                            optimizer.zero_grad()
                            out_dim, _ = W.shape
                            assert out_dim % 4 == 0
                            fs = out_dim // 4

                            loss = 0
                            for k in range(4):
                                W_i = W[k*fs:(k+1)*fs, :]
                                loss += torch.norm(torch.matmul(W_i,
                                                W_i.transpose(0, 1)) - \
                                              torch.eye(W_i.shape[0],
                                                        device=device)) ** 2
                            loss.backward()
                            optimizer.step()
                            if i % 1000 == 0:
                                print("Iteration {}: loss {:.3f}".format(i, \
                                                                 loss.item()))
                        print('... Orthogonal initialization done')

                    else:
                        optimizer = torch.optim.SGD([W], lr=lr, momentum=0)
                        print('Orthogonally initializing the hidden-to-hidden '
                              'weights ... ')
                        for i in range(n_iter):
                            optimizer.zero_grad()
                            loss = torch.norm(torch.matmul(W, W.transpose(0,1)) - \
                                   torch.eye(W.shape[0], device=device))**2
                            loss.backward()
                            optimizer.step()
                            if i%1000==0:
                                print("Iteration {}: loss {:.3f}".format(i, \
                                    loss.item()))
                        print('... Orthogonal initialization done')

    def init_kaiming_sparse_layer(self, weights, mask, bias=None):
        """Apply a kaiming initialization for a sparse layer.

        This implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:

            https://git.io/fhnxV

        However, here I apply a it to a sparse layer, for which the fan-in is
        neuron-specific based on the existing masks. After initialization, the
        masks are re-applied.

        Args:
            weights (torch.Tensor): The weights.
            bias (torch.Tensor or None, optional): The biases.

        Returns:
            (tuple): Tuple containing...

            -**new_weights**: The new weights.
            -**new_bias**: The new biases. Can be None if no biases are used.
        """
        assert mask.shape == weights.shape
        assert torch.all(torch.eq(mask[mask==0], weights[weights==0]))
        post_size, pre_size = weights.shape
        new_weights = torch.zeros_like(weights)
        new_bias = None

        ### Initialize the weights.
        fan_in = (mask != 0).sum(axis=1).float()
        bound = torch.sqrt(2./fan_in) # neuron-specific
        for n in range(post_size):
            # Set uniform init between [-bound, bound]
            new_weights[n, :] = (torch.rand(pre_size) * 2 - 1) * bound[n]
        assert new_weights.shape == weights.shape

        ### Initialize the biases.
        if bias is not None:
            new_bias = torch.zeros_like(bias)
            assert new_bias.shape == bias.shape
            for n in range(post_size):
                # Set uniform init between [-bound, bound]
                new_bias[n] = (torch.rand(1) * 2 - 1) * bound[n]
            new_bias = nn.Parameter(new_bias)
            assert new_bias.shape == bias.shape

        ### Reapply the masks.
        new_weights *= mask
        assert torch.all(torch.eq(mask[mask==0], new_weights[new_weights==0]))

        # Transform into parameters.
        new_weights = nn.Parameter(new_weights)
        if bias is not None:
            new_bias = nn.Parameter(new_bias)

        return new_weights, new_bias

    def init_kaiming(self):
        """Initialize the layers with Kaiming initialization.

        If provided as an argument when constructing the class, this should
        be already the case for fully-connected networks. However, this function
        is useful to apply **after applying connectivity masks**.
        """
        print('Doing Kaiming initialization after masking.')
        for layer, mask in zip(self._fc_layers_pre, self.masks_fc_layers_pre):
            layer.weight, bias = self.init_kaiming_sparse_layer(layer.weight, 
                mask, bias=layer.bias if self._use_bias else None)
            layer.weight.grad = None
            if self._use_bias:
                layer.bias = bias
                layer.bias.grad = None
        for i, layer in enumerate(self._rec_layers):
            # If the hidden-to-hidden layers have fc outputs, the recurrent
            # hh masks will also contain masks for those layers but the ih masks
            # not, so we need to correct for that.
            ih_mask_idx_coeff = 1
            if not self._use_lstm and self._fc_rec_output:
                ih_mask_idx_coeff = 2
            if hasattr(layer, 'weight'):
                mask_fc = self.masks_rec_hh[i]
                layer.weight, bias = self.init_kaiming_sparse_layer(
                    layer.weight, mask_fc, 
                    bias=layer.bias if self._use_bias else None)
                layer.weight.grad = None
                if self._use_bias:
                    layer.bias = bias
                    layer.bias.grad = None
            else:
                mask_hh = self.masks_rec_hh[i]
                layer.weight_hh_l0, bias_hh = self.init_kaiming_sparse_layer(\
                    layer.weight_hh_l0, mask_hh, bias=layer.bias_hh_l0 if \
                    self._use_bias else None)
                layer.weight_hh_l0.grad = None
                layer.weight_ih_l0, bias_ih = self.init_kaiming_sparse_layer(\
                    layer.weight_ih_l0, self.masks_rec[int(i/ih_mask_idx_coeff)], 
                    bias=layer.bias_ih_l0 if self._use_bias else None)
                layer.weight_ih_l0.grad = None
                if self._use_bias:
                    layer.bias_hh_l0 = bias_hh
                    layer.bias_hh_l0.grad = None
                    layer.bias_ih_l0 = bias_ih
                    layer.bias_ih_l0.grad = None
                layer.flatten_parameters()
        for layer, mask in zip(self._fc_layers, self.masks_fc_layers):
            layer.weight, bias = self.init_kaiming_sparse_layer(layer.weight, 
                mask, bias=layer.bias if self._use_bias else None)
            layer.weight.grad = None
            if self._use_bias:
                layer.bias = bias
                layer.bias.grad = None

    @property
    def use_lstm(self):
        """Getter for read-only attribute :attr:`use_lstm`."""
        return self._use_lstm

    @property
    def use_bias(self):
        """Getter for read-only attribute :attr:`use_bias`."""
        return self._use_bias

    def save_weights(self, path):
        """Save the hidden-to-hidden weights of the RNN."""
        if len(self._rec_layers) > 1:
            warn('The save weights function only works when the rnn has only '
                 'one recurrent layer. Skipping.')
            pass
        weights_hh = self._rec_layers[0].weight_hh_l0
        pd.DataFrame(weights_hh.detach().cpu().numpy()).to_csv(
            path, header=False, index=False)

    def has_same_architecture(self, net, return_same_weight_sign=False):
        """Check if the network has the same architecture as another one.

        Args:
            net: The network to compare to.
            return_same_weight_sign (bool, optional): If activated, whether the
                two networks have the same weight signs will also be returned.

        Returns:
            (....): Tuple containing:

            - **same_architecture**: Whether they have the same architecture.
            - **same_weights**: Whether they have the same weights.
            - **same_weight_signs**: Whether the signs are the same. Only
                returned if ``return_same_weight_sign`` is ``True``.
        """
        ### Check general properties.
        same_architecture = True
        same_weights = True
        same_weight_sign = True
        assert type(net) == type(self)
        if not net.use_lstm == self.use_lstm:
            same_architecture = False
        if not net._use_bias == self._use_bias:
            same_architecture = False
        if not net._full_connectivity == self._full_connectivity:
            same_architecture = False
        if not net.use_lstm and not self.use_lstm:
            if not net._fc_rec_output == self._fc_rec_output:
                same_architecture = False

        ### Check layer dimensions.
        if not net._fc_layers_pre_size == self._fc_layers_pre_size:
            same_architecture = False
        if not net._fc_layers_size == self._fc_layers_size:
            same_architecture = False
        if not net._rec_layers_size == self._rec_layers_size:
            same_architecture = False

        ### Check masks.
        if same_architecture:
            for mask1, mask2 in zip(net.masks_fc_layers_pre, \
                    self.masks_fc_layers_pre):
                if not torch.all(torch.eq(mask1.cpu(), mask2.cpu())).item(): 
                    same_architecture = False
            for mask1, mask2 in zip(net.masks_rec, self.masks_rec):
                if not torch.all(torch.eq(mask1.cpu(), mask2.cpu())).item(): 
                    same_architecture = False
            for mask1, mask2 in zip(net.masks_rec_hh, self.masks_rec_hh):
                if not torch.all(torch.eq(mask1.cpu(), mask2.cpu())).item(): 
                    same_architecture = False
            for mask1, mask2 in zip(net.masks_fc_layers, self.masks_fc_layers):
                if not torch.all(torch.eq(mask1.cpu(), mask2.cpu())).item(): 
                    same_architecture = False

        if not same_architecture:
            if return_same_weight_sign:
                return False, False, False
            else:
                return False, False

        ### Check weights.
        for layer1, layer2 in zip(net._fc_layers_pre, self._fc_layers_pre):
            if not torch.all(torch.eq(layer1.weight.cpu(), \
                    layer2.weight.cpu())).item(): 
                same_weights = False
            if not torch.all(torch.eq(torch.sign(layer1.weight).cpu(), \
                    torch.sign(layer2.weight).cpu())).item(): 
                same_weight_sign = False
            if self._use_bias:
                if not torch.all(torch.eq(layer1.bias.cpu(), \
                        layer2.bias.cpu())).item(): 
                    same_weights = False
                if not torch.all(torch.eq(torch.sign(layer1.bias).cpu(), \
                        torch.sign(layer2.bias).cpu())).item(): 
                    same_weight_sign = False
        for layer1, layer2 in zip(net._rec_layers, self._rec_layers):
            if hasattr(layer1, 'weight'):
                if not torch.all(torch.eq(layer1.weight.cpu(), \
                        layer2.weight.cpu())).item(): 
                    same_weights = False
                if not torch.all(torch.eq(torch.sign(layer1.weight).cpu(), \
                        torch.sign(layer2.weight).cpu())).item(): 
                    same_weight_sign = False
                if self._use_bias:
                    if not torch.all(torch.eq(layer1.bias.cpu(), \
                            layer2.bias.cpu())).item(): 
                        same_weights = False
                if self._use_bias:
                    if not torch.all(torch.eq(torch.sign(layer1.bias).cpu(), \
                            torch.sign(layer2.bias).cpu())).item(): 
                        same_weight_sign = False
            else:
                if not torch.all(torch.eq(layer1.weight_hh_l0.cpu(), \
                        layer2.weight_hh_l0.cpu())).item(): 
                    same_weights = False
                if not torch.all(torch.eq(\
                        torch.sign(layer1.weight_hh_l0).cpu(), \
                        torch.sign(layer2.weight_hh_l0).cpu())).item(): 
                    same_weight_sign = False
                if not torch.all(torch.eq(layer1.weight_ih_l0.cpu(), \
                        layer2.weight_ih_l0.cpu())).item(): 
                    same_weights = False
                if not torch.all(torch.eq(\
                        torch.sign(layer1.weight_ih_l0).cpu(), \
                        torch.sign(layer2.weight_ih_l0).cpu())).item(): 
                    same_weight_sign = False
                if self._use_bias:
                    if not torch.all(torch.eq(layer1.bias_hh_l0.cpu(), \
                            layer2.bias_hh_l0.cpu())).item(): 
                        same_weights = False
                    if not torch.all(torch.eq(
                            torch.sign(layer1.bias_hh_l0).cpu(), \
                            torch.sign(layer2.bias_hh_l0).cpu())).item(): 
                        same_weight_sign = False
                    if not torch.all(torch.eq(layer1.bias_ih_l0.cpu(), \
                            layer2.bias_ih_l0.cpu())).item(): 
                        same_weights = False
                    if not torch.all(torch.eq(
                            torch.sign(layer1.bias_ih_l0).cpu(), \
                            torch.sign(layer2.bias_ih_l0).cpu())).item(): 
                        same_weight_sign = False
        for layer1, layer2 in zip(net._fc_layers, self._fc_layers):
            if not torch.all(torch.eq(layer1.weight.cpu(), \
                    layer2.weight.cpu())).item(): 
                same_weights = False
            if not torch.all(torch.eq(torch.sign(layer1.weight).cpu(), \
                    torch.sign(layer2.weight).cpu())).item(): 
                same_weight_sign = False
            if self._use_bias:
                if not torch.all(torch.eq(layer1.bias.cpu(), \
                        layer2.bias.cpu())).item(): 
                    same_weights = False
                if not torch.all(torch.eq(torch.sign(layer1.bias).cpu(), \
                        torch.sign(layer2.bias).cpu())).item(): 
                    same_weight_sign = False

        if return_same_weight_sign:
            return same_architecture, same_weights, same_weight_sign
        else:
            return same_architecture, same_weights

    def copy_connectivity(self, net, copy_weights=False,
                          copy_weights_sign=False, device='cpu'):
        """Copy the connectivity of an existing network.

        It is expected that the current network and the provided one have
        identical architectures and only differ in the specific topology
        (i.e. masks) and weight magnitudes.

        Args:
            net: The network object from which to copy the connectivity
            copy_weights (bool, optional): Whether the weight magnitudes should
                also be copied or initialized at random.
            copy_weights_sign (bool, optional): If activated, the student weight
                signs will match those of the teacher.
            device: The device where to put the copied net.
        """
        # Need to take care of weights, number of weights, masks hooks etc.
        ### Make the connectivity identical.
        has_same_architecture, has_same_weights = \
            self.has_same_architecture(net)

        if not has_same_architecture or (not has_same_weights and \
                (copy_weights or copy_weights_sign)):
            assert net.use_lstm == self.use_lstm
            assert net.use_bias == self.use_bias
            assert net._n_in == self._n_in

            ### Copy the topology.
            # Overwrite properties.
            self._full_connectivity = net._full_connectivity
            self._fc_rec_output = net._fc_rec_output
            self._fc_layers_pre_size = net._fc_layers_pre_size
            self._fc_layers_size = net._fc_layers_size
            self._rec_layers_size = net._rec_layers_size
            self.initialize_layers(net.masks_fc_layers_pre, net.masks_fc_layers,
                                   net.masks_rec, net.masks_rec_hh)

            ### Copy weights or weight sign.
            if copy_weights:
                self.copy_weights(net)
            elif copy_weights_sign:
                self.copy_weights_sign(net)

        has_same_architecture, has_same_weights = \
            self.has_same_architecture(net)
        assert has_same_architecture
        if copy_weights:
            assert has_same_weights
        else:
            assert not has_same_weights

        # Move to the right device.
        self.to(device)

        msg = ''
        if copy_weights:
            msg = ' and weights'
        elif copy_weights_sign:
            msg = ' and weight signs'
        print('Copied network architecture%s.' % msg)

    def copy_weights(self, net):
        """Copy the weights of an existing network.

        It is expected that the current network and the provided one have
        identical architectures and topology. The gradients will be set to zero.

        Args:
            net: The network object from which to copy the connectivity.
        """
        assert self.has_same_architecture(net)[0]
        for layer1, layer2 in zip(net._fc_layers_pre, self._fc_layers_pre):
            layer2.weight = layer1.weight
            layer2.weight.grad = None
            if self._use_bias:
                layer2.bias = layer1.bias
                layer2.bias.grad = None
        for layer1, layer2 in zip(net._rec_layers, self._rec_layers):
            if hasattr(layer1, 'weight'):
                layer2.weight = layer1.weight
                layer2.weight.grad = None
                if self._use_bias:
                    layer2.bias = layer1.bias
                    layer2.bias.grad = None
            else:
                layer2.weight_hh_l0 = layer1.weight_hh_l0
                layer2.weight_hh_l0.grad = None
                layer2.weight_ih_l0 = layer1.weight_ih_l0
                layer2.weight_ih_l0.grad = None
                if self._use_bias:
                    layer2.bias_hh_l0 = layer1.bias_hh_l0
                    layer2.bias_hh_l0.grad = None
                    layer2.bias_ih_l0 = layer1.bias_ih_l0
                    layer2.bias_ih_l0.grad = None
        for layer1, layer2 in zip(net._fc_layers, self._fc_layers):
            layer2.weight = layer1.weight
            layer2.weight.grad = None
            if self._use_bias:
                layer2.bias = layer1.bias
                layer2.bias.grad = None

    def copy_weights_sign(self, net):
        """Copy the weight signs of an existing network.

        This function leaves the weight magnitudes intact, but sets the signs
        to be the same as the corresponding weight in the teacher.
        It is expected that the current network and the provided one have
        identical architectures and topology. The gradients will be set to zero.

        Args:
            net: The network object from which to copy the connectivity.
        """
        assert self.has_same_architecture(net)[0]
        for layer1, layer2 in zip(net._fc_layers_pre, self._fc_layers_pre):
            layer2.weight = nn.Parameter(torch.abs(layer2.weight) * \
                                         torch.sign(layer1.weight))
            layer2.weight.grad = None
            if self._use_bias:
                layer2.bias = nn.Parameter(torch.abs(layer2.bias) * \
                                           torch.sign(layer1.bias))
                layer2.bias.grad = None
        for layer1, layer2 in zip(net._rec_layers, self._rec_layers):
            if hasattr(layer1, 'weight'):
                layer2.weight = nn.Parameter(torch.abs(layer2.weight) * \
                                             torch.sign(layer1.weight))
                layer2.weight.grad = None
                if self._use_bias:
                    layer2.bias = nn.Parameter(torch.abs(layer2.bias) * \
                                               torch.sign(layer1.bias))
                    layer2.bias.grad = None
            else:
                layer2.weight_hh_l0 = nn.Parameter(\
                                            torch.abs(layer2.weight_hh_l0) * \
                                            torch.sign(layer1.weight_hh_l0))
                layer2.weight_hh_l0.grad = None
                layer2.weight_ih_l0 = nn.Parameter(\
                                            torch.abs(layer2.weight_ih_l0) * \
                                            torch.sign(layer1.weight_ih_l0))
                layer2.weight_ih_l0.grad = None
                if self._use_bias:
                    layer2.bias_hh_l0 = nn.Parameter(\
                                            torch.abs(layer2.bias_hh_l0) * \
                                            torch.sign(layer1.bias_hh_l0))
                    layer2.bias_hh_l0.grad = None
                    layer2.bias_ih_l0 = nn.Parameter(\
                                            torch.abs(layer2.bias_ih_l0) * \
                                            torch.sign(layer1.bias_ih_l0))
                    layer2.bias_ih_l0.grad = None
        for layer1, layer2 in zip(net._fc_layers, self._fc_layers):
            layer2.weight = nn.Parameter(torch.abs(layer2.weight) * \
                                         torch.sign(layer1.weight))
            layer2.weight.grad = None
            if self._use_bias:
                layer2.bias = nn.Parameter(torch.abs(layer2.bias) * \
                                           torch.sign(layer1.bias))
                layer2.bias.grad = None

    def prune_hh_weights(self, p=0.85):
        """Prune the hidden-to-hidden weights.

        Pruning is performed by simply removing the weights with the smallest 
        absolute value according to the provided desired pruning level.

        Note that the native PyTorch function ``l1_unstructured```can only be
        applied to fully connected layers, and not to masked layers with zero
        weights.

        Args:
            p (float, optional): The fraction of weights to prune.
        """
        for layer in self._rec_layers:
            if hasattr(layer, 'weight_hh_l0'):
                nn.utils.prune.l1_unstructured(layer, 'weight_hh_l0', p)
                W = layer.weight_hh_l0
                num_pruned_weights = int(np.floor(W.numel() * p))
                assert (W == 0).sum() >= num_pruned_weights

    def to(self, *args, **kwargs):
        """Override the to() method.

        With this function the masks are also put on the correct device, as
        well as the backward hooks. See PyTorch documentation for correct
        documentation.
        """
        self = super().to(*args, **kwargs)
        out = torch._C._nn._parse_to(*args, **kwargs)
        device = out[0]

        # Put the masks in the right device.
        for l, mask in enumerate(self.masks_fc_layers_pre):
            self.masks_fc_layers_pre[l] = mask.to(device)
        for l, mask in enumerate(self.masks_rec):
            self.masks_rec[l] = mask.to(device)
        for l, mask in enumerate(self.masks_rec_hh):
            self.masks_rec_hh[l] = mask.to(device)
        for l, mask in enumerate(self.masks_fc_layers):
            self.masks_fc_layers[l] = mask.to(device)

        # Re-register the hooks.
        self.register_hook(self.masks_fc_layers_pre, self._fc_layers_pre,
            device=device)
        self.register_hook(self.masks_rec, self._rec_layers,
            conn='weight_ih_l0', device=device)
        self.register_hook(self.masks_rec_hh, self._rec_layers,
            conn='weight_hh_l0', device=device)
        self.register_hook(self.masks_fc_layers, self._fc_layers, device=device)

        return self

    def save_logs(self, writer, i, norm='inf'):
        """ Save some logs to the tensorboard.

        Three things are logged for all layers:
            - **the weight infinity norms**
            - **the masked weight infinity norms**
            - **the weight gradient infinity norms**

        Args:
            writer: The tensorboard writer.
            i (int): The current iteration.
            norm (str): The type of norm to be used.
        """
        if norm == 'inf':
            norm = float('inf')

        # Pre-recurrent fc layers.
        for l, layer in enumerate(self._fc_layers_pre):
            # Weight norms.
            lnorm = torch.norm(layer.weight, p=norm)
            writer.add_scalar('network/pre_fc_%i_weight_norm'%l, lnorm, i)

            # Masked weight norms.
            masked_lnorm =  torch.norm(layer.weight *
                                    (1 - self.masks_fc_layers_pre[l]), norm)
            writer.add_scalar('network/pre_fc_%i_masked_weight_norm'%l,
                              masked_lnorm, i)

            # Gradient norms.
            gnorm = torch.norm(layer.weight.grad, p=norm)
            writer.add_scalar('network/pre_fc_%i_weight_grad_norm'%l, gnorm, i)

        # Recurrent layers.
        for l, layer in enumerate(self._rec_layers):
            if hasattr(layer, 'weight_hh_l0'):
                # Weight norms.
                lnorm_hh = torch.norm(layer.weight_hh_l0, p=norm)
                lnorm_ih = torch.norm(layer.weight_ih_l0, p=norm)
                writer.add_scalar('network/rec_%i_hh_weight_norm'%l, lnorm_hh,
                    i)
                writer.add_scalar('network/rec_%i_ih_weight_norm'%l, lnorm_ih,
                    i)

                # Masked weight norms.
                masked_hh_lnorm =  torch.norm(layer.weight_hh_l0 *
                                        (1 - self.masks_rec_hh[l]), norm)
                masked_ih_lnorm =  torch.norm(layer.weight_ih_l0 *
                                        (1 - self.masks_rec[l]), norm)
                writer.add_scalar('network/rec_%i_hh_masked_weight_norm'%l,
                    masked_hh_lnorm, i)
                writer.add_scalar('network/rec_%i_ih_masked_weight_norm'%l,
                    masked_ih_lnorm, i)

                # Gradient norms.
                gnorm_hh = torch.norm(layer.weight_hh_l0.grad, p=norm)
                gnorm_ih = torch.norm(layer.weight_ih_l0.grad, p=norm)
                writer.add_scalar('network/rec_%i_hh_weight_grad_norm'%l,
                    gnorm_hh, i)
                writer.add_scalar('network/rec_%i_ih_weight_grad_norm'%l,
                    gnorm_ih, i)
            else:
                # Weight norms.
                lnorm_hh = torch.norm(layer.weight, p=norm)
                writer.add_scalar('network/rec_%i_ho_weight_norm'%l, lnorm_hh, 
                    i)

                # Masked weight norms.
                masked_hh_lnorm =  torch.norm(layer.weight *
                                        (1 - self.masks_rec_hh[l]), norm)
                writer.add_scalar('network/rec_%i_ho_masked_weight_norm'%l,
                    masked_hh_lnorm, i)

                # Gradient norms.
                gnorm_hh = torch.norm(layer.weight.grad, p=norm)
                writer.add_scalar('network/rec_%i_ho_weight_grad_norm'%l,
                    gnorm_hh, i)

        # Post-recurrent fc layers.
        for l, layer in enumerate(self._fc_layers):
            # Weight norms.
            lnorm = torch.norm(layer.weight, p=norm)
            writer.add_scalar('network/post_fc_%i_weight_norm'%l, lnorm, i)

            # Masked weight norms.
            masked_lnorm =  torch.norm(layer.weight *
                                    (1 - self.masks_fc_layers[l]), norm)
            writer.add_scalar('network/post_fc_%i_masked_weight_norm'%l,
                              masked_lnorm, i)

            # Gradient norms.
            gnorm = torch.norm(layer.weight.grad, p=norm)
            writer.add_scalar('network/post_fc_%i_weight_grad_norm'%l, gnorm, i)