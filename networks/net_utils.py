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
# @title          :networks/net_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Utils for building networks
---------------------------

Useful functions for generating networks.

"""
import numpy as np
import torch

from networks.mlp import MLP
from networks.rnn import RNN
from utils import misc

def generate_network(config, shared, device, n_in=10, n_out=10, 
                     is_teacher=False, rseed=None):
    """Create a RNN network.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        device: Torch device.
        n_in (int): Input size.
        n_out (int): Output size.
        is_teacher (bool, optional): Whether the network to be generated is a
            teacher network.
        rseed (int or None, optional): The random seed for reproducibility.

    Returns:
        An RNN instance.
    """

    def get_config(attr):
        """Small helper function to extra the correct hyperparameter options
        depending on whether a teacher or not is being constructed."""
        prefix = ''
        if is_teacher:
            prefix = 'teacher_'
        return getattr(config, prefix + attr)

    # Extract relevant network sizes.
    n_hidden = misc.str_to_ints(get_config('rnn_arch'))

    if (is_teacher and config.teacher_use_mlp) or (not is_teacher and \
            config.use_mlp):
        ### Basic MLP.
        net = MLP(n_in=n_in, 
                  n_out=n_out,
                  hidden_sizes=n_hidden,
                  use_bias=not get_config('dont_use_bias'),
                  activation=get_config('net_act'),
                  output_per_ts=shared.output_per_ts,
                  n_ts=shared.n_ts,
                  kaiming_init=get_config('use_kaiming_init')
              ).to(device)
    else:
        # Determine the size of fully connected layers.
        pre_fc_layers = misc.str_to_ints(get_config('rnn_pre_fc_layers'))
        post_fc_layers = [*misc.str_to_ints(get_config('rnn_post_fc_layers')), \
            n_out]

        if len(pre_fc_layers) > 0:
            print('Prepending additional fully-connected layers to the main ' +
                  'network.')
        if len(post_fc_layers) > 1:
            print('Adding additional fully-connected layers to main network.')

        # Build the sparsity masks.
        masks_fc_layers_pre, masks_rec, masks_rec_hh, masks_fc_layers = \
            build_all_masks(config, n_in, pre_fc_layers, n_hidden,
                            post_fc_layers, is_teacher=is_teacher, rseed=rseed)

        # Generate the network.
        net = RNN(n_in=n_in,
                  rec_layers_size=n_hidden,
                  fc_layers_pre_size=pre_fc_layers,
                  fc_layers_size=post_fc_layers,
                  activation=get_config('net_act'),
                  use_lstm=not get_config('use_vanilla_rnn'),
                  use_bias=not get_config('dont_use_bias'),
                  fc_rec_output=get_config('fc_rec_output'),
                  masks_fc_layers_pre=masks_fc_layers_pre,
                  masks_rec_hh=masks_rec_hh,
                  masks_rec=masks_rec, 
                  masks_fc_layers=masks_fc_layers,
                  kaiming_init=get_config('use_kaiming_init')
              ).to(device)

        # Even if the kaiming init was already applied upon layer generation,
        # it was done before applying masks. So if the network is not
        # fully connected if has to be redone here.
        if get_config('use_kaiming_init') and not net._full_connectivity:
            net.init_kaiming()
        # If both inits are selected, the orthogonal init will override
        # the kaiming init in hidden-to-hidden weights.
        if get_config('orthogonal_hh_init'):
            net.init_hh_weights_orthogonal()

    if config.use_same_capacity_mlp:
        # Modify the size of the hidden layer
        num_weights = net.get_num_weights()
        raise NotImplementedError

    return net


def build_all_masks(config, n_in, pre_fc_layers, n_hidden, post_fc_layers,
                    is_teacher=False, rseed=None):
    """Build masks for all layers given the config.

    Args:
        config (Namespace): The configuration.
        n_in (int): The input dimensionality.
        pre_fc_layers (list): The list of sizes of the pre-recurrent fully-
            connected layers.
        n_hidden (list): The list of recurrent layer sizes.
        post_fc_layers (list): The list of sizes of the post-recurrent fully-
            connected layers.
        rseed (None or int, optional): The random seed.

    Returns:
        (tuple): Tuple containing:

        - **masks_fc_layers_pre**: The pre-fc masks.
        - **masks_rec**: The input-to-hidden recurrent masks.
        - **masks_rec_hh**: The hidden-to-hidden recurrent masks.
        - **masks_fc_layers**: The post-fc masks.
    """
    def get_config(attr):
        """Small helper function to extra the correct hyperparameter options
        depending on whether a teacher or not is being constructed."""
        prefix = ''
        if is_teacher:
            prefix = 'teacher_'
        return getattr(config, prefix + attr)

    if rseed is None:
        if is_teacher:
            rseed = config.data_random_seed
        else:
            rseed = config.arch_random_seed
    rstate = np.random.RandomState(rseed)

    masks_fc_layers_pre = None
    masks_fc_layers = None
    masks_rec_hh = None
    masks_rec = None

    ### Get pre-recurrent fc layers masks.
    pre_size = n_in
    if get_config('fc_sparsity') not in [-1, 1]:

        if len(pre_fc_layers) > 0:
            masks_fc_layers_pre = []
            for l in range(len(pre_fc_layers)):
                post_size = pre_fc_layers[l]
                mask = build_mask(pre_size, post_size, rstate,
                                  p=get_config('fc_sparsity'))
                masks_fc_layers_pre.append(mask)
                pre_size = post_size
        else:
            # If no initial pre-recurrent fc layers exist, we need to mask
            # the input weights to the first recurrent layer.
            masks_rec = [None]*len(n_hidden)
            post_size = n_hidden[0]
            mask = build_mask(pre_size, post_size, rstate,
                              p=get_config('fc_sparsity'))
            masks_rec[0] = mask
            pre_size = post_size

        if get_config('no_sparse_input_output'):
            # Make sure input fully-connected layer is not masked.
            if len(pre_fc_layers) > 0:
                masks_fc_layers_pre[0] = None
            else:
                masks_rec[0] = None

    ### Get hidden-to-hidden recurrent masks.
    if get_config('rec_sparsity') not in [-1, 1]:
        masks_rec_hh = []
        for l in range(len(n_hidden)):
            mask = build_mask(n_hidden[l], n_hidden[l], rstate,
                              p=get_config('rec_sparsity'))
            if get_config('recurrency_level') != -1:
                # If the recurrency level is specific, the mask is customized.
                mask = build_recurrency_mask(n_hidden[l], n_hidden[l], rstate,
                              p=get_config('rec_sparsity'),
                              r=get_config('recurrency_level'))
            masks_rec_hh.append(mask)

            if get_config('use_vanilla_rnn') and \
                    get_config('fc_rec_output'):
                # For vanilla RNNs with fully-connected outputs, we consider
                # the output layer of the RNN as part of the recurrent
                # computation and therefore apply the same mask as for the
                # hidden-to-hidden weights.
                masks_rec_hh.append(mask)

    ### Get fc layers between recurrent layers.
    if get_config('fc_sparsity') not in [-1, 1]:
        # For the connections between recurrent layers, the sparsity level
        # of the fully-connected layers is used.
        if masks_rec is None:
            masks_rec = [None]*len(n_hidden)

        for l in range(len(n_hidden)):
            if l == 0 and masks_rec[l] is not None:
                # If the initial mask has already been set above (i.e. in
                # the case where there are no fully-connected pre layers
                # and the input is fully-connected), we skip.
                continue
            post_size = n_hidden[l]
            mask = build_mask(pre_size, post_size, rstate,
                              p=get_config('fc_sparsity'))
            masks_rec[l] = mask
            pre_size = post_size

    ### Get post-recurrent fc layers masks.
    if get_config('fc_sparsity') not in [-1, 1]:

        masks_fc_layers = []
        for l in range(len(post_fc_layers)):
            post_size = post_fc_layers[l]
            mask = build_mask(pre_size, post_size, rstate,
                              p=get_config('fc_sparsity'))
            masks_fc_layers.append(mask)
            pre_size = post_size

        if get_config('no_sparse_input_output'):
            # Make sure output fully-connected layer is not masked.
            masks_fc_layers[-1] = None

    # Sanity checks.
    assert masks_fc_layers_pre is None or len(masks_fc_layers_pre) == \
            len(pre_fc_layers)
    if get_config('use_vanilla_rnn') and  get_config('fc_rec_output'):
        assert masks_rec_hh is None or len(masks_rec_hh) == len(n_hidden)*2
    else:
        assert masks_rec_hh is None or len(masks_rec_hh) == len(n_hidden)
    assert masks_rec is None or len(masks_rec) == len(n_hidden)
    assert masks_fc_layers is None or len(masks_fc_layers) == \
            len(post_fc_layers)

    return masks_fc_layers_pre, masks_rec, masks_rec_hh, masks_fc_layers 


def build_mask(n_in, n_out, rstate, p=1):
    """Build a sparsity mask with the desired dimensions.

    Importantly, we would like layers with identical number of neurons and
    sparsity levels to have an identical number of connections. For this reason
    we always set the number of existing connections as the floor of
    multiplying the total number of connections by p.

    Args:
        n_int (int): The input size, which will define the number of columns.
        n_out (int): The output size, which will define the number of rows.
        rstate: The random state.
        p (float, optional): The desired sparsity level. A value of ``1``
            indicates a fully-connected layer.

    Returns:
        (torch.Tensor): The mask.
    """
    num_nonmasked_conn = int(np.floor(n_in * n_out * p))
    rand_mask = torch.tensor(rstate.rand(n_out, n_in), dtype=torch.float32)

    # Flatten and sort the values.
    vals, _ = torch.sort(torch.flatten(rand_mask))

    # Compute the highest accepted value to be considered as connection.
    highest_accepted_val = vals[num_nonmasked_conn].item()

    # Build the mask.
    mask = torch.empty_like(rand_mask)
    mask[rand_mask < highest_accepted_val] = 1
    mask[rand_mask >= highest_accepted_val] = 0
    
    # Sometimes, due to numerical issues not the exact same number of
    # connections exists. In this case, we correct.
    if not mask.sum() == num_nonmasked_conn:
        num_extra_conn = mask.sum() - num_nonmasked_conn
        while num_extra_conn != 0:
            post = rstate.randint(n_out)
            pre = rstate.randint(n_in)
            if num_extra_conn > 0:
                # Remove connections.
                if mask[pre, post] == 1:
                    mask[pre, post] = 0
                    num_extra_conn -= 1
            elif num_extra_conn < 0:
                # Add connections.
                if mask[pre, post] == 0:
                    mask[pre, post] = 1
                    num_extra_conn += 1

    assert mask.sum() == num_nonmasked_conn
    return mask


def build_recurrency_mask(n_in, n_out, rstate, p=1, r=1):
    """Build a sparsity mask with the desired dimensions and recurrency level.

    Args:
        (....): See docstring of function :func:`build_mask`.
        r (float, optional): The desired recurrency level.

    Returns:
        (torch.Tensor): The mask.
    """
    mask = rstate.rand(n_out, n_in)

    # masks_rec_hh = []
    # # When the recurrency level is determined, the index of the
    # # recurrent neurons has a meaning, where lower indices
    # # indicate more upstream neurons in the hierarchy, and
    # # viceversa. For this reason, input neurons connect to the
    # # first indices, and output neurons to the last indices.

    # # Input mask.
    # input_idx = np.arange(n_rnn_connected_to_input)

    # # Recurrent layers.
    # for l in range(len(n_hidden)):
    #     layer_mask = torch.zeros((n_hidden[l], n_hidden[l]))

    #     # Number of connections to set.
    #     num_connections = int(n_hidden[l] * n_hidden[l] * \
    #         get_config('rec_sparsity'))
    #     num_tri_connections =  int((n_hidden[l]**2 - n_hidden[l])/2)

    #     # Add connection to the neuron with subsequent index.
    #     # This makes the recurrent layer fully feedforward.
    #     layer_mask[1:, :-1] = torch.eye(n_hidden[l]-1)

    #     # Determine the number of connections to allocate in the
    #     # upper (feedback) and lower (feedforward) triangles.
    #     num_avail_upper_tri_connections = int(\
    #         (n_hidden[l]**2 - n_hidden[l])/2)
    #     # Because the lower triangle also has the largest
    #     # diagonal filled with ones, it cannot be used to add
    #     # further connections so we need (n_hidden[l] - 1).
    #     num_avail_lower_tri_connections = int(\
    #         ((n_hidden[l] - 1)**2 - (n_hidden[l] - 1))/2)
    #     num_off_diag_connections = n_hidden[l]**2 * \
    #         get_config('rec_sparsity') - n_hidden[l]
    #     num_upper_tri_connections = int(num_off_diag_connections * \
    #             get_config('recurrency_level')/2)
    #     num_lower_tri_connections = int(num_off_diag_connections * \
    #             (1 - get_config('recurrency_level')/2))

    #     if num_avail_lower_tri_connections < \
    #             num_lower_tri_connections:
    #         raise ValueError('The required recurrency level is '
    #                          'too low for the desired level of '
    #                          'recurrent connectivity.')

    #     # Randomly select the connections in the triangles.
    #     ratio_ut_conn = num_upper_tri_connections / \
    #         num_tri_connections
    #     # Because the number of lower tri connections cannot be
    #     # in the diagonal lower to the main one, we need to
    #     # scale up the ratio_lt_conn to compensate for this.
    #     ratio_lt_conn = num_lower_tri_connections / \
    #         num_tri_connections * (n_hidden[l]/(n_hidden[l]-1))
    #     rand_mask = torch.rand((n_hidden[l], n_hidden[l]))
    #     rand_mask_ut = torch.zeros_like(rand_mask)
    #     rand_mask_lt = torch.zeros_like(rand_mask)
    #     rand_mask_ut[rand_mask < ratio_ut_conn ] = 1.
    #     rand_mask_lt[rand_mask < ratio_lt_conn ] = 1.

    #     layer_mask += torch.tril(rand_mask_lt, diagonal=-2)
    #     layer_mask += torch.triu(rand_mask_ut, diagonal=1)

    #     assert layer_mask.max() <= 1.

    #     # Values to check:
    #     # layer_mask.sum()/layer_mask.numel() == \
    #     #       get_config('rec_sparsity')
    #     # torch.triu(rand_mask_ut, diagonal=1).sum() / \
    #     #       ((torch.triu(rand_mask_ut, diagonal=1).sum() + \
    #     #        torch.tril(rand_mask_lt, diagonal=-2)) / 2) \
    #     #       == get_config('recurrency_level')

    #     masks_rec_hh.append(layer_mask)

    # # Output mask.
    # output_idx = np.arange(n_hidden[-1])[\
    #     -n_rnn_connected_to_output:]

    raise NotImplementedError


def generate_mask_input(binary_idx, shape):
    """Generate a binary mask with rows of ones or zeros, according to the
    binary_idx.

    Args:
        binary_idx: binary index list indicating whether the corresponding row
            will contain zeros or ones
        shape: shape of the returned mask

    Returns: 
        (np.array): A binary input mask.
    """
    mask = np.ones(shape)
    return (mask.T * binary_idx).T

def generate_mask_output(binary_idx, shape):
    """Generate a binary mask with columns of ones or zeros, according to the
    binary_idx.

    Args:
        binary_idx: binary index list indicating whether the
            corresponding column
            will contain zeros or ones
        shape: shape of the returned mask

    Returns: 
        (np.array): A binary output mask.
    """
    mask = np.ones(shape)
    return mask * binary_idx


def extract_hh_weights(net):
    """Extract hidden-to-hidden weights.

    This function extracts the hidden-to-hidden weights.
    Note, if the main network uses LSTM weights, then this function
    automatically decomposes them since they are stored in a concatenated form.

    Args:
        mnet (networks.rnn): The network.

    Returns:
        (list): A list with the extracted weight tensors.
    """

    ret = []
    for layer in net._rec_layers:
        W = layer.weight_hh_l0

        # LSTM weight matrices are stored such that the hidden-to-hidden 
        # matrices for the 4 gates are concatenated.
        if net._use_lstm:
            out_dim, _ = W.shape
            assert out_dim % 4 == 0
            fs = out_dim // 4

            ret.append(W[:fs, :])
            ret.append(W[fs:2*fs, :])
            ret.append(W[2*fs:3*fs, :])
            ret.append(W[3*fs:, :])
        else:
            ret.append(W)

    if net._use_lstm:
        assert len(ret) == len(net._rec_layers_size)*4
    else:
        assert len(ret) == len(net._rec_layers_size)

    return ret

def set_connectivity_fraction(prob_mat, conn_ratio=None):
    """Function to set a specified connectivity ratio in recurrent connections.

    This function is a recurrent function whose stop case corresponds to
    whenever the mean connectivity of the matrix is greater than the desired
    one (`conn_ratio`).

    Args:
        prob_mat (np.array): The initial connectivity probabilities.
        conn_ratio (float, optional): The desired average connectivity 
            probability.

    Returns:
        (np.array): The connectivity matrix with scaled probabilities.
    """
    # Add minimal connection probability to the neurons that have 0 connections
    prob_mat[prob_mat==0] += 1e-6

    curr_ratio = prob_mat.mean()
    if conn_ratio is None:
        return prob_mat
    elif conn_ratio == 1:
        return np.ones_like(prob_mat)
    elif conn_ratio < curr_ratio or np.abs(conn_ratio - curr_ratio) <= 1e-5: 
        # Ending case: reduce connections, or desired value is close enough.
        # If the number of connections needs to be reduced, simply
        # scale the probabilities accordingly.
        prob_mat = prob_mat * conn_ratio / curr_ratio
        prob_mat[prob_mat > 1] = 1.
        return prob_mat
    else: 
        # If the number of connections needs to be increased, a simple
        # scaling cannot be done, since the max probability value is 1.
        if np.sum(prob_mat * conn_ratio / curr_ratio > 1) > 0:
            # If some neuron should be scaled beyond 1, fix it by adding more
            # connectivity to other neurons.
            prob_mat_scaled = np.copy(prob_mat * conn_ratio / curr_ratio)

            # Set a probability of 1 for all neurons whose connectivity
            # will be saturated (i.e. whose current connectivity probability
            # satisfies p_i/conn_ratio > 1) and then call this function again.
            prob_mat_scaled[prob_mat * conn_ratio / curr_ratio > 1] = 1.
            prob_mat_scaled = set_connectivity_fraction(prob_mat_scaled, \
                conn_ratio=conn_ratio)
            return prob_mat_scaled
        else:
            return prob_mat * conn_ratio / curr_ratio
