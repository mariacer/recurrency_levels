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
# @title          :bio_rnn/bio_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.7
"""
Utilities for bio-like RNNs.
"""
import numpy as np

from networks.rnn import RNN

def generate_network(config, shared, device, n_in=10, n_out=10,
                     is_teacher=False):
    """Create a RNN microcircuit network.

    This function is inspired by :func:`networks.net_utils.generate_network`.

    It builds a certain custom connectivity based on a set of input masks.

    For simplicity, microcircuit RNNs can only handle a single recurrent layer.
    The input and output masks are also only applied around the recurrent layer,
    i.e. they might not correspond to the actual input and output layers of the
    entire network if a pre-recurrent fully-connected layer is added, or if
    there is more than one fully-connected output layer.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        device: Torch device.
        n_in (int): The input size.
        n_out (int): The output size.
        is_teacher (bool, optional): Whether the network to be generated is a
            teacher network.

    Returns:
        An RNN instance.
    """
    
    if config.use_mlp or (is_teacher and config.teacher_use_mlp):
        raise ValueError('MLP option not compatible with bio-microcircuit.')

    def get_config(attr):
        """Small helper function to extra the correct hyperparameter options
        depending on whether a teacher or not is being constructed."""
        prefix = ''
        if is_teacher:
            prefix = 'teacher_'
        return getattr(config, prefix + attr)

    # Extract hyperparameters.
    n_hidden = misc.str_to_ints(get_config('rnn_arch'))
    post_fc_layers = misc.str_to_ints(get_config('rnn_post_fc_layers'))
    pre_fc_layers = misc.str_to_ints(get_config('rnn_pre_fc_layers'))
    xz_length = get_config('xz_length_microcircuit')
    y_length = get_config('y_length_microcircuit')

    if get_config('fc_rec_output'):
        raise NotImplementedError('Fully connected output layers within '+
            'the recurrent vanilla RNN layers are not implemented for ' +
            'the microcircuit RNN.')

    # Compute the number of input and output neurons that will connect to its
    # neighboring layers given the total numer of neurons and the fraction of
    # input and output.
    n_rnn_connected_to_input = int(get_config('input_fraction')*n_hidden[0])
    n_rnn_connected_to_output = int(get_config('output_fraction')*n_hidden[-1])

    # Define the input and output sizes surrounding the recurrent layers.
    n_out_rnn = n_out
    n_in_rnn = n_in

    if len(post_fc_layers) > 0:
        print('Adding additional fully-connected layers to main network.')
        n_out_rnn = post_fc_layers[0]

    if (not use_full_connectivity and len(pre_fc_layers) > 0) or \
        (not use_full_connectivity and len(post_fc_layers) > 0):
        raise NotImplementedError

    if len(pre_fc_layers) > 0:
        print('Prepending additional fully-connected layers to the main ' +
              'network.')
        n_in_rnn = pre_fc_layers[-1]
    # config.input_dim = n_in
    # config.out_dim = n_out

    if len(n_hidden) > 1:
        raise ValueError('Only one hidden recurrent layer is supported'+
                         'for the microcircuit RNN.')

    # Given the desired number of neurons and the experimentally found
    # neuronal density, deduce the correct dimensions of the brain
    # region containing the microcircuit.
    neuron_density = 25e-6  # neurons / micron**3
    if xz_length is None and y_length is None:
        xz_length = (n_hidden[0]/neuron_density)**(1/3)
        y_length = (n_hidden[0]/neuron_density)**(1/3)
    elif xz_length is None:
        xz_length = (n_hidden[0] / (y_length * neuron_density))**(1/2)
    elif y_length is None:
        xz_length = (n_hidden[0] / (xz_length**2 * neuron_density))

    # Determine 3D locations of the neurons in a fictional microcolumn.
    xyz_coordinates = generate_neuron_coordinates(n=n_hidden[0],
                                                  x=xz_length,
                                                  y=y_length,
                                                  z=xz_length)

    # Load experimental data.
    # It is given in an array form with (n_y,n_r) dimensions, where
    # n_y corresponds to the number of bins in the vertical dimensions 
    # (i.e. perpendicular to cortex surface) and n_r is the number of
    # bins in the radial dimension (i.e. parallel to cortex surface). 
    path = '../datasets'
    if get_config('random_microcircuit_connectivity'):
        if get_config('use_clusters'):
            con_prob_mat = np.load(path +
                '/connectivity_data/con_prob_mat_cluster_random.npy')
        else:
            con_prob_mat = np.load(path +
                           '/connectivity_data/con_prob_mat_random.npy')
    else:
        if get_config('use_clusters'):
            con_prob_mat = np.load(path +
                        '/connectivity_data/con_prob_mat_cluster.npy')
        else:
            con_prob_mat = np.load(path +
                               '/connectivity_data/con_prob_mat.npy')

    # Scale connectivity probabilities if asked.
    if get_config('scale_connectivity_prob') > 0:
        con_prob_mat *= get_config('scale_connectivity_prob')
    elif get_config('rec_sparsity') > 0:
        con_prob_mat = set_connectivity_fraction(con_prob_mat, \
            conn_ratio=get_config('rec_sparsity')) 
        assert np.abs(con_prob_mat.mean() - get_config('rec_sparsity')) \
            < 1e-5
    assert con_prob_mat.max() <= 1.

    if get_config('use_clusters'):
        xz_bins = np.load(path + '/connectivity_data/xz_bins_cluster.npy')
        y_bins = np.load(path + '/connectivity_data/y_bins_cluster.npy')
    else:
        xz_bins = np.load(path + '/connectivity_data/xz_bins.npy')
        y_bins = np.load(path + '/connectivity_data/y_bins.npy')

    # Get indices of input and output neurons in the column.
    input_idx = get_neurons_idc(xyz_coordinates, group='input',
                          nb_neurons=n_rnn_connected_to_input)
    output_idx = get_neurons_idc(xyz_coordinates, group='output',
                          nb_neurons=n_rnn_connected_to_output)

    # Generate binary connectivity masks for all layers.
    input_mask = generate_mask_input(input_idx, shape=(n_hidden[0], n_in_rnn))
    output_mask = generate_mask_output(output_idx,
                                       shape=(n_out_rnn, n_hidden[0]))

    if get_config('use_clusters'):
        masks_rec_hh = generate_cluster_connectivity_matrix(xyz_coordinates,
                                con_prob_mat=con_prob_mat,
                                xz_bins=xz_bins,
                                y_bins=y_bins,
                                cn=get_config('nb_clusters'),
                                pc=get_config('within_cluster_prob_scaling'),
                                po=get_config('outside_cluster_prob_scaling'))
    else:
        masks_rec_hh = generate_connectivity_matrix(xyz_coordinates,
                                        con_prob_mat=con_prob_mat,
                                        xz_bins=xz_bins,
                                        y_bins=y_bins)

    if get_config('use_full_connectivity'):
        input_mask = np.ones_like(input_mask)
        output_mask = np.ones_like(output_mask)
        masks_rec_hh = np.ones_like(masks_rec_hh)

    # Generate the network.
    net = RNN(n_in=n_in,
              rec_layers_size=n_hidden,
              fc_layers_pre_size=pre_fc_layers,
              fc_layers_size=[*post_fc_layers, n_out],
              activation=get_config('net_act'),
              masks_fc_layers_pre=torch.from_numpy(input_mask).float(),
              masks_rec_hh=[torch.from_numpy(masks_rec_hh).float()],
              masks_fc_layers=torch.from_numpy(output_mask).float(),
              use_bias=not get_config('dont_use_bias'),
              use_lstm=not get_config('use_vanilla_rnn'),
              fc_rec_output=get_config('fc_rec_output')
          ).to(device)

    return net

def generate_connectivity_matrix(xyz_coord, con_prob_mat, xz_bins, y_bins):
    """Generate connectivity matrix.

    Generate a binary mask of connectivities for the neuron coordinates given
    by xyz_coord, according to the connectivity probabilities in con_prob_mat.

    Args:
        xyz_coord (np.ndarray): An nx3 matrix with n the number of neurons.
            Each row gives the xyz coordinates of each neuron.
        con_prob_mat (np.ndarray): connectivity probabilities in cilindrical 
            coordinates: the first axis corresponds with distances along the y 
            coordinate, whereas the second axis correspond with radial distance 
            from the y-axis.
        xz_bins (np.ndarray): radial distance measures corresponding to the 
            upper bounds of the bins in con_prob_mat (in \mu m).
        y_bins (np.ndarray): vertical distance measures corresponding to the 
            upper bounds of the bins in con_prob_mat (in \mu m).

    Returns:
        (np.ndarray): A binary nxn mask of connectivities for all pairs of 
            neurons.
    """
    n = xyz_coord.shape[0] # number of neurons
    connectivity_matrix = np.random.rand(n, n)

    for i in range(n):
        for j in range(n):

            # All diagonal entries should be zero (no recurrent connections to
            # the neurons themselves).
            if i == j:
                connectivity_matrix[i, j] = 0
                continue

            # Calculate the distance between neurons i and j.
            y_distance = xyz_coord[j, 1] - xyz_coord[i, 1]  
            # not abs, because the vertical direction matters
            radial_distance = np.sqrt((xyz_coord[i, 0] - xyz_coord[j, 0])**2 +
                                      (xyz_coord[i, 2] - xyz_coord[j, 2])**2)

            # find y bin index
            y_bin_idx = len(y_bins) - np.sum(y_distance < y_bins)
            # find xz bin index
            xz_bin_idx = len(xz_bins) - np.sum(radial_distance < xz_bins)

            # if index is out of range, take the last bin of the con_prob_mat
            y_bin_idx = min(y_bin_idx, len(y_bins) - 1)
            xz_bin_idx = min(xz_bin_idx, len(xz_bins) - 1)

            # sample binary connection according to connectivity probability in
            # con_prob_mat
            if con_prob_mat[y_bin_idx, xz_bin_idx] > connectivity_matrix[i,j]:
                connectivity_matrix[i, j] = 1
            else:
                connectivity_matrix[i, j] = 0

    return connectivity_matrix

def generate_cluster_connectivity_matrix(xyz_coord, con_prob_mat, xz_bins, 
                                         y_bins, pc, po, cn):
    """Generate connectivity matrix with the clustered approach.

    Generate a binary mask of connectivities for the neuron coordinates given
    by xyz_coord, according to the connectivity probabilities in con_prob_mat.

    Args:
        xyz_coord (np.ndarray): An nx3 matrix with n the number of neurons.
            Each row gives the xyz coordinates of each neuron.
        con_prob_mat (np.ndarray): connectivity probabilities in cilindrical
            coordinates: the first axis corresponds with distances along the y
            coordinate, whereas the second axis correspond with radial distance
            from the y-axis.
        xz_bins (np.ndarray): radial distance measures corresponding to the
            upper bounds of the bins in con_prob_mat (in \mu m).
        y_bins (np.ndarray): vertical distance measures corresponding to the
            upper bounds of the bins in con_prob_mat (in \mu m).
        pc (float): probability scaling factor for within-cluster neurons
        po (float): probability scaling factor for outside-cluster neurons
        cn (int): number of clusters

    Returns:
        (np.ndarray): A binary nxn mask of connectivities for all pairs of
            neurons.
    """
    n = xyz_coord.shape[0] # number of neurons
    connectivity_matrix = np.zeros((n, n))
    clusters = np.random.randint(low=0, high=cn, size=n)

    for i in range(n-1):
        for j in range(i+1, n):

            # Calculate the distance between neurons i and j.
            y_distance1 = xyz_coord[j, 1] - xyz_coord[i, 1]
            y_distance2 = -y_distance1
            # not abs, because the vertical direction matters
            radial_distance = np.sqrt((xyz_coord[i, 0] - xyz_coord[j, 0])**2 +
                                      (xyz_coord[i, 2] - xyz_coord[j, 2])**2)

            # find y bin index
            y_bin_idx1 = len(y_bins) - np.sum(y_distance1 < y_bins)
            y_bin_idx2 = len(y_bins) - np.sum(y_distance2 < y_bins)
            # find xz bin index
            xz_bin_idx = len(xz_bins) - np.sum(radial_distance < xz_bins)

            # if index is out of range, take the last bin of the con_prob_mat
            y_bin_idx1 = min(y_bin_idx1, len(y_bins) - 1)
            y_bin_idx2 = min(y_bin_idx2, len(y_bins) - 1)
            xz_bin_idx = min(xz_bin_idx, len(xz_bins) - 1)

            # sample binary connection according to connectivity probability in
            # con_prob_mat
            r1 = np.random.rand()
            r2 = np.random.rand()

            if clusters[i] == clusters[j]:
                if r1/pc < con_prob_mat[y_bin_idx1, xz_bin_idx]:
                    connectivity_matrix[i, j] = 1
                if r2/pc < con_prob_mat[y_bin_idx2, xz_bin_idx]:
                    connectivity_matrix[j,i] = 1
            else:
                if r1/po < con_prob_mat[y_bin_idx1, xz_bin_idx]:
                    connectivity_matrix[i, j] = 1
                if r2/po < con_prob_mat[y_bin_idx2, xz_bin_idx]:
                    connectivity_matrix[j,i] = 1

    return connectivity_matrix


def generate_neuron_coordinates(x=360, y=360, z=360, n=None):
    """Generate an xyz_coord matrix with the coordinates of n neurons uniformly
    distributed in the cube x y z.

    Args:
        x (float): length of the x-side of the cube in \mu m
        y (float): length of the y-side of the cube in \mu m
        z (float): length of the z-side of the cube in \mu m
        n (int): number of neurons in the cube. If None, the average neuron 
            density (25e-6 neurons / \mu m^3) is used to compute n. A warning 
            will be raised when the provided n differs too much from the average
            neuron density.

    Returns:
        (np.ndarray): nx3 matrix with neuron coordinates in \mu m.
    """
    density = 25e-6
    if n is None:
        n = int(x*y*z*density)
    else:
        if (n > 1.4 * x*y*z*density or n < 0.6 * x*y*z*density):
            print('WARNING: the number of hidden recurrent neurons does not '
                  'match the anatomical data.')

    # Populate the xyz coordinates via a uniform distribution over [0,1].
    xyz_coord = np.random.rand(n, 3)
    # Scale the columns such that they are in the x, y and z range
    xyz_coord = xyz_coord * np.array([[x, y, z]])

    return xyz_coord


def get_neurons_idc(xyz_coord, nb_neurons=None, y_threshold=None, 
                    group='input'):
    """Get indices of either input or output neurons.

    Return the list with indices of neurons in the xyz_coord matrix that have
    the lowest (resp. highers) y coordinates (nb_neurons) or with an y 
    coordinate below (resp. above) the y_threshold, which indicates that they 
    are input (resp. output) neurons. 

    Importantly, only one of the arguments [nb_neurons, y_threshold] should be 
    specified (either you chose for a fixed number of input neurons or you chose 
    a threshold which results in a flexible number of input neurons). 
    The default option is nb_neurons=200.

    Args:
        xyz_coord (np.ndarray): nx3 matrix with neuron coordinates in \mu m
            (with n the amount of neurons).
        nb_neurons (int): number of input neurons that need to be selected. The
            <nb_neurons> neurons with the smallest y value will be selected.
        y_threshold (float): the y threshold in \mu m which indicates
            the barrier
            between input and hidden neurons
        group`(str, optional): The group of neurons whose indices are to be 
            returned. Can be either `input` or `output`.

    Returns:
        (nd.array): The binary indices of the target neurons.
    """
    use_fixed = False
    if (nb_neurons is None and y_threshold is None):
        nb_neurons = 200
        use_fixed = True
    elif (nb_neurons is not None and y_threshold is not None):
        raise ValueError('Only one of the arguments [nb_neurons, y_threshold'
                         'should be specified.')
    elif nb_neurons is not None:
        use_fixed = True

    if nb_neurons > xyz_coord.shape[0]:
        raise ValueError('The specified number of input neurons <nb_neurons>'
                         'is greater than the provided neuron coordinates.')

    if use_fixed:
        if group == 'input':
            idx = np.argsort(xyz_coord[:, 1])[:nb_neurons]
        elif group == 'output':
            idx = np.argsort(xyz_coord[:, 1])[-nb_neurons:]
        return idx2binary_idx(idx, xyz_coord.shape[0])
    else:
        if group == 'input':
            return xyz_coord[:, 1] < y_threshold
        elif group == 'output':
            return xyz_coord[:, 1] > y_threshold

  
def idx2binary_idx(idx, max_idx):
    """Convert an array of integer indices to an array of binary indices on the
    corresponding row indices.

    Args:
        idx: array of integer indices
        max_idx: the maximum index (length of the array of which indices are
            provided).

    Returns: 
        (np.array): The binary indices of length max_idx.
    """
    binary_idx = np.zeros(max_idx)
    for i in idx:
        binary_idx[i] = 1
    return binary_idx
