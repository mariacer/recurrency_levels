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
# @title          :feedforwardness/utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Utils for feedforwardness measures and studies
"""
import numpy as np
import os
import pickle
import time
import torch
import warnings

def compute_directedness(A, logger, feedforward=False, n_connections=None,
                         n_neurons=None):
    r"""Compute level of directedness of the network given the adjacency matrix.

    We compute directedness as done in 
    `Reimann et al. <https://www.frontiersin.org/articles/10.3389/fncom.2017.00048/full>`_.
    Specifically, we compute:

    .. math::

        d = \sum_i^N (deg_{in;i} - deg_{out;i})^2

    where $N$ is the dimensionality of the matrix $A$ and $deg_{in;i}$ and
    $deg_{out;i}$ denote the in- and out-degrees of the neuron $i$, i.e. they
    correspond to the sum of $A_{:i}$ and $A_{i:}$, respectively.
    Note that this function only works for square A matrices, and therefore can't
    be applied to an LSTM.

    Args:
        A (torch.tensor): The adjacency matrix. We assume that A_ij corresponds
            to neuron i connecting to neuron j, i.e. rows indicate pre-synaptic
            neurons and columns post-synaptic neurons.
        logger: The logger.
        feedforward (boolean, optional): If ``True``, the directedness measure
            will be computed for an equivalent graph, with same number of
            neurons and connections, but exhibiting perfectly feedforward 
            connectivity. It will correspond to the graph that, with the
            provided amount of resources, will result in the higher
            directedness. Note that this value can then be used for
            normalization.
        n_connections (int): The number of desired connections. Only used if
            we are constructing a feedforward graph.
        n_neurons (int): The number of desired neurons. Only used if
            we are constructing a feedforward graph.

    Returns:
        (int): The directedness.
    """
    if A is not None:
        assert A.shape[0] == A.shape[1] # not the case in LSTMs

    if n_connections is not None or n_neurons is not None:
        assert feedforward
    else:
        n_connections = A.sum()
        n_neurons = A.shape[0]

    if feedforward:

        # Build an adjacency matrix for a perfectly directed graph.
        if A is not None:
            A = torch.zeros_like(A)
        else:
            A = torch.zeros(n_neurons, n_neurons)
        count_conn = 0
        for post in range(n_neurons):
            for pre in range(n_neurons):
                if pre == post:
                    continue
                if pre <= post:
                    # Self-connections mean 1 out degree and 1 in degree and
                    # therefore don't increase the directedness measure.

                    # Furthermore, we don't allow connection back down the
                    # hierarchy (pre < post).
                    continue
                else:
                    if not count_conn >= n_connections:
                        A[post, pre]  = 1
                        count_conn += 1
                if count_conn >= n_connections:
                    break
            if count_conn >= n_connections:
                break

        # It might be that a purely feedforward connectivity is not possible
        # if the number of connections is too high. In that case, we fill up
        # the diagonal of the adjacency matrix with ones, since self-connections
        # don't decrease the directedness measure.
        if count_conn < n_connections:
            logger.debug('The number of desired connections is too high to ' + 
                          'allow for perfect feedforwardness. Adding self-' +
                          'connections to the graph. ' +
                          '(n = %i, c = %.3f)' % (n_neurons, n_connections))
            diag_idx = 0
            while count_conn < n_connections and diag_idx < A.shape[0]:
                A[diag_idx, diag_idx] = 1
                count_conn += 1
                diag_idx += 1

        if count_conn < n_connections:
            logger.debug('The number of desired connections is still too ' +
                          'high after adding self-connections. Need to ' +
                          'implement an algorithm to obtain highest ' +
                          'possible directedness in recurrent graphs.')
            # Ideas: connect every time nodes with the lowest (degin - degout),
            # since this will affect the least the overall square sum.

        if not A.sum() == n_connections:
            logger.debug('The number of desired connections ' + 
                         '%i could not be met while ' % n_connections +
                         'maintaining between-neuron feedforwardness ' +
                         '(self-connections were added). The total number of ' +
                         'existing connections is %i.' % A.sum())

    deg_in = A.sum(axis=0) # sum along rows
    deg_out = A.sum(axis=1) # sum along columns 

    d = ((deg_in - deg_out)**2).sum()

    return int(d.item())

def iteratively_compute_number_cycles(A, logger, N=None,
                                      correct_multiple_counts=True,
                                      max_time=60.):
    r"""Compute the number of closed walks up to a given length.

    This is an alternative to the function :func:`compute_number_closed_walks`
    which instead of using the adjacency matrix, uses an iterative algorithm
    to count the number of cycles. Importantly, this algorithm only counts each
    cycle once, and therefore automatically applies the corrective term that
    is optional to the other function.

    This algorithm is based on the Deep First Search algorithm and is inspired
    by 
    'this blog <https://nlogn.in/count-all-possible-paths-between-two-vertices-of-a-directed-graph/>'_.

    Note that this algorithm will in general count every cycle of length `n` `n`
    times (ex: node 1 > node 2 > node 1, and node 2 > node 1 > node 2 will be
    counted as two separate cycles). This can be corrected for by dividing the
    number of cycles of a given length by its length, according to the
    input `correct_multiple_counts`.

    Args:
        (...): See docstring of function :func:`compute_number_closed_walks`.
        correct_multiple_counts (boolean): If ``True``, the fact that a given
            cycle of length `n` is counted `n` times is corrected for.
        max_time (float): The maximum number of seconds the count may take.

    Returns:
        (....): Tuple containing:

        - **num_cycles**: The number of cycles.
        - **num_cycles_of_len**: The number of cycles of a given length. Each
            element ``num_cycles_of_len[i]`` corresponds to the number of cycles
            of length ``i+1``.
    """
    start = time.process_time()

    if N is None:
        N = A.shape[0]
    else:
        assert N <= A.shape[0]

    # For each node, find number of paths starting and finishing there.
    all_paths = []
    for i in range(A.shape[0]):
        n_cycles_i, cycles_i = count_cycles(A, [i], logger, max_length=N,
            all_paths=[], start_time=start, max_time=max_time)
        all_paths.extend(cycles_i)
    if max_time is not None and start is not None:
        if time.process_time() - start > max_time:
            logger.debug('Maxtime in "iteratively_compute_number_cycles". '+
                          'Breaking.')

    n_cycles_of_len = np.zeros(N)
    for cycle in all_paths:
        # Why do we subtract 2?
        # 1) Because lengths start at 1 and indexes at 0
        # 2) Because the cycles contain the number of nodes in the cycle, and so
        #    the number of edges is its length minus 1
        n_cycles_of_len[len(cycle)-2] += 1

    if correct_multiple_counts:
        for i, n in enumerate(n_cycles_of_len):
            n_cycles_of_len[i] = n / (i + 1)
    
    n_cycles = np.sum(n_cycles_of_len)

    return n_cycles, list(n_cycles_of_len)

def count_cycles(A, path, logger, all_paths=None, max_length=-1,
                 start_time=None, max_time=None,
                 revisit_non_initial_node=False):
    """Count cycles that start with the provided path.

    Stop conditions:
    1) initial vertex
    2) a vertex you've already seen (if ``revisit_non_initial_node=False``)
    3) longer length than desired

    Args:
        A (torch.Tensor): The adjacency matrix.
        path (list): The path visited so far.
        all_paths (list): List keeping track of the cycles.
        logger: The logger.
        max_length (int): The maximum cycle length.
        start_time (float): The start time of the process.
        max_time (float): The maximum time of the process.
        revisit_non_initial_node (boolean): If ``True``, nodes that don't
            correspond to the initial nodes of a given path can be repeated.
            If ``False``, whenever a certain non-initial node is encountered
            twice, the path is not counted as a cycle.

    Returns:
        (....): Tuple containing:

        - **count**: The cycles count.
        - **cycles**: The list with the nodes in the cycles.
    """
    if max_length == -1:
        max_length = A.shape[0]
    if len(path) > A.shape[0] or len(path) > max_length:
        return 0, all_paths

    count = 0
    child_nodes = torch.where(A[:, path[-1]] == 1)[0]
    for i in child_nodes:
        if max_time is not None and start_time is not None:
            if time.process_time() - start_time > max_time:
                break
        # If the cycle is completed, add a cycle count.
        if i == path[0]:
            count += 1
            cycle = [pp for pp in path]
            cycle.append(i)
            all_paths.append(cycle)
        else:
            # If the vertex was already visited, don't explore this tree.
            if i in path and not revisit_non_initial_node:
                continue
            # Else, increase the path and explore the tree.
            else:    
                path_aux = [pp for pp in path]
                path_aux.append(i)
                cc, all_paths_aux = count_cycles(A, path_aux, logger,
                                                 max_length=max_length,
                                                 all_paths=[], 
                                                 start_time=start_time,
                                                 max_time=max_time)
                count += cc
                all_paths.extend(all_paths_aux)

    return count, all_paths

def compute_number_closed_walks(A, logger, N=None, apply_correction=True,
            fc=False, tuples_lookup_filename='tuples_summing_to.pickle',
            dirname='combinatorics', return_walks=False, max_time=60.):
    r"""Compute the number of closed walks up to a given length.

    Because we are looking for closed walks, we only look at the diagonal of the
    different powers of the adjacency matrix :math:`A`. Recall that the power
    :math:`n` of the adjacency matrix has the property that the element
    :math:`a^n_ij` determines the number of walks from neuron :math:`i` to
    neuron :math:`j` that have length :math:`n`.

    When the corrective term is applied, the fact that smaller length closed
    walks can be accounted for several times is corrected. For example, when
    counting the number of closed walks of length 6, this will not only count
    cycles (i.e. only initial and final vertex is repeated), but also closed
    walks where more than one vertices are repeated (for example, it can find
    four different closed walks built from two different closed walks of length
    3). Therefore, this number of possible combinations will be subtracted. It
    is however, still unclear whether this step is necessary to properly
    quantify recurrency or not.

    We use the formula derived by Christian in
    `this document <https://www.overleaf.com/project/60c0cf0fd9c392483e3f34c1>`_.

    Args:
        A (torch.tensor): The adjacency matrix. We assume that A_ij corresponds
            to neuron i connecting to neuron j, i.e. rows indicate pre-synaptic
            neurons and columns post-synaptic neurons.
        logger: The logger.
        N (int): The maximal length of the walk to be considered. If None, the
            dimensionality of the matrix :math:`A` will be used, i.e. the number
            of neurons.
        apply_correction (bool, optional): If True, a corrective term will be
            applied to the computation of the total number of closed walks such
            that smaller length ones are not accounted for several times.
        fc (bool, optional): Whether a fully-connected network should be
            considered. If True, the topology in the adjancecy matrix A is
            ignored.
        tuples_lookup_filename (str): The name of the file where to store the
            tuples lookup table.
        dirname (str): The name of the file where to store the files.
        return_walks (boolean, optional): If ``True``, the total number of walks
            of any given length is also returned.
        max_time (float): The maximum time of the process.

    Returns:
        (....): Tuple containing:

        - **num_cycles**: The number of cycles.
        - **num_cycles_of_len**: The number of cycles of a given length. Each
            element ``num_cycles_of_len[i]`` corresponds to the number of cycles
            of length ``i+1``.
        - **num_walks_of_len**: Analogous to ``num_cycles_of_len`` but for open
            and closed walks. Only returned if ``return_walks`` is ``True``.
    """
    start = time.process_time()

    if N is None:
        N = A.shape[0]

    # Filenames where to store computed things.
    tuples_lookup_path = os.path.join(dirname, tuples_lookup_filename)

    # If files have already been stored, load to save computation.
    if os.path.exists(tuples_lookup_path):
        with open(tuples_lookup_path, 'rb') as f:
            tuples_lookup = pickle.load(f)
        max_tuple_stored = np.max([int(key) for key in tuples_lookup.keys()])
    else:
        tuples_lookup = dict()
        max_tuple_stored = 0
    corrected_lookup = dict()

    # Initialize for power of 1.
    if fc:
        A = torch.ones_like(A) # fully connected network

    # Compute the powers of A and accumulate the number of closed walks, as well
    # as the total number of walks of a given length (closed and not).
    cycles = []
    walks = []
    for n in range(1, N+1):
        if time.process_time() - start > max_time:
            cycles.append(np.nan)
            walks.append(np.nan)
            continue
        if n == 1:
            A_n = torch.matrix_power(A, 1)
        else:
            A_n = torch.mm(A_n, A)
        if apply_correction:
            corrected_cycles, corrected_lookup, tuples_lookup = \
                get_correction_term(n, logger, A_n=A_n, 
                                    corrected_lookup=corrected_lookup, 
                                    tuples_lookup=tuples_lookup,
                                    start_time=start, max_time=max_time)
            cycles.append(corrected_cycles.sum().item())
            walk = None
            if return_walks:
                raise NotImplementedError
                # walk = ...
            walks.append(walk)
        else:
            cycles.append(torch.diag(A_n).sum().item())
            walks.append(A_n.sum().item())
        #assert cycles[-1] >= 0
    if max_time is not None and start is not None:
        if time.process_time() - start > max_time:
            logger.debug('Maxtime in "compute_number_closed_walks". ' +
                         'Breaking.')

    # Store a pickle file with the set of tuples summing to a value if it
    # contains more information than the previously stored one.
    max_tuple = np.max([int(key) for key in tuples_lookup.keys()])
    if max_tuple > max_tuple_stored:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(tuples_lookup_path, 'wb') as f:
            pickle.dump(tuples_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

    assert len(cycles) == N
    assert len(walks) == N

    if return_walks:
        return np.sum(cycles), cycles, walks
    else:
        return np.sum(cycles), cycles


def get_correction_term(n, logger, A_n=None, corrected_lookup=None,
                        tuples_lookup=None, start_time=None, max_time=None):
    r"""For a given adjacency matrix and cycle length, compute the
    corrective term that has to be applied for the cycle counts for each neuron.

    This function compues the following quantity for each neuron :math:`i`:
    
    .. math::

        \hat{a_i^{(n)}} = diag(A^n)_i - c_i^{(n)}

    Where :math:`c_i^{(n)}` is computed according to:

    .. math::

        c_i^{(n)} = \sum_{S^{(n)} \in \Sigma^{(n)}} 
            \frac{\mid S^{(n)} \mid !}{\prod_{c \in count(S^{(n)})c!}}
            \prod_{s \in S^{(n)} \hat{a}_i^{(s)}

    Args:
        n (int): The length of the cycles.
        logger: The logger.
        A_n (torch.Tensor): The adjacency matrix to the power of ``n``.
        corrected_lookup (dict): A dictionary containing, for each cycle
            length, the correction to apply for the number of cycles for `A`. 
        tuples_lookup (dict): A lookup table with tuples that sum up to a given
            integer value.
        start_time (float): The start time of the process.
        max_time (float): The maximum time of the process.

    Returns:
        (list): The list of correction terms for each neuron.
    """
    if corrected_lookup == None:
        corrected_lookup = dict()
    if tuples_lookup == None:
        tuples_lookup = dict()

    # Initial recursion state.
    if str(n) in corrected_lookup.keys():
        return corrected_lookup[str(n)], corrected_lookup, tuples_lookup
    elif n == 1:
        corrected_lookup[str(1)] = torch.diag(A_n)
        return torch.diag(A_n), corrected_lookup, tuples_lookup
    else:
        assert A_n is not None

    # Get all possible tuples of integers that sum up to n.
    # This corresponds to $\Sigma^{(n)}$.
    sigma_n, tuples_lookup = get_tuples_summing_to(n, lookup=tuples_lookup)

    N = A_n.shape[0]
    c = torch.zeros(N, device=A_n.device)
    for i in range(N):
        c_i = 0
        for s_n in sigma_n:
            s_n_permutations = get_number_permutations(s_n)
            aux = s_n_permutations
            for s in s_n:
                c_s, corrected_lookup, tuples_lookup = get_correction_term(
                        s, logger, corrected_lookup=corrected_lookup, 
                        tuples_lookup=tuples_lookup,
                        start_time=start_time, max_time=max_time)
                aux *=  c_s[i]
                #assert c_s[i] >= 0
                if max_time is not None and start_time is not None:
                    if time.process_time() - start_time > max_time:
                        break
            c_i += aux
            if max_time is not None and start_time is not None:
                if time.process_time() - start_time > max_time:
                    break

        if max_time is not None and start_time is not None:
            if time.process_time() - start_time > max_time:
                break
        c[i] = c_i
        #assert c_i >= 0

    corrected_cycles = torch.diagonal(A_n) - c
    corrected_lookup[str(n)] = corrected_cycles

    return corrected_cycles, corrected_lookup, tuples_lookup

def store_tuples_summing_to(n, filename = 'tuples_summing_to.pickle', 
                            dirname='combinatorics'):
    """Get set of tuples summing all the way up to n, and store as pickle.

    This functions is meant to be run before running the actual counting
    algorithm, such that the correction for the number of subcycles that add
    up to a cycle of length n is easily computed.

    Args:
        n (int): The number up to which to compute the tuples summing up to it.
        filename (str): The name of the pickle file where to store the tuples.
        dirname (str): The directory where to store the pickle.
    """

    tuples_lookup_path = os.path.join(dirname, filename)

    # Check if files have already been stored, and if yes, load to save
    # computation.
    if os.path.exists(tuples_lookup_path):
        with open(tuples_lookup_path, 'rb') as f:
            lookup = pickle.load(f)
        max_tuple_stored = np.max([int(key) for key in lookup.keys()])
    else:
        lookup = dict()
        max_tuple_stored = 0

    tuples, lookup = get_tuples_summing_to(n, lookup=lookup)

    if n > max_tuple_stored:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(tuples_lookup_path, 'wb') as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_tuples_summing_to(n, lookup=None):
    """Get a list of all tuples of integers summing up to a given integer.

    Args:
        n (int): The value to which they should sum up.
        lookup (dict): A lookup table to check tuples that sum up to a given
            value. Can be used to sidestep computing too much recursion.

    Returns:
        (....): Tuple including:
        
        - **tuples**: The list of all tuples, without duplicates, and without
            including different possible permutations.
        - **lookup**: The lookup table with computed values.
    """
    if lookup is None:
        lookup = dict()
    elif lookup is not None and str(n) in lookup.keys():
        return lookup[str(n)], lookup

    # Initial condition for the recursion.
    if n == 1:
        return [[1]], {'1': np.array([1])}

    def tuple_in_list(tp, lst):
        """Check if a tuple is present in a given list.

        Args:
            tp (tuple): The tuple.
            lst (list): The list.

        Returns:
            (boolean): Whether it is present or not.
        """
        for el in lst:
            if np.array_equal(el, tp):
                return True
        return False

    # Recursion
    tuples = []
    for i in range(1, int(np.floor(n/2)+1)):
        new_tuple = np.array(np.sort([i, n-i]))
        if not tuple_in_list(new_tuple, tuples):
            tuples.append(new_tuple)
        tuples_n_i, lookup = get_tuples_summing_to(n-i, lookup)
        for tp in tuples_n_i:
            new_tuple = list(tp)
            new_tuple.append(i)
            new_tuple = np.sort(new_tuple)
            if not tuple_in_list(new_tuple, tuples):
                tuples.append(new_tuple)

    lookup[str(n)] = tuples

    ### Sanity check. 
    # Use stars and bars theorem 1 to verify results.
    # https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
    total_tuples = 0
    for k in range(2, n+1):
        total_tuples += binomial_coefficient(n-1, k-1)
    # Compute number of tuples WITH permutations.
    perm_tuples = 0
    for tp in tuples:
        perm_tuples += get_number_permutations(tp)
    assert perm_tuples == total_tuples

    return tuples, lookup

def binomial_coefficient(n, k):
    r"""Compute the binomial coefficient.

    This function computes:

    .. math::

        \frac{n!}{k! (n-k)!}
    """
    return np.math.factorial(n) / (np.math.factorial(k) * \
            np.math.factorial(n-k))

def get_number_permutations(tup):
    r"""Get the number of possible permutations of a given tuple of integers.

    This is computed according to:

    .. math::

        \frac{\mid tup \mid !}{\prod_{c \in count(tup)}c!} 

    Args:
        tup (tuple): The tuple of interest.

    Returns:
        (int): The number of possible permutations.
    """
    k = len(tup)

    _, counts = np.unique(tup, return_counts=True)

    perm = np.math.factorial(k)
    for c in counts:
        perm /= np.math.factorial(c)
    assert (perm - int(perm)) == 0

    return int(perm)


def analyze_feedforwardness(net, config, logger, teacher=False):
    """Analyze the feedforwardness of the network.

    Args:
        net: The network.
        config: The configuration.
        logger: The logger.
        teacher (bool, optional): Whether the provided network is a teacher.

    Returns:
        (dict): A dictionary containing the computed statistics.
    """
    msg = '' if not teacher else ' of the teacher'
    logger.info('Computing feedforwardness statistics' + msg + '...')

    if len(net.masks_rec_hh) > 1:
        logger.debug('Only the first recurrent layer is being taken into ' +
                      'account for the computation of the feedforwardness.')

    metrics = dict()

    ### Compute directedness according to the HBP metric.
    metrics['feedforwardness'] = compute_directedness(net.masks_rec_hh[0], \
                                                      logger)

    ### Compute number of cycles recursively.
    cycles_recursive = [None, None]
    if not config.dont_use_recursion:
        cycles_recursive = iteratively_compute_number_cycles(\
                                        net.masks_rec_hh[0], logger,
                                        max_time=config.feedforwardness_maxtime,
                                        correct_multiple_counts=False)
    metrics['cycles_recursive'] = float(cycles_recursive[0])
    metrics['cycles_recursive_list'] = list(cycles_recursive[1])
    # Correct for the cycle length.
    corrected_cycles_list = [n / (i + 1) for i, n in \
                                    enumerate(metrics['cycles_recursive_list'])]
    metrics['cycles_recursive_corr'] = float(np.sum(corrected_cycles_list))
    metrics['cycles_recursive_corr_list'] = list(corrected_cycles_list)

    ### Compute number of cycles with adjacency matrix.
    cycles_adjacency_corr = [None, None]
    cycles_adjacency = [None, None]
    ratio_cycles = [None, None]
    if not config.dont_use_adjacency:
        combinatorics_dirname = 'combinatorics'
        if config.hpsearch:
            combinatorics_dirname = '../../../../feedforwardness/combinatorics'

        # Compute the number of cycles with correction.
        cycles_adjacency_corr = compute_number_closed_walks(net.masks_rec_hh[0],
                                logger, max_time=config.feedforwardness_maxtime,
                                dirname=combinatorics_dirname)

        # Compute the number of cycles without correction.
        cycles_adjacency = compute_number_closed_walks(net.masks_rec_hh[0],
                                logger, max_time=config.feedforwardness_maxtime,
                                dirname=combinatorics_dirname,
                                apply_correction=False,
                                return_walks=True)

        # Also compute the ratio of cycles vs. walks. This can only be computed
        # without correction since computing the number of walks is not
        # implemented using the correction.
        ratio_cycles[0] = float(cycles_adjacency[0] / \
                                   np.sum(cycles_adjacency[2]))
        ratio_cycles[1] = [float(cycles_adjacency[1][i]/cycles_adjacency[2][i])\
                        if cycles_adjacency[2][i] != 0 else 0 \
                        for i in range(len(cycles_adjacency[1]))]

    metrics['cycles_adjacency_corr'] = float(cycles_adjacency_corr[0])
    metrics['cycles_adjacency_corr_list'] = list(cycles_adjacency_corr[1])
    metrics['cycles_adjacency'] = float(cycles_adjacency[0])
    metrics['cycles_adjacency_list'] = list(cycles_adjacency[1])
    metrics['ratio_cycles_adjacency'] = float(ratio_cycles[0])
    metrics['ratio_cycles_adjacency_list'] = list(ratio_cycles[1])

    logger.info('Computing feedforwardness statistics' + msg + '... Done.')

    return metrics