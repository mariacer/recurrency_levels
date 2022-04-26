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
# @title          :feedforwardness/feedforwardness_study.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Study of feedforwardness in small clusters of a larger network.
---------------------------------------------------------------

In this script, we generate large networks and perform an artificial
"patch-clamping" experiment, i.e. we randomly select small clusters of neurons
and compute the feedforwardness within these clusters. We do this as a control
to verify that strong feedforwardness in small clusters tells us something
about the feedforwardness of the entire network.
"""
import __init__

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from numpy.random import seed
import os
import pandas as pd
from scipy.stats import pearsonr
import torch
import time

from feedforwardness import utils
from utils import misc

def generate_random_adjacency_matrix(n, p, num_conn=None,
                                     self_connections=True):
    """Generate a random adjacency matrix.

    Args:
        n (int): Number of neurons in the recurrent layer.
        p (float): Connection probability (in the 0-1 range).
        num_conn (int or None): The number of connections desired. If provided,
            it overrides the value given by `p`.
        self_connections (bool, optional): Whether self connections are allowed.

    Returns:
        (torch.Tensor): The adjacency matrix.
    """
    # Generate the adjacency matrix of the overall network.
    A_aux = torch.rand((n, n))
    A = torch.zeros_like(A_aux)

    if num_conn is not None:
        while A.sum() < num_conn:
            # Chose some random pre-synaptic and post-synaptic neurons.
            pre_idx = torch.randint(0, n, (1,))
            post_idx = torch.randint(0, n, (1,))

            # If the proposed connection doesn't exist, add it.
            if A[pre_idx, post_idx] == 0:
                if pre_idx == post_idx and not self_connections:
                    continue
                else:
                    A[pre_idx, post_idx] = 1.

        assert A.sum() == num_conn
    else:
        if self_connections:
            A[A_aux <= p] = 1
        else:
            # We need to correct p to account for the fact that no 
            # self-connections can exist. We have p = x/(n**2) where x is the 
            # number of off-diagonal connections. Then we need to construct the
            # random matrix using: p' = x/(n**2 - n) = pn/(n-1)
            p = p*n / (n-1)
            A[A_aux <= p] = 1
            # Delete self-connections.
            A[range(n), range(n)] = 0

    assert self_connections or torch.diag(A).sum() == 0
    assert np.all(np.equal(np.unique(A), np.array([0., 1.]))) or \
        np.all(np.equal(np.unique(A), np.array([0.]))) or \
        np.all(np.equal(np.unique(A), np.array([1.])))

    return A

def analyze_network(n=1000, p=0.8, n_clusters=200, size_clusters=10,
                    self_connections=False):
    """Analyze one network.

    Args:
        n (int): Number of neurons in the recurrent layer.
        p (float): Connection probability (in the 0-1 range).
        n_clusters (int): Number of clusters to analyze.
        size_clusters (int): The size of the clusters.
        self_connections (bool): If True, self connections are allowed.
    """
    A = generate_random_adjacency_matrix(n, p, 
        self_connections=self_connections)

    ### Analyze randomly selected clusers.
    cluster_dir = []
    cluster_p = []
    cluster_self = []
    for i in range(n_clusters):
        neuron_ids = torch.randperm(n)[:size_clusters]
        A_cluster = A[neuron_ids, :]
        A_cluster = A_cluster[:, neuron_ids]
        cluster_dir.append(utils.compute_directedness(A_cluster))
        cluster_p.append(A_cluster.sum()/A_cluster.numel())
        cluster_self.append(torch.diag(A_cluster).sum())

    # When not allowing self-connections, the effective connection probability
    # of subsampled clusters is smaller than that of the original matrix, since
    # this one was constructed by using a surrogate connection probability that
    # takes into account the matrix size. Thus by changing the matrix size, the
    # effective connection probability changes.
    # In other words, we used a surrogate probability p' that depends on n:
    # p' = x/(n**2 - n) = p*n/(n-1)
    # and the lower the n, the lower this probability will be.
    # Therefore, for the case where no self-connections are applied, we
    # generate the random clusters according to the effective connectivity of
    # the subsamples given by:
    # p'' = p' * (ns**2 - ns) / (ns**2) = p' * (ns - 1) / ns
    # where ns is the size of the subclusters.
    if not self_connections:
        p_small = p * n / (n - 1)
        p = p_small * (size_clusters - 1) / size_clusters

    ### Compare to randomly generated small clusters.
    random_cluster_dir = []
    random_cluster_p = []
    random_cluster_self = []
    for i in range(n_clusters):
        A_cluster = generate_random_adjacency_matrix(size_clusters, p, 
            self_connections=self_connections)
        random_cluster_dir.append(utils.compute_directedness(A_cluster))
        random_cluster_p.append(A_cluster.sum()/A_cluster.numel())
        random_cluster_self.append(torch.diag(A_cluster).sum())

    ### Plot the different distributions.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(cluster_dir, color='r', label='subcluster', alpha=0.5)
    ax1.hist(random_cluster_dir, color='b', label='random cluster', alpha=0.5)
    ax1.set_xlabel('cluster directedness')
    ax2.hist(cluster_p, color='r', label='subcluster', alpha=0.5)
    ax2.hist(random_cluster_p, color='b', label='random cluster', alpha=0.5)
    ax2.set_xlabel('cluster connection density')
    ax3.hist(cluster_self, color='r', label='subcluster', alpha=0.5)
    ax3.hist(random_cluster_self, color='b', label='random cluster', alpha=0.5)
    ax3.set_xlabel('num self connections')
    ax3.legend()

    print('Subampled cluster p: %.2f, Random cluster p: %.2f'% \
        (np.mean(cluster_p), np.mean(random_cluster_p)))

def compare_large_vs_small_clusters(n_large=1000, n_small=10, p=0.8, 
                                    n_clusters=200, self_connections=False):
    """Compare properties of large vs small random clusters.

    Args:
        n_large (int): The size of the large clusters.
        n_small (int): The size of the small clusters.
        (....): See docstring of function :func:`analyze_network`.
    """

    p_large_cluster = []
    self_large_cluster = []
    large_directedness = []
    for i in range(n_clusters):
        A = generate_random_adjacency_matrix(n_large, p, 
            self_connections=self_connections)
        p_large_cluster.append(A.sum()/A.numel())
        self_large_cluster.append(torch.diag(A).sum())
        large_directedness.append(utils.compute_directedness(A))

    p_small_cluster = []
    self_small_cluster = []
    small_directedness = []
    for i in range(n_clusters):
        A = generate_random_adjacency_matrix(n_small, p, 
            self_connections=self_connections)
        p_small_cluster.append(A.sum()/A.numel())
        self_small_cluster.append(torch.diag(A).sum())
        small_directedness.append(utils.compute_directedness(A))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(small_directedness, color='r', label='small', alpha=0.5)
    ax1.hist(large_directedness, color='b', label='large', alpha=0.5)
    ax1.set_xlabel('cluster directedness')
    ax2.hist(p_small_cluster, color='r', label='small', alpha=0.5)
    ax2.hist(p_large_cluster, color='b', label='large', alpha=0.5)
    ax2.set_xlabel('cluster connection density')
    ax3.hist(self_small_cluster, color='r', label='small', alpha=0.5)
    ax3.hist(self_large_cluster, color='b', label='large', alpha=0.5)
    ax3.set_xlabel('num self connections')
    ax3.legend()
    print('Large p: %.2f, Small p: %.2f'% \
        (np.mean(p_large_cluster), np.mean(p_small_cluster)))

def compare_feedforwardness_measures(n_values, n_nets=10, p_range=[0.3], 
                                     self_connections=False, path='out/',
                                     filename='feedforwardness_comparison.csv'):
    """Compare feedforwardness measures across a range of network sizes.

    Args:
        n_values (np.array): The network sizes to explore.
        n_nets (int): Number of networks to explore for each size.
        p (list): The connection probabilities to explore.
    """
    all_results = dict()
    all_results['size'] = []
    all_results['p'] = []
    all_results['connections'] = []
    all_results['directedness'] = []
    all_results['directedness normalized'] = []
    all_results['directedness time'] = []
    all_results['cycles (adjacency)'] = []
    all_results['cycles (adjacency) per length'] = []
    all_results['walks (adjacency) per length'] = []
    all_results['ratio cycles per length'] = []
    all_results['cycles (adjacency) normalized'] = []
    all_results['cycles (adjacency) time'] = []
    all_results['cycles (iterative)'] = []
    all_results['cycles (iterative) normalized'] = []
    all_results['cycles (iterative) time'] = []

    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        print('Loading existing .csv file.')
        df_old = pd.read_csv(filepath)
    else:
        df_old = pd.DataFrame()

    for p in p_range:
        for n in n_values:
            print('Exploring networks of size %i and connectivity %.2f' %(n, p))
            for i in range(n_nets):
                print('    Network %i/%i'% (i + 1, n_nets))
                results = compute_feedforwardness_measures(n, p, 
                        self_connections=self_connections)
                for key in results.keys():
                    all_results[key].append(results[key])
                all_results['size'].append(n)
                all_results['p'].append(p)

                # Append to old df.
                df = pd.DataFrame.from_dict(all_results)
                df.set_index('size')
                df = pd.concat([df_old, df])

                # Store csv in file.
                if not os.path.exists(path):
                    os.mkdir(path)
                df.to_csv(index=False, path_or_buf=os.path.join(path, \
                    'feedforwardness_comparison.csv'))

            # Plot results.
            fig, ax = plt.subplots(1, 3, figsize=[14, 4.8])

            directedness = [val for i, val in enumerate(df['directedness'\
                ].tolist()) if df['size'].tolist()[i] == n and df['p'\
                ].tolist()[i] == p]
            cycles_adjacency = [val for i, val in enumerate(df['cycles ' + \
                '(adjacency)'\
                ].tolist()) if df['size'].tolist()[i] == n and df['p'\
                ].tolist()[i] == p]
            cycles_recursive = [val for i, val in enumerate(df['cycles ' + \
                '(iterative)'\
                ].tolist()) if df['size'].tolist()[i] == n and df['p'\
                ].tolist()[i] == p]

            ax[0].scatter(directedness, cycles_adjacency)
            ax[0].set_xlabel('directedness')
            ax[0].set_ylabel('cycles (adjacency)')

            ax[1].scatter(directedness, cycles_recursive)
            ax[1].set_xlabel('directedness')
            ax[1].set_ylabel('cycles (iterative)')

            ax[2].scatter(cycles_adjacency, cycles_recursive)
            ax[2].set_xlabel('cycles (adjacency)')
            ax[2].set_ylabel('cycles (iterative)')

            if not os.path.exists('out'):
                os.mkdir('out')
            plt.savefig('out/ff_comparison_net_size_' + str(n) + '_conn_' + \
                        str(p) + '.png')

def plot_feedforwardness_measures_across_resources(path='out/',
                                     filename='feedforwardness_comparison.csv',
                                     time=False, plot_heatmaps=True):
    """Plot heatmaps of the feedforwardness measures for different resources.

    Here, resources means number of connections and number of neurons.

    Args:
        (....): See docstring of function
            :func:`compare_feedforwardness_measures`.
        time (bool, optional): Whether to plot the computational time or not.
        plot_heatmaps (bool, optional): Whether to plot heatmaps.
    """

    ### Load the .csv file.
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        print('Loading existing .csv file.')
        df = pd.read_csv(filepath)
    else:
        raise ValueError('The file %s doesnt exist.' % filepath)

    ### Construct data matrices.
    sizes = np.sort(np.unique(df['size'].tolist()))
    ps = np.sort(np.unique(df['p'].tolist()))

    msg = ' time' if time else ''

    # Directedness.
    directedness = np.zeros((len(sizes), len(ps)))
    directedness_n = np.zeros((len(sizes), len(ps)))
    cycles_rec = np.zeros((len(sizes), len(ps)))
    cycles_adj = np.zeros((len(sizes), len(ps)))
    directedness_std = np.zeros((len(sizes), len(ps)))
    directedness_n_std = np.zeros((len(sizes), len(ps)))
    cycles_rec_std = np.zeros((len(sizes), len(ps)))
    cycles_adj_std = np.zeros((len(sizes), len(ps)))
    all_cycle_ratios = {}
    for i, s in enumerate(sizes):
        all_cycle_ratios[str(s)] = {}
        for j, p in enumerate(ps):
            all_cycle_ratios[str(s)][str(p)] = {}
            dirct = df[(df['size'] == s) & (df['p'] == p)]['directedness' + msg]
            cyrec = df[(df['size'] == s) & (df['p'] == p)]['cycles ' + \
                '(iterative)' + msg]
            cyadj = df[(df['size'] == s) & (df['p'] == p)]['cycles ' + \
                '(adjacency)' + msg]
            clratio = df[(df['size'] == s) & (df['p'] == p)]['ratio cycles ' + \
                'per length']
            clratio = np.array(clratio)
            directedness[i, j] = np.mean(dirct)
            cycles_rec[i, j] = np.mean(cyrec)
            cycles_adj[i, j] = np.mean(cyadj)
            directedness_std[i, j] = np.std(dirct)
            cycles_rec_std[i, j] = np.std(cyrec)
            cycles_adj_std[i, j] = np.std(cyadj)

            clratio_aux = np.zeros((len(clratio), s))
            for ii in range(len(clratio)):
                clratio_aux[ii, :] = misc.str_to_float_or_list(clratio[ii])
            all_cycle_ratios[str(s)][str(p)]['mean'] = \
                np.nanmean(clratio_aux, axis=0)
            all_cycle_ratios[str(s)][str(p)]['std'] = \
                np.nanstd(clratio_aux, axis=0)

            if not time:
                dirct_n = df[(df['size'] == s) & (df['p'] == p)][\
                    'directedness normalized' + msg]
            directedness_n[i, j] = np.mean(dirct_n)
            directedness_n_std[i, j] = np.std(dirct_n)

    def plot_mat(mean_matrix, std_matrix, metric_name='', time=False):
        msg = ' time' if time else ''

        fig, ax = plt.subplots(1,2, figsize=(12, 5))
        im0 = ax[0].imshow(mean_matrix.T)
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(std_matrix.T)
        fig.colorbar(im1, ax=ax[1])

        ax[0].set_ylabel('connectivity')
        ax[0].set_title(metric_name + msg + ' mean')
        ax[0].set_xticks(list(ax[0].get_xticks()[1:-1]))
        ax[0].set_yticks(list(ax[0].get_yticks()[1:-1]))
        ax[0].set_yticklabels(ps)
        ax[0].set_xticklabels(sizes)
        ax[0].set_xlabel('size')

        ax[1].set_title(metric_name + msg + ' std')
        ax[1].set_xticks(list(ax[1].get_xticks()[1:-1]))
        ax[1].set_yticks(list(ax[1].get_yticks()[1:-1]))
        ax[1].set_yticklabels(ps)
        ax[1].set_xticklabels(sizes)
        ax[1].set_xlabel('size')

        fig.savefig(os.path.join(path, msg + '%s_resource_range.png' % \
                                 metric_name))

    if plot_heatmaps:
        plot_mat(directedness, directedness_std, metric_name='directedness', \
                                                 time=time)
        if not time:
            # Time should be quite identical for both directedness measures.
            plot_mat(directedness_n, directedness_n_std, \
                metric_name='directedness normalized')
        plot_mat(cycles_rec, cycles_rec_std, metric_name='cycles (recursive)',\
                                             time=time)
        plot_mat(cycles_adj, cycles_adj_std, metric_name='cycles (adjacency)',\
                                             time=time)

    # Plot cycles per length.
    cmapp = plt.get_cmap('Oranges', len(ps)+1)
    color = [cmapp(i/len(ps)) for i in range(1, len(ps)+1)]
    for i, s in enumerate(sizes):
        plt.figure()
        for j, p in enumerate(ps):
            xrang = range(1, len(all_cycle_ratios[str(s)][str(p)]['mean']) + 1)

            plt.plot(xrang, all_cycle_ratios[str(s)][str(p)]['mean'], 
                label='p = %.2f' % p, color=color[j])
            plt.fill_between(xrang, \
                all_cycle_ratios[str(s)][str(p)]['mean']- \
                    all_cycle_ratios[str(s)][str(p)]['std'],
                all_cycle_ratios[str(s)][str(p)]['mean']+ \
                    all_cycle_ratios[str(s)][str(p)]['std'],
                alpha=0.1, color=color[j])
        plt.plot(xrang, [1/s for _ in xrang], '--k', label='chance')
        plt.xlabel('cycle length')
        plt.ylabel('ratio of cycles per length')
        plt.title('size n = %i' % s)
        plt.legend()
        plt.savefig('out/s%i_p%.2f.png' % (s, p))


def compute_feedforwardness_measures(n=100, p=0.3, self_connections=False,
                                     correct_n_closed_walks=False):
    """Compute different measures of feedforwardness.

    We randomly generate a recurrent layer and compute its feedforwardness
    according to two different metrics: 1) the one used in the Riemann paper
    (see function :func:`networks.net_utils.compute_directedness`) and 2) one
    based on the powers of the adjacency matrix (see function
    :func:`networks.net_utils.compute_number_closed_walks`).
    We then study whether, for networks of same size and same number of
    connections, these normalized measures correlate.

    Args:
        (....): See docstring of function :func:`analyze_network`.
        correct_n_closed_walks (bool, optional): If ``True``, the computation
            of the number of closed walks will be corrected for the fact that
            lower length closed walks can be used to composed walks of longer
            length.
        normalize (bool, optional): Whether to normalize the values by the
            higher one obtained across samples.
        fc_normalize (bool, optional): Whether to normalize the number of closed
            walks by the results obtained for a fully connected network.

    Returns:
        (dict): A dictionary containing feedforwardness measures and runtimes.
    """
    results = dict()

    # To ensure that all networks are comparable, we use the exact same number
    # of connections.
    num_conn = int(n*n*p)
    A = generate_random_adjacency_matrix(n, p, num_conn=num_conn, 
        self_connections=self_connections)
    assert A.sum() == num_conn
    results['connections'] = num_conn 
    A_fc = torch.ones_like(A)

    # Compute directedness values.
    start = time.process_time()
    results['directedness'] = utils.compute_directedness(A)
    results['directedness time'] = time.process_time() - start
    results['directedness normalized'] = results['directedness'] / \
        utils.compute_directedness(A, feedforward=True)

    start = time.process_time()
    results['cycles (adjacency)'], cw_counts, wk_counts = \
        utils.compute_number_closed_walks(A, return_walks=True,
        apply_correction=correct_n_closed_walks)
    results['cycles (adjacency) per length'] = cw_counts
    results['walks (adjacency) per length'] = wk_counts
    results['ratio cycles per length'] = [i/ii for i,ii in zip(cw_counts, wk_counts)]
    results['cycles (adjacency) time'] = time.process_time() - start
    cycles_fc, _ = utils.compute_number_closed_walks(A_fc,\
        apply_correction=correct_n_closed_walks)
    results['cycles (adjacency) normalized'] = results['cycles (adjacency)'] / \
        cycles_fc

    start = time.process_time()
    results['cycles (iterative)'], c_counts = \
        utils.iteratively_compute_number_cycles(A)
    results['cycles (iterative) time'] = time.process_time() - start
    cycles_fc, _ = utils.iteratively_compute_number_cycles(A_fc)
    results['cycles (iterative) normalized'] = results['cycles (iterative)'] / \
        cycles_fc

    return results

def study_feedforwardness_measures(n_range, p_range, n_clusters=10,
                                   self_connections=False, plot=True,
                                   correct_n_closed_walks=False,
                                   normalize=True):
    """Study the feedforwarness measures across a range of parameters.

    Args:
        n_range (range): The range of number of neurons to try,
        p_range (range): The range of connection density to try.
        (....): See docstring of function :func:`analyze_network`.

    Returns:
        (np.array): The matrix with average correlation coefficients.
    """

    m = np.zeros((len(n_range), len(p_range)))
    m_ncw = np.zeros((len(n_range), len(p_range)))
    m_ncw_fc = np.zeros((len(n_range), len(p_range)))
    m_dr = np.zeros((len(n_range), len(p_range)))
    for i, n in enumerate(n_range):
        for j, p in enumerate(p_range):
            R, ncw, dr = compare_feedforwardness_measures(n=int(n), p=p, 
                     n_clusters=n_clusters, self_connections=self_connections,
                     correct_n_closed_walks=correct_n_closed_walks,
                     normalize=normalize)
            m[i, j] = R
            m_ncw[i, j] = ncw
            m_dr[i, j] = dr

            normalize = True
            _, ncw_fc, _ = compare_feedforwardness_measures(n=int(n), p=p, 
                     n_clusters=n_clusters, self_connections=self_connections,
                     correct_n_closed_walks=correct_n_closed_walks,
                     normalize=normalize, fc_normalize=True)
            m_ncw_fc[i, j] = ncw_fc

    if plot:
        # plt.figure()
        # plt.imshow(m)
        # plt.yticks(range(m.shape[0]), n_range)
        # plt.xticks(range(m.shape[1]), p_range)
        # plt.ylabel('number of neurons')
        # plt.xlabel('connection density')
        # plt.title('Pearsons correlation')
        # plt.colorbar()

        plt.figure()
        plt.imshow(m_ncw, norm=LogNorm(vmin=np.min(m_ncw), vmax=np.max(m_ncw)))
        plt.yticks(range(m.shape[0]), n_range)
        plt.xticks(range(m.shape[1]), p_range)
        plt.ylabel('number of neurons')
        plt.xlabel('connection density')
        plt.title('number of closed walks')
        plt.colorbar()

        plt.figure()
        plt.imshow(m_ncw_fc, norm=LogNorm(vmin=np.min(m_ncw_fc), vmax=np.max(m_ncw_fc)))
        plt.yticks(range(m.shape[0]), n_range)
        plt.xticks(range(m.shape[1]), p_range)
        plt.ylabel('number of neurons')
        plt.xlabel('connection density')
        plt.title('ratio of closed walks vs. fc network')
        plt.colorbar()

        # plt.figure()
        # plt.imshow(m_dr)
        # plt.yticks(range(m.shape[0]), n_range)
        # plt.xticks(range(m.shape[1]), p_range)
        # plt.ylabel('number of neurons')
        # plt.xlabel('connection density')
        # plt.title('directedness')
        # plt.colorbar()

    return m

if __name__=='__main__':

    # Define parameters.
    p = 0.3
    n_large = 100  # size of the large network
    n_small = 10   # size of the subnetworks
    self_connections = True  # whether self-connections are allowed
    n_clusters = 1  # number of subnetworks to study
    correct_n_closed_walks = True

    np.random.seed(42)
    torch.manual_seed(32)

    ############################################################################
    ######## Compare obtained density in random large and small clusters #######
    ############################################################################
    # print('Comparing obtained connectivity density in random large and ' +
    #       'small clusters.')
    # compare_large_vs_small_clusters(n_large=n_large, n_small=n_small, p=p, 
    #                                 n_clusters=n_clusters,
    #                                 self_connections=self_connections)

    ############################################################################
    ######### Compare obtained density in large and subsampled clusters ########
    ############################################################################
    # print('\nComparing obtained connectivity density in random large and ' +
    #       'small SUBSAMPLED clusters.')
    # analyze_network(n=n_large, p=p, size_clusters=n_small, 
    #                 n_clusters=n_clusters, self_connections=self_connections)

    ############################################################################
    ############ Compare feedforwardness measures for random networks ##########
    ############################################################################
    # print('\nComparing feedforwardness measures.')
    p_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    n_range = [6, 7, 8, 9]
    compare_feedforwardness_measures(n_range, p_range=p_range, n_nets=5,
                              self_connections=self_connections)

    print('\nPlotting feedforwardness measures.')
    plot_feedforwardness_measures_across_resources(plot_heatmaps=False)
    # plot_feedforwardness_measures_across_resources(time=True)


    ############################################################################
    ############# Study feedforwardness measures for random networks ###########
    ############################################################################
    # study_feedforwardness_measures(np.arange(10, 100, 10), 
    #                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    #                            n_clusters=n_clusters,
    #                            self_connections=self_connections, 
    #                            correct_n_closed_walks=correct_n_closed_walks)
    # study_feedforwardness_measures(np.arange(10, 100, 10), 
    #                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    #                            n_clusters=n_clusters,
    #                            self_connections=self_connections, 
    #                            correct_n_closed_walks=correct_n_closed_walks,
    #                            normalize=False)



    plt.show()
