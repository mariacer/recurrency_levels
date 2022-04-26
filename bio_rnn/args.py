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
# @title          :bio_rnn/args.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.7
"""
Command line arguments for Microcircuit experiments
"""

def microcircuit_args(parser, net='main', dy_length_microcircuit=None,
             dxz_length_microcircuit=None, dscale_connectivity_prob=-1,
             dnb_clusters=10, dwithin_cluster_prob_scaling=1.5,
             doutside_cluster_prob_scaling=0.5):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options whenever dealing with microcircuits.

    Arguments specified in this function:
        - `random_microcircuit_connectivity`
        - `y_length_microcircuit`
        - `xz_length_microcircuit`
        - `scale_connectivity_prob`
        - `use_clusters`
        - `nb_clusters`
        - `within_cluster_prob_scaling`
        - `outside_cluster_prob_scaling`
        - `use_full_connectivity`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        net (str, optional): The network that is being defined. Either `main`
            or `teacher`.
        dy_length_microcircuit: Default value of option 
            `y_length_microcircuit`.
        dxz_length_microcircuit: Default value of option 
            `xz_length_microcircuit`.
        dnb_clusters: Default value of option `nb_clusters`.
        dwithin_cluster_prob_scaling: Default value of option 
            `within_cluster_prob_scaling`.
        doutside_cluster_prob_scaling: Default value of option 
            `outside_cluster_prob_scaling`.

    Returns:
        The created argument group, in case more options should be added.
    """
    heading = 'Microcircuit network options'

    pfx = ''
    sfx = ''
    if net == 'teacher':
        pfx = 'teacher_'
        sfx = ' in the teacher'

    mgroup = parser.add_argument_group(heading)
    mgroup.add_argument('--%srandom_microcircuit_connectivity',
                        action='store_true',
                        help='Use a uniform connectivity matrix ' + sfx + 
                             ' (resulting in random rnn connectivity) instead ' 
                             'of the anatomical gradients experimentally '
                             'observed.')
    mgroup.add_argument('--%suse_full_connectivity'%pfx, action='store_true',
                        help='If True, the anatomical data will be overriden '
                             'and the generated microcircuit will have full '
                             'connectivity.')
    mgroup.add_argument('--%sy_length_microcircuit'%pfx,
                        type=float, default=dy_length_microcircuit,
                        help='The length (in micron) along the y-axis of the '
                             'brain volume containing the neuronal ' 
                             'microcircuit. This information is needed for ' 
                             'determining the connectivity probabilities '+sfx+
                             ' from experimental anatomical data. If '
                             'unspecified, the number '
                             'of rnn neurons (rnn_arch) will be used together'
                             'with the average neuron density to calculate'
                             'the dimensions of a cube containing rnn_arch '
                             'neurons on average.')
    mgroup.add_argument('--%sxz_length_microcircuit'%pfx, type=float,
                        default=dxz_length_microcircuit,
                        help='The length (in micron) '
                             'along the x-axis and the z-axis '
                             'of the brain volume '
                             'containing the neuronal microcircuit. This'
                             'information is needed for determining the '
                             'connectivity probabilities '+ sfx + 'from '
                             'experimental anatomical data. If unspecified, '
                             'the number of rnn neurons (rnn_arch) will be used '
                             'together with the average neuron density to '
                             'calculate the dimensions of a cube containing '
                             'rnn_arch neurons on average.')
    mgroup.add_argument('--%sscale_connectivity_prob'%pfx, type=float,
                        default=dscale_connectivity_prob,
                        help='When > 0, the anatomical recurrent connectivity '
                             'probabilities ' + sfx + 'will be scaled by this '
                             'factor, to influence the amount of connections '
                             'the microcircuit will have.')
    mgroup.add_argument('--%suse_clusters'%pfx, action='store_true',
                        help='Flag indicating whether the clustered approach'
                             'should be used to generate the connectivity'
                             'matrix of the microcircuit' + sfx + '.')
    mgroup.add_argument('--%snb_clusters'%pfx, type=int, default=dnb_clusters,
                        help='Number of clusters to be used for generating '
                             'the connectivity matrix of the microcircuit' +
                             sfx + '.')
    mgroup.add_argument('--%swithin_cluster_prob_scaling'%pfx, type=float,
                        default=dwithin_cluster_prob_scaling,
                        help='If use_clusters is True, the connection '
                             'probability of neurons within the same cluster'
                             '(randomly assigned)' + idx + ' will be scaled by '
                             'this value.')
    mgroup.add_argument('--%soutside_cluster_prob_scaling'%pfx, type=float,
                        default=doutside_cluster_prob_scaling,
                        help='If use_clusters is True, the connection '
                             'probability of neurons that are not in the '
                             'same cluster' + sfx +
                             '(randomly assigned) will be scaled by this '
                             'value.')

    return mgroup

def check_invalid_args(config):
    """Sanity check for command-line arguments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    if config.rec_sparsity != 1. and \
            config.scale_connectivity_prob > 0:
        raise ValueError('The fraction of recurrent connections and ' + 
                         'the scaling of the connection probabilities ' +
                         'cannot be provided simultaneously.')