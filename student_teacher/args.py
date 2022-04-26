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
# @title          :student_teacher/args.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.7
"""
Command line arguments for Student-Teacher experiments
"""
import numpy as np

from utils import misc
from utils import args

def teacher_rnn_args(parser, doutput_size=5, dinput_size=10,
                     dnum_train=1000, dnum_test=100, dnum_val=0, dn_ts_in=100,
                     dn_ts_out=-1):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options to a main network. These arguments are used
    for making a teacher rnn network that can be used in the student-teacher
    regression setting.

    Arguments specified in this function:
        - `teacher_output_size`
        - `teacher_input_size`
        - `teacher_num_train`
        - `teacher_num_test`
        - `teacher_num_val`
        - `teacher_n_ts_in`
        - `teacher_n_ts_out`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        doutput_size: Default value of option `output_size`.
        dinput_size: Default value of option `input_size`.
        dnum_train: Default value of option `num_train`.
        dnum_test: Default value of option `num_test`.
        dnum_val: Default value of option `num_val`.
        dn_ts_in: Default value of option `n_ts_in`.
        dn_ts_out: Default value of option `n_ts_out`.

    Returns:
        The created argument group, in case more options should be added.
    """
    heading = 'Teacher network options'

    agroup = parser.add_argument_group(heading)
    agroup.add_argument('--teacher_output_size', type=int, default=doutput_size,
                        help='Output size of the teacher network.')
    agroup.add_argument('--teacher_input_size', type=int, default=dinput_size,
                        help='Input size of the teacher network.')
    agroup.add_argument('--teacher_num_train', type=int, default=dnum_train,
                        help='number of training samples to be created by the '
                             'teacher network.')
    agroup.add_argument('--teacher_num_test', type=int, default=dnum_test,
                        help='number of testing samples to be created by the '
                             'teacher network.')
    agroup.add_argument('--teacher_num_val', type=int, default=dnum_val,
                        help='number of validation samples to be created by the '
                             'teacher network.')
    agroup.add_argument('--teacher_n_ts_in', type=int, default=dn_ts_in,
                        help='The number of input timesteps')
    agroup.add_argument('--teacher_n_ts_out', type=int, default=dn_ts_out,
                        help='The number of output timesteps. Can be greater'
                             'than ``n_ts_in``. In this case, the inputs at '
                             'time greater than ``teacher_n_ts_in`` will be '
                             'zero. A value equal to ``-1`` means that the '
                             'same number of output steps will be used as '
                             'in the input.')

def student_teacher_training_args(parser):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    arguments related to training in the student-teacher setup.

    Arguments specified in this function:
        - `compute_late_mse`
        - `use_same_resources`
        - `set_identical_topology`
        - `set_same_weight_sign`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    heading = 'Student-teacher training options'

    agroup = parser.add_argument_group(heading)
    agroup.add_argument('--compute_late_mse', action='store_true',
                        help='If activated, the MSE will only be computed ' 
                             'over late timesteps, i.e. from timestep ``n`` '
                             'to timestep ``T``, where ``T`` is the length '
                             'of the output sequences and ``n`` is the '
                             'number of neurons in the hidden-to-hidden '
                             'recurrent layers.')
    agroup.add_argument('--use_same_resources', action='store_true',
                        help='If activated, the student will use the same '
                             'amount of resources as the teacher, i.e. the '
                             'same architecture and sparsity levels will be '
                             'set.')
    agroup.add_argument('--set_identical_topology', action='store_true',
                        help='If activated, the topology of the students '
                             'will be identical to that of the teacher, i.e. '
                             'the same connections as in the teacher exist.')
    agroup.add_argument('--set_same_weight_sign', action='store_true',
                        help='If activated, the topology of the students '
                             'will be identical to that of the teacher, and '
                             'the initial weights will have the same sign ' 
                             'as that of the teacher.')
    return agroup

def check_invalid_args(config):
    """Sanity check for command-line arguments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    n_ts_out = config.teacher_n_ts_out
    if n_ts_out == -1:
        n_ts_out = config.teacher_n_ts_in
    num_recurrent_neurons = np.sum(misc.str_to_ints(config.rnn_arch))
    if num_recurrent_neurons >= n_ts_out and config.compute_late_mse:
        raise ValueError('The number of recurrent neurons exceeds the length '
                         'of the output sequences, thus the option '
                         '``compute_late_mse`` cannot be activated.')
    if (config.set_identical_topology or config.set_same_weight_sign) and \
            config.use_same_capacity_mlp:
        raise ValueError('The student cannot be identical or share weights ' +
                         'if it is an MLP.')
    args.check_invalid_network_args(config, net='teacher')


def post_process_args(config):
    """Post process the command line arguments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    if config.set_same_weight_sign:
        config.set_identical_topology = True
    if config.use_same_resources:
        config.rnn_arch = config.teacher_rnn_arch
        config.rnn_pre_fc_layers = config.teacher_rnn_pre_fc_layers
        config.rnn_post_fc_layers = config.teacher_rnn_post_fc_layers
        config.fc_rec_output = config.teacher_fc_rec_output
        config.fc_sparsity = config.teacher_fc_sparsity
        config.rec_sparsity = config.teacher_rec_sparsity
        config.dont_use_bias = config.teacher_dont_use_bias
        config.use_vanilla_rnn = config.teacher_use_vanilla_rnn
