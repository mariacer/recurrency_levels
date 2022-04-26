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
# @title          :utils/args.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.7
"""
Command line arguments for experiments
--------------------------------------

All command-line arguments and default values are handled in this module.
"""
import argparse
from datetime import datetime
import warnings

import bio_rnn.args as mc_args
import student_teacher.args as st_args
from utils import misc

def parse_cmd_arguments(experiment='audioset', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        experiment (str, optional): The name of the experiment being run.
        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """

    description = 'Training RNNs on sequential tasks.'

    parser = argparse.ArgumentParser(description=description)

    miscellaneous_args(parser)
    rnn_args(parser)
    feedforwardness_args(parser)
    if 'audioset' in experiment:
        eval_args(parser, dval_set_size=5000)
        train_args(parser)
    elif 'pos_tagging' in experiment:
        eval_args(parser)
        train_args(parser, dlr='5e-3')
    elif 'student_teacher' in experiment:
        eval_args(parser)
        train_args(parser, dn_iter=500)
    else:
        eval_args(parser)
        train_args(parser)

    if 'student_teacher' in experiment:
        rnn_args(parser, net='teacher')
        st_args.teacher_rnn_args(parser)
        st_args.student_teacher_training_args(parser)
    if 'microcircuit' in experiment:
        mc_args.microcircuit_args(parser)
        if 'student_teacher' in experiment:
            # Add microcircuit options for teacher as well.
            mc_args.microcircuit_args(parser, net='teacher')

    # Overwrite the provided arguments if default required, or list provided.
    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    # if hasattr(config, 'clip_grad_norm') and config.clip_grad_norm != -1:
    #     raise NotImplementedError
    # if hasattr(config, 'clip_grad_value') and config.clip_grad_value != -1:
    #     raise NotImplementedError

    post_process_args(config)
    check_invalid_args(config)
    if 'microcircuit' in experiment:
        mc_args.check_invalid_args(config)
    if 'student_teacher' in experiment:
        st_args.post_process_args(config)
        st_args.check_invalid_args(config)

    return config

def miscellaneous_args(parser, dout_dir=None):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for miscellaneous arguments.

    Arguments specified in this function:
        - `out_dir`
        - `no_cuda`
        - `deterministic_run`
        - `show_plots`
        - `data_random_seed`
        - `random_seed`
        - `save_weights`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dout_dir (optional): Default value of option `out_dir`. If :code:`None`,
            the default value will be `./out/run_<YY>-<MM>-<DD>_<hh>-<mm>-<ss>`
            that contains the current date and time.

    Returns:
        The created argument group, in case more options should be added.
    """
    if dout_dir is None:
        dout_dir = './out/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ### Miscellaneous arguments
    agroup = parser.add_argument_group('Miscellaneous options')
    agroup.add_argument('--out_dir', type=str, default=dout_dir,
                        help='Where to store the outputs of this simulation.')
    agroup.add_argument('--no_cuda', action='store_true',
                            help='Flag to disable GPU usage.')
    agroup.add_argument('--deterministic_run', action='store_true',
                        help='Enable deterministic CuDNN behavior. Note, that' +
                             'CuDNN algorithms are not deterministic by ' +
                             'default and results might not be reproducible ' +
                             'unless this option is activated. Note, that ' +
                             'this may slow down training significantly!')  
    agroup.add_argument('--show_plots', action='store_true',
                        help='Whether plots should be shown.')
    agroup.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    agroup.add_argument('--data_random_seed', type=int, metavar='N',
                    default=42,
                    help='The data is randomly generated at every ' +
                     'run. This seed ensures that the randomness ' +
                     'during data generation is decoupled from the ' +
                     'training randomness. Default: %(default)s.')
    agroup.add_argument('--hpsearch', action='store_true',
                        help='Flag indicating that a hyperparameter search'
                             'is going on. This is needed to select the correct'
                             ' relative path to the datasets dictionary.')
    agroup.add_argument('--save_logs', action='store_true',
                        help='Flag indicating that tensorboard plots should be '
                             'made of the current run.')
    agroup.add_argument('--save_weights', action='store_true',
                        help='Flag indicating that the hidden-to-hidden weights'
                             'of the rnn should be saved to a .csv file '
                             'after training.')

    return agroup

def train_args(parser, dlr='1e-3', dbatch_size=64, dn_iter='5000',
               dadam_beta1='0.9', show_clip_grad_value=False,
               show_clip_grad_norm=False, show_momentum=True,
               dclip_grad_value=-1, dclip_grad_norm=-1):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for options to configure network training.

    Arguments specified in this function:
        - `batch_size`
        - `n_iter`
        - `lr`
        - `reservoir`
        - `momentum`
        - `weight_decay`
        - `dont_use_adam`
        - `adam_beta1`
        - `clip_grad_value`
        - `clip_grad_norm`
        - `orthogonal_hh_reg`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dlr: Default value for option `lr`.
        dbatch_size: Default value for option `batch_size`.
        dn_iter: Default value for option `n_iter`.
        dadam_beta1: Default value for option `adam_beta1`.
        show_clip_grad_value: Whether the `clip_grad_value` argument should be
            shown.
        show_clip_grad_norm: Whether the `clip_grad_norm` argument should be
            shown.
        show_momentum: Whether the `momentum` argument should be
            shown.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Training options.
    agroup = parser.add_argument_group('Training options')
    agroup.add_argument('--batch_size', type=int, metavar='N',
                        default=dbatch_size,
                        help='Training batch size. Default: %(default)s.')
    agroup.add_argument('--n_iter', type=str, metavar='N', default=dn_iter,
                        help='Number of training iterations per task. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--lr', type=str, default=dlr,
                        help='Learning rate of optimizer(s). Default: ' +
                             '%(default)s. If a list of 2 floats is provided,'
                             '2 optimizers will be created, one for the fully-'
                             'connected layers and one for the recurrent ' +
                             'layers (ex: --lr=[0.1,0.3].')
    agroup.add_argument('--reservoir', action='store_true',
                        help='If active, all the parameters except those of ' +
                             'the output layer will be frozen, and the ' +
                             'resulting network will correspond to a ' +
                             'reservoir.')
    agroup.add_argument('--weight_decay', type=str, default='0',
                        help='Weight decay of the optimizer(s). Default: ' +
                             '%(default)s.')
    agroup.add_argument('--dont_use_adam', action='store_true',
                        help='Use SGD rather than Adam optimizer.')
    agroup.add_argument('--adam_beta1', type=str, default=dadam_beta1,
                        help='The "beta1" parameter when using torch.optim.' +
                             'Adam as optimizer. Default: %(default)s.')
    agroup.add_argument('--orthogonal_hh_reg', type=float, default=-1,
                        help='If "-1", no orthogonal regularization will ' +
                             'be applied. Otherwise, the hidden-to-' +
                             'hidden weights of the recurrent layers are ' +
                             'regularized to be orthogonal with the ' +
                             'given regularization strength. ' +
                             'Default: %(default)s')
    agroup.add_argument('--clip_grad_value', type=float,
                        default=dclip_grad_value,
                        help='Clip the values of each gradient with the '
                             'specified clip threshold. Default -1 means no '
                             'clipping.')
    agroup.add_argument('--clip_grad_norm', type=float,
                        default=dclip_grad_norm,
                        help='Clip the norms of each gradient with the '
                             'specified clip threshold. Default -1 means no '
                             'clipping.')

    if show_momentum:
        agroup.add_argument('--momentum', type=str, default='0.0',
                            help='Momentum of the optimizer (only used in ' +
                                 'SGD and RMSprop). Default: %(default)s.')
    # if show_clip_grad_value:
    #     agroup.add_argument('--clip_grad_value', type=float, default=-1,
    #                     help='If not "-1", gradients will be clipped using ' +
    #                          '"torch.nn.utils.clip_grad_value_". Default: ' +
    #                          '%(default)s.')
    # if show_clip_grad_norm:
    #     agroup.add_argument('--clip_grad_norm', type=float, default=-1,
    #                     help='If not "-1", gradient norms will be clipped ' +
    #                          'using "torch.nn.utils.clip_grad_norm_". ' +
    #                          'Default: %(default)s.')

    return agroup

def rnn_args(parser, net='main', drnn_arch='32', drec_sparsity=-1,
             dfc_sparsity=-1, dnet_act='tanh', dprune_fraction=-1,
             drecurrency_level=-1, darch_random_seed=42):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options to a main network.

    Arguments specified in this function:
        - `rnn_arch`
        - `rnn_pre_fc_layers`
        - `rnn_post_fc_layers`
        - `rec_sparsity`
        - `fc_sparsity`
        - `no_sparse_input_output`
        - `fc_rec_output`
        - `net_act`
        - `dont_use_bias`
        - `use_vanilla_rnn`
        - `use_mlp`
        - `use_same_capacity_mlp`
        - `recurrency_level`
        - `orthogonal_hh_init`
        - `use_kaiming_init`
        - `prune_fraction`
        - `arch_random_seed`

    Note that these arguments will be applied to a teacher network generating
    a dataset if the argument ``net`` is set to ``teacher``.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        net (str, optional): The network that is being defined. Either `main`
            or `teacher`.
        drnn_arch: Default value of option `rnn_arch`.
        drec_sparsity: Default value of option `rec_sparsity`.
        dfc_sparsity: Default value of option `fc_sparsity`.
        dnet_act: Default value of option `net_act`.
        drecurrency_level: Default value of option `recurrency_level`.
        darch_random_seed: Default value of option `arch_random_seed`.

    Returns:
        The created argument group, in case more options should be added.
    """
    heading = 'Recurrent network options'

    pfx = ''
    if net == 'teacher':
        pfx = 'teacher_'

    agroup = parser.add_argument_group(heading)
    agroup.add_argument('--%srnn_arch'%pfx, type=str, default=drnn_arch,
                        help='Specifies the dimension of the hidden' +
                             '(recurrent) layer of the recurrent network.' +
                             'Default: %(default)s.')
    agroup.add_argument('--%srnn_pre_fc_layers'%pfx, type=str,
                        default='',
                        help='If using an "RNN" network, ' +
                             'this will specify the sizes of all initial ' +
                             'fully-connected latyers. If left empty, ' +
                             'there will be no initial fully-connected ' +
                             'layers and the first layer is going to be ' +
                             'a recurrent layer. Default: %(default)s.')
    agroup.add_argument('--%srnn_post_fc_layers'%pfx, type=str, default='',
                        help='If using an "RNN" network, ' +
                             'this will specify the sizes of all final ' +
                             'hidden fully-connected layers. Note, the ' +
                             'output layer is also fully-connected, even ' +
                             'if this option is left empty. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--%srec_sparsity'%pfx, type=float,
                        default=drec_sparsity,
                        help='The fraction of the recurrent neurons that '
                             'will be connected to other recurrent units. '
                             'A value of -1 means full connectivity. '
                             'Default: %(default)s.')
    agroup.add_argument('--%sfc_sparsity'%pfx, type=float,
                        default=dfc_sparsity,
                        help='The fraction of the fully-connected neurons that '
                             'will be connected to neighboring layers. '
                             'A value of -1 means full connectivity. '
                             'Default: %(default)s.')
    agroup.add_argument('--%sno_sparse_input_output'%pfx, action='store_true',
                        help='This ensures that the input are output layers '
                             'are fully-connected, irrespective of the '
                             'specified sparsity levels for fully-connected '
                             'and recurrent layers.')
    agroup.add_argument('--%sfc_rec_output'%pfx, action='store_true',
                        help='If active when a vanilla RNN is being used, ' +
                             'the RNN layers will consist of a recurrent '+
                             'layer with a fully connected output, as is ' +
                             'normally done in Elman networks. Otherwise, ' +
                             'vanilla RNN layers consist only of a recurrent ' +
                             'computation.')
    agroup.add_argument('--%snet_act'%pfx, type=str, default=dnet_act,
                        help='Activation function used in the network. ' +
                             'Default: %(default)s.',
                        choices=['relu', 'tanh', 'linear'])
    agroup.add_argument('--%sdont_use_bias'%pfx, action='store_true',
                        help='If activated, no bias will be used.')
    agroup.add_argument('--%suse_vanilla_rnn'%pfx, action='store_true',
                        help='Whether vanilla rnn cells should be used. ' +
                             'Otherwise, LSTM cells are used.')
    agroup.add_argument('--%suse_mlp'%pfx, action='store_true',
                        help='If activated, a feedforward network will be used '
                             'to process the data instead of an RNN. In this '
                             'case the sizes of the hidden layer are specified '
                             'by ``rnn_arch``.')
    agroup.add_argument('--%suse_same_capacity_mlp'%pfx, action='store_true',
                        help='If activated, the network will be feedforward '
                             'network with a single hidden layer and a similar '
                             'number of connections to the equivalent rnn.')
    agroup.add_argument('--%srecurrency_level'%pfx, type=float,
                        default=drecurrency_level,
                        help='Determines the level of desired recurrency '
                             'in the recurrent layers. Specifically, given a '
                             'square matrix of recurrent weights, it specifies '
                             'the number of connections in the lower triangle '
                             '(i.e. feedforward connections) vs. in the upper '
                             'triangle (i.e. feedback connections). A value '
                             'of ``1`` therefore indicates purely recurrent '
                             'connectivity while a value of ``0`` indicates '
                             'purely feedforward connectivity. Only applicable '
                             'if masks are used (i.e. if the input, output or '
                             'recurrent connectivity fractions are set). '
                             'If ``-1`` the value is ignored. '
                             'Default: %(default)s.')
    agroup.add_argument('--%sorthogonal_hh_init'%pfx, action='store_true',
                        help='Initialize hidden-to-hidden weights of ' +
                             'recurrent layers orthogonally. If ' +
                             '"use_kaiming_init" is already activated, ' +
                             'the kaiming init of hidden-to-hidden layers ' +
                             'will be overwriden by the orthogonal init.')
    agroup.add_argument('--%suse_kaiming_init'%pfx, action='store_true',
                        help='Initialize all layers with a Kaiming ' +
                             'initialization.')
    if net == 'main':
        agroup.add_argument('--prune_fraction', type=float,
                            default=dprune_fraction,
                            help='If > 0, the specified fraction of the '
                                 'hidden-to-hidden weights of the rnn will be '
                                 'pruned. Only works with 1 recurrent layer.')
        agroup.add_argument('--arch_random_seed', type=int, 
                           default=darch_random_seed,
                            help='The random seed to be used to generate the ' +
                                 'masks and topology of the networks.')
                            # For a teacher network, this value is 
                            # "data_random_seed".
    return agroup


def eval_args(parser, dval_iter=250, dval_batch_size=256, dval_set_size=0):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for validation and testing options.

    Arguments specified in this function:
        - `val_iter`
        - `val_batch_size`
        - `val_set_size`
        - `plot_samples`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dval_iter (int): Default value of argument `val_iter`.
        dval_batch_size (int): Default value of argument `val_batch_size`.
        dval_set_size (int): Default value of argument `val_set_size`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Eval arguments
    agroup = parser.add_argument_group('Evaluation options')
    agroup.add_argument('--val_iter', type=int, metavar='N', default=dval_iter,
                        help='How often the validation should be performed ' +
                             'during training. Default: %(default)s.')
    agroup.add_argument('--val_batch_size', type=int, metavar='N',
                        default=dval_batch_size,
                        help='Batch size during validation/testing. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--val_set_size', type=int, metavar='N',
                        default=dval_set_size,
                        help='If unequal "0", a validation set will be ' +
                             'extracted from the training set (hence, ' +
                             'reducing the size of the training set). ' +
                             'This can be useful for efficiency reasons ' +
                             'if the validation set is smaller than the ' +
                             'test set. If the training is influenced by ' +
                             'generalization measures on the data (e.g., ' +
                             'a learning rate schedule), then it is good ' +
                             'practice to use a validation set for this. ' +
                             'It is also desirable to select ' +
                             'hyperparameters based on a validation set, ' +
                             'if possible. Default: %(default)s.')
    agroup.add_argument('--plot_samples', action='store_true',
                        help='Plot inputs, outputs and predictions at test '
                             'time.')

    return agroup

def feedforwardness_args(parser, dfeedforwardness_maxtime=60.):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for options to compute the level of recurrency.

    Arguments specified in this function:
        - `feedforwardness_maxtime`
        - `dont_use_adjacency`
        - `dont_use_recursion`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dfeedforwardness_maxtime: Default value for option 
            `feedforwardness_maxtime`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Training options.
    agroup = parser.add_argument_group('Feedforwardness options')
    agroup.add_argument('--feedforwardness_maxtime', type=float,
                        default=dfeedforwardness_maxtime,
                        help='Maximum allowed number of seconds to compute ' +
                             'any one of the feedforwardness measures. ' +
                              'Default: %(default)s.')

    ### Powers of the adjacency matrix.
    agroup.add_argument('--dont_use_adjacency', action='store_true',
                        help='Do not compute the number of cycles using the ' +
                             'powers of the adjacency matrix.')

    ### Recursive algorithm.
    agroup.add_argument('--dont_use_recursion', action='store_true',
                        help='Do not compute the number of cycles using the ' +
                             'recursive algorithm.')

def check_invalid_args(config):
    """Sanity check for command-line arguments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """

    # Training arguments.
    if not isinstance(config.lr, float) and len(config.lr)!= 2:
        raise ValueError('The specified learning rate has to either be a '
                         'single global value, or a list of two floats '
                         'specifying the learning rate in fully-connected '
                         'and recurrent parameters.')
    if not isinstance(config.momentum, float) and len(config.momentum)!= 2:
        raise ValueError('The specified momentum has to either be a '
                         'single global value, or a list of two floats '
                         'specifying the learning rate in fully-connected '
                         'and recurrent parameters.')
    if not isinstance(config.adam_beta1, float) and len(config.adam_beta1)!= 2:
        raise ValueError('The specified Adam beta value has to either be a '
                         'single global value, or a list of two floats '
                         'specifying the learning rate in fully-connected '
                         'and recurrent parameters.')
    if not isinstance(config.weight_decay, float) and \
            len(config.weight_decay)!= 2:
        raise ValueError('The specified weight_decay has to either be a '
                         'single global value, or a list of two floats '
                         'specifying the learning rate in fully-connected '
                         'and recurrent parameters.')
    if config.prune_fraction not in [-1., 0.] and \
            config.rec_sparsity not in [-1., 1]:
        raise NotImplementedError('The pruning is only currently implemented ' +
                                  'for fully-connected networks. For sparse ' +
                                  'layers, the pruning percentage would have '+
                                  'to be applied to the actual number of ' +
                                  'existing connections.')

    # Network arguments.
    check_invalid_network_args(config)

def check_invalid_network_args(config, net='main'):
    """Check arguments related to network architecture.

    This function can check both main and teacher networks depending on input.

    Args:
        (....): See docstring of function :func:`check_invalid_args`.
        net (str, optional): The network that is being defined. Either `main`
            or `teacher`.
    """

    pfx = ''
    if net == 'teacher':
        pfx = 'teacher_'

    if getattr(config, pfx + 'rec_sparsity') not in [-1, 1] and \
            (getattr(config, pfx + 'rec_sparsity') > 1. or \
            getattr(config, pfx + 'rec_sparsity')  < 0.):
        raise ValueError('The sparsity level of recurrent layers should be '
                         'between 0 and 1!')
    if getattr(config, pfx + 'fc_sparsity')  not in [-1, 1] and \
            (getattr(config, pfx + 'fc_sparsity') > 1. or \
            getattr(config, pfx + 'fc_sparsity')  < 0.):
        raise ValueError('The sparsity level of fully-connected layers should '
                         'be between 0 and 1!')
    if getattr(config, pfx + 'fc_sparsity') not in [-1, 1] and \
            getattr(config, pfx + 'use_mlp'):
        raise NotImplementedError('Sparse MLP not implemented yet.')
    if getattr(config, pfx + 'rec_sparsity') not in [-1, 1] and not \
            getattr(config, pfx + 'use_vanilla_rnn'):
        raise NotImplementedError('Sparse LSTM not implemented yet.')

    if getattr(config, pfx + 'recurrency_level') != -1:
        if getattr(config, pfx + 'rec_sparsity') == 1:
            raise ValueError('Recurrency levels are not compatible with '
                'full connectivity.')
    if not getattr(config, pfx + 'use_vanilla_rnn') and \
            getattr(config, pfx + 'fc_rec_output'):
        raise ValueError('Fully-connected connectivity in the output of the ' +
                         'recurrent layers is only a valid option for ' + 
                         'vanilla RNNs.')
    if getattr(config, pfx + 'use_same_capacity_mlp') and \
            isinstance(misc.str_to_float_or_list(\
            getattr(config, pfx + 'rnn_arch')), list):
        raise NotImplementedError('Constructing an MLP with equivalent ' +
                                  'capacity is only implemented for single ' +
                                  'hidden layer networks.')
    if getattr(config, pfx + 'use_same_capacity_mlp') and \
            getattr(config, pfx + 'use_mlp') :
        raise ValueError('The options "use_same_capacity_mlp" and "use_mlp" ' +
                         'cannot be simultaneously active as they lead to ' +
                         'different behavior.')

def post_process_args(config):
    """Post process the command line arguments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    config.n_iter = misc.str_round_to_int(config.n_iter)
    config.lr = misc.str_to_float_or_list(config.lr)
    config.adam_beta1 = misc.str_to_float_or_list(config.adam_beta1)
    config.weight_decay = misc.str_to_float_or_list(config.weight_decay)
    config.momentum = misc.str_to_float_or_list(config.momentum)


if __name__=='__main__':
    pass