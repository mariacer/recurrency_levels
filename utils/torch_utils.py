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
# @title          :utils/torch_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.6.8
"""
A collection of helper functions that should capture common functionalities
needed when working with PyTorch.

Some are based on implementations by Christian Henning in another project:

    https://github.com/mariacer/cl_in_rnns
"""
import math
import torch
import torch.nn as nn 

def init_params(weights):
    """Initialize the weights according to kaiming uniform initialization.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    Args:
        weights: The module with weights be initialized.
    """
    fans = {}
    for name, w in weights.named_parameters():
        is_bias = True if name.startswith('bias') else False
        layer_name = name[len('bias'):] if is_bias else name[len('weight'):]

        if not is_bias:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            # Save the fans for initializing the corresponding bias afterwards.
            fans[layer_name] = nn.init._calculate_fan_in_and_fan_out(w)
        else:
            fan_in, _ = fans[layer_name]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)

def get_optimizer(params, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False):
    """Create an optimizer instance for the given set of parameters. Default
    optimizer is :class:`torch.optim.SGD`.

    Args:
        params: The parameters passed to the optimizer.
        lr: Learning rate.
        momentum (optional): Momentum (only applicable to
            :class:`torch.optim.SGD` and :class:`torch.optim.RMSprop`.
        weight_decay (optional): L2 penalty.
        use_adam: Use :class:`torch.optim.Adam` optimizer.
        adam_beta1: First parameter in the `betas` tuple that is passed to the
            optimizer :class:`torch.optim.Adam`:
            :code:`betas=(adam_beta1, 0.999)`.
        use_rmsprop: Use :class:`torch.optim.RMSprop` optimizer.
        use_adadelta: Use :class:`torch.optim.Adadelta` optimizer.
        use_adagrad: Use :class:`torch.optim.Adagrad` optimizer.

    Returns:
        Optimizer instance.
    """
    if use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, betas=[adam_beta1, 0.999],
                                     weight_decay=weight_decay)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(params, lr=lr,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
    elif use_adadelta:
        optimizer = torch.optim.Adadelta(params, lr=lr,
                                         weight_decay=weight_decay)
    elif use_adagrad:
        optimizer = torch.optim.Adagrad(params, lr=lr,
                                        weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    return optimizer


class OptimizerList(object):
    """Optimizer list with different optimization properties.

    Class for a list of optimizers, that behaves as a single optimizer,
    i.e. zero_grad and step works as expected. The OptimizerList will contain
    3 optimizers: one for the fully-connected weights, one for the recurrent
    weights and one for the output weights (which only has different
    properties to that of the fully-connected weights if we are dealing with
    a reservoir).
    """
    def __init__(self, net, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False, reservoir=False):
        """Initialize the class.

        Args:
            (...): See docstring of function :func:`get_optimizer_from_net`.
        """
        if not isinstance(lr, list):
            assert reservoir
            lr = [lr for i in range(2)]
        else:
            assert len(lr) == 2

        if not isinstance(momentum, list):
            momentum = [momentum for i in range(2)]

        if not isinstance(weight_decay, list):
            weight_decay = [weight_decay for i in range(2)]

        if not isinstance(adam_beta1, list):
            adam_beta1 = [adam_beta1 for i in range(2)]

        # Gather all pre-recurrent weights, recurrent weights, post-recurrent
        # weights and output weights in separate lists.
        fc_param = []
        for layer in net._fc_layers_pre:
            fc_param.append(layer.weight)
            if net._use_bias:
                fc_param.append(layer.bias)
        for layer in net._fc_layers[:-1]:
            fc_param.append(layer.weight)
            if net._use_bias:
                fc_param.append(layer.bias)
        for layer in net._rec_layers:
            if hasattr(layer, 'weight'): # only if linear layer
                fc_param.append(layer.weight)
                if net._use_bias:
                    fc_param.append(layer.bias)
            elif hasattr(layer, 'weight_ih_l0'): # only if input-to-rec layer
                fc_param.append(layer.weight_ih_l0)
                if net._use_bias:
                    fc_param.append(layer.bias_ih_l0)
        rec_param = []
        for layer in net._rec_layers:
            if hasattr(layer, 'weight_hh_l0'): # only for recurrent layers
                rec_param.append(layer.weight_hh_l0)
                if net._use_bias:
                    rec_param.append(layer.bias_hh_l0)
        output_param = nn.ParameterList(net._fc_layers[-1].parameters())

        # Add fully-connected learning rate for the output layer.
        lr.append(lr[0])
        if reservoir:
            # In a reservoir network, only the output weights are learned.
            lr[:2] = [0., 0.]

        fc_optimizer = get_optimizer(fc_param, lr=lr[0],
                                               momentum=momentum[0],
                                               weight_decay=weight_decay[0],
                                               use_adam=use_adam,
                                               adam_beta1=adam_beta1[0],
                                               use_rmsprop=use_rmsprop,
                                               use_adadelta=use_adadelta,
                                               use_adagrad=use_adagrad)
        rec_optimizer = get_optimizer(rec_param, lr=lr[1],
                                        momentum=momentum[1],
                                        weight_decay=weight_decay[1],
                                        use_adam=use_adam,
                                        adam_beta1=adam_beta1[1],
                                        use_rmsprop=use_rmsprop,
                                        use_adadelta=use_adadelta,
                                        use_adagrad=use_adagrad)
        output_optimizer = get_optimizer(output_param, lr=lr[2],
                                        momentum=momentum[0],
                                        weight_decay=weight_decay[0],
                                        use_adam=use_adam,
                                        adam_beta1=adam_beta1[0],
                                        use_rmsprop=use_rmsprop,
                                        use_adadelta=use_adadelta,
                                        use_adagrad=use_adagrad) 

        self._optimizer_list = [fc_optimizer, rec_optimizer, output_optimizer]


    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()

    def step(self, i=None):
        """
        Perform a step on the optimizer of layer i. If i is None, a step is
        performed on all optimizers.
        """
        if i is None:
            for optimizer in self._optimizer_list:
                optimizer.step()
        else:
            self._optimizer_list[i].step()


def get_optimizer_from_net(net, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False, reservoir=False):
    """Create an optimizer instance for the given network. 
    If `lr` is a float, one optimizer for all parameters is returned. If `lr`
    is a list with 2 floats, an OptimizerList will be returned which contains 3
    optimizers, separate for the input weights, recurrent weights and
    output weights. The optimzier for the fully connected and output weights are
    normally identical, except in the case where the `reservoir` option is
    selected, in which case they are different, and the only non-zero learning
    rate will be that of the outputs. Default optimizer is 
    :class:`torch.optim.SGD`.

    Args:
        net: The rnn network.
        lr: Learning rate.
        momentum (optional): Momentum (only applicable to
            :class:`torch.optim.SGD` and :class:`torch.optim.RMSprop`.
        weight_decay (optional): L2 penalty.
        use_adam: Use :class:`torch.optim.Adam` optimizer.
        adam_beta1: First parameter in the `betas` tuple that is passed to 
            the optimizer :class:`torch.optim.Adam`:
            :code:`betas=(adam_beta1, 0.999)`.
        use_rmsprop: Use :class:`torch.optim.RMSprop` optimizer.
        use_adadelta: Use :class:`torch.optim.Adadelta` optimizer.
        use_adagrad: Use :class:`torch.optim.Adagrad` optimizer.
        reservoir (bool): Whether the resulting network is a reservoir.

    Returns:
        Optimizer instance or OptimizerList.
    """
    if isinstance(lr, float) and not reservoir:
        params = net.parameters()
        optimizer = get_optimizer(params, lr=lr, momentum=momentum,
                             weight_decay=weight_decay,
                             use_adam=use_adam,
                             adam_beta1=adam_beta1,
                             use_rmsprop=use_rmsprop,
                             use_adadelta=use_adadelta,
                             use_adagrad=use_adagrad)
    elif isinstance(lr, list) or (isinstance(lr, float) and reservoir):
        optimizer = OptimizerList(net, lr=lr, momentum=momentum,
                             weight_decay=weight_decay,
                             use_adam=use_adam,
                             adam_beta1=adam_beta1,
                             use_rmsprop=use_rmsprop,
                             use_adadelta=use_adadelta,
                             use_adagrad=use_adagrad,
                             reservoir=reservoir)
    else:
        raise ValueError('Expected lr to be a float or list, got {}'.
                         format(type(lr)))
    return optimizer
