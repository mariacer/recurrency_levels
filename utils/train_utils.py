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
# @title          :utils/train_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Helper functions for training RNNs
----------------------------------

Set of helper functions that will be used when training an RNN.
"""
import numpy as np
import torch
from torch.nn import functional as F

from networks.net_utils import extract_hh_weights

def sequential_nll(loss_type='ce', reduction='sum', return_per_ts=False):
    r"""Create a custom NLL function for sequential tasks.

    We consider a network that has an output shape of ``[T, B, C]``, where ``T``
    is the length of the sequence (number of timesteps), ``B`` is the batch size
    and ``C`` is the number of output channels (e.g., number of classes for a 
    classification problem).

    We first derive the likelihood for a dataset with :math:`N` samples
    :math:`p(\mathcal{D} \mid W) = \prod_{n=1}^N p(\mathbf{y}_n \mid W)`,
    where :math:`W` are the weights of the neural network. Each sample in the
    dataset can be decomposed into targets per timestep as follows
    :math:`\mathbf{y}_n = (y_n^{(1)}, \dots, y_n^{(T)})` (note, individual
    :math:`y_n^{(t)}` might still be vectors of targets, e.g., in case of the
    `copy task <https://arxiv.org/pdf/1410.5401.pdf>`__).

    We adopt the following decomposition of the joint:

    .. math::

        p(\mathbf{y}_n \mid W) = p(y_n^{(1)}, \dots, y_n^{(T)} \mid W) = \
            \prod_{t=1}^T p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W)

    We now claim, that

    .. math::

        p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W) \
            \approx p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W)

    where :math:`\mathbf{h}_n^{(t-1)}` is the RNN state used to compute the
    likelihood of :math:`y_n^{(t)}`.

    We need to define a loss function that minimizes the negative-log-likelihood
    (NLL)

    .. math::

        \text{NLL} = \sum_{n=1}^N \text{NLL}_n = \
            \sum_{n=1}^N \sum_{t=1}^T \text{NLL}_n^{(t)}

    with :math:`\text{NLL}_n^{(t)} \equiv - \
    \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W)`.

    Note:
        The NLL sums over all samples in the dataset. Hence, whenever one has to
        compute the NLL over a minibatch, it should be renormalized by dividing
        by the batch size and multiplying by the dataset size!

    In the following, we consider how :math:`\text{NLL}_n^{(t)}` can be computed
    for different kinds of problems.

    **Mean-squared-error loss**

    In case of a mean-squared error loss, we assume the likelihood to be
    Gaussian with diagonal covariance. The quantity :math:`y_n^{(t)}` is now a
    ``C``-dimensional target vector. We denote the actual network output for
    timestep :math:`t` by :math:`z_n^{(t)}`, which is also ``C``-dimensional.

    Note:
        Network outputs should be logits, i.e., spanning the whole real line
        since Gaussians have full support.

    We can then define the per sample loss function as follows

    .. math::
        L_n = \frac{1}{T} \sum_{t=1}^T \frac{1}{C} \
            \sum_{c=1}^C m^{(t,c)} (y_n^{(t,c)} - z_n^{(t,c)})^2

    The mask :math:`m^{(t,c)}` shall introduce a greater importance for some
    timesteps/classes.

    Since the optimized loss function represents a MSE, we consider a Gaussian
    likelihood where the masking values enter implicitly as variances:

    .. math::

        \text{NLL}_n^{(t)} &\equiv \
            - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \\
            &= \text{const.} + \frac{1}{2} \sum_{c=1}^C \frac{1}{\sigma_{t,c}^2}
            \big(y_n^{(t,c)} - z_n^{(t,c)} \big)^2 \\
            &= \text{const.} + \frac{1}{C T} \sum_{c=1}^C m^{(t,c)}
            \big(y_n^{(t,c)} - z_n^{(t,c)} \big)^2 \\

    where :math:`m^{(t,c)} = \frac{CT}{2\sigma_{t,c}^2}`, such that if we choose
    :math:`\sigma_{t,c}^2 = \frac{CT}{2m^{(t,c)}}` then the NLL perfectly
    corresponds to the loss function chosen above.

    Hence, by defining a mask :math:`m^{(t,c)}` we implicitly define the
    variances of the Gaussian likelihood function. For instance, a mask value
    :math:`m^{(t,c)} = 0` would correspond to an infinite variance, which makes
    sense since we don't care about the output value in that case.

    **Cross-entropy loss**

    In this case, the target :math:`y_n^{(t)}` represents the target class
    (the label) for timestep :math:`t`. We consider :math:`z_n^{(t)}` as the
    ``C``-dimensional logit output of our network and consider the likelihood
    for each label defined by a softmax

    .. math::

        \tilde{z}_n^{(t)} = \text{softmax}(\beta^{(t)} z_n^{(t)})

    where :math:`\beta^{(t)}` determines the inverse temperature for timestep
    :math:`t`, which we can use to introduce varying importances for timesteps.

    We can now express the NLL for sample :math:`n` at timestep :math:`t` as
    follows

    .. math::

        \text{NLL}_n^{(t)} &\equiv \
            - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \\
            &= - \log \tilde{z}_n^{(t,y_n^{(t)})} \\
            &= - \sum_{c=1}^C [y_n^{(t)} = c] \log \tilde{z}_n^{(t,c)} \\
            &= - \sum_{c=1}^C [y_n^{(t)} = c] \log \
                \big( \text{softmax}(\beta^{(t)} \mathbf{z}_n^{(t)})_c \big)

    where :math:`[\cdot]` denotes the Iverson bracket. One possibility to
    incorporate some kind of masking is to set different temperature values
    :math:`\frac{1}{\beta^{(t)}}` for different timesteps. For instance:

    - Setting :math:`\beta^{(t)} = 0` is equivalent to saying that we don't care
      about the prediction (it will also stop gradient flow through the
      network).
    - Setting :math:`\beta^{(t)}` to a large value corresponds to requesting
      sharper and more confident predictions.

    **Binary cross-entropy loss**

    In this case, the target :math:`y_n^{(t)}` is a binary ``C``-dimensional
    vector that determines the label of **independent** binary decisions, i.e.,
    we aim to minimize

    .. math::

        \text{NLL}_n^{(t)} = - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \
            = - \sum_{c=1}^C \log p(y_n^{(t,c)} \mid \mathbf{h}_n^{(t-1)}, W)

    Again, we consider :math:`z_n^{(t)}` as the ``C``-dimensional logit output
    of the network and define the likelihood per decision via a sigmoid

    .. math::

        \tilde{z}_n^{(t,c)} = \text{sigmoid}(\beta^{(t,c)} z_n^{(t,c)})

    where :math:`\beta^{(t,c)}` is the inverse temperature per
    timestep/decision.

    We can then define the likelihood using the binary-cross entropy

    .. math::

        \text{NLL}_n^{(t)} = \sum_{c=1}^C \big( \
             - y_n^{(t,c)} \log \tilde{z}_n^{(t,c)} \
             - (1 - y_n^{(t,c)}) \log (1 - \tilde{z}_n^{(t,c)}) \
        \big)

    The same considerations as for the usual cross-entropy (see above) apply
    here for setting the inverse temperature.

    Note:
        All loss types require network outputs to be provided as logits!

    Args:
        loss_type (str): Determines which kind of loss function is used. The
            following options are available:

            - ``'ce'``: A function handle that uses the cross-entropy loss is
              returned.
            - ``'bce'``: A function handle that uses the binary cross-entropy
              loss is returned.
            - ``'mse'``: A function handle that uses the mean-squared-error loss
              is returned.
        reduction (str): Whether the NLL loss should be summed ``'sum'`` or
            meaned ``'mean'`` across the batch dimension.
        return_per_ts (bool, optional): Whether the loss per timestep should
            alse be returned.

    Returns:
        (func): A function handle.

        Based on the chosen ``loss_type``, the returned functions might have
        different signatures in terms of keyword arguments. Though, the
        positional arguments are the same for each function.

        - ``'ce'``: A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, mask=None)
        - ``'bce'``: A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, beta=None)
        - ``'mse'``: : A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, beta=None)

        The keyword arguments have the following meaning:

        - **ts_factors** (torch.Tensor, optional): A list of factors,
          one for each timestep
          :math:`\log p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W)`.
          These factors are multiplied before the timesteps are summed together.
          The tensor should be of shape ``[num_timesteps, 1]`` or
          ``[num_timesteps, batch_size]``.

          Note:
              Setting ``ts_factors`` to ``0`` should be identical to setting
              ``mask`` or ``beta`` values to zero (at least the gradient
              computation through the loss is identical, the actuall NLL value
              might be different. Keep in mind, that for the MSE loss the NLL is
              anyway only computed up to additive constants.
        - **mask** (torch.Tensor, optional): The mask :math:`m^{(t,c)}` that can
          be applied per timestep/channel. The shape of the mask must either
          be identical to the one of ``Y`` or broadcastable with respect to
          ``Y``.
        - **beta** (torch.Tensor, optional): Contains the inverse temperatures
          :math:`\beta^{(t)}` per timestep for the ``'ce'`` loss and the
          inverse temperatures per timestep/channel for the ``'bce'`` loss.
          The shape of ``beta`` must be broadcastable with respect to ``Y``.
          In addition, ``beta`` is expected to fulfill ``beta.shape[2] == 1``
          in case of the ``'ce'`` loss.
    """
    assert reduction in ['sum', 'mean']
    assert loss_type in ['ce', 'bce', 'mse']

    def custom_nll_mse(Y, T, data, allowed_outputs, empirical_fisher,
                       ts_factors=None, mask=None):
        assert np.all(np.equal(list(Y.shape), list(T.shape)))

        nll = (Y - T)**2
        if mask is not None:
            nll = mask * nll

        nll = nll.mean(dim=2)

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        if reduction == 'mean':
            nll_reduced = nll.mean()
        else:
            nll_reduced = nll.mean(dim=0).sum()

        if return_per_ts:
            return nll_reduced, nll.mean(dim=1)
        else:
            return nll_reduced

    def custom_nll_ce(Y, T, data, allowed_outputs, empirical_fisher,
                      ts_factors=None, beta=None):
        # We expect targets to be either given as labels or as 1-hot encodings.
        assert len(Y.shape) == 3 and \
            np.all(np.equal(list(Y.shape[:2]), list(T.shape[:2]))) and \
            (len(T.shape) == 2 or Y.shape[2] == T.shape[2])

        if len(T.shape) == 2:
            labels = T
        else:
            labels = torch.argmax(T, 2)

        if beta is None:
            log_sm = F.log_softmax(Y, dim=2)
        else:
            log_sm = F.log_softmax(Y * beta, dim=2)
        # We need to swap dimensions from [T, B, C] to [T, C, B].
        # See documentation of method `F.nll_loss`.
        log_sm = log_sm.permute(0, 2, 1)
        nll = F.nll_loss(log_sm, labels, reduction='none')
        assert len(nll.shape) == 2

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        # Sum across time-series dimension.
        nll_reduced = nll.sum(dim=0)

        if reduction == 'mean':
            nll_reduced = nll_reduced.mean()
        else:
            nll_reduced = nll_reduced.sum()

        if return_per_ts:
            return nll_reduced, nll.mean(dim=1)
        else:
            return nll_reduced

    def custom_nll_bce(Y, T, data, allowed_outputs, empirical_fisher,
                       ts_factors=None, beta=None):
        # T is expected to be binary vector.
        assert np.all(np.equal(list(Y.shape), list(T.shape)))

        if beta is None:
            nll = F.binary_cross_entropy_with_logits(Y, T, reduction='none')
        else:
            nll = F.binary_cross_entropy_with_logits(beta * Y, T,
                                                     reduction='none')
        assert len(nll.shape) == 3

        # Sum accross channel dimension.
        nll = nll.sum(dim=2)

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        # Sum across time-series dimension.
        nll_reduced = nll_reduced.sum(dim=0)

        if reduction == 'mean':
            nll_reduced = nll_reduced.mean()
        else:
            nll_reduced = nll_reduced.sum()
        
        if return_per_ts:
            return nll_reduced, nll.mean(dim=1)
        else:
            return nll_reduced

        return nll_reduced

    if loss_type == 'ce':
        return custom_nll_ce
    elif loss_type == 'bce':
        return custom_nll_bce
    else:
        assert loss_type == 'mse'
        return custom_nll_mse


def init_summary_dict(classification=True):
    """ Initialize the summary dictionary.

    Args:
        classification (boolean): Whether it is a classificaiton task. If not,
            no accuracy items will be added to the dictionary.

    Returns:
        (dict): The empty summary dictionary.
    """
    summary_dict = {
        'loss_train_last': -1,
        'loss_test_last': -1,
        'loss_train_best': -1,
        'loss_test_best': -1,
    }

    if classification:
        summary_dict['acc_train_last'] = -1
        summary_dict['acc_train_best'] = -1
        summary_dict['acc_test_last'] = -1
        summary_dict['acc_test_best'] = -1

    return summary_dict


def orthogonal_regularizer(config, mnet):
    r"""Compute orthogonal regularizer.

    .. math::

        \lambda \sum_i \lVert W_i^T W_i - I \rVert^2

    This function computes the orthogonal regularizer for all hidden-to-hidden
    weights as returned by function :func:`extract_hh_weights`.

    Args:
        config (argparse.Namespace): The user configuration.
        mnet (networks.rnn): The network.

    Returns:
        (torch.Tensor): A scalar tensor representing the orthogonal regularizer.
    """
    assert config.orthogonal_hh_reg > 0

    # Weights to be regularized.
    weights = extract_hh_weights(mnet)

    reg = 0
    for W in weights:
        # Compute Frobenius norm of W^T W - I
        reg += torch.norm(torch.matmul(W, W.transpose(0,1)) - \
                          torch.eye(W.shape[0], device=W.device))**2

    return config.orthogonal_hh_reg * reg