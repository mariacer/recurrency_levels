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
# @title          :utils/train.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Functions to train RNNs
-----------------------

Set of functions to train an RNN on sequential tasks.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.torch_utils import get_optimizer_from_net
from utils.train_utils import init_summary_dict, orthogonal_regularizer
import numpy as np
import os
import json

def preprocess_inputs(X, shared):
    """Preprocess inputs for POS tagging.

    If word embeddings are used (attribute ``shared.word_emb_lookups``), 
    then inputs are first translated using the word embeddings.

    Args:
        X: The inputs.
        shared: A Namespace with miscellaneous information.

    Returns:
        The preprocessed inputs.
    """ 
    if hasattr(shared, 'word_emb_lookups'):
        X = shared.word_emb_lookups.forward(X)

    return X.float()

def test(dhandler, net, device, config, logger, writer, shared,
         loss_func=None, accuracy_func=None, samples_to_plot=4,
         pre_training=False, plot_samples=False):
    """Test the network on the provided datset.

    Args:
        (....): See docstring of function :func:`train`.
        plot_samples (bool, optional): Whether to plot samples.
        samples_to_plot (int, optional): The number of samples to plot.
        pre_training (bool, optional): Indicates whether this testing is
            taking place before training.

    Returns:
        (tuple): Tuple containing:

        - **loss** (list): The test loss.
        - **accuracy** (list): The test accuracy. ``None`` for 
          non-classification experiments.
    """
    net.eval()

    with torch.no_grad():

        # Reset the data.
        dhandler.reset_batch_generator()

        # Get all test data
        sample_ids = dhandler.get_test_ids()
        X = dhandler.input_to_torch_tensor( \
            dhandler.get_test_inputs(), device, mode='inference',
            sample_ids=sample_ids)
        X = preprocess_inputs(X, shared)
        T = dhandler.output_to_torch_tensor( \
            dhandler.get_test_outputs(), device, mode='inference',
            sample_ids=sample_ids)
        batch_size = X.shape[1]

        # Make predictions.
        Y_logits = net.forward(X)

        # Plot samples if required.
        if plot_samples:
            inputs = dhandler._flatten_array(X[:, :samples_to_plot, :].cpu(),
                                             ts_dim_first=True)
            outputs = dhandler._flatten_array(T[:, :samples_to_plot, :].cpu(),
                                              ts_dim_first=True)
            predictions = dhandler._flatten_array(\
                          Y_logits[:, :samples_to_plot, :].cpu(),
                          ts_dim_first=True)
            figname = os.path.join(config.out_dir, 'teacher_data')
            if pre_training:
                figname += '_pre_train'
            dhandler.plot_samples('teacher data', inputs, num_samples_per_row=2,
                                  outputs=outputs, predictions=predictions,
                                  filename=figname, show=False)

        # Compute loss and accuracy.
        loss = loss_func(Y_logits, T, dhandler, None, None, sample_ids)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss/batch_size
        if accuracy_func is None:
            accuracy = None
        else:
            accuracy = accuracy_func(Y_logits, T, dhandler, sample_ids)

    return loss, accuracy

def train(dhandler, net, device, config, logger, writer, shared, 
          loss_func=None, accuracy_func=None):
    """Train the network on the provided datset.

    Args:
        dhandlers: The dataset handler.
        net: The network.
        device: Torch device (cpu or gpu).
        config (argparse.Namespace): The command line arguments.
        logger: Console (and file) logger.
        writer: The tensorboard summary writer.
        task_loss_func (fct_handler): The function to compute task loss.
        accuracy_func (fct_handler): The function to compute the accuracy, only
            necessary for classification tasks. ``None```otherwise.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.

    Returns:
        (dict): A dictionary containing training information.
    """
    logger.info('Training the network...')
    dhandler.reset_batch_generator()
    classification = not accuracy_func is None

    ### Determine parameters to be optimized.
    params = net.parameters()

    ### Define the optimizer.
    optimizer = get_optimizer_from_net(net, lr=config.lr,
        adam_beta1=config.adam_beta1, use_adam=not config.dont_use_adam,
        weight_decay=config.weight_decay, reservoir=config.reservoir)

    ### Start training.
    total_samples = 0
    summary_dict = init_summary_dict(classification=classification)
    loss_train = []
    acc_train = []
    # results = {'accuracy': [], 'loss': []}
    mat = []
    task_loss_per_ts = None
    train_losses = []
    test_losses = []
    for curr_iter in range(config.n_iter):
        net.train()

        # set optimizer to zero
        optimizer.zero_grad()

        ### Grab the current batch.
        batch = dhandler.next_train_batch(config.batch_size, return_ids=True)
        X = dhandler.input_to_torch_tensor(batch[0], device, mode='train',
                                           sample_ids=batch[2])
        T = dhandler.output_to_torch_tensor(batch[1], device, mode='train',
                                            sample_ids=batch[2])
        batch_size = X.shape[1]
        total_samples += batch_size
        X_before_preprocessing = X
        X = preprocess_inputs(X, shared)

        ### Pass through the network.
        Y_logits = net.forward(X)

        ### Compute the loss and accuracy.
        task_loss = loss_func(Y_logits, T, dhandler, None, None, batch[2])
        if isinstance(task_loss, tuple):
            task_loss_per_ts = task_loss[1]
            mat.append(task_loss_per_ts.cpu().detach().numpy())
            task_loss = task_loss[0]
        loss = task_loss

        # Add orthogonal regularization to hidden-to-hidden weights.
        ortho_loss = None
        if config.orthogonal_hh_reg > 0:
            ortho_loss = orthogonal_regularizer(config, net)
            loss += ortho_loss

        loss = loss/batch_size
        task_loss = task_loss/batch_size
        if accuracy_func is None:
            accuracy = None
        else:
            accuracy = accuracy_func(Y_logits, T, dhandler, batch[2])
        loss_train.append(loss.item())
        acc_train.append(accuracy)

        ### Backpropagate and update the weights.
        loss.backward()
        if config.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(net.parameters(), config.clip_grad_norm)

        if config.clip_grad_value > 0:
            nn.utils.clip_grad_value_(net.parameters(), config.clip_grad_value)

        optimizer.step()

        ### Log training progress.
        if curr_iter % 10 == 0:
            if config.save_logs:
                net.save_logs(writer, curr_iter)
                writer.add_scalar('train/loss', loss, curr_iter)
                if accuracy_func is not None:
                    writer.add_scalar('train/accuracy', accuracy, curr_iter)
            test_loss, acc = test(dhandler, net, device, config, logger, writer,
                                  shared, loss_func=loss_func,
                                  accuracy_func=accuracy_func)
            test_losses.append(test_loss.item())
            train_losses.append(task_loss.item())

            # Log.
            acc_msg = ''
            ort_msg = ''
            if accuracy_func is not None:
                acc_msg = 'Accuracy: {:.3f}, '
            if config.orthogonal_hh_reg > 0:
                 ort_msg = ', Ortho loss: {:.3f}'
            msg = 'Training step {}: ' + acc_msg + 'Task loss: {:.5f}, ' + \
                'Test task loss: {:.5f}' + ort_msg

            if accuracy_func is not None:
                logger.info(msg.format(curr_iter, accuracy, task_loss, \
                    test_loss, ortho_loss))
            else:
                logger.info(msg.format(curr_iter, task_loss, test_loss,
                            ortho_loss))            

            # Save summary dict.
            file_path = os.path.join(config.out_dir, 'summary_dict.json')
            with open(file_path, 'w') as f:
                json.dump(summary_dict, f)

    # Log final results.
    logger.info('Trained with %i samples.' % total_samples)
    summary_dict['loss_train_last'] = loss_train[-1]
    summary_dict['loss_train_best'] = np.min(np.array(loss_train))
    summary_dict['loss_train_all'] = train_losses
    summary_dict['loss_test_all'] = test_losses
    if accuracy_func is not None:
        summary_dict['acc_train_last'] = acc_train[-1]
        summary_dict['acc_train_best'] = np.max(np.array(acc_train))

    if config.save_logs and task_loss_per_ts is not None:
        # Plot per timestep loss and save to tensorboard.
        fig, ax = plt.subplots()
        mat = np.array(mat)
        ax.imshow(mat)
        ax.set_xlabel('time')
        ax.set_ylabel('iters')
        writer.add_figure('train/loss_per_ts_fig', fig, close=True)

    # Save summary dict.
    file_path = os.path.join(config.out_dir, 'summary_dict.json')
    with open(file_path, 'w') as f:
        json.dump(summary_dict, f)

    return summary_dict