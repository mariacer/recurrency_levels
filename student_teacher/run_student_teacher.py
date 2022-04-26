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
# @title          :student_teacher/run_student_teacher.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Main script from which to run Student-Teacher experiments
---------------------------------------------------------
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from argparse import Namespace
from warnings import warn
import os

from networks.net_utils import generate_network
import student_teacher.st_utils as dutils
from utils.args import parse_cmd_arguments
import utils.sim_utils as sutils
import feedforwardness.utils as futils

def run(config=None):
    """Run the script."""

    ### Setup.
    if config is None:
        config = parse_cmd_arguments(experiment='student_teacher')
    device, writer, logger = sutils.setup_environment(config,
        script_name=globals()['__file__'])
    logger.info('Running student-teacher experiment.')

    ### Get the datahandler and loss function.
    dhandler = dutils.get_dhandler(config)
    loss_func = dutils.get_loss_func(config, device, logger)
    shared = Namespace()
    shared.feature_size = config.teacher_input_size
    shared.output_per_ts = True
    shared.n_ts = None

    ### Initialize the network.
    net = generate_network(config, shared, device, n_in=shared.feature_size,
                                                   n_out=dhandler.out_shape[0])
    if config.set_identical_topology:
        net.copy_connectivity(dhandler.teacher_rnn, device=device,
                              copy_weights_sign=config.set_same_weight_sign)
    
    # Sanity checks.
    same_topology, same_weights, same_signs = net.has_same_architecture(\
        dhandler.teacher_rnn, return_same_weight_sign=True)
    if config.set_identical_topology:
        assert same_topology
    if config.set_same_weight_sign:
        assert same_signs

    summary_dict = sutils.run_experiment(net, config, shared, dhandler, device, 
                                         logger, writer, loss_func, None)

    return summary_dict

if __name__=='__main__':
    run()
