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
# @title          :bio_rnn/run_audioset.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# python_version  :3.7
"""
Audioset experiments with a microcircuit
----------------------------------------
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from argparse import Namespace

from bio_rnn.bio_utils import generate_network
import real_world_benchmarks.audioset_utils as dutils
from utils.args import parse_cmd_arguments
import utils.sim_utils as sutils

def run():
    """Run the script."""

    ### Setup.
    config = parse_cmd_arguments(experiment='audioset_microcircuit')
    device, writer, logger = sutils.setup_environment(config,
        script_name=globals()['__file__'])   
    logger.info('Running Audioset experiment with a microcircuit.')

    ### Get the datahandler and loss function.
    dhandler = dutils.get_dhandler(config)
    loss_func = dutils.get_loss_func(config, device, logger)
    accuracy_func = dutils.get_accuracy_func(config) 
    shared = Namespace()
    shared.feature_size = dhandler.in_shape[0]
    shared.output_per_ts = False
    shared.n_ts = 10

    ### Initialize the network.
    net = generate_network(config, shared, device, n_in=shared.feature_size,
                                                   n_out=dhandler.out_shape[0])

    summary_dict = sutils.run_experiment(net, config, shared, dhandler, device, 
                                         logger, writer, loss_func,
                                         accuracy_func)

    return summary_dict

if __name__=='__main__':
	run()
