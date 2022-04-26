#!/usr/bin/env python3
# Copyright 2020 Alexander Meulemans
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
# @title          :utils/config_utils.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :03/11/2020
# @version        :1.0
# python_version  :3.7
"""
Script for utilities for dealing with the configs.
"""
import os
import sys

def _override_cmd_arg(config):
    """Override command line arguments.

    Args:
        config: The command line configuration.
    """
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)

def _args_to_cmd_str(cmd_dict, script_name):
    """Translate a dictionary of argument names to values into a string that
    can be typed into a console.

    Copied from <https://github.com/chrhenning/hypercl>_.

    Args:
        cmd_dict: Dictionary with argument names as keys, that map to a value.
        script_name (str): The name of the script.

    Returns:
        A string of the form:
            python3 train.py --ARG1=VAL1 ... --out_dir=OUT_DIR
    """
    script_name = os.path.basename(script_name)
    cmd_str = 'python3 %s' % script_name

    for k, v in cmd_dict.items():
        if k == 'out_dir':
            # Write the output directory at the very end of the command line.
            out_dir = v
            continue
        if type(v) == bool:
            cmd_str += ' --%s' % k if v else ''
        else:
            cmd_str += ' --%s=%s' % (k, str(v))
    cmd_str += ' --%s=%s' % ('out_dir', str(out_dir))

    return cmd_str
