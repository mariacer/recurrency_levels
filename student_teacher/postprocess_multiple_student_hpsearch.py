#!/usr/bin/env python3
# Copyright 2020 by Alexander Meulemans
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
# @title           :student_teacher/postprocess_multiple_student_hpsearch.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :27/07/2021
# @version         :1.0
# @python_version  :3.7
"""
Postprocessing file for multiple student hpsearches
---------------------------------------------------

This script mainly copies the histogram figures of multiple student teacher
experiments into a common folder for ease of inspection.
"""
import __init__

import argparse
import os
from shutil import copyfile

def run():
    """Run the post-processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('hpsearch_dir', type=str,
                        help='Directory of the hpsearch to post-process.')
    args = parser.parse_args()

    if not os.path.exists(args.hpsearch_dir):
        raise ValueError('The requested hpsearch directory does not exist.')

    # Create the directory where to gather the figures.
    figure_dir = os.path.join(args.hpsearch_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    # Copy one by one the figures into that folder.
    num_failed = 0
    num_total = 0
    for el in os.listdir(args.hpsearch_dir):
        src = os.path.join(args.hpsearch_dir, el)
        if os.path.isdir(src):
            dst = os.path.join(figure_dir, el) + '.png'
            src_file = os.path.join(os.path.join(src, 'tmp'),'results_test.png')
            src_file2 = os.path.join(os.path.join(src, 'tmp'),'results_train.png')
            num_total += 1
            try:
                copyfile(src_file, dst)
                copyfile(src_file2, dst)
            except:
                num_failed += 1
                pass
    print('Number of failed copies: %i/%i.' % (num_failed, num_total))

if __name__=='__main__':
    run()   