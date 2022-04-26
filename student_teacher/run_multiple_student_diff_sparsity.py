#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
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
# @title          :student_teacher/run_multiple_student_diff_sparsity.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :12/04/2021
# @version        :1.0
# python_version  :3.7
"""
Compare students with different levels of sparsity.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import matplotlib.pyplot as plt
import numpy as np
import os

from student_teacher.run_multiple_student_teacher import run

def plot_perf_vs_rec_sparsity(results, out_dir, test=True):
    """Get scatter plot of performance vs. recurrent sparsity level.

    Args:
        results (dict): The results dictionary.
        out_dir (str): The directory where to save the figures.
        test (boolean, optional): Whether test or training accuracies should
            be plotted.
    """

    key = 'loss_test_last'
    mode = 'test'
    if not test:
        key = 'loss_train_last'
        mode = 'train'

    sp_teacher = results['rec_sparsity_teacher'][0]
    plt.figure()
    y_vals = results[key].tolist()[:-2]
    y_vals /= np.max(y_vals)
    plt.scatter(results['rec_sparsity_student'].tolist()[:-2], y_vals)
    plt.xlabel('student sparsity')
    plt.ylabel('%s loss (AU)'%mode)
    plt.plot([sp_teacher, sp_teacher], plt.gca().get_ylim(), '-r',
             label='teacher sparsity')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'perf_vs_rec_sparsity_%s.png' % mode))

if __name__ == '__main__':
    # Define the feature and values to be used.
    key = 'rec_sparsity'
    values = np.linspace(0, 1, 11)

    # Run the experiment with multiple students.
    _, results, out_dir = run(config_module='configs.vanilla_varying_sparsity',
                              student_key=key, values=values, plot=False,
                              return_all_results=True)

    # Plot the results.
    plot_perf_vs_rec_sparsity(results, out_dir, test=False)
    plot_perf_vs_rec_sparsity(results, out_dir)


