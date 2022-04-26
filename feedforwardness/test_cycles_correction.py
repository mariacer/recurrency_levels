#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
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
# @title          :feedforwardness/test_cycles_correction.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :12/06/2021
# @version        :1.0
# python_version  :3.7
"""
Test the correction of Christian for the recurrency metric, as well as the
recursive algorithm to count cycles.
"""
import __init__
import numpy as np
import torch

from feedforwardness.utils import compute_number_closed_walks, \
    iteratively_compute_number_cycles, count_cycles

np.random.seed(42)
torch.manual_seed(32)

N = 4 # The number of nodes.
threshold = 0.5 # The fraction of missing connections.

# Generate a random adjacency matrix.
A = torch.rand(N, N)
A[A > threshold] = 1
A[A <= threshold] = 0
A = torch.tensor([[1., 0., 0., 0., 0.],
                [0., 0., 0., 1., 1.],
                [1., 0., 0., 1., 0.],
                [0., 1., 0., 0., 1.],
                [1., 0., 1., 1., 1.]])
print(A)

######## Recurrency metric correction #########

if False:
    cycles, _ = compute_number_closed_walks(A, N=N, apply_correction=True)
    print('Cycles: ', cycles)

######## Recursive algorithm #########

if True:
    for n in range(1, 5):
        cycles, _ = iteratively_compute_number_cycles(A, N=n)
        print('Cycles: ', cycles)
