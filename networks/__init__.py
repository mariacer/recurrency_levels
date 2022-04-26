import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
trgt_dir = os.path.abspath('..')
if curr_dir == 'networks' and '..' not in sys.path:
    sys.path.insert(0, '..')
