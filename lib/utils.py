from __future__ import print_function
from __future__ import absolute_import
import errno
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def init_directories(exp_name, root_dir=None):
    if not root_dir:
        root_dir = os.environ['OUTPUT_DIR']
    
    dirs = dict()
    names = ['samples', 'ckpt'] #, 'plots', 'logs']
    for n in names:
        dirs[n] = os.path.join(root_dir, n, exp_name)     
    
    return dirs

def create_directories(dirs, train, save_codes):
    if train:
        keys = [k for k in dirs.keys() if (k!= 'codes' and k!= 'data')]
        for k in keys:
            mkdir_p(dirs[k])
    
    if save_codes:
        mkdir_p(dirs['codes'])