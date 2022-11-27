import os
import sys

import numpy as np

eps = np.finfo(float).eps


def get_data_sample(data):
    counter = 0
    try:
        if isinstance(data, list):
            for topo in data:
                yield topo
                counter += 1
        elif isinstance(data, dict):
            for topo_list in data.values():
                for topo in topo_list:
                    yield topo
                    counter += 1
        else:
            # should be a single sample
            yield data
            counter += 1
    finally:
        print(f'---- Processed a total of {counter} samples ----')


def iter_mat_indices(n, get_sym=False):
    for i in range(n):
        for j in range(n):
            if not get_sym and i == j:
                continue
            yield i, j


class print_to_file:
    def __init__(self, save_dir, filename):
        self.save_dir = save_dir
        self.filename = filename
        self.file = None
        self.sys_stdout = None

    def __enter__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.sys_stdout = sys.stdout
        self.file = open(os.path.join(self.save_dir, f'{self.filename}.txt'), 'w')
        sys.stdout = self.file

    def __exit__(self, *_):
        sys.stdout = self.sys_stdout
        self.file.close()
