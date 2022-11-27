import gzip
import os
import pickle


def pickle_write(file_path, data, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return


def pickle_gzip_write(file_path, data):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return


def pickle_read(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_gzip_read(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
