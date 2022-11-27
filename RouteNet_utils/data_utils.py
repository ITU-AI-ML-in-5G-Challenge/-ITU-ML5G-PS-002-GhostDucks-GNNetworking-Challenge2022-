import argparse
import logging
import os
import pathlib
import pickle
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm

from RouteNet_Fermi.data_generator import network_to_hypergraph, hypergraph_to_input_data
from RouteNet_Fermi.datanetAPI import DatanetAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def approximate_dataset_size(data_dir, is_pkl):
    data_dir = pathlib.Path(data_dir)

    if is_pkl:
        total_files = len(list(data_dir.rglob('*.pkl')))
    else:
        files = data_dir.rglob('*.tar.gz')
        total_files = 0
        for f in files:
            strs = f.name[:-len('.tar.gz')].split('_')
            n = int(strs[-1]) - int(strs[-2]) + 1
            total_files += n

    return total_files


def original_data_generator(data_dir, shuffle, seed, training):
    """
    This function is a copy of RouteNet_Fermi.data_generator.generator
    """
    try:
        data_dir = data_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle, seed=seed)
    it = iter(tool)
    num_samples = 0

    for sample in it:
        num_samples += 1
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
        HG = network_to_hypergraph(G=G, R=R, T=T, P=P)

        ret = hypergraph_to_input_data(HG)
        num_samples += 1
        if training:
            if G.number_of_nodes() > 10:
                print("WARNING: The topology must have at most 10 nodes")
                continue
            if not all(8000 <= qs <= 64000 for qs in ret[0]["queue_size"]):
                print("WARNING: All Queue Sizes must be between 8000 and 64000")
                continue
            if not all(10000 <= c <= 400000 for c in ret[0]["capacity"]):
                print("WARNING: All Link Capacities must be between 8000 and 64000")
                continue
            if not all(256 <= t / p <= 2000 or np.isclose(t / p, 256, rtol=0.001) or np.isclose(t / p, 2000, rtol=0.001)
                       for t, p in zip(ret[0]["traffic"], ret[0]["packets"])):
                print("WARNING: All Packet Sizes must be between 256 and 2000")
                continue
        yield ret, sample.data_set_file


def store_ds_as_pkl(input_data_dir, output_data_dir, print_every=50):
    logging.info(f'input data dir: {os.path.abspath(input_data_dir)}')
    logging.info(f'output data dir: {os.path.abspath(output_data_dir)}')

    if not os.path.exists(input_data_dir):
        logging.error('input dir does not exist')
        raise Exception('no input data dir')

    gen = original_data_generator(data_dir=input_data_dir, shuffle=False, seed=None, training=False)
    start_time = time.time()

    prev_file = ''
    counter = 0
    i = 0
    for (sample, sample_dir) in tqdm(gen):
        if sample_dir == prev_file:
            counter += 1
        else:
            prev_file = sample_dir
            counter = 0

        out_file_dir = pathlib.Path(
            str(output_data_dir) + prev_file.removeprefix(str(input_data_dir)).replace('.tar.gz', f'_s_{counter}.pkl'))

        if not os.path.exists(os.path.dirname(out_file_dir)):
            os.makedirs(os.path.dirname(out_file_dir))
            logging.info(f'created output dir: {os.path.abspath(os.path.dirname(out_file_dir))}')

        if not out_file_dir.exists():
            i += 1
            with open(out_file_dir, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if i == 1 or i % print_every == 0:
                runtime = time.time() - start_time
                print(f'processed {i} samples in {runtime:.2f} seconds')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='../datasets/')
    parser.add_argument('--output_data_dir', type=str, default='../datasets_pkl/')
    parser.add_argument('--print_every', type=int, default=50)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    store_ds_as_pkl(args.input_data_dir, args.output_data_dir, args.print_every)
