import copy
import datetime
import os
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors

import data_exploration.utils as data_exploration_utils
from RouteNet_Fermi import datanetAPI
from common.utils import pickling
from common.utils.pickling import pickle_read
from data_exploration import (
    explore_topology,
    explore_traffic_mat,
    explore_performance_mat,
    explore_link_utilization,
    explore_port_utilization,
)
from prepare_submission_data import pkl_to_tar, copy_sample_locally

# Range of the maximum average lambda | traffic intensity used
# max_avg_lambda_range = [min_value,max_value]
max_avg_lambda_range = [10, 10000]

# List of the network topology sizes to use
net_sizes = [50, 75, 100, 130, 170, 200, 240, 260, 280, 300]


def get_data(cfg: Namespace) -> defaultdict:
    samples_dict = defaultdict(list)
    if cfg.input_data_type == 'pkl':
        s_t = time.time()
        for net_size in cfg.topology_sizes:
            path_dir = os.path.join(cfg.source_data_path, str(net_size), 'samples.pkl')
            data = pickling.pickle_read(path_dir)
            samples_dict[net_size] = data
        print(f'Loading pickles takes {time.time() - s_t:.3f}')
    elif cfg.input_data_type == 'raw':
        s_t = time.time()
        reader = datanetAPI.DatanetAPI(cfg.source_data_path)
        for sample in tqdm.tqdm(reader):
            if cfg.split_filename:
                sample.data_set_file = sample.data_set_file.split('/')[-1]
            samples_dict[sample.get_network_size()].append(sample)
        print(f'Loading raw takes {time.time() - s_t:.3f}')
    else:
        raise Exception('Invalid read_from')

    if cfg.save_data_to_pkl:
        for key, val in samples_dict.items():
            path_dir = os.path.join(cfg.source_data_path, str(key)).replace(cfg.source_data_path, cfg.save_pkl_path)
            file_dir = os.path.join(path_dir, 'samples.pkl')
            pickling.pickle_write(file_dir, val, create_dir=True)
    return samples_dict


def get_slice_indices(conf: Union[str, list]) -> list:
    assert isinstance(conf, str)
    if conf.lower() == 'all':
        chosen_sizes = net_sizes
    else:
        chosen_sizes = [int(t) for t in conf.split(',') if int(t) in net_sizes]
        assert len(chosen_sizes) == len(conf.split(','))
    print(f'Topology Sizes for Data Exploration {chosen_sizes}')
    return chosen_sizes


def get_link_stats_of_generated_datasets(args):
    for i in range(100):
        tmp_args = copy.deepcopy(args)
        tmp_args.source_data_path = os.path.join(tmp_args.source_data_path, str(i))
        if os.path.isdir(tmp_args.source_data_path):
            samples_dict = get_data(tmp_args)
            explore_link_utilization.explore(samples_dict, f'all_{i}')


def get_topology_stats(args, samples_dict):
    save_path = os.path.join(args.stats_save_dir, 'topology_stats')

    if args.get_topology_stats_all:
        with data_exploration_utils.print_to_file(save_path, 'topology_stats_all'):
            explore_topology.explore_topology_features(save_path, samples_dict, name='all')

    if args.get_topology_stats_per_node_count:
        with data_exploration_utils.print_to_file(save_path, 'topology_stats_per_node_count'):
            for name, samples in samples_dict.items():
                print(f'processing graph of size {name}')
                explore_topology.explore_topology_features(save_path, samples, name)

    if args.get_topology_stats_per_sample:
        with data_exploration_utils.print_to_file(save_path, 'topology_stats_per_sample'):
            for sample in data_exploration_utils.get_data_sample(samples_dict):
                t_name = sample.data_set_file.replace('.', '_').replace('/', '_')
                print('processing sample ', t_name)
                explore_topology.explore_topology_features(save_path, sample, name=t_name)


def get_traffic_mat_stats(args, samples_dict):
    save_path = os.path.join(args.stats_save_dir, 'traffic_mat_stats')

    if args.get_traffic_mat_stats_all:
        with data_exploration_utils.print_to_file(save_path, 'traffic_mat_stats_all'):
            explore_traffic_mat.explore(save_path, samples_dict, name='all')

    if args.get_traffic_mat_stats_per_node_count:
        with data_exploration_utils.print_to_file(save_path, 'traffic_mat_stats_per_node_count'):
            for name, samples in samples_dict.items():
                print(f'processing graph of size {name}')
                explore_traffic_mat.explore(save_path, samples, name=name)

    if args.get_traffic_mat_stats_per_sample:
        with data_exploration_utils.print_to_file(save_path, 'traffic_mat_stats_per_sample'):
            for sample in data_exploration_utils.get_data_sample(samples_dict):
                t_name = sample.data_set_file.replace('.', '_').replace('/', '_')
                print('processing sample ', t_name)
                explore_traffic_mat.explore(save_path, sample, name=t_name)


def get_performance_mat_stats(args, samples_dict):
    save_path = os.path.join(args.stats_save_dir, 'performance_mat_stats')

    if args.get_performance_mat_stats_all:
        with data_exploration_utils.print_to_file(save_path, 'performance_mat_stats_all'):
            explore_performance_mat.explore(save_path, samples_dict, name='all')

    if args.get_performance_mat_stats_per_node_count:
        with data_exploration_utils.print_to_file(save_path, 'performance_mat_stats_per_node_count'):
            for name, samples in samples_dict.items():
                print(f'processing graph of size {name}')
                explore_performance_mat.explore(save_path, samples, name=name)

    if args.get_performance_mat_stats_per_sample:
        with data_exploration_utils.print_to_file(save_path, 'performance_mat_stats_per_sample'):
            for sample in data_exploration_utils.get_data_sample(samples_dict):
                t_name = sample.data_set_file.replace('.', '_').replace('/', '_')
                print('processing sample ', t_name)
                explore_performance_mat.explore(save_path, sample, name=t_name)


def get_link_load_stats(args, samples_dict):
    save_path = os.path.join(args.stats_save_dir, 'link_load_stats')

    if args.get_link_load_stats_all:
        with data_exploration_utils.print_to_file(save_path, 'link_load_stats_all'):
            explore_link_utilization.explore(save_path, samples_dict, 'all')

    if args.get_link_load_stats_per_node_count:
        with data_exploration_utils.print_to_file(save_path, 'link_load_stats_per_node_count'):
            for name, samples in samples_dict.items():
                print(f'processing graph of size {name}')
                explore_link_utilization.explore(save_path, samples, name=name)

    if args.get_link_load_stats_per_sample:
        with data_exploration_utils.print_to_file(save_path, 'link_load_stats_per_sample'):
            for sample in data_exploration_utils.get_data_sample(samples_dict):
                t_name = sample.data_set_file.replace('.', '_').replace('/', '_')
                explore_link_utilization.explore(save_path, sample, name=t_name)


def get_port_utilization_stats(args, samples_dict):
    save_path = os.path.join(args.stats_save_dir, 'port_utilization_stats')

    if args.get_port_utilization_stats_all:
        with data_exploration_utils.print_to_file(save_path, 'port_utilization_stats_all'):
            explore_port_utilization.explore(save_path, samples_dict, 'all')

    if args.get_port_utilization_stats_per_node_count:
        with data_exploration_utils.print_to_file(save_path, 'port_utilization_stats_per_node_count'):
            for name, samples in samples_dict.items():
                print(f'processing graph of size {name}')
                explore_port_utilization.explore(save_path, samples, name=name)

    if args.get_port_utilization_stats_per_sample:
        with data_exploration_utils.print_to_file(save_path, 'port_utilization_stats_per_sample'):
            for sample in data_exploration_utils.get_data_sample(samples_dict):
                t_name = sample.data_set_file.replace('.', '_').replace('/', '_')
                explore_port_utilization.explore(save_path, sample, name=t_name)


def prepare_submission_data_from_list(pkls, tgt_dir, name):
    # prepare target dir
    graphs_dir = tgt_dir / 'graphs'
    routes_dir = tgt_dir / 'routings'
    tgt_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    routes_dir.mkdir(exist_ok=True)

    src_tars = []
    for i, pkl_path in enumerate(pkls):
        tar_path, idx_in_tar = pkl_to_tar(pkl_path)
        if not tar_path.exists():
            raise FileNotFoundError(f'pickle: {pkl_path}\n tar: {tar_path}')

        src_tars.append(tar_path)
        copy_sample_locally(tar_path, idx_in_tar, i, tgt_dir, graphs_dir, routes_dir, name)

    with open(tgt_dir.parent / 'orig_tarfiles.txt', 'w') as fp:
        for tar_path in src_tars:
            fp.write(str(tar_path) + '\n')


def explore_closest(args: Namespace) -> None:
    # fixme - dirty code
    data_path = '/mnt/ext-10g/shared/Projects/GNNetworkingChallenge/trained_oracle_models/6.44/'
    val_embeddings_pkl = os.path.join(data_path, 'sample_embeddings_27-6.44_new/val_min_max_mean.pkl')
    val_losses_csv = os.path.join(data_path, 'losses_27-6.44/val_sample_loss_27-6.44.csv')
    train_embeddings_pkl = os.path.join(data_path, 'sample_embeddings_27-6.44_new/train_min_max_mean.pkl')
    train_losses_csv = os.path.join(data_path, 'losses_27-6.44/train_sample_loss_27-6.44.csv')

    val_embs = pickle_read(val_embeddings_pkl)
    train_embs = pickle_read(train_embeddings_pkl)

    # train_losses = pd.read_csv(train_losses_csv)
    val_losses = pd.read_csv(val_losses_csv)

    # mapping from path to index of embedding
    # p2e_tr = {p: i for i, p in enumerate(train_embs['paths'])}
    p2e_val = {p: i for i, p in enumerate(val_embs['paths'])}

    ntop = 1
    sort_key = 'loss'  # loss - according to current 100-model weighted loss
    sorted_val = val_losses.sort_values(by=sort_key, ascending=False)

    iival = [p2e_val[sorted_val.iloc[i].path] for i in range(ntop)]
    # indices of current train samples
    # iitr = set(p2e_tr[p] for p in train_losses['path'].values)
    val_embeddings = val_embs['embeddings'][iival]
    val_paths = [val_embs['paths'][i] for i in iival]

    n_neighbors = 10
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(train_embs['embeddings'])
    nn_dist, nn_idx = nn.kneighbors(val_embs['embeddings'][iival])

    closest_samples = [train_embs['paths'][i] for i in nn_idx[0]]

    extracted_files_path = Path(os.path.join(args.stats_save_dir, 'samples'))
    prepare_submission_data_from_list(closest_samples, extracted_files_path, '')
    args.source_data_path = extracted_files_path


def set_logging(args: Namespace) -> None:
    if not args.plot_to_gui:
        import matplotlib
        matplotlib.use('Agg')

    os.makedirs(args.stats_save_dir)
    print(f'Save dir - {args.stats_save_dir}')


def parse_args() -> Namespace:
    def to_bool(val):
        return val.lower() != 'false'

    default_save_dir = os.path.join('./exploration_results',
                                    datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S') + '')
    parser = ArgumentParser()

    parser.add_argument('-topology_sizes', type=str, default='50',
                        # parser.add_argument('-topology_sizes', type=str, default='all',
                        help='sizes divided by "," or "all"')
    parser.add_argument('-input_data_type', type=str, default='pkl',
                        help='read data from {pkl, raw}')
    parser.add_argument('-save_data_to_pkl', type=to_bool, default=False)
    parser.add_argument('-source_data_path', type=str,
                        default='./validation_dataset_pkl',
                        help='{./validation_dataset, ./validation_dataset_pkl,'
                             '/home/yakovl/dev/GNNetworkingChallenge/datasets/random_multi}')
    parser.add_argument('-save_pkl_path', type=str, default='./dataset_pkl')
    parser.add_argument('-stats_save_dir', type=str, default=default_save_dir)

    parser.add_argument('-get_topology_stats_all', type=to_bool, default=False)
    parser.add_argument('-get_topology_stats_per_node_count', type=to_bool, default=False)
    parser.add_argument('-get_topology_stats_per_sample', type=to_bool, default=False)

    parser.add_argument('-get_traffic_mat_stats_all', type=to_bool, default=False)
    parser.add_argument('-get_traffic_mat_stats_per_node_count', type=to_bool, default=False)
    parser.add_argument('-get_traffic_mat_stats_per_sample', type=to_bool, default=False)

    parser.add_argument('-get_performance_mat_stats_all', type=to_bool, default=False)
    parser.add_argument('-get_performance_mat_stats_per_node_count', type=to_bool, default=False)
    parser.add_argument('-get_performance_mat_stats_per_sample', type=to_bool, default=False)

    parser.add_argument('-get_link_load_stats_all', type=to_bool, default=False)
    parser.add_argument('-get_link_load_stats_per_node_count', type=to_bool, default=False)
    parser.add_argument('-get_link_load_stats_per_sample', type=to_bool, default=False)

    parser.add_argument('-get_port_utilization_stats_all', type=to_bool, default=False)
    parser.add_argument('-get_port_utilization_stats_per_node_count', type=to_bool, default=False)
    parser.add_argument('-get_port_utilization_stats_per_sample', type=to_bool, default=False)

    parser.add_argument('-split_filename', type=to_bool, default=False)
    parser.add_argument('-explore_closest', type=to_bool, default=False)
    parser.add_argument('-plot_to_gui', type=to_bool, default=False)

    args = parser.parse_args()
    args.topology_sizes = get_slice_indices(args.topology_sizes)

    return args


if __name__ == '__main__':
    args = parse_args()
    set_logging(args)
    if args.explore_closest:
        explore_closest(args)
    samples_dict = get_data(args)

    get_topology_stats(args, samples_dict)
    get_traffic_mat_stats(args, samples_dict)
    get_performance_mat_stats(args, samples_dict)
    get_link_load_stats(args, samples_dict)
    get_port_utilization_stats(args, samples_dict)
