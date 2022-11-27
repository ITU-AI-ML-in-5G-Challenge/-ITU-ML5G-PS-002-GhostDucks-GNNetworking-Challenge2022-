import copy
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_exploration.utils import get_data_sample, iter_mat_indices

time_distribution_types = ['Poisson', 'CBR', 'ON-OFF']


def is_routing_path_symmetric(sample):
    routing_mat = sample.get_routing_matrix()
    nodes = sample.get_network_size()

    route_lengths = []
    symmetric = asymmetric = 0
    for i, j in iter_mat_indices(nodes):
        if routing_mat[i, j] == routing_mat[j, i][::-1]:
            symmetric += 1
        else:
            asymmetric += 1
        route_lengths.append(len(routing_mat[i, j]))

    return symmetric, asymmetric, route_lengths


def time_dist_to_str(time_dist):
    # see RouteNet_Fermi.datanetAPI.TimeDist
    if time_dist == 0 or time_dist == 1:
        return time_distribution_types[time_dist]
    elif time_dist == 4:
        return time_distribution_types[2]
    else:
        raise Exception('Invalid time_dist value')


def get_flows_per_link(sample):
    flows_per_link = Counter()
    for i, j in iter_mat_indices(sample.get_network_size()):
        curr_route = sample.get_routing_matrix()[i][j]
        for k in range(len(curr_route) - 1):
            link_id = tuple(sorted((curr_route[k], curr_route[k + 1])))
            flows_per_link[link_id] += 1
    return Counter(flows_per_link.values())


def get_flows_stats(sample):
    traff_mat = sample.get_traffic_matrix()
    num_nodes = sample.get_network_size()

    # number of flows per node
    num_flows = []
    for i, j in iter_mat_indices(num_nodes):
        num_flows.append(len(traff_mat[i, j]['Flows']))
        if len(traff_mat[i, j]['Flows']) != 1:
            raise Exception('more than 1 flow per src-dst pair')

    # time_distribution of each flow
    sample_stats_list = []
    for i, j in iter_mat_indices(num_nodes):
        for flow in traff_mat[i, j]['Flows']:
            flow_stats = copy.deepcopy(flow)
            flow_stats['TimeDist'] = time_dist_to_str(flow['TimeDist'])

            if flow_stats['TimeDist'] == 'ON-OFF':
                flow_stats['TimeOnOff'] = \
                    (flow_stats['TimeDistParams']['AvgTOn'], flow_stats['TimeDistParams']['AvgTOff'])

            size_dist_params = flow_stats['SizeDistParams']

            flow_stats['NumPackageSizeCandidates'] = int(size_dist_params['NumCandidates'])

            flow_stats['PackageSizes'] = \
                tuple([size_dist_params[f'Size_{i}'] for i in range(flow_stats['NumPackageSizeCandidates'])])
            flow_stats['PackageProbs'] = \
                tuple([size_dist_params[f'Prob_{i}'] for i in range(flow_stats['NumPackageSizeCandidates'])])
            flow_stats['AvgPktSize'] = size_dist_params['AvgPktSize']
            del flow_stats['TimeDistParams']
            del flow_stats['SizeDistParams']
            del flow_stats['SizeDist']
            sample_stats_list.append(flow_stats)

    return sample_stats_list


def explore(save_path, data, name=''):
    figs_path = os.path.join(save_path, 'figs')
    os.makedirs(figs_path, exist_ok=True)
    sample_stats = []
    route_lengths = []
    symmetric = asymmetric = 0
    flows_per_link = Counter()

    for sample in get_data_sample(data):
        # check if routing symmetric
        t_sym, t_asym, t_len = is_routing_path_symmetric(sample)
        route_lengths.extend(t_len)
        symmetric += t_sym
        asymmetric += t_asym
        sample_stats.extend(get_flows_stats(sample))
    print(f'Symmetric routes count: {symmetric}')
    print(f'Asymmetric routes count: {asymmetric}')

    for sample in get_data_sample(data):
        flows_per_link.update(get_flows_per_link(sample))
    plt.figure()
    plt.hist(flows_per_link.keys(), weights=flows_per_link.values(), bins=range(max(flows_per_link.keys())))
    plt.title(f'flows per link of graphs of size {name}')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'flows per link of graphs of size {name}.png'))
    plt.close()

    y = np.bincount(np.array(route_lengths, dtype=int))
    ii = np.nonzero(y)[0]
    total_count = np.sum(y)
    print('Route Lengths')
    for val, count in zip(ii, y[ii]):
        print(f'length: {val}\t count: {count}\t ratio: {count / total_count:.3f}')

    df = pd.DataFrame(sample_stats)

    # print time dist
    time_dist = df['TimeDist'].value_counts(normalize=True)
    print('Packets Time Distribution')
    for t_dist in time_distribution_types:
        print(f'{t_dist.ljust(8)} \t{time_dist[t_dist]:.3f}')

    on_off_time_dist = df['TimeOnOff'].value_counts(normalize=True).to_dict()
    print('ON-OFF Times Distribution')
    for k, v in on_off_time_dist.items():
        print(*k, sep=', ', end=' ')
        print(f'\t{v}')

    # print ToS
    tos_dist = df['ToS'].value_counts(normalize=True).to_dict()
    print('ToS Distribution')
    for i in range(3):
        t_tos = tos_dist.get(i, 0)
        print(f'ToS {i}\t{t_tos:.3f}')

    # print avg bandwidth
    avg_bw = np.array(df['AvgBw'].to_list(), dtype=float)
    plt.figure()
    plt.hist(avg_bw, bins=range(math.floor(min(avg_bw)), math.ceil(max(avg_bw))))
    plt.title(f'bandwidth hist of graphs of size {name}')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'average_bandwidth_{name}.png'))
    plt.close()

    high_val_bw = avg_bw[avg_bw > 3000]
    if high_val_bw.size > 0:
        plt.figure()
        plt.hist(high_val_bw, bins=range(math.floor(min(high_val_bw)), math.ceil(max(high_val_bw))))
        plt.title(f'high average bandwidth histogram of graphs of size {name}')
        plt.show()
        plt.savefig(os.path.join(figs_path, f'high_average_bandwidth_{name}.png'))
        plt.close()

    num_candidates = df['NumPackageSizeCandidates'].value_counts(normalize=True).to_dict()
    for i in range(6):
        print(f'{i} Package Size Candidates {num_candidates.get(i, 0):.3f}')

    # packet size is between 256 and 2000 bits
    pkg_sizes = df['PackageSizes'].value_counts(normalize=True).to_dict()
    print('Package Sizes Distribution')
    for k, v in pkg_sizes.items():
        print(f'{" ".join(str(t) for t in k)}\t{v}')

    pkg_probs = df['PackageProbs'].value_counts(normalize=True).to_dict()
    print('Package Probabilities Distribution')
    for k, v in pkg_probs.items():
        print(*k, sep=', ', end=' ')
        print(f'\t{v}')
