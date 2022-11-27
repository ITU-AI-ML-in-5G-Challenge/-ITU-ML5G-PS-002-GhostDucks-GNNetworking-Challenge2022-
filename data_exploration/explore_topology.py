import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_exploration.utils import get_data_sample

schedulingPolicy_t = ['FIFO', 'SP', 'WFQ', 'DRR']
schedulingPolicy_weighted = ['WFQ', 'DRR']


def get_node_properties(data):
    # preprocess raw data

    def get_sample_node_props(sample):
        # extract properties of a single topology
        nodes_properties = []
        for node_id in range(sample.get_network_size()):
            node_props = copy.deepcopy(sample.get_node_properties(node_id))
            node_props['networkSize'] = sample.get_network_size()

            buffer_sizes = node_props['bufferSizes']
            if isinstance(buffer_sizes, str):
                node_props['bufferSizes'] = [int(t) for t in buffer_sizes.split(',')]
                node_props['bufferSizeTotal'] = sum([int(t) for t in buffer_sizes.split(',')])
            else:
                node_props['bufferSizeTotal'] = buffer_sizes

            if node_props['schedulingPolicy'] in schedulingPolicy_weighted:
                node_props['schedulingWeights'] = [float(t) for t in node_props['schedulingWeights'].split(',')]
                node_props['bufferSizes_schedulingWeights'] = \
                    node_props['bufferSizes'] + node_props['schedulingWeights']

            nodes_properties.append(node_props)

        return nodes_properties

    topo_props = []

    for sample in get_data_sample(data):
        topo_props.extend(get_sample_node_props(sample))

    return pd.DataFrame(topo_props)


def get_link_bandwidth_stats(data):
    # Link bandwidth stats (must be between 10000 and 400000 and in multiples of 1000)

    def get_sample_bandwidth_stats(sample):
        edge_bandwidth = dict()
        for node_id in range(sample.get_network_size()):
            for neighbor_id in sample.get_topology_object()[node_id]:
                key = tuple(sorted([node_id, neighbor_id]))
                if key in edge_bandwidth:
                    continue
                try:
                    tmp_bandwidth = sample.get_srcdst_link_bandwidth(node_id, neighbor_id)
                except:
                    try:
                        tmp_bandwidth = sample.topology_object[node_id][neighbor_id]['bandwidth']
                    except:
                        raise Exception('get_sample_bandwidth_stats ERROR')
                edge_bandwidth[key] = tmp_bandwidth
        return np.fromiter(edge_bandwidth.values(), dtype=int)

    topo_props = []
    counter = 0
    if isinstance(data, list):
        for topo in data:
            topo_props.extend(get_sample_bandwidth_stats(topo))
            counter += 1
    elif isinstance(data, dict):
        for topo_list in data.values():
            for topo in topo_list:
                topo_props.extend(get_sample_bandwidth_stats(topo))
                counter += 1
    else:
        # should be a single sample
        topo_props.extend(get_sample_bandwidth_stats(data))
        counter += 1

    return np.array(topo_props)


def explore_topology_features(save_path, data, name):
    df = get_node_properties(data)
    figs_path = os.path.join(save_path, 'figs')
    os.makedirs(figs_path, exist_ok=True)

    # print Policy Distribution
    scheduling_policy_count_norm = df['schedulingPolicy'].value_counts(normalize=True)
    print('Scheduling Policy Distribution')
    for policy in schedulingPolicy_t:
        t_count_norm = scheduling_policy_count_norm[policy] if policy in scheduling_policy_count_norm else 0
        print(f'{policy.ljust(4)} \t'
              f'{t_count_norm:.3f}')

    # print buffer sizes and schedule weights of each policy
    for sch_policy in schedulingPolicy_t:
        print(f'{sch_policy} Buffer Size Distribution')
        tmp_df = df.loc[df["schedulingPolicy"] == sch_policy]
        print(tmp_df['bufferSizes'].value_counts(normalize=True).sort_index().to_string())

        if sch_policy in schedulingPolicy_weighted:
            if tmp_df['schedulingWeights'].size != 0:
                print(tmp_df['schedulingWeights'].value_counts(normalize=True).sort_index().to_string())
                print(tmp_df['bufferSizes_schedulingWeights'].value_counts(normalize=True).sort_index().to_string())

    link_bandwidth = get_link_bandwidth_stats(data)
    plt.hist(link_bandwidth, bins=range(min(link_bandwidth), max(link_bandwidth), 1000))
    plt.title(f'link bandwidth of {name} nodes networks')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'link_bandwidth_{name}.png'))
    plt.close()

    y = np.bincount(link_bandwidth)
    ii = np.nonzero(y)[0]
    total_count = np.sum(y)
    print('Per Link Bandwidth Stats')
    for val, count in zip(ii, y[ii]):
        print(f'val: {val}\t count: {count}\t ratio: {count / total_count:.3f}')
