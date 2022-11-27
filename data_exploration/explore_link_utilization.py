import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from data_exploration.utils import get_data_sample, iter_mat_indices, eps

hist_step = 0.02


def get_link_load(data: defaultdict) -> np.array:
    all_link_loads = []
    all_flows_per_link = []

    for sample in get_data_sample(data):

        link_capacities, routes, avg_bw = [dict() for _ in range(3)]

        for i, j in iter_mat_indices(sample.get_network_size()):
            # extract link bandwidth - the matrix is symmetrical (bw(i->j) == bw(j->i))
            try:
                link_properties = sample.get_link_properties(i, j)
                link_properties_sym = sample.get_link_properties(j, i)
            except:
                try:
                    link_properties = sample.topology_object[i][j]
                    link_properties_sym = sample.topology_object[j][i]

                except:
                    raise Exception('Can not parse input data')

            if link_properties:
                # if nodes are connected by link
                link_capacities[(i, j)] = link_properties['bandwidth']
                # validate that symmetrical
                assert link_properties['bandwidth'] == link_properties_sym['bandwidth']

            # extract routing path
            curr_route = sample.get_routing_matrix()[i, j]
            if len(curr_route):
                routes[(i, j)] = curr_route

            # extract average bandwidth between source and destination (there is always a single flow for each src-dst)
            avg_bw[(i, j)] = sample.get_traffic_matrix()[i, j]['Flows'][0]['AvgBw']

        # extract average bandwidth off all edges
        avg_bw_per_link = defaultdict(lambda: 0)
        flows_per_link = defaultdict(lambda: 0)
        for src_dst, route in routes.items():
            for i in range(len(route) - 1):
                link_id = tuple(sorted((route[i], route[i + 1])))
                avg_bw_per_link[link_id] += avg_bw[src_dst]
                flows_per_link[link_id] += 1

        # calculate load per link - as in RouteNet-Fermi
        load_per_link = dict()
        for src_dst, bw in avg_bw_per_link.items():
            load = (1 / link_capacities[src_dst]) * bw
            load_per_link[src_dst] = load
            all_link_loads.append(load)

        for nflows in flows_per_link.values():
            all_flows_per_link.append(nflows)

    return np.asarray(all_link_loads), np.asarray(all_flows_per_link)


def explore(save_path: str, data: defaultdict, name: str = '') -> None:
    figs_path = os.path.join(save_path, 'figs')
    os.makedirs(figs_path, exist_ok=True)

    link_loads, flows_per_link = get_link_load(data)

    plt.figure()
    bins = np.arange(start=0 - hist_step / 2,
                     stop=np.ceil(np.max(link_loads)) + hist_step / 2 + eps,
                     step=hist_step)
    plt.hist(link_loads, bins=bins)
    plt.title(f'Link loads hist of graphs of size {name}')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'link_loads_{name}.png'))
    plt.close()

    plt.figure()
    plt.hist(link_loads, bins=bins)
    plt.title(f'Link loads log hist of graphs of size {name}')
    plt.show()
    plt.yscale('log')
    plt.savefig(os.path.join(figs_path, f'link_loads_logs_{name}.png'))
    plt.close()

    plt.figure()
    bins = np.arange(start=0 - hist_step / 2,
                     stop=np.ceil(np.max(flows_per_link)) + hist_step / 2 + eps,
                     step=hist_step)
    plt.hist(flows_per_link, bins=bins)
    plt.title(f'Num flows per link hist of graphs of size {name}')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'flows_per_link_{name}.png'))
    plt.close()
