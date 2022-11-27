import copy
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_exploration.utils import get_data_sample, eps

utilization_bins_step = 0.05
losses_bins_step = 0.02
pkt_size_bins = 10
avg_port_bins = 50
max_q_bins = 50


def get_flow_stats(data: defaultdict) -> tuple[[pd.DataFrame] * 4]:
    port_utilization_list = []
    qos_lists = [[] for _ in range(3)]

    for sample in get_data_sample(data):
        all_node_stats = sample.get_port_stats()
        for node_id, node_stats in enumerate(all_node_stats):
            for link_id, port_utilization_orig in node_stats.items():
                port_utilization = copy.deepcopy(port_utilization_orig)
                port_utilization['id'] = (node_id, link_id)
                del port_utilization['qosQueuesStats']
                port_utilization_list.append(port_utilization)

                qos_queue_stats = port_utilization_orig['qosQueuesStats']
                for idx, qos_port_stats in enumerate(qos_queue_stats):
                    qos_lists[idx].append(qos_port_stats)

    return pd.DataFrame(port_utilization_list), *[pd.DataFrame(t) for t in qos_lists]


def plot_histograms(save_path: str, port_util_df: pd.DataFrame, name: str = '') -> None:
    utilization = np.asarray(port_util_df['utilization'], dtype=float)
    bins = np.arange(start=0 - utilization_bins_step / 2,
                     stop=1 + utilization_bins_step / 2 + eps,
                     step=utilization_bins_step)
    plt.figure()
    plt.hist(utilization, bins=bins)
    plt.title(f'Port utilization of graphs of size {name}, bins {utilization_bins_step}')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(save_path, f'port_utilization_{name}.png'))
    plt.close()

    losses = np.asarray(port_util_df['losses'], dtype=float)
    bins = np.arange(start=0 - losses_bins_step / 2,
                     stop=1 + losses_bins_step / 2 + eps,
                     step=losses_bins_step)
    plt.figure()
    plt.hist(losses, bins=bins)
    plt.title(f'Port losses of graphs of size {name}, bins {losses_bins_step}')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(save_path, f'port_losses_{name}.png'))
    plt.close()

    avg_pkt_size = np.asarray(port_util_df['avgPacketSize'], dtype=float)
    bins = np.arange(start=np.floor(np.min(avg_pkt_size)) - pkt_size_bins / 2,
                     stop=np.ceil(np.max(avg_pkt_size)) + pkt_size_bins / 2 + eps,
                     step=pkt_size_bins)
    plt.figure()
    plt.hist(avg_pkt_size, bins=bins)
    plt.title(f'Port avg packet size of graphs of size {name}, bins {pkt_size_bins}')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(save_path, f'port_avg_pkt_size_{name}.png'))
    plt.close()

    if 'qos' in name:
        avg_port_occ = np.asarray(port_util_df['avgPortOccupancy'], dtype=float)
        bins = np.arange(start=np.floor(np.min(avg_port_occ)) - avg_port_bins / 2,
                         stop=np.ceil(np.max(avg_port_occ)) + avg_port_bins / 2 + eps,
                         step=avg_port_bins)
        plt.figure()
        plt.hist(avg_port_occ, bins=bins)
        plt.title(f'Avg port occupancy of graphs of size {name}, bins {avg_port_bins}')
        plt.yscale('log')
        plt.show()
        plt.savefig(os.path.join(save_path, f'port_avg_occupancy_{name}.png'))
        plt.close()

        max_q_occupancy = np.asarray(port_util_df['maxQueueOccupancy'], dtype=float)
        bins = np.arange(start=np.floor(np.min(max_q_occupancy)) - max_q_bins / 2,
                         stop=np.ceil(np.max(max_q_occupancy)) + max_q_bins / 2 + eps,
                         step=max_q_bins)
        plt.figure()
        plt.hist(max_q_occupancy, bins=bins)
        plt.title(f'Max queue occupancy of graphs of size {name}, bins {max_q_bins}')
        plt.yscale('log')
        plt.show()
        plt.savefig(os.path.join(save_path, f'port_max_q_occupancy_{name}.png'))
        plt.close()


def explore(save_path: str, data: defaultdict, name: int) -> None:
    figs_path = os.path.join(save_path, 'figs')
    port_util_dfs = get_flow_stats(data)
    df_types = ('total', 'qos_0', 'qos_1', 'qos_2')

    for port_util_df, df_type in zip(port_util_dfs, df_types):
        tmp_save_path = os.path.join(figs_path, df_type)
        os.makedirs(tmp_save_path, exist_ok=True)
        plot_histograms(tmp_save_path, port_util_df, '_'.join((str(name), df_type)))
