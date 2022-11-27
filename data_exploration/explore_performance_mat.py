import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_exploration.utils import get_data_sample, iter_mat_indices, eps

packets_drop_bins_step = 0.05  # fixme - move to argparse
packets_delay_bins_step = 0.1  # fixme - move to argparse


def get_flow_stats(data: defaultdict) -> pd.DataFrame:
    flows_list = []
    for sample in get_data_sample(data):
        for i, j in iter_mat_indices(sample.get_network_size()):
            flows_list.append(sample.get_performance_matrix()[i, j]['Flows'][0])  # each src-dst have only one flow

    return pd.DataFrame(flows_list)


def explore(save_path: str, data: defaultdict, name: str) -> None:
    figs_path = os.path.join(save_path, 'figs')
    os.makedirs(figs_path, exist_ok=True)

    flows_df = get_flow_stats(data)

    packets_drop = np.asarray(flows_df['PktsDrop'], dtype=float)  # this is the ratio of dropped packages
    bins = np.arange(start=0 - packets_drop_bins_step / 2,
                     stop=1 + packets_drop_bins_step / 2 + eps,
                     step=packets_drop_bins_step)
    plt.figure()
    plt.hist(packets_drop, bins=bins)
    plt.title(f'Packets drop of graphs of size {name}, bins {packets_drop_bins_step}')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'performance_mat_packets_drop_{name}.png'))
    plt.close()

    average_delay = np.asarray(flows_df['AvgDelay'], dtype=float)
    bins = np.arange(start=0 - packets_delay_bins_step / 2,
                     stop=np.ceil(np.max(average_delay)) + packets_delay_bins_step / 2 + eps,
                     step=packets_delay_bins_step)
    plt.figure()
    plt.hist(average_delay, bins=bins)
    plt.title(f'Packets delay of graphs of size {name}, bins {packets_delay_bins_step}')
    plt.yscale('log')
    plt.show()
    plt.savefig(os.path.join(figs_path, f'performance_mat_packets_delay_{name}.png'))
    plt.close()
