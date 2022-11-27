import argparse
from common.utils.pickling import pickle_write
import pandas as pd
from RouteNet_Fermi.datanetAPI import DatanetAPI
import tqdm
from data_exploration.explore_link_utilization import get_link_load
from data_exploration.explore_performance_mat import get_flow_stats


class DataReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __iter__(self):
        tool = DatanetAPI(self.data_dir, shuffle=False, seed=None)
        it = iter(tool)

        prev_file = ''
        counter = 0
        i = 0

        for sample in it:
            sample_path = sample.data_set_file

            if sample_path == prev_file:
                counter += 1
            else:
                prev_file = sample_path
                counter = 0

            sample_id = sample_path.replace('.tar.gz', f'_s_{counter}.pkl')
            yield sample_id, sample


def collect_sample_stats(paths):
    reader = DataReader(paths)
    percentiles = [.1, .25, .5, .7, .8, .9]
    sample_ids, results = [], []
    for sample_id, sample in tqdm.tqdm(reader, desc='collecting stats'):
        # link loads
        link_loads = pd.Series(get_link_load([sample])[0], name='LinkLoads')
        # delay, drops
        flow_stats = get_flow_stats([sample])

        stats = pd.concat((link_loads.describe(percentiles=percentiles),
                           flow_stats[['PktsDrop', 'AvgDelay']].describe(percentiles=percentiles)), axis=1)

        # flatten the stats to create one row per sample
        results.append(pd.concat([stats[c] for c in stats.columns]).values)
        sample_ids.append(sample_id)
        # results[sample_id] = pd.concat([stats[c] for c in stats.columns])

    results = pd.DataFrame(results, index=sample_ids,
                           columns=[f'{c}_{r}' for c in stats.columns for r in stats.index])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help='files with samples list', required=True)
    parser.add_argument('-o', '--save-path', help='output file path', default='./stats.csv')
    args = parser.parse_args()

    stats = collect_sample_stats(args.data_path)
    stats.to_csv(args.save_path)
    #pickle_write(args.save_path, stats)
