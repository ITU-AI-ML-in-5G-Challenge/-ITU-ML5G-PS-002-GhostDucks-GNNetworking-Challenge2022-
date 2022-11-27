import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from common.utils.pickling import pickle_write, pickle_read


def embed_min_max_mean(x):
    return np.concatenate((x.min(0), x.max(0), x.mean(0)))


def embed_sample(path, path_aggregate):
    smp = pickle_read(path)

    ps = smp['path_state']  # [F, maxlen, 32]
    # average over the path length dimension, ignoring zero places
    mask = np.expand_dims(ps.sum(-1) != 0, -1)  # [F, maxlen, 1]
    if path_aggregate == 'mean':
        ps = ps.mean(1, where=mask)  # [F, 32]
    elif path_aggregate == 'sum':
        ps = ps.sum(1, where=mask)
    elif path_aggregate == 'sumcpc':
        cpc = np.expand_dims(smp['path_capacity'], -1)
        ps = np.divide(ps, cpc/10000, where=mask)
        ps = ps.sum(1, where=mask)
    else:
        raise RuntimeError('path_aggregate invalid')
    ps = embed_min_max_mean(ps)  # [96]

    lns = embed_min_max_mean(smp['link_state'])  # [96]
    qs = embed_min_max_mean(smp['queue_state'])  # [96]
    return np.concatenate((ps, lns, qs)), smp['fpath']


def embed_paths(paths, save_path, path_aggregate):
    emb_dim = len(embed_sample(paths[0], path_aggregate=path_aggregate)[0])
    emb = np.zeros((len(paths), emb_dim))

    sample_paths = []
    for i, p in tqdm(enumerate(paths)):
        emb[i], fpath = embed_sample(path=p, path_aggregate=path_aggregate)
        sample_paths.append(fpath)

    if save_path is not None:
        print('saving embeddings to:', save_path)
        pickle_write(save_path, {'embeddings': emb, 'paths': sample_paths})

    return emb, paths


def embed_dir(emb_root, save_path, path_aggregate):
    paths = list(sorted(Path(emb_root).glob('**/*.pkl')))
    return embed_paths(paths, save_path, path_aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--embeddings-path', required=True)
    parser.add_argument('-o', '--output-file', default='embeddings.pkl')
    parser.add_argument('-pa', '--path-aggregate', choices=['mean', 'sum', 'sumcpc'], default='mean')
    args = parser.parse_args()

    embed_dir(args.embeddings_path, args.output_file, args.path_aggregate)
