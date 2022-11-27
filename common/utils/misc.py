from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union


def load_samples(path):
    with open(path, 'r') as f:
        samples = [x.strip() for x in f.readlines()]
    return samples


def save_samples(sample_paths, path):
    with open(path, 'w') as fp:
        for smp in sample_paths:
            fp.write(smp + '\n')


def net_size(nflows):
    return ((1+np.sqrt(1+4*nflows))/2).astype(np.int32)


def load_sample_loss_csv(path):
    df = pd.read_csv(path)
    if len(df.columns) == 5:
        df.columns = ['idx', 'path', 'loss', 'flows', 'net_size']
    else:
        df.columns = ['idx', 'path', 'loss', 'flows']
    df = df.drop(['idx'], axis=1)
    if 'net_size' not in df.columns:
        df['net_size'] = net_size(df['flows'])
    df['wloss'] = df.loss * df.flows / df.flows.sum()
    df['dset'] = df.path.str.split('/').str[7]    # assuming path starts with /mnt/ext/.../generated...
    return df


def load_sample_losses(losses_dir=None, loss_files=None):
    if loss_files is None:
        losses_dir = Path(losses_dir)
        loss_files = list(sorted(losses_dir.glob('*_sample_loss*.csv')))

    data = []
    for p in loss_files:
        df = load_sample_loss_csv(p)
        df['loss_file'] = p.name
        data.append(df)

    df = pd.concat(data)
    return df


def splice_to_end(samples: pd.DataFrame, rm: Union[pd.DataFrame, pd.Series], add: Union[list[str], pd.DataFrame, pd.Series]):
    keep = samples.drop(rm.index, axis=0)
    if isinstance(add, list):
        add = pd.Series(add)
    elif isinstance(add, pd.DataFrame):
        add = add.path
    new100 = pd.concat((keep.path, add))
    return new100


def splice_and_shuffle(samples: pd.DataFrame, rm: Union[pd.DataFrame, pd.Series], add: Union[list[str], pd.DataFrame, pd.Series]):
    keep = samples.drop(rm.index, axis=0)
    if isinstance(add, list):
        add = pd.Series(add)
    elif isinstance(add, pd.DataFrame):
        add = add.path
    new100 = pd.concat((keep.path, add))
    new100 = new100.sample(frac=1)
    return new100


def save_list(samples: Union[pd.DataFrame, pd.Series], save_dir: Path):
    if isinstance(samples, pd.DataFrame):
        samples = samples.path
    assert len(samples) == 100

    if not save_dir.exists():
        save_dir.mkdir()

    samples.to_csv(save_dir / 'samples.txt', index=False, header=False)