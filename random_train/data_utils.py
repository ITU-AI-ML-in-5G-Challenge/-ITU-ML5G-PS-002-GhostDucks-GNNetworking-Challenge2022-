import os
import numpy as np
from pathlib import Path

def get_all_pkl_samples(data_dir):
    samples = []
    for root, _, files in os.walk(data_dir):
        samples.extend([os.path.join(os.path.abspath(root), f) for f in files if f.endswith("pkl")])
    samples.sort()
    return samples

def get_batch_pkl_samples(data_dir, batches_lst = []):
    if not batches_lst:
        samples = get_all_pkl_samples(data_dir)
    else:
        samples = []
        data_dir = Path(data_dir)
        for b in batches_lst:
            batch_dir = data_dir / b
            samples = samples + list(batch_dir.glob('**/*.pkl'))
        samples = [str(s) for s in samples]
        samples.sort()
    return samples


def sample_dataset(sample_val, smaple_mode, data_dir =None, task_path = None,
                   Nsets=1, batches_lst = []):
    """
    get sampels from file or create random n samples
    Nset - #number of different random sets

    """
    # get samples
    if smaple_mode=='len':
        N = int(sample_val)
        all_samples = np.array(get_batch_pkl_samples(data_dir, batches_lst = batches_lst))
        if Nsets>1:
            samples = [sorted(np.random.choice(all_samples, N, replace=False)) for _ in range(Nsets)]
        else:
            samples = sorted(np.random.choice(all_samples, N, replace=False))
    elif smaple_mode=='file':
        file_path = sample_val
        with open(file_path, 'r') as f:
            samples = [x.strip() for x in f.readlines()]
    elif smaple_mode=='list':
        samples = sample_val
    # save samples
    if task_path:
        task_file_path = task_path / 'samples'
        if not task_file_path.exists():
            with open(task_file_path, 'w') as f:
                f.write('\n'.join(samples))
    return samples


def sample_replace(sub_samples, all_samples, k =1, Nsets =1, shuffle = True, fix_repalce = False):
    """
    k number of candidate to replace
    n number of sets
    """
    samples = []
    # random replace candidate
    sub_samples = np.array(sub_samples)
    # get replace unique and diff samples
    set_diff = np.array(list(set(all_samples) - set(sub_samples)))
    if fix_repalce:
        idx_rep = np.random.choice(len(sub_samples), size=k, replace=False)
        idx_same = np.full(len(sub_samples), True)
        idx_same[idx_rep] = False
        idx_same = np.where(idx_same)[0]
        for i in range(Nsets):
            samples_rep = sorted(np.random.choice(set_diff, size=k, replace=False))
            jsamples = np.concatenate((sub_samples[idx_same], samples_rep))
            if shuffle:
                jsamples = np.random.permutation(jsamples)
            else:
                jsamples = sorted(jsamples)
            # np.where([s[-3:] != 'pkl' for s in jsamples])
            samples.append(jsamples)
    else:
        # random replace candidate
        for i in range(Nsets):
            idx_rep = np.random.choice(len(sub_samples), size=k, replace=False)
            idx_same = np.full(len(sub_samples), True)
            idx_same[idx_rep] = False
            idx_same = np.where(idx_same)[0]

            samples_rep = sorted(np.random.choice(set_diff, size=k, replace=False))
            jsamples = np.concatenate((sub_samples[idx_same], samples_rep))
            if shuffle:
                jsamples = np.random.permutation(jsamples)
            else:
                jsamples = sorted(jsamples)
            # np.where([s[-3:] != 'pkl' for s in jsamples])
            samples.append(jsamples)
    return samples




