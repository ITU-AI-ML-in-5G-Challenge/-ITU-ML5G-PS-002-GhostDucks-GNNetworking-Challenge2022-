import os
from pathlib import Path
import shutil
import tarfile
import argparse
import re
import random

def pkl_to_tar(pkl_path):
    suffix = re.search('_s_(\d+)\.pkl', pkl_path)

    tar_path = pkl_path.replace('generated_datasets_pkl', 'generated_datasets')
    tar_path = tar_path.replace(suffix.group(0), '.tar.gz')
    idx_in_tar = int(suffix.group(1))
    return Path(tar_path), idx_in_tar


def delete_lines(path, keep):
    with open(path, 'r') as fp:
        line = list(fp.readlines())[keep]

    with open(path, 'w') as fp:
        fp.write(line)


def prepare_submission_data(samples_path, tgt_dir, name):
    with open(samples_path, 'r') as fp:
        pkls_orig = [s.strip() for s in fp.readlines()]

    # datanetAPI training is done using a fixed sample order given by random.Random(1234).shuffle().
    # our training, uses our pkl data generator without any shuffling.
    # so in order for our pkl-trained result to reproduce using tar.gz training with datanetAPI data generator
    # we need to reorder our examples to reflect the datanetAPI sample ordering
    print('reshuffling original order to fit with Random(1234)')
    order = list(range(100))
    random.Random(1234).shuffle(order)
    m = {v: i for i, v in enumerate(order)}
    pkls = [pkls_orig[m[i]] for i in range(100)]

    print(f'found {len(pkls)} pickle samples in {samples_path.name}')

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


def copy_sample_locally(src, idx_in_tar, tgt_idx, data_dir, graphs_dir, routes_dir, name):
    cwd = os.getcwd()

    with tarfile.open(src, 'r:gz') as tar:
        dir_info = tar.next()
        dir_name = dir_info.name[:-1] if dir_info.name.endswith('/') else dir_info.name

        # unzip the archive and rename the top directory
        tar.extractall(data_dir)

    new_name = f'results_datasets_{name}_smp_{tgt_idx:03d}_{tgt_idx}_{tgt_idx}'
    extracted = data_dir / new_name
    shutil.move(data_dir / dir_name, extracted)

    # remove everything except the pkl sample line
    with open(extracted / 'input_files.txt', 'r') as fp:
        nlines = len(fp.readlines())

    if nlines > 1:
        delete_lines(extracted / 'input_files.txt', keep=idx_in_tar)
        delete_lines(extracted / 'linkUsage.txt', keep=idx_in_tar)
        delete_lines(extracted / 'simulationResults.txt', keep=idx_in_tar)
        delete_lines(extracted / 'stability.txt', keep=idx_in_tar)
        delete_lines(extracted / 'traffic.txt', keep=idx_in_tar)

    # copy original graph and routing, rename the files to avoid duplicates and update new input_files.txt
    with open(extracted / 'input_files.txt', 'r') as fp:
        input_files_line = fp.readline().strip()

    tgt_g = f'graph_{tgt_idx:05d}.txt'
    tgt_r = f'routing_{tgt_idx:05d}'
    _, graph_name, route_name = input_files_line.split(';')
    shutil.copyfile(src.parent / 'graphs' / graph_name, graphs_dir / tgt_g)
    shutil.copytree(src.parent / 'routings' / route_name, routes_dir / tgt_r)

    # write new input_files.txt
    tgt_input_files_line = f'{tgt_idx};{tgt_g};{tgt_r}'
    with open(data_dir / new_name / 'input_files.txt', 'w') as fp:
        fp.write(tgt_input_files_line + '\n')

    # zip the updated directory structure
    os.chdir(data_dir)
    with tarfile.open(f'{new_name}.tar.gz', "w:gz") as tgt_tar:
        tgt_tar.add(new_name)

    shutil.rmtree(new_name)
    os.chdir(cwd)

    return new_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--samples_path', help='path to samples file', required=True)
    parser.add_argument('-n', '--name', help='name of data', default='sub')
    parser.add_argument('-o', '--output_dir', help='path where to save tarfiles', default='.')
    args = parser.parse_args()

    p_samples = Path(args.samples_path)
    tgt_dir = Path(args.output_dir) / 'data_tars'
    name = args.name

    prepare_submission_data(p_samples, tgt_dir, name)
