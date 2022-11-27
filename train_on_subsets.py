import datetime
import os
from time import time
import pandas as pd
from pathlib import Path
from joblib.parallel import Parallel, delayed
from argparse import Namespace, ArgumentParser
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def train_on_file_set(samples_file, name, save_path, val_data_path,
                      epochs=20, steps_per_epoch=2000, val_steps_during_train=130,
                      gpu='', seed=None, shuffle=False):
    print(f'Experiment id:{name}')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import custom_train

    save_path = Path(save_path)
    task_name = f'{name}_' + datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    task_path = save_path / task_name
    if not task_path.exists():
        task_path.mkdir(parents=True)

    args = Namespace()
    args.task_name = task_name
    args.save_path = save_path
    args.sample_train_mode = 'file'
    args.sample_val = samples_file
    # args.data_dir = train_data_path
    args.data_dir = '/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl'
    args.save_best_only = False
    args.load_from_ckpt = True
    args.ckpt_weights_dir = '/mnt/ext/shared/Projects/GNNetworkingChallenge/RouteNet_Fermi/initial_weights/initial_weights'

    print('shuffle train: ', shuffle)
    print('gpu: ', gpu)
    ret = custom_train.main(args, args.data_dir, val_steps_during_train=val_steps_during_train,
                            epochs=epochs, steps_per_epoch=steps_per_epoch, check_size=False, test_path=val_data_path,
                            log_sample_loss=True, seed=seed, shuffle_train=shuffle)

    return ret


def main(train_root='.', n_workers=1, n_gpus=8, random_seed=False, shuffle=False, nval=130):
    val_data_path = '/mnt/ext/shared/Projects/GNNetworkingChallenge/validation_dataset_pkl'
    epochs = 20
    steps_per_epoch = 2000
    val_steps_during_train = nval

    DEBUG = False
    if DEBUG:
        print('##### DEBUG ###################################')
        print('##### DEBUG ###################################')
        print('##### DEBUG ###################################')
        val_data_path = '/home/yakovl/dev/GNNetworkingChallenge/datasets/small_validation_dataset_pkl'
        epochs = 4
        steps_per_epoch = 10
        val_steps_during_train = 3

    if random_seed:
        print('##### RANDOM SEED USED ###################################')
        print('##### RANDOM SEED USED ###################################')
        print('##### RANDOM SEED USED ###################################')

    train_root = Path(train_root)
    job_files = sorted(train_root.glob('**/samples.txt'))

    job_args = []
    for i, path in enumerate(job_files):
        if random_seed:
            seed = random.randint(0, 1000000000)
        else:
            seed = None

        job_args.append({'samples_file': path,
                         'name': path.parent.name,
                         'save_path': path.parent,
                         'val_data_path': val_data_path,
                         'gpu': str(i % n_gpus) if n_gpus > 0 else '-1',
                         'epochs': epochs,
                         'steps_per_epoch': steps_per_epoch,
                         'val_steps_during_train': val_steps_during_train,
                         'seed': seed,
                         'shuffle': shuffle,
                        })

    if n_workers > 1:
        results = Parallel(n_jobs=n_workers, verbose=51, )(delayed(train_on_file_set)(**args) for args in job_args)
    else:
        results = []
        for args in job_args:
            print(args['name'])
            results.append(train_on_file_set(**args))

    if nval > 0:
        report = []
        for args, ret in zip(job_args, results):
            best = min(ret['logs'].values(), key=lambda x: x['loss'])
            last = max(ret['logs'].values(), key=lambda x: x['epoch'])
            report.append({
                'name': args['name'],
                'best_loss': best['loss'],
                'last_loss': last['loss'],
                'best_epoch': best['epoch'],
            })

        report = pd.DataFrame(report)
        print(report)
        report.to_csv(Path(train_root) / f'results.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-path', help='root directory where to look for sample files', default='.')
    parser.add_argument('-w', '--num-workers', help='number of joblib workers', type=int, default=1)
    parser.add_argument('-ngpu', '--num-gpus', help='number of gpus', type=int, default=1)
    parser.add_argument('-rndseed', '--random-seed', help='randomize seed', action='store_true')
    parser.add_argument('-shuffle', '--shuffle', help='shuffle train set', action='store_true')
    parser.add_argument('-nval', '--nval', type=int, help='val steps', default=130)
    args = parser.parse_args()

    main(train_root=args.data_path, n_workers=args.num_workers, n_gpus=args.num_gpus,
         random_seed=args.random_seed, shuffle=args.shuffle, nval=args.nval)
