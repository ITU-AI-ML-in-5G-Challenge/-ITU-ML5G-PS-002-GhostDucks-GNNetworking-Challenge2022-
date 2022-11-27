from argparse import ArgumentParser
import random_train.train as trainer
from pathlib import Path
import datetime
import common.utils.pickling as pickling
import random_train.data_utils as data_utils
from tqdm import tqdm
from joblib import Parallel, delayed
import copy
import random_train.utils as train_utils
import numpy as np
from evaluate import my_evaluate as n_evaluate

def my_bool(s):
    return s != 'False'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-d', '--data-dir', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl/')
    parser.add_argument('--batches', default='all', help='options: {all, list of batches, file filepath}')
    parser.add_argument('-t', '--test-data-path', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/validation_dataset_pkl/')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--log_sample_loss', type=my_bool, default=False)
    parser.add_argument('-task_name', type=str, default='default_task')
    parser.add_argument('--train_mode', type=str, choices=['n_random', 'n_permute','random_replace','train_single'], default='n_random')
    parser.add_argument('--k_replace', type=int, default=5, help = 'number of sample to be replaced')
    parser.add_argument('--n_trainings', type = int, default=1)
    parser.add_argument('-w_train','--n_workers_train', type=int, default=1)
    parser.add_argument('-w_val','--n_workers_val', type=int, default=1)
    parser.add_argument('-sample_mode','--sample_train_mode', type = str, choices=['all','file','len'], default = 'all')
    parser.add_argument('--sample_val', default=100) # par gets value accoridng to smaple mode - file path for 'file,' int for 'len'.
    parser.add_argument('--ngpus_train', type=int, default=-1)
    parser.add_argument('--ngpus_val', type=int, default=-1)
    parser.add_argument('-nval', '--val-steps', type=int, default=0)
    parser.add_argument('-epochs', '--epochs', type=int, default=20)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.005)
    parser.add_argument('-steps', '--epoch-steps', type=int, default=2000)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--decay-epochs', type=int, default=0)
    parser.add_argument('--use_clearml', type=my_bool, default=False)
    parser.add_argument('--load_from_ckpt', type=my_bool, default=True)
    parser.add_argument('--save_best_only', type=my_bool, default=False)
    parser.add_argument('--ckpt_weights_dir', type=str, default='../RouteNet_Fermi/initial_weights/initial_weights')
    parser.add_argument('-final_eval', '--final_evaluation', type=my_bool, default=False)
    parser.add_argument('-shuf', '--shuffle_train', type=my_bool, default=False)
    args = parser.parse_args()
    args.task_name = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S_') + args.task_name
    return args


"""
evaluate
"""
def n_evaluate_samples(args, test_dirs = None):
    """
    Get per sample evaulaiton on test and train dataset
    """
    if not test_dirs:
        test_root = Path(args.save_path)
        test_dirs = sorted(list(x for x in test_root.iterdir() if x.is_dir())) if args.n_trainings>1 else [test_root]
    test_data_path = args.test_data_path
    jobs = [(test_dir, test_data_path) for test_dir in test_dirs]
    n_workers = args.n_workers_val
    parrallel =  n_workers > 1
    if parrallel:
        Parallel(n_jobs=n_workers, verbose=51)(delayed(evaluate_samples_single)(*args) for args in jobs)
    else:
        for test_dir in tqdm(test_dirs):
            evaluate_samples_single(test_dir, test_data_path)




def evaluate_samples_single(test_dir, test_data_path, checkpoint = 'best'):
    """
    test_dir - test directory
    test_data_path - test val set, get original val or train set using sample file.
    checkpoint - how to choose snapshot
    """
    import custom_train
    custom_evaluate = custom_train.evaluate

    def log_loss(mean_loss, sample_losses, eval_dir, ckpt, ds):
        # save as pkl
        eval_val_fp = eval_dir / '{}_{}.pkl'.format(ds, ckpt)
        pickling.pickle_write(eval_val_fp, [mean_loss, sample_losses])

        # save as txt
        f_sample_loss = eval_dir / '{}_{}_sample'.format(ds, ckpt)
        with open(f_sample_loss, 'w') as fp:
            for i, (path, loss, count) in enumerate(sample_losses):
                fp.write(f'{i},{str(path)},{loss},{count}\n')

        f_mean_loss = eval_dir / '{}_{}_mean'.format(ds, ckpt)
        with open(f_mean_loss, 'a') as fp:
            fp.write(f'{ckpt_nm[-1]}: {mean_loss:.4f}\n')

    def get_checkpoint(test_dir, checkpoint):
        if checkpoint == 'last':
            # get last ckpt
            ckpt_fps = (test_dir / 'modelCheckpoints').glob('*.index')
            ckpt_nm = sorted([f.name.split('.index')[0] for f in ckpt_fps])
            ckpt_fp = test_dir / 'modelCheckpoints' / ckpt_nm[-1]
        elif checkpoint == 'best':
            res = train_utils.get_eval_loss_single(test_dir)
            best_idx = res[1][res[0].index('best epoch')]
            ckpt_fp = list((test_dir / 'modelCheckpoints').glob(f'{best_idx}*.index'))[0]
            ckpt_nm = ckpt_fp.name
            ckpt_fp = str(ckpt_fp)
        return ckpt_fp, ckpt_nm

    last_ckpt_fp, ckpt_nm = get_checkpoint(test_dir, checkpoint)
    # validation loss
    eval_dir = test_dir / 'Loss'
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True)

    mean_loss, sample_losses = custom_evaluate(last_ckpt_fp, test_path=test_data_path)
    log_loss(mean_loss, sample_losses, eval_dir, ckpt_nm[-1], 'val')

    # train loss
    sample_file = test_dir / 'samples'
    samples_files = data_utils.sample_dataset(sample_file, 'file', task_path = test_dir)
    mean_loss, sample_losses = custom_evaluate(last_ckpt_fp, files=samples_files)
    log_loss(mean_loss, sample_losses, eval_dir, ckpt_nm[-1], 'train')

"""
train
"""
def random_replace_training(args):
    """
    given init samples set (namely model that gets best)
    replace one sample with k options
    train model with each and choose next best model
    """
    #init step
    NGPUS = args.ngpus_train
    save_path_ = Path(args.save_path)
    all_samples = get_all_train_samples(args.data_dir, datasets=args.batches)
    if args.sample_train_mode == 'file':
        sample_file = Path(args.sample_val)
    elif args.sample_train_mode == 'len':
        task_name = 'Phase#0_Exp#0'
        trainer.train_single(args, task_name=task_name)
        ckpt_path = str(next(save_path_.glob(task_name + '*')))
        n_workers = min(args.n_workers_val, args.epochs)
        n_gpus = args.ngpus_val
        n_evaluate([ckpt_path], args.test_data_path, 'val', n_workers, n_gpus)
        test_dir = list(save_path_.iterdir())[0]
        sample_file = test_dir / 'samples'

    samples_best = data_utils.sample_dataset(sample_file, 'file')
    # get k random sets
    samples = data_utils.sample_replace(samples_best, all_samples,
                                          k=args.k_replace, Nsets=args.n_trainings-1)
    samples.append(np.random.permutation(samples_best))

    ii = 1
    args.sample_train_mode = 'list'
    while True:
        # define jobs
        jobs = []
        save_path = save_path_ / 'Phase#{}'.format(ii)
        if save_path.exists():
            ii += 1
            continue
        args.save_path = str(save_path)
        for i in range(args.n_trainings):
            args = copy.deepcopy(args)
            args.sample_val = samples[i]
            if NGPUS > 0:
                jobs.append({'train_args': args, 'task_name': 'Exp#{}_'.format(i), 'gpu': str(i % NGPUS)})
            else:
                jobs.append({'train_args': args, 'task_name': 'Exp#{}_'.format(i)})

        #train models
        trainer.n_trainings_caller(jobs)

        # evaluate
        test_dirs = [x for x in save_path.iterdir() if x.is_dir()] + [sample_file.parent]
        res = [train_utils.get_eval_loss_single(dir) for dir in test_dirs]
        col_idx = res[0][0].index('best loss')
        best = [res[i][1][col_idx] for i in range(len(res))]
        # choose best
        best_idx = np.argmin(best)
        best_path = save_path / 'best_path.txt'
        with open(best_path, 'w') as f:
            f.write(str(test_dirs[best_idx]) + ' ' + f'Loss:{best[best_idx]}')

        ii += 1
        # get k random sets
        sample_file = test_dirs[best_idx] / 'samples'
        samples_best = data_utils.sample_dataset(sample_file, 'file')
        samples = data_utils.sample_replace(samples_best, all_samples,
                                              k=args.k_replace, Nsets=args.n_trainings-1)
        samples.append(np.random.permutation(samples_best))


def n_random(args):
    N = args.n_trainings
    batches_lst = [] if args.batches=='all' else [s for s in args.batches.split(' ')]
    samples = data_utils.sample_dataset(args.sample_val, 'len', data_dir=args.data_dir,
                                          Nsets=N, batches_lst=batches_lst)
    NGPUS = args.ngpus_train
    args.sample_train_mode = 'list'
    save_path = Path(args.save_path)
    dirs_idx = []
    if save_path.exists():
        dirs = [x for x in save_path.iterdir() if x.is_dir()]
        dirs_idx = sorted([int(d.name.split('#')[1].split('_')[0]) for d in dirs])

    jobs = []
    for i in range(N):
        if dirs_idx.count(i)>0:
            continue
        args = copy.deepcopy(args)
        args.sample_val = samples[i]
        if NGPUS>0:
            jobs.append( {'train_args': args, 'task_name': 'Exp#{}_'.format(i), 'gpu': str(i % NGPUS) })
        else:
            jobs.append({'train_args': args, 'task_name': 'Exp#{}_'.format(i)})
    trainer.n_trainings_caller(jobs)

def n_permute(args):
    jobs = []
    sample_file = Path(args.sample_val)
    samples = data_utils.sample_dataset(sample_file, 'file')
    args.sample_train_mode = 'list'
    NGPUS = args.ngpus_train
    N = args.n_trainings
    for i in range(N):
        args = copy.deepcopy(args)
        args.sample_val = samples = np.random.permutation(samples)
        if NGPUS > 0:
            jobs.append({'train_args': args, 'task_name': 'Exp#{}_'.format(i), 'gpu': str(i % NGPUS)})
        else:
            jobs.append({'train_args': args, 'task_name': 'Exp#{}_'.format(i)})
    trainer.n_trainings_caller(jobs)

def get_all_train_samples(data_dir, datasets):
    if datasets.find('file')==0:
        file_path = datasets.split(' ')[1]
        with open(file_path, 'r') as f:
            samples = [x.strip() for x in f.readlines()]
    else:
        datasets = [] if datasets=='all' else [s for s in datasets.split(' ')]
        samples = data_utils.get_batch_pkl_samples(data_dir, batches_lst=datasets)
    return samples




if __name__ == '__main__':
    args = parse_args()
    if args.test_only:
        n_evaluate_samples(args)
    else:
        if args.train_mode=='train_single':
            trainer.train_single(args, gpu='0')
        elif args.train_mode=='n_random':
            n_random(args)
        elif args.train_mode=='n_permute':
            n_permute(args)
        elif args.train_mode=='random_replace':
            random_replace_training(args)
