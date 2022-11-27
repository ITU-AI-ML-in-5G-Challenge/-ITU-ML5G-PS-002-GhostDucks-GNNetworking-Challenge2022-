import os
import yaml
from pathlib import Path
import datetime
from joblib import Parallel, delayed
import copy
from evaluate import my_evaluate as n_evaluate
import socket

def n_trainings_caller(jobs, eval = True):
    args = jobs[0]['train_args']
    n_workers = min(args.n_workers_train, len(jobs))
    parrallel =  n_workers > 1
    if parrallel:
        Parallel(n_jobs=n_workers, verbose=51, backend='multiprocessing')(delayed(train_single)(**job_args) for job_args in jobs)
    else:
        for i, job_args in enumerate(jobs):
            train_single(**job_args)
    if eval:
        save_path = Path(args.save_path)
        test_data_path = args.test_data_path
        n_workers = min(args.n_workers_val, len(jobs))
        n_gpus = args.ngpus_val
        ckpt_paths = []
        for job in jobs:
            ckpt_paths.append( str(next(save_path.glob(job['task_name'] + '*'))))
        n_evaluate(ckpt_paths, test_data_path, 'val', n_workers, n_gpus)




def train_single(train_args, task_name = '', gpu=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import custom_train

    args = train_args
    args.task_name = task_name + datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    print(f'Start training:{args.task_name}')
    nsamples = custom_train.get_nsamples_(args)
    print(f'approximate number of samples: {nsamples}')

    epoch_steps = args.epoch_steps if args.epoch_steps > 0 else nsamples
    config = vars(args)
    config['nsamples'] = nsamples
    config['epoch_steps_actual'] = epoch_steps
    decay_steps = epoch_steps * args.decay_epochs
    config['decay_steps'] = decay_steps
    config['hostname'] = socket.gethostname()
    # save config
    config_ = copy.deepcopy(config)
    if args.sample_train_mode == 'list':
        config_['sample_val'] = None
    print(yaml.dump(config_))
    task_path = Path(args.save_path) / args.task_name
    if not task_path.exists():
        task_path.mkdir(parents=True)
    with open(task_path / 'config.yaml', 'w') as fp:
        yaml.dump(config_, fp)

    # if args.use_clearml:
    #     log_utils.init_clearml(args.task_name)

    custom_train.main(args, args.data_dir, final_evaluation=args.final_evaluation, val_steps_during_train=args.val_steps, check_size=False,
         steps_per_epoch=epoch_steps, epochs=args.epochs, test_path=args.test_data_path, lr=args.learning_rate,
         decay=args.decay, decay_steps=decay_steps, log_sample_loss=args.log_sample_loss, shuffle_train=args.shuffle_train)
