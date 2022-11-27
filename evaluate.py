import os
import time
from argparse import ArgumentParser
from pathlib import Path
from joblib.parallel import Parallel, delayed
import pandas as pd
from common.utils.pickling import pickle_write


def evaluate_one(checkpoint_path, test_data_path, gpu='', name=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # print('gpu ', gpu)
    # mean_loss = 0.12345
    # sample_losses = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    # return mean_loss, sample_losses

    ckpt = Path(checkpoint_path)
    output_dir = ckpt.parent.parent / f'eval'
    out_file = output_dir / f'{name}_eval_{ckpt.name}.log'
    if out_file.exists():
        print('skip already exists', output_dir.name, out_file)
        return None

    from custom_train import evaluate as custom_evaluate
    if os.path.isfile(test_data_path):
        with open(test_data_path, 'r') as fp:
            files = [s.strip() for s in fp.readlines()]
        mean_loss, sample_losses = custom_evaluate(checkpoint_path, files=files)
    else:
        mean_loss, sample_losses = custom_evaluate(checkpoint_path, test_data_path)

    if not output_dir.exists():
        output_dir.mkdir()
    sample_loss_path = output_dir / f'{name}_sample_loss_{ckpt.name}.csv'
    sample_losses.to_csv(sample_loss_path)

    with open(out_file, 'w') as fp:
        fp.write(f'{ckpt.name}: {mean_loss:.4f}\n')

    # print(f'{output_dir.parent}: {ckpt.name}: {mean_loss:.3f}')

    return None #mean_loss, sample_losses


def collect_results(eval_dir, name):
    eval_dir = Path(eval_dir)
    files = list(sorted(eval_dir.glob(f'{name}_eval_*.log')))
    epoch_losses = []
    for f in files:
        with open(f, 'r') as fp:
            line = fp.readline().strip()
        ep = int(line.split('-')[0])
        loss = float(line.split(':')[1])
        epoch_losses.append({'epoch': ep, 'loss': loss, 'ckpt': line.split(':')[0]})

    epoch_losses = pd.DataFrame(epoch_losses)
    epoch_losses.to_csv(eval_dir.parent / f'{name}_eval.log', index=False)

    # val_sample_loss_all.pkl
    out = {}
    out['epoch_losses'] = epoch_losses.set_index('epoch', drop=True)['loss'].to_dict()
    out['sample_losses'] = {}

    prefix = f'{name}_sample_loss_'
    files = list(sorted(eval_dir.glob(f'{prefix}*.csv')))
    for f in files:
        ep = int(f.name[len(prefix):].split('-')[0])
        df = pd.read_csv(f).iloc[:,1:]
        out['sample_losses'][ep] = df

    out['epoch_losses2'] = epoch_losses
    pickle_write(eval_dir.parent / f'{name}_sample_loss_all.pkl', out)


def already_evaluated(ckpt, name):
    ckpt = Path(ckpt)
    eval_file = ckpt.parent.parent / 'eval' / f'{name}_eval_{ckpt.name}.log'
    return eval_file.exists()


def find_checkpoints(ckpt_paths, name, allow_evaluated=False):
    ckpts = []
    for p in ckpt_paths:
        if os.path.exists(p + '.index'):
            ckpts.append(p)
        elif os.path.isdir(p):
            ckpts.extend([str(cc).replace('.index','')
                          for cc in sorted(Path(p).glob('**/modelCheckpoints/*.index'))])

    if not allow_evaluated:
        nall = len(ckpts)
        ckpts = [c for c in ckpts if not already_evaluated(c, name)]
        print(f'found {len(ckpts)} new checkpoints and {nall - len(ckpts)} already evaluated')
    return ckpts


def my_evaluate(ckpt_paths, test_data_path, name, n_workers, n_gpus):
    ckpts = find_checkpoints(ckpt_paths, name)
    if len(ckpts) == 0:
        print('no new checkpoints found')
        return

    results = []
    for i in range(0, len(ckpts), n_workers):
        ickpts = ckpts[i:i+n_workers]
        job_args = []
        for j, ckpt in enumerate(ickpts):
            job_args.append({'checkpoint_path': ckpt,
                             'test_data_path': test_data_path,
                             'gpu': str(j % n_gpus) if n_gpus > 0 else '-1',
                             'name': name,
                             })

        if n_workers > 1:
            iresults = Parallel(n_jobs=n_workers, verbose=51)(delayed(evaluate_one)(**iargs) for iargs in job_args)
        else:
            iresults = []
            for iargs in job_args:
                results.append(evaluate_one(**iargs))

        results.extend(iresults)

    # collect results
    eval_dirs = set(Path(path).parent.parent / 'eval' for path in ckpts)
    for eval_dir in eval_dirs:
        collect_results(eval_dir, name)

    #
    # for iargs, (mean_loss, sample_losses) in zip(job_args, results):
    #     ckpt = Path(iargs['checkpoint_path'])
    #     output_dir = ckpt.parent.parent / f'eval'
    #     if not output_dir.exists():
    #         output_dir.mkdir()
    #     sample_loss_path = output_dir / f'{name}_sample_loss_{ckpt.name}.csv'
    #     sample_losses.to_csv(sample_loss_path)
    #
    #     out_file = os.path.join(output_dir, f'{name}_eval.log')
    #     with open(out_file, 'a') as fp:
    #         fp.write(f'{ckpt.name}: {mean_loss:.4f}\n')
    #
    #     print(f'{output_dir.parent}: {ckpt.name}: {mean_loss:.3f}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint-path', nargs='+')
    parser.add_argument('-data', '--test-data-path', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/validation_dataset_pkl')
    parser.add_argument('-orig', '--orig', action='store_true')
    parser.add_argument('-name', '--name', default='val')
    parser.add_argument('-w', '--workers', type=int, default=1)
    parser.add_argument('-ngpus', '--num-gpus', help='number of gpus', type=int, default=0)
    parser.add_argument('-wait', '--wait', action='store_true',
                        help='wait for checkpoints even if there arent any (runs forever)')
    parser.add_argument('-cleanup', '--cleanup', action='store_true')

    args = parser.parse_args()

    if args.orig:
        # original evaluate
        from RouteNet_Fermi import evaluate
        evaluate(args.checkpoint_path)
    elif args.cleanup:
        # collect and save results from all epochs
        ckpts = find_checkpoints(args.checkpoint_path, args.name, allow_evaluated=True)
        eval_dirs = set(Path(path).parent.parent / 'eval' for path in ckpts)
        for eval_dir in eval_dirs:
            collect_results(eval_dir, args.name)
    elif args.wait:
        while True:
            print('WAIT mode, running forever')
            my_evaluate(args.checkpoint_path, args.test_data_path, args.name, args.workers, args.num_gpus)
            print('evaluation wait ..')
            time.sleep(2)
    else:
        my_evaluate(args.checkpoint_path, args.test_data_path, args.name, args.workers, args.num_gpus)
