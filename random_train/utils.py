from pathlib import Path
import numpy as np
import common.utils.pickling as pickling


def get_sample_loss(data_path):
    if type(data_path)==str:
        data_path = Path(data_path)
        exp_dirs = [x for x in data_path.iterdir() if x.is_dir()]
        exp_dirs.sort()
    else:
        exp_dirs = data_path

    for i, exp_dir in enumerate(exp_dirs):
        sample_loss_fp = exp_dir / 'val_sample_loss_all.pkl'
        if sample_loss_fp.exists():
            sample_loss = pickling.pickle_read(sample_loss_fp)



def get_eval_loss_multi(data_path):
    """
    data path str or list of pathes
    """
    import pandas as pd # TODO there is bug with import pandas at runtime.
    if type(data_path)==str:
        data_path = Path(data_path)
        exp_dirs = [x for x in data_path.iterdir() if x.is_dir()]
        exp_dirs.sort()
    else:
        exp_dirs = data_path
    columns = list(range(1, 21)) + ['final loss', 'best loss', 'best epoch', 'name', 'path']
    N = len(exp_dirs)
    df_res = pd.DataFrame(columns=columns, index=range(N))
    for i, exp_dir in enumerate(exp_dirs):
        try:
            r = get_eval_loss_single(exp_dir)
            if not r:
                continue
            df_res.loc[i,r[0]] = r[1]
        except Exception:
            continue
        filt = df_res['best loss'].isnull()
        df_res = df_res[~filt]
    return df_res

def get_eval_loss_single(data_dir, mode = 'eval'):
    data_dir = Path(data_dir)
    name = data_dir.name
    path = data_dir.parent
    if mode == 'train_val':
        ckpt_fps = list((data_dir / 'modelCheckpoints').glob('*.index'))
        ckpt_fps.sort()
        epochs_idx = [int(s.name.split('-')[0]) for s in ckpt_fps]
        mean_loss = [float(s.name.split('.index')[0].split('-')[1]) for s in ckpt_fps]
    elif mode == 'eval':
        val_eval_fps = sorted(list((data_dir / 'eval').glob('val_eval*')))
        epochs_idx = []
        mean_loss = []
        for val_fp in  val_eval_fps:
            epochs_idx.append(int(val_fp.name.split('_')[-1].split('-')[0]))
            with open(val_fp, 'r') as f:
                log = f.read().splitlines()
                mean_loss.append(float(log[0].split(': ')[1]))
    if epochs_idx[-1] != 20:
        return
    final = mean_loss[-1]
    min_idx = np.argmin(mean_loss)
    best_epoch = epochs_idx[min_idx]
    best = mean_loss[min_idx]
    cols = epochs_idx + ['final loss', 'best loss', 'best epoch'] + ['name', 'path']
    cols_val = mean_loss + [final, best, best_epoch] + [name,path]
    return cols, cols_val




def main():
    data_dir = '/mnt/ext/users/eliyahus/Projects/Simulations/Datacom/GNNC22/NoShuffle/clustering/reproduce/Exp#29_2022-10-25__01-44-09/'
    get_eval_loss_multi(data_dir)
    get_eval_loss_single(data_dir)
    get_sample_loss(data_dir)

if __name__ == '__main__':
    main()