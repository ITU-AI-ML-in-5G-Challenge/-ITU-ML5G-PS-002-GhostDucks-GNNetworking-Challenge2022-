from argparse import ArgumentParser
import random_train.train as trainer
from pathlib import Path
import datetime
import common.utils.pickling as pickling
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import pandas as pd
import copy
import numpy as np
import math

def my_bool(s):
    return s != 'False'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-d', '--data-dir', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl/')
    parser.add_argument('--batches', default='all', help='list of batches')
    parser.add_argument('-t', '--test-data-path', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/validation_dataset_pkl/')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--log_sample_loss', type=my_bool, default=False)
    parser.add_argument('-task_name', type=str, default='default_task')
    parser.add_argument('--train', type=my_bool, default=False)
    parser.add_argument('--k_replace', type=int, default=5, help = 'number of sample to be replaced')
    parser.add_argument('--n_trainings', type = int, default=150)
    parser.add_argument('-w_train','--n_workers_train', type=int, default=1)
    parser.add_argument('-w_val','--n_workers_val', type=int, default=1)
    parser.add_argument('-sample_mode','--sample_train_mode', type = str, choices=['all','file','len'], default = 'all')
    parser.add_argument('--sample_val', default=100) # par gets value accoridng to sample mode - file path for 'file,' int for 'len'.
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
    parser.add_argument('--rnd_state_pkl', type=str, default='')
    # clustering
    parser.add_argument('--embed_dir', type=str, nargs=2, default=['oracle_models/6.15/sample_embeddings_43-6.15/', 'oracle_models/5.89/sample_embeddings_80-5.89/'])
    parser.add_argument('--embed_ext', type=str, default='')
    parser.add_argument('--embed_mode', type=str, choices=['nearest_embed','clustering_lows', 'nearest_embed#oracles' ], default='nearest_embed#oracles')
    parser.add_argument('--permute', type=my_bool, default=True)
    parser.add_argument('--topk', type = int, default=5)

    args = parser.parse_args()
    args.task_name = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S_') + args.task_name
    return args


# args n_low_cent 20 tresh 0.05 topk 10
# args cent_mode lowK cent_K 20 low_tresh 0.05 topk 10
def get_samples_clustering(train_embed_obj, val_embed_obj, val_df, args, mode='my_clustering', Ntrain=1, permute=False,
                            batches_lst=[]):
    train_embed = train_embed_obj['embeddings']
    train_paths = train_embed_obj['paths']
    if batches_lst:
        train_batches = [p.split('/')[7] for p in train_paths]
        train_batches = np.array(train_batches)
        idx = np.where(np.isin(train_batches, batches_lst))[0]
        train_embed = train_embed[idx]
        train_paths = list(np.array(train_paths)[idx])

    val_embed = val_embed_obj['embeddings']
    if mode == 'clustering_lows':
        cent_mode = args['cent_mode']
        if cent_mode == 'lowK':
            cent_args = {'mode': cent_mode, 'K': args['cent_K'], 'low_tresh': args['low_tresh']}
        elif cent_mode == 'allK':
            cent_args = {'mode': cent_mode, 'K': args['cent_K']}

        sample_mode = args['sample_mode']
        topk = args['topk']
        sample_args = args['sample_args'] if 'sample_args' in args else None

        # get centroids
        val_idxs_c = get_centroids(val_df, val_embed, **cent_args)

        # get topk
        C = val_embed[val_idxs_c]
        X = train_embed
        X_paths = np.array(train_paths)
        df = assign_x(C, X, X_paths, k=topk, dist='cosine')
        df = df.sort_values(by=['assign', 'dist'])

        samples = sample(df, Ntrain, permute, sample_mode, **sample_args)
    return samples

def sample(df, size, permute, sample_mode, Nsamples = 100, k_common = 0):
    def get_path_counts(df):
        path_counts = df.paths.value_counts()
        f = lambda x: path_counts[x]
        df['paths_count'] = list(df.paths.apply(f))
        return df

    def get_others(df, paths):
        filt = df.paths.isin(paths)
        return df[~filt]

    samples = []
    if sample_mode == 'sample_one':
        for i in range(size):
            # sample
            # sample ones
            samples_1 = df.groupby('assign').sample(1).paths  # exclude also all ones
            samples_1 = samples_1.unique()
            if len(samples_1) < Nsamples:
                df_others = df[~df.paths.isin(samples_1)]
                Ngroups = len(df_others['assign'].unique())
                n_samples = math.ceil((Nsamples - len(samples_1)) / Ngroups)
                Nsamples_2 = Nsamples - len(samples_1)
                flag = True
                while flag:
                    samples_2 = df_others.groupby('assign').sample(n_samples, replace=True).paths
                    samples_2 = samples_2.unique()
                    flag = False if len(samples_2) >= Nsamples_2 else True
                samples_2 = np.random.choice(samples_2, size=Nsamples - len(samples_1), replace=False)
                jsamples = list(np.concatenate([samples_1, samples_2]))
                if permute:
                    jsamples = np.random.permutation(jsamples)
            samples.append(jsamples)
    elif sample_mode=='sample_common':
        samples = []

        df = get_path_counts(df)
        df = df.sort_values(by=['paths_count', 'dist'], ascending=False)
        sampls_1 = df.groupby(['assign']).head(1).paths.unique()  # exclude also all ones
        # from others take all k commin
        df = get_others(df, sampls_1)
        df = get_path_counts(df)
        sample_2 = df[df.paths_count >= k_common].paths.unique()

        df = get_others(df, sample_2)
        df = get_path_counts(df)
        N = Nsamples - (len(sampls_1) + len(sample_2))
        sample_3 = df[df.paths_count == k_common-1].paths.unique()
        for i in range(size):
            rnd_sample_3 = np.random.choice(sample_3, size=N, replace=False)
            jsamples = np.concatenate([sampls_1, sample_2, rnd_sample_3])
            if permute:
                    jsamples = np.random.permutation(jsamples)
            samples.append(jsamples)
    return samples

# clustering utils
def get_centroids(val_df, embed, mode = 'lowK', K = 20, low_tresh = None):
    """
    returns sub list of val embeddings
    take all hard and sample from lows
    """
    if mode == 'lowK':
        tresh = low_tresh
        # high loss
        filt = val_df.wloss >= tresh
        h_idx = np.where(filt)[0]
        others_idx = np.where(~filt)[0]

        # get low loss
        low_embed = embed[others_idx]
        kmeans = KMeans(n_clusters=K).fit(low_embed)
        low_c = kmeans.cluster_centers_
        dist = cdist(low_c, low_embed, 'euclidean')  # this is the cosine distance not similarity!
        l_idx = dist.argmin(axis=1)
        l_idx = np.unique(l_idx)
        val_idxs_c = np.concatenate([h_idx, l_idx])
    elif mode == 'allK':
        kmeans = KMeans(n_clusters=K).fit(embed)
        C = kmeans.cluster_centers_
        dist = cdist(C, embed, 'euclidean')  # this is the cosine distance not similarity!
        val_idxs_c = dist.argmin(axis=1)
        val_idxs_c = np.unique(val_idxs_c)
    val_idxs_c.sort()
    return val_idxs_c

def assign_x(C, X, X_paths, k=5, dist='cosine'):
    dist = cdist(C, X, 'cosine')  # this is the cosine distance not similarity!
    # find k-clostest for each c in C
    index_array = np.argpartition(dist, kth=k, axis=1)
    k_idx_array = index_array[:, :k]

    # create df for all x
    NC = C.shape[0]

    # dist
    k_dist = (np.take_along_axis(dist, index_array, axis=1))[:, :k]
    k_dist = k_dist.flatten()
    # assign
    assign = np.arange(NC)
    assign = np.repeat(assign, k)
    # paths
    X_idx = k_idx_array.flatten()
    paths = X_paths[X_idx]
    data = {'paths': paths, 'dist': k_dist, 'assign': assign}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['assign', 'dist'])
    return df





def get_samples_nearest_embed(train_embed_obj,val_embed_obj , mode='minloss', topk=25, Ntrain=1, permute = False, batches_lst = []):

    Nsamples = 100
    train_embed = train_embed_obj['embeddings']
    train_paths = train_embed_obj['paths']
    if batches_lst:
        train_batches = [p.split('/')[7] for p in train_paths]
        train_batches = np.array(train_batches)
        idx = np.where(np.isin(train_batches, batches_lst))[0]
        train_embed = train_embed[idx]
        train_paths = list(np.array(train_paths)[idx])

    if mode == 'minloss':
        samples = []
        # calculate distance between train and test
        dist = cdist(train_embed, val_embed_obj['embeddings'], 'cosine')
        assign = dist.argmin(axis=1)
        min_dist = dist.min(axis=1)

        # use pandas for easy sampling
        data = {'paths': train_paths, 'min_dist': min_dist, 'assign': assign}
        df = pd.DataFrame(data)
        df = df.sort_values(by=['assign', 'min_dist'])

        # take only topk
        df = df.groupby('assign').head(topk)

        for i in range(Ntrain):
            # sample ones
            samples_1 = df.groupby('assign').sample(1).paths  # exclude also all ones
            if len(samples_1)<Nsamples:
                df_others = df[~df.paths.isin(samples_1)]
                Ngroups = len(df_others['assign'].unique())
                n_samples = math.ceil((Nsamples - len(samples_1)) / Ngroups)
                Nsamples_2 = Nsamples - len(samples_1)
                flag = True
                while flag:
                    samples_2 = df_others.groupby('assign').sample(n_samples, replace=True).paths
                    samples_2 = samples_2.unique()
                    flag = False if len(samples_2)>=Nsamples_2 else True
                samples_2 = np.random.choice(samples_2, size=Nsamples - len(samples_1), replace=False)
                jsamples = list(np.concatenate([samples_1, samples_2]))
            else:
                jsamples = np.random.choice(samples_1, size=Nsamples, replace=False)
                if permute:
                    jsamples = np.random.permutation(jsamples)
            samples.append(jsamples)
        return samples




def get_samples_nearest_embed_oracles(embed_dirs, topk=5, Ntrain=1, permute = False,
                                      batches_lst = [], save_path = ''):
    def get_sub_batches(train_embed_obj, batches_lst):
        train_embed = train_embed_obj['embeddings']
        train_paths = train_embed_obj['paths']
        if batches_lst:
            train_batches = [p.split('/')[7] for p in train_paths]
            train_batches = np.array(train_batches)
            idx = np.where(np.isin(train_batches, batches_lst))[0]
            train_embed = train_embed[idx]
            train_paths = list(np.array(train_paths)[idx])
        return train_embed, train_paths

    def get_df_train(train_embed, val_embed, train_paths):
        # calculate distance between train and test
        dist = cdist(train_embed, val_embed, 'cosine')
        assign = dist.argmin(axis=1)
        min_dist = dist.min(axis=1)

        # use pandas for easy sampline
        data = {'paths': train_paths, 'min_dist': min_dist, 'assign': assign}
        df = pd.DataFrame(data)
        return df
    Nsamples = 100

    if len(embed_dirs) != 2:
        raise RuntimeError(f'expected 2 embedding directories - one for each oracle, got {len(embed_dirs)}')

    embed_dir1 = embed_dirs[0]
    embed_dir2 = embed_dirs[1]
    train_embed_obj1, val_embed_obj1 = get_embeddings(embed_dir1)
    train_embed_obj2, val_embed_obj2 = get_embeddings(embed_dir2)

    train_embed1, train_paths1  = get_sub_batches(train_embed_obj1, batches_lst)
    train_embed2, train_paths2 = get_sub_batches(train_embed_obj2, batches_lst)

    val_df1 = get_loss_df(embed_dir1)
    val_df2 = get_loss_df(embed_dir2)
    val_df = pd.merge(val_df1, val_df2[['path', 'loss', 'wloss']], on='path', suffixes=['_1', '_2'])

    filt = val_df.wloss_1 <= val_df.wloss_2
    idx1 = np.where(filt)[0]
    idx2 = np.where(~filt)[0]

    df_train1 = get_df_train(train_embed1, val_embed_obj1['embeddings'][idx1], train_paths1)
    df_train2 = get_df_train(train_embed2, val_embed_obj2['embeddings'][idx2], train_paths2)

    samples = []
    df_train1 = df_train1.sort_values(by=['assign', 'min_dist'])
    df_train2 = df_train2.sort_values(by=['assign', 'min_dist'])

    # take only topk
    df1 = df_train1.groupby('assign').head(topk)
    df2 = df_train2.groupby('assign').head(topk)

    # save sample pool
    if save_path:
        save_path = Path(args.save_path)
        if not save_path.exists():
            save_path.mkdir()

        smp1 = df1.copy()
        smp2 = df2.copy()
        oracle1_name = embed_dir1.split('_')[-1]
        oracle2_name = embed_dir2.split('_')[-1]
        smp1['oracle'] = oracle1_name
        smp2['oracle'] = oracle2_name
        smp1['cluster'] = smp1['oracle'] + '_c' + smp1['assign'].astype(str)
        smp2['cluster'] = smp2['oracle'] + '_c' + smp2['assign'].astype(str)
        samples_pool = pd.concat([smp1, smp2])

        pool = {'samples': samples_pool,
                'topk': topk,
                'embed_dir1': embed_dir1,
                'embed_dir2': embed_dir2,
                }

        pool_fp = save_path / 'topk_samples_pool.pkl'
        print('saving samples pool to:', str(pool_fp))
        pickling.pickle_write(pool_fp, pool)


    for i in range(Ntrain):
        # sample ones
        samples_ones_1 = df1.groupby('assign').sample(1).paths
        samples_ones_2 = df2.groupby('assign').sample(1).paths
        samples_ones = np.unique(np.concatenate([samples_ones_1, samples_ones_2]))

        if len(samples_ones)<Nsamples:
            df_others1 = df1[~df1.paths.isin(samples_ones)]
            df_others2 = df2[~df2.paths.isin(samples_ones)]

            samples_two1 = df_others1.groupby('assign').sample(1).paths
            samples_two2 = df_others2.groupby('assign').sample(1).paths
            samples_two = np.unique(np.concatenate([samples_two1, samples_two2]))
            samples_two = np.random.choice(samples_two, size=Nsamples - len(samples_ones), replace=False)

            jsamples = list(np.concatenate([samples_ones, samples_two]))
            if permute:
                jsamples = np.random.permutation(jsamples)
        samples.append(jsamples)
    return samples


def get_loss_df(embed_dir, set='val'):
    val_sample_dir = Path(embed_dir).parent / 'eval'
    val_loss_fp = next(val_sample_dir.glob(f'{set}_sample_loss*'))
    print(f'loading {set} losses from: ', val_loss_fp)
    val_df = pd.read_csv(str(val_loss_fp))if val_loss_fp.name.endswith('csv') else pickling.pickle_read(val_loss_fp)
    Nflows = val_df.flows.sum()
    val_df['wloss'] = (val_df.loss * val_df.flows) / Nflows
    return val_df


def get_embeddings(embed_dir, embed_ext=''):
    embed_dir = Path(embed_dir)

    train_embedding_fp = embed_dir / f'train_min_max_mean_{embed_ext}.pkl' \
        if embed_ext else embed_dir / 'train_min_max_mean.pkl'
    val_embedding_fp = embed_dir / f'val_min_max_mean_{embed_ext}.pkl' \
        if embed_ext else embed_dir / 'val_min_max_mean.pkl'

    # load embed
    print(f'loading train embeddings from: ', train_embedding_fp)
    train_embed = pickling.pickle_read(train_embedding_fp)
    print(f'loading val embeddings from: ', val_embedding_fp)
    val_embed = pickling.pickle_read(val_embedding_fp)
    return train_embed, val_embed


def clustering_train(args):
    embed_dir = args.embed_dir
    embed_ext = args.embed_ext
    N = args.n_trainings
    # set random state
    if args.rnd_state_pkl:
        rnd_state = pickling.pickle_read(args.rnd_state_pkl)
        np.random.set_state(rnd_state)
    else:
        save_path = Path(args.save_path)
        if not save_path.exists():
            save_path.mkdir()
        rnd_state = np.random.get_state()
        rnd_fp = save_path / 'rnd_state.pkl'
        pickling.pickle_write(rnd_fp, rnd_state)

    # get samples
    batches_lst = [] if args.batches == 'all' else [s for s in args.batches.split(' ')]
    if args.embed_mode == 'nearest_embed':
        train_embed, val_embed = get_embeddings(embed_dir, embed_ext=embed_ext)
        samples = get_samples_nearest_embed(train_embed, val_embed, mode=args.embed_mode, topk=args.topk, Ntrain=args.n_trainings,
                               permute=args.permute, batches_lst = batches_lst)
    elif args.embed_mode == 'nearest_embed#oracles':
        samples = get_samples_nearest_embed_oracles(embed_dir, topk=args.topk, Ntrain=args.n_trainings,
                                                    permute=args.permute, batches_lst = batches_lst, save_path=save_path)

    elif args.embed_mode.find('clustering')>-1:
        train_embed, val_embed = get_embeddings(args.embed_dir, embed_ext=args.embed_ext)
        val_df = get_loss_df(args.embed_dir)
        # clustering_args = {'cent_mode': 'lowK', 'low_tresh': 0.05, 'cent_K': 20, 'topk': args.topk, 'sample_mode': 'sample_one'}
        clustering_args = {'cent_mode': 'lowK', 'low_tresh': 0.05, 'cent_K': 20, 'topk': args.topk,
                           'sample_mode': 'sample_common', 'sample_args': {'k_common': 3}}
        samples = get_samples_clustering(train_embed, val_embed, val_df, clustering_args, mode=args.embed_mode, Ntrain=args.n_trainings,
                                    permute=False, batches_lst=batches_lst)

    samples = fix_samples_path(samples, args.data_dir)
    if args.train:
        # train models
        NGPUS = args.ngpus_train
        args.sample_train_mode = 'list'

        jobs = []
        for i in range(N):
            args = copy.deepcopy(args)
            args.sample_val = samples[i]
            if NGPUS>0:
                jobs.append( {'train_args': args, 'task_name': 'Exp_{}_'.format(i), 'gpu': str(i % NGPUS) })
            else:
                jobs.append({'train_args': args, 'task_name': 'Exp_{}_'.format(i)})
        trainer.n_trainings_caller(jobs)
    else:
        # save generated subsets
        print('saving subsets to:', args.save_path)
        save_path = Path(args.save_path)
        for i, samples in enumerate(samples):
            task_data_dir = save_path / 'Exp_{:04d}'.format(i)
            if not task_data_dir.exists():
                task_data_dir.mkdir()
            samples_path = task_data_dir / 'samples.txt'
            with open(samples_path, 'w') as f:
                f.write('\n'.join(samples))


def fix_samples_path(samples, data_dir):
    header = '/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl/'
    for samples_set in samples:
        for i, path in enumerate(samples_set):
            if path.startswith(header):
                samples_set[i] = str(Path(data_dir) / path[len(header):])

    return samples


if __name__ == '__main__':
    args = parse_args()
    clustering_train(args)

