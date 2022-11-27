import os
from argparse import ArgumentParser
import pickle

from pathlib import Path
from common.utils.misc import load_samples
import tensorflow as tf
from RouteNet_Fermi.data_generator import input_fn, get_all_pkl_samples
from RouteNet_Fermi.model import RouteNet_Fermi
from tqdm import tqdm


def collect_outputs_slow(model, dataset):
    model_outputs = []
    for (x, y) in tqdm(dataset):
        out = model(x, training=False)
        model_outputs.append(out)

    return model_outputs


def collect_outputs(model, dataset, batched):
    if batched:
        model_outputs = model.predict(dataset, verbose=1)
    else:
        model_outputs = collect_outputs_slow(model, dataset)

    return model_outputs


def extract_features(ckpt_path, data_dir, output_dir):
    data_dir = os.path.abspath(data_dir)

    if output_dir is None:
        output_dir = os.path.join('.', f'sample_embeddings_{os.path.basename(ckpt_path)}')
    output_dir = os.path.abspath(output_dir)

    # data
    if os.path.isdir(data_dir):
        files = get_all_pkl_samples(data_dir)
        dsdir = data_dir
    else:
        files = load_samples(data_dir)
        dsdir = '/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl'
        assert(all(f.startswith(dsdir) for f in files))

    dataset = input_fn(files, shuffle=False, use_pkl=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # prepare model
    model = RouteNet_Fermi(aux_config={})
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  run_eagerly=False)
    model.load_weights(ckpt_path).expect_partial()

    # extract all the fwd pass data
    print('extracting features')
    model_outputs = collect_outputs(model, dataset, batched=True)

    # save results
    for i, (fpath, output, path_state, path_shape, link_state, queue_state, cpc) in \
            tqdm(enumerate(zip(files, *model_outputs)), desc='saving outputs'):
        path_state = tf.reshape(path_state, path_shape)
        out = {
            'idx': i,
            'fpath': fpath,
            'output': output.numpy(),
            'path_state': path_state.numpy(),
            'link_state': link_state.numpy(),
            'queue_state': queue_state.numpy(),
            'path_capacity': tf.squeeze(cpc, axis=-1).to_tensor(0.).numpy()     # nflows x max_path_len
        }

        if not fpath.startswith(dsdir):
            raise RuntimeError(f'fpath.startswith(dsdir)  fpath  {fpath}  dsdir   {dsdir}')

        out_path = fpath.replace(dsdir, output_dir)
        if Path(out_path).absolute() == Path(fpath).absolute():
            raise RuntimeError('trying to overwrite original data pickles! source path == destination path')
        if os.path.exists(out_path):
            raise RuntimeError(f'output path already exists: {out_path}')

        d = os.path.dirname(out_path)
        os.makedirs(d, exist_ok=True)

        with open(out_path, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint-path', required=True)
    parser.add_argument('-data', '--input-data-dir', required=True)
    parser.add_argument('-outdir', '--output-data-dir', default=None)
    args = parser.parse_args()

    extract_features(args.checkpoint_path, args.input_data_dir, args.output_data_dir)
