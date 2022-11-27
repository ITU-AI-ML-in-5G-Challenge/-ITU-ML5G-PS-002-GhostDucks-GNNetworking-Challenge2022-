from pathlib import Path
from sys import stderr
import warnings
from tqdm import tqdm
from RouteNet_utils import log_utils
from argparse import ArgumentParser, ArgumentTypeError
import yaml
import datetime

warnings.filterwarnings("ignore")
seed_value = 69420
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(seed_value)
#tf.random.set_random_seed(seed_value)
tf.keras.utils.set_random_seed(seed_value)
#tf.config.experimental.enable_op_determinism()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# the following 3 lines limit the cpu usage of the training to use a single cpu
# in order to free the rest of cpus to other tasks.
# But if doing validation as part of training, comment out these 3 lines, since val graphs
# are large and need lots of compute
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.get_logger().setLevel('INFO')

gpus = tf.config.list_physical_devices('GPU')
print(len(gpus), 'Physical GPUs')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from RouteNet_Fermi.data_generator import input_fn, get_all_pkl_samples, create_generator, input_fn_from_gen
from random_train.data_utils import sample_dataset
from RouteNet_Fermi.model import RouteNet_Fermi
from RouteNet_utils.data_utils import approximate_dataset_size


def init_seed(seed_value = 69420):
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)


def main(args, train_path, final_evaluation=False,
         val_steps_during_train=20, epochs=20, steps_per_epoch=2000, lr=0.005,
         decay=1, decay_steps=0,
         check_size=True, test_path='./validation_dataset', log_sample_loss=False,
         return_model=False, optimizer='adam', seed=None, reset_optimizer=False,
         shuffle_train=True):
    """
    Trains and evaluates the model with the provided dataset.
    The model will be trained for 20 epochs.
    At each epoch a checkpoint of the model will be generated and stored at the folder ckpt_dir which will be created
    automatically if it doesn't exist already.
    Training the model will also generate logs at "./logs" that can be opened with tensorboard.

    Parameters
    ----------
    train_path
        Path to the training dataset
    final_evaluation, optional
        If True after training the model will be validated using all of the validation dataset, by default False
    """
    if seed is not None:
        init_seed(seed)

    # randomness debug
    # print('seed_value:', seed_value)
    # print('numpy random hash:', np.random.get_state()[1].sum())
    # print('PYTHONHASHSEED env:', os.environ['PYTHONHASHSEED'])
    # print('tf random:', tf.random.uniform([10]).numpy().sum())
    # print('python random: ', sum(random.random() for _ in range(100)))
    # exit(0)

    if not os.path.exists(train_path):
        print(f"ERROR: the provided training path \"{os.path.abspath(train_path)}\" does not exist!", file=stderr)
        return None
    if not os.path.exists(test_path):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(test_path), file=stderr)
        return None

    task_path = Path(args.save_path) / args.task_name
    log_path = task_path / 'logs'
    if not os.path.exists(log_path):
        print("INFO: Logs folder created at ", os.path.abspath(log_path))
        os.makedirs(log_path)

    # Check dataset size
    if check_size:
        dataset_size = approximate_dataset_size(train_path, 'pkl' in train_path)
        print('dataset size:', dataset_size)
        if not dataset_size:
            print(f"ERROR: The dataset has no valid samples!", file=stderr)
            return None
        elif (dataset_size > 100):
            print(f"ERROR: The dataset can only have up to 100 samples (currently has {dataset_size})!", file=stderr)
            return None

    use_pkl = True
    if args.sample_train_mode != 'all':
        samples = sample_dataset(args.sample_val, args.sample_train_mode, train_path, task_path)
        ds_train = input_fn(shuffle=shuffle_train, training=True, use_pkl=use_pkl, samples=samples)
    else:
        ds_train = input_fn(train_path, shuffle=shuffle_train, training=True, use_pkl=use_pkl)

    ds_train = ds_train.repeat()

    test_gen, test_gen_args = create_generator(test_path, shuffle=False, use_pkl=use_pkl)
    ds_test = input_fn_from_gen(test_gen, test_gen_args)

    if decay < 1 and decay_steps > 0:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay, staircase=True)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    print(optimizer)


    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    metrics = [log_utils.CountSamples(aggregate=False), log_utils.SampleMAPE()] if log_sample_loss else None
    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  run_eagerly=False,
                  metrics=metrics)

    if args.load_from_ckpt:
        print('loading weights from ', args.ckpt_weights_dir)
        model.load_weights(args.ckpt_weights_dir)
        if reset_optimizer:
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            print('optimizer after ckpt load: ', optimizer)

            model.compile(loss=loss_object,
                          optimizer=optimizer,
                          run_eagerly=False,
                          metrics=metrics)

    ckpt_dir = task_path / 'modelCheckpoints'
    latest = tf.train.latest_checkpoint(ckpt_dir)

    if latest is not None:
        print(f"ERROR: Found a pretrained models, please clear or remove the {ckpt_dir} directory and try again!")
        return None
    else:
        print("INFO: Starting training from scratch...")

    ckpt_metric = 'val_loss' if val_steps_during_train != 0 else 'loss'
    filepath = os.path.join(ckpt_dir, "{epoch:02d}-{" + ckpt_metric + ":.2f}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=1,
        mode="min",
        monitor=ckpt_metric,
        save_best_only=args.save_best_only,
        save_weights_only=True,
        save_freq='epoch')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, update_freq=1)
    # rbl_callback = log_utils.ReportBatchLoss()
    time_callback = log_utils.ReportEpochTime()
    val_logger = log_utils.TestLossCollectorCB(log_sample_loss=log_sample_loss)

    model.fit(ds_train,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_test,
              validation_steps=val_steps_during_train,
              callbacks=[cp_callback, tensorboard_callback, time_callback, val_logger],
              use_multiprocessing=True,
              verbose=1)

    if val_steps_during_train > 0 and ('pkl' in test_path):
        files = test_gen(*test_gen_args).samples[:val_steps_during_train]
        val_logger.save_report(task_path, 'val', files)

    if final_evaluation:
        print("Final evaluation:")
        model.evaluate(ds_test)

    ret = {'logs': val_logger.logs.copy()}
    if return_model:
        ret['model'] = model

    return ret

def evaluate_slow(ds_test, model, loss_object):
    lengths = []
    losses = []
    for x, y in tqdm(ds_test):
        # loss = model.test_on_batch(x, tf.expand_dims(y,-1))
        y_pred = model(x)
        loss = loss_object(y, y_pred)
        losses.append(loss)
        lengths.append(y.shape)

    mean_loss = np.mean(losses)
    return mean_loss, losses, lengths


def evaluate(ckpt_path, test_path = None, files = None):
    """
    Loads model from checkpoint and trains the model.

    Parameters
    ----------
    ckpt_path
        Path to the checkpoint. Format the name as it was introduced in tf.keras.Model.load_weights.
    """

    if not test_path and not files:
        if not os.path.exists(test_path):
            print("ERROR: Validation dataset not found at the expected location:",
                  os.path.abspath(test_path), file=stderr)
            return None

    use_pkl = True
    if files is None:
        files = get_all_pkl_samples(test_path)
        if len(files) == 0:
            raise RuntimeError('no pkl files found in data directory')
    elif len(files) == 0:
        raise RuntimeError('empty files list')

    #use_pkl = 'pkl' in test_path if test_path else 'pkl' in str(files[0])
    ds_test = input_fn(files, shuffle=False, use_pkl=use_pkl)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  metrics=[log_utils.CountSamples(aggregate=False), log_utils.SampleMAPE()],
                  run_eagerly=False)

    # stored_weights = tf.train.load_checkpoint(ckpt_path)
    model.load_weights(ckpt_path).expect_partial()

    # Evaluate model
    cb = log_utils.TestLossCollectorCB(log_sample_loss=True)
    mean_loss = model.evaluate(ds_test, use_multiprocessing=True, workers=20, callbacks=[cb])[0]
    epoch_losses, batch_losses = cb.get_report(files, epoch=0)
    assert(epoch_losses[0] == mean_loss)
    return mean_loss, batch_losses[0]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_nsamples_(args):
    """
    get number of samples
    """
    sample_mode = args.sample_train_mode
    is_pkl = 'pkl' in args.data_dir
    if sample_mode == 'all':
        nsamples = approximate_dataset_size(args.data_dir, is_pkl=is_pkl)
    elif sample_mode == 'len':
        nsamples = int(args.sample_val)
    elif sample_mode == 'file':
        file_path = args.sample_val
        with open(file_path, 'r') as f:
            samples = [x.strip() for x in f.readlines()]
            nsamples = len(samples)
    elif sample_mode=='list':
        nsamples = len(args.sample_val)
    return nsamples

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./datasets_pkl')
    parser.add_argument('-t', '--test_data_path', default='./validation_dataset_pkl')
    parser.add_argument('-o', '--save_path', default='./')
    parser.add_argument('-task_name', '--task_name', type=str, default='default_task')
    parser.add_argument('-sample_mode','--sample_train_mode', type = str, choices=['all','file','len'], default = 'all')
    parser.add_argument('-sample_val', '--sample_val', default=100) # size of train set if sample_train
    parser.add_argument('-nval', '--val_steps', type=int, default=20)
    parser.add_argument('-epochs', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
    #parser.add_argument('-sz', '--check-size', action='store_true')
    parser.add_argument('-steps', '--epoch_steps', type=int, default=-1)
    parser.add_argument('-decay', '--decay', type=float, default=1)
    parser.add_argument('-decay_ep', '--decay-epochs', type=int, default=0)
    parser.add_argument('-clearml', '--use_clearml', type=str2bool, default=True)
    parser.add_argument('-load_from_ckpt', '--load_from_ckpt', type=str2bool, default=True)
    parser.add_argument('-save_best_only', '--save_best_only', type=str2bool, default=False)
    parser.add_argument('-final_eval', '--final_evaluation', type=str2bool, default=False)
    parser.add_argument('-ckpt', '--ckpt_weights_dir', type=str, default='./RouteNet_Fermi/initial_weights/initial_weights')
    parser.add_argument('-opt', '--optimizer', default='adam', choices={'adam', 'sgd'})
    parser.add_argument('-reset_opt', '--reset_optimizer', action='store_true', help='resets optimizer when loading from checkpoint')
    parser.add_argument('-shuf', '--shuffle_train', type=str2bool, default=True)
    args = parser.parse_args()
    args.task_name = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S_') + args.task_name
    return args

if __name__ == '__main__':
    args = parse_args()

    nsamples = get_nsamples_(args)
    print(f'approximate number of samples: {nsamples}')
    epoch_steps = args.epoch_steps if args.epoch_steps > 0 else nsamples
    config = vars(args)
    config['nsamples'] = nsamples
    config['epoch_steps_actual'] = epoch_steps
    decay_steps = epoch_steps * args.decay_epochs
    config['decay_steps'] = decay_steps

    print(yaml.dump(config))
    task_path = Path(args.save_path) / args.task_name
    if not task_path.exists():
        task_path.mkdir(parents=True)
    with open(task_path / 'config.yaml', 'w') as fp:
        yaml.dump(config, fp)

    if args.use_clearml:
        log_utils.init_clearml(args.task_name)

    main(args, args.data_dir, final_evaluation=args.final_evaluation, val_steps_during_train=args.val_steps, check_size=False,
         steps_per_epoch=epoch_steps, epochs=args.epochs, test_path=args.test_data_path, lr=args.learning_rate,
         decay=args.decay, decay_steps=decay_steps, optimizer=args.optimizer, reset_optimizer=args.reset_optimizer,
         shuffle_train=args.shuffle_train)

