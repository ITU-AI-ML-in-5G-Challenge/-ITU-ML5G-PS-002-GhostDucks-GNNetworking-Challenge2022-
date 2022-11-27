import os
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from common.utils.pickling import pickle_write


class SampleMAPE(tf.keras.metrics.Metric):
    """Metric which calculates MAPE per batch"""

    def __init__(self, name='sample_mape', dtype=tf.float32, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.err = self.add_weight(name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        err = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
        self.err.assign(tf.cast(err, dtype=self.dtype))

    def result(self):
        return self.err


class ReportBatchLoss(tf.keras.callbacks.Callback):

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        """
        batch: batch number
        logs - a dictionary with data
        self.params - training params
        """
        step = batch + self.epoch * self.params['steps']
        tf.summary.scalar(name='running_average_loss', data=logs['loss'], step=step)
        tf.summary.scalar(name='train_mape', data=tf.reduce_mean(logs['sample_mape']), step=step)


class TestLossCollectorCB(tf.keras.callbacks.Callback):
    """ callback to collect individual batch losses """

    def __init__(self, log_sample_loss):
        self.logs = {}
        self.epoch = 0
        self.log_sample_loss = log_sample_loss

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_test_end(self, logs):
        rec = self.logs.setdefault(self.epoch,
                                   {'epoch': self.epoch, 'batch_losses': [], 'counts': [], 'all_losses': []})
        rec.update(logs)

    def on_test_batch_end(self, batch, logs=None):
        if self.log_sample_loss:
            rec = self.logs.setdefault(self.epoch,
                                       {'epoch': self.epoch, 'batch_losses': [], 'counts': [], 'all_losses': []})

            sample_mape, count, agg_loss = logs['sample_mape'], logs['count'], logs['loss']
            rec['batch_losses'].append(sample_mape)
            rec['counts'].append(count)
            rec['all_losses'].append(agg_loss)
            # _agg_loss = np.average(rec['batch_losses'], weights=rec['counts'])
            # assert (np.allclose(_agg_loss, agg_loss))

    def get_sample_losses(self, rec, files):
        losses = pd.DataFrame([(p, l, c) for p, l, c in zip(files, rec['batch_losses'], rec['counts'])],
                              columns=['path', 'loss', 'flows'])
        losses['net_size'] = ((1+np.sqrt(1+4*losses['flows']))/2).astype(np.int32)
        return losses

    def get_best(self):
        return min(self.logs.values(), key=lambda x: x['loss'])

    def get_report(self, files, epoch=None, best_epoch=False):
        rec = None
        if best_epoch:
            rec = self.get_best()
        elif epoch is not None:
            rec = self.logs[epoch]

        if rec is not None:
            batch_losses = {rec['epoch']: self.get_sample_losses(rec, files)}
            epoch_losses = {rec['epoch']: rec['loss']}
        else:
            batch_losses = {epoch: self.get_sample_losses(rec, files) for epoch, rec in self.logs.items()}
            epoch_losses = {rec['epoch']: rec['loss'] for rec in self.logs.values()}

        return epoch_losses, batch_losses

    def save_report(self, save_dir, name, files, epoch=None, best_epoch=None):
        ep_losses, smp_losses = self.get_report(files, epoch, best_epoch)

        # save eval result
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # save everything to pickle
        path = os.path.join(save_dir, f'{name}_sample_loss_all.pkl')
        pickle_write(path, {'epoch_losses': ep_losses, 'sample_losses': smp_losses})

        # save to text file the epoch test losses
        path = os.path.join(save_dir, f'{name}_eval.log')
        pd.DataFrame({'epoch': ep_losses.keys(), 'loss': ep_losses.values()}).to_csv(path, index=False)

        # save as csv the sample losses of best epoch
        if self.log_sample_loss:
            rec = self.get_best()
            path = os.path.join(save_dir, f'./{name}_sample_loss_best_ep_{rec["epoch"]}_loss_{rec["loss"]:.2f}.csv')
            sample_losses = self.get_sample_losses(rec, files)
            sample_losses.to_csv(path)


class CountSamples(tf.keras.metrics.Metric):
    """Metric which counts the number of examples seen"""

    def __init__(self, name='count', aggregate=False, dtype=tf.int64, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.count = self.add_weight(name, initializer='zeros')
        self.aggregate = aggregate

    def update_state(self, y_true, y_pred, sample_weight=None):
        first_tensor = tf.nest.flatten(y_true)[0]
        batch_size = tf.shape(first_tensor)[0]
        if self.aggregate:
            self.count.assign_add(tf.cast(batch_size, dtype=self.dtype))
        else:
            self.count.assign(tf.cast(batch_size, dtype=self.dtype))

    def result(self):
        return self.count


class ReportEpochTime(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_start_time = None
        self.test_start_time = None
        self.train_begin_time = None

    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        # at the epoch start
        self.epoch_start_time = time.time()

    def on_test_begin(self, logs=None):
        # start of validation
        self.test_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # at the epoch end - after training and validation step
        epoch_end_time = time.time()
        train_time = self.test_start_time - self.epoch_start_time
        test_time = epoch_end_time - self.test_start_time
        total_time = epoch_end_time - self.epoch_start_time
        tf.summary.scalar(name='time/train', data=train_time, step=epoch)
        tf.summary.scalar(name='time/test', data=test_time, step=epoch)
        tf.summary.scalar(name='time/total', data=total_time, step=epoch)

    def on_train_end(self, logs=None):
        total_train_time = time.time() - self.train_begin_time
        print(f'total train time is {total_train_time:.2f} seconds')


def init_clearml(task_name):
    from clearml import Task
    os.environ['no_proxy'] = '10.0.0.0/8'  # this prevents proxy problems when connecting to clearml server
    clearml_project = 'GNN_Challenge_2022'
    Task.init(project_name=clearml_project, task_name=task_name)
