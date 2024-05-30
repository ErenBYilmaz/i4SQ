from math import inf
from typing import Callable, Any, List

from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.utils import tf_utils

MonitoredQuantity = Any
Batch = int
Epoch = int  # counting starts with 0


class MissingQuantityError(Exception):
    pass


class MissingQuantityErrorAfterBatch(Exception):
    def __init__(self, batch, logs, quantity_name):
        available_keys = list(map(str, logs.keys()))
        super().__init__("Batch {0}: Missing quantity: {1}. Available are {2}".format(batch,
                                                                                      quantity_name,
                                                                                      available_keys))
        self.batch = batch
        self.logs = logs


class MissingQuantityErrorAfterEpoch(Exception):
    def __init__(self, epoch, logs, quantity_name):
        available_keys = list(map(str, logs.keys()))
        super().__init__("Epoch {0}: Missing quantity: {1}. Available are {2}".format(epoch,
                                                                                      quantity_name,
                                                                                      available_keys))
        self.epoch = epoch
        self.logs = logs


class MonitorMetric(Callback):
    def __init__(self,
                 monitor='val_loss',
                 termination_condition_batch_end: Callable[[MonitoredQuantity, Batch], bool] = None,
                 termination_condition_epoch_end: Callable[[MonitoredQuantity, Epoch], bool] = None,
                 verbose=1, ):
        super().__init__()
        self.verbose = verbose
        self._supports_tf_logs = True
        self.monitor = monitor
        self.termination_condition_batch_end = termination_condition_batch_end
        self.termination_condition_batch_end_fulfilled = False
        self.termination_condition_epoch_end = termination_condition_epoch_end
        self.termination_condition_epoch_end_fulfilled = False

    def on_batch_end(self, batch: Batch, logs=None):
        if self.termination_condition_batch_end is None or self.model.stop_training:
            return
        metric_value = self.get_metric_from_logs(batch, logs)
        if self.termination_condition_batch_end(metric_value, batch):
            if self.verbose:
                print('Batch {0}: Termination condition fulfilled for {1} = {2}'.format(batch, self.monitor, metric_value))
            self.model.stop_training = True
            self.termination_condition_batch_end_fulfilled = True

    def get_metric_from_logs(self, batch, logs):
        logs = logs or {}
        metric_value = logs.get(self.monitor)
        if metric_value is None:
            raise MissingQuantityErrorAfterBatch(batch, logs, self.monitor)
        metric_value = tf_utils.sync_to_numpy_or_python_type(metric_value)
        return metric_value

    def on_epoch_end(self, epoch: Epoch, logs=None):
        if self.termination_condition_epoch_end is None or self.model.stop_training:
            return
        metric_value = self.get_metric_from_logs(epoch, logs)
        if self.termination_condition_epoch_end(metric_value, epoch):
            if self.verbose:
                print('Epoch {0}: Termination condition fulfilled for {1} = {2}'.format(epoch, self.monitor, metric_value))
            self.model.stop_training = True
            self.termination_condition_epoch_end_fulfilled = True


class EarlyStopping(MonitorMetric):
    def early_stopping_condition(self, new_value: MonitoredQuantity, epoch: Epoch) -> bool:
        self._history.append(new_value)
        if self.patience <= 0:
            return True
        elif self.larger_result_is_better and new_value >= self.best_value:
            self.best_value = new_value
            self.best_epoch = epoch
            return False
        elif not self.larger_result_is_better and new_value <= self.best_value:
            self.best_value = new_value
            self.best_epoch = epoch
            return False
        elif epoch + 1 >= self.min_epochs and self.best_epoch + self.patience <= epoch:
            # for example with a patience of 3 and best epoch 2, we skip this part in epoch 3 and 4 but not 5
            # then again, if in epoch 5 we had a better result, we would not even get here
            return True
        else:
            return False

    def __init__(self, larger_result_is_better: bool, patience: int = 1, min_epochs=1, monitor='val_loss', ):
        self.min_epochs = min_epochs
        self.patience = patience
        if patience <= 0:
            print('WARNING: setting patience to values <= 0 will results in immediate termination')
        self.larger_result_is_better = larger_result_is_better
        self._history: List[MonitoredQuantity] = []
        if larger_result_is_better:
            self.best_value: MonitoredQuantity = -inf
        else:
            self.best_value: MonitoredQuantity = inf
        self.best_epoch = -1
        MonitorMetric.__init__(self,
                               monitor=monitor,
                               termination_condition_epoch_end=self.early_stopping_condition, )
