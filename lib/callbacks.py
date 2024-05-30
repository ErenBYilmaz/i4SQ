import math
from typing import Optional, Dict, Any, List

from lib import dl_backend
from lib.image_processing_tool import TrainableImageProcessingTool

try:
    from tensorflow.python.keras.callbacks import CallbackList as BaseCallbackList

    # If we inherit from the tensorflow callback list, tensorflow will know how to handle it
    bcl_from_tf = True
except ImportError:
    class BaseCallbackList:
        pass


    bcl_from_tf = False


class AnyCallback:
    """Implements the keras callback interface without the need to import tensorflow"""

    def __init__(self):
        super().__init__()
        self.model: Optional[TrainableImageProcessingTool] = None
        self.params: Optional[Dict[str, Any]] = None

    def set_model(self, model: TrainableImageProcessingTool):
        assert isinstance(model, TrainableImageProcessingTool), type(model)
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        self.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.on_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def set_params(self, params):
        self.params = params

    @staticmethod
    def _implements_train_batch_hooks():
        return True

    @staticmethod
    def _implements_test_batch_hooks():
        return True

    @staticmethod
    def _implements_predict_batch_hooks():
        return True


class TerminateOnNaN(AnyCallback):
    """
    Callback that terminates training when a NaN loss is encountered.
    Inspired by keras.callbacks.TerminateOnNaN
    """

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            loss = float(loss)
            # loss = tf_utils.sync_to_numpy_or_python_type(loss)
            if math.isnan(loss) or math.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True


class CallbackList(BaseCallbackList):
    """
    Container abstracting a list of callbacks.
    Copied and modified from keras.callbacks.CallbackList
    """

    def __init__(self,
                 callbacks: List[AnyCallback] = None,
                 model: TrainableImageProcessingTool = None,
                 **params):
        if callbacks is None:
            callbacks = []
        self.callbacks: List[AnyCallback] = callbacks
        if bcl_from_tf:
            self._add_default_callbacks(params.get('add_history', False),
                                        params.get('add_progbar', False), )
        self.model = None
        self.params = None
        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

        self._supports_tf_logs = all(
            getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks)
        assert self._supports_tf_logs
        self._batch_hooks_support_tf_logs = all(
            getattr(cb, '_supports_tf_logs', False)
            for cb in self.callbacks
            if cb._implements_train_batch_hooks()
            or cb._implements_test_batch_hooks()
            or cb._implements_predict_batch_hooks())
        assert self._batch_hooks_support_tf_logs

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model: TrainableImageProcessingTool):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        logs = logs
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`.
        """
        logs = logs
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class RecordBatchMetrics(AnyCallback):
    def __init__(self, cumulative: bool, only_metrics: Optional[List[str]] = None, ):
        super().__init__()
        if only_metrics is not None:
            if len(only_metrics) == 0:
                raise ValueError(only_metrics)
            if len(only_metrics) != len(set(only_metrics)):
                raise ValueError(only_metrics)
        self.cumulative = cumulative
        self.only_metrics = only_metrics
        self.metrics = []

    def on_test_batch_end(self, batch, logs=None):
        self.metrics.append(logs if self.only_metrics is None else {m: logs[m] for m in self.only_metrics})
        if not self.cumulative:
            self.reset_metrics()

    def set_model(self, model: Any):
        self.model = model  # could be a TrainableImageProcessingTool or a tensorflow or torch model

    def reset_metrics(self):
        dl_backend.b().reset_model_metrics(model=self.model,
                                           only_metrics=self.only_metrics)

    def compute_metrics_on_batches(self, batches: list, model: Any):
        def data_generator():
            for batch in batches:
                yield batch

        model.evaluate(x=data_generator(),
                       verbose=0,
                       callbacks=[self],
                       steps=len(batches),
                       return_dict=True, )
        return self.metrics


class RecordBatchLosses(RecordBatchMetrics):
    def __init__(self):
        super().__init__(only_metrics=['loss'], cumulative=False)
