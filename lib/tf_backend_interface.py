from typing import Optional, List, Union, Tuple, Type

import tensorflow
from tensorflow.python.keras.metrics import Recall, Metric, CategoricalCrossentropy, CategoricalAccuracy, MeanSquaredError, MeanAbsoluteError
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, UnknownError, InternalError

from lib.backend_interface import DeepLearningBackendInterface
from lib.memory_control import tf_memory_leak_cleanup
from lib.metrics import Specificity, PPV, NPV, F1ScoreWithoutResetStatesWarning, genantscore_custom_loss, genantscore_regression_accuracy, MeanSquaredErrorByVariance
from lib.util import copy_and_rename_method
from tensorflow.keras.models import Sequential


class TensorflowBackendInterface(DeepLearningBackendInterface):

    def multi_task_weighted_metrics(self, num_classes: int) -> list:
        return [CategoricalCrossentropy(name='wcce'), CategoricalAccuracy(name='wcacc')]

    def metrics_for_binary_task(self, threshold: Optional[float] = 0.5) -> list:
        return [tensorflow.keras.metrics.BinaryAccuracy(name='acc', threshold=threshold),
                tensorflow.keras.metrics.AUC(name='roc', curve='ROC'),
                tensorflow.keras.metrics.AUC(name='aps', curve='PR'),
                Recall(thresholds=threshold, name='sens'),
                Specificity(thresholds=threshold, name='spec'),
                PPV(thresholds=threshold, name='ppv'),
                NPV(thresholds=threshold, name='npv'),
                ]

    def weighted_metrics_for_binary_task(self):
        return [tensorflow.keras.metrics.BinaryCrossentropy(name='ce'),
                tensorflow.keras.metrics.BinaryAccuracy(name='wacc')]

    def multi_task_metrics(self, num_classes: int) -> list:
        metrics: List[Union[str, Metric]] = [tensorflow.keras.metrics.CategoricalCrossentropy(name='cce'),
                                             tensorflow.keras.metrics.CategoricalAccuracy(name='cacc'),
                                             F1ScoreWithoutResetStatesWarning(num_classes=num_classes, average='macro', name='macro_f1'),
                                             tensorflow.keras.metrics.AUC(curve='ROC',
                                                                          multi_label=True,
                                                                          num_labels=num_classes,
                                                                          name=f'roc_avg')]
        for c_idx in range(num_classes):
            metrics.append(tensorflow.keras.metrics.AUC(curve='ROC',
                                                        multi_label=True,
                                                        num_labels=num_classes,
                                                        label_weights=self.one_hot_encode(num_classes, c_idx),
                                                        name=f'roc_c{c_idx}'))
        return metrics

    def genant_score_custom_loss_metrics(self) -> list:
        return [copy_and_rename_method(genantscore_custom_loss, "custom")]

    def weighted_genant_score_regression_accuracy_metrics(self) -> list:
        return [copy_and_rename_method(genantscore_regression_accuracy, 'gsr_acc'), ]

    def regression_metrics(self) -> list:
        return [
            tensorflow.keras.metrics.MeanSquaredError(name='mse'),
            tensorflow.keras.metrics.MeanAbsoluteError(name='mae'),
            tensorflow.keras.metrics.RootMeanSquaredError(name='rmse'),
        ]

    def regression_weighted_metrics(self) -> list:
        return [MeanAbsoluteError(name='wmae'),
                MeanSquaredError(name='wmse'),
                MeanSquaredErrorByVariance(name='wmse_by_var'), ]

    def gpu_error_classes(self) -> Tuple[Type[Exception], ...]:
        return (ResourceExhaustedError, UnknownError, InternalError)

    def empty_model(self, loss):
        m = Sequential()
        m.compile(loss=loss)
        return m

    def categorical_crossentropy_loss(self):
        return 'categorical_crossentropy'

    def binary_crossentropy_loss(self):
        return 'binary_crossentropy'

    def mse_loss(self):
        return 'mean_squared_error'

    def clear_session(self):
        tensorflow.keras.backend.clear_session()

    def limit_memory_usage(self):
        import lib.memory_control
        lib.memory_control.MemoryLimiter.limit_memory_usage()

    def memory_leak_cleanup(self):
        tf_memory_leak_cleanup()
        if self.memory_limit_bytes() is not None:
            assert self.current_gpu_usage_bytes() <= 0.5 * self.memory_limit_bytes(), self.memory_stats()

    @classmethod
    def current_gpu_usage_bytes(cls):
        assert len(tensorflow.config.experimental.list_physical_devices('GPU')) == 1
        current_gpu_usage_bytes = tensorflow.config.experimental.get_memory_info('GPU:0')['current']
        return current_gpu_usage_bytes

    @classmethod
    def memory_limit_bytes(cls):
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        result = 0
        for gpu in gpus:
            config = tensorflow.config.experimental.get_virtual_device_configuration(gpu)
            if config is None:
                assert len(gpus) == 1, len(gpus)  # I have not thought about any other case yet
                return None
            result += config[0].memory_limit
        return result * 1024 * 1024  # limit is in MB

    def reset_model_metrics(self,
                            model: tensorflow.keras.models.Model,
                            only_metrics: Optional[List[str]] = None):
        reset_metrics_count = 0
        for m in model.metrics:
            if only_metrics is None or m.name in only_metrics:
                m.reset_state()
                reset_metrics_count += 1
        if only_metrics is not None:
            assert reset_metrics_count == len(only_metrics)
        if reset_metrics_count == 0:
            raise RuntimeError((only_metrics, model.metrics))

    def reuse_training_workers(self) -> bool:
        return True

    def memory_stats(self):
        return {'current': self.current_gpu_usage_bytes(), 'max': self.memory_limit_bytes()}
