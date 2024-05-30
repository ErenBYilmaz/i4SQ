from typing import Optional, Tuple, Type, List

import torch
from torch.cuda import OutOfMemoryError
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryRecall, BinarySpecificity, BinaryPrecision, MulticlassAUROC, MulticlassF1Score

from lib.backend_interface import DeepLearningBackendInterface
from lib.torch_metrics import WeightedCategoricalCrossEntropy, BinaryMetricWrapper, WeightedCategoricalAccuracy, NamedMetricWrapper, NamedMetric, WeightedGenantScoreRegressionAccuracy, \
    WeightedMeanAbsoluteError, WeightedMeanSquaredError, NotImplementedMetric, GenantScoreCustomLoss, CategoricalCrossEntropy, ProbToClassIdxWrapper, OneVsAllAUROC


class TorchBackendInterface(DeepLearningBackendInterface):
    def metrics_for_binary_task(self, threshold: Optional[float] = 0.5) -> List[NamedMetric]:
        return [
            NamedMetricWrapper(BinaryAccuracy(threshold=threshold), name='acc'),
            NamedMetricWrapper(BinaryAUROC(), name='roc'),
            NamedMetricWrapper(BinaryAveragePrecision(), name='aps'),
            NamedMetricWrapper(BinaryRecall(threshold=threshold), name='sens'),
            NamedMetricWrapper(BinarySpecificity(threshold=threshold), name='spec'),
            NamedMetricWrapper(BinaryPrecision(threshold=threshold), name='ppv'),
            NotImplementedMetric(threshold=threshold, name='npv'),
        ]

    def weighted_metrics_for_binary_task(self) -> List[NamedMetric]:
        return [
            BinaryMetricWrapper(WeightedCategoricalCrossEntropy(name='_ce'), name='ce'),
            BinaryMetricWrapper(WeightedCategoricalAccuracy(name='_wacc', num_classes=2), name='wacc'),
        ]

    def multi_task_weighted_metrics(self, num_classes: int) -> List[NamedMetric]:
        return [WeightedCategoricalCrossEntropy(name='wcce'),
                WeightedCategoricalAccuracy(name='wcacc', num_classes=num_classes)]

    def multi_task_metrics(self, num_classes: int) -> List[NamedMetric]:
        metrics: List[NamedMetric] = [CategoricalCrossEntropy(name='cce'),
                                      ProbToClassIdxWrapper(Accuracy(task="multiclass", num_classes=num_classes), name='cacc'),
                                      NamedMetricWrapper(MulticlassF1Score(num_classes=num_classes, average='macro'), name='macro_f1'),
                                      NamedMetricWrapper(MulticlassAUROC(num_classes=num_classes,
                                                                         average='macro'),
                                                         name='roc_avg',
                                                         wrapped_requires_class_idx_targets=True)]
        for c_idx in range(num_classes):
            metrics.append(OneVsAllAUROC(num_classes=num_classes, one_cls_idx=c_idx, name=f'roc_c{c_idx}'))
        return metrics

    def genant_score_custom_loss_metrics(self) -> List[NamedMetric]:
        return [GenantScoreCustomLoss(name="custom")]

    def weighted_genant_score_regression_accuracy_metrics(self) -> List[NamedMetric]:
        return [WeightedGenantScoreRegressionAccuracy(name='gsr_acc')]

    def regression_metrics(self) -> List[NamedMetric]:
        return [
            NamedMetricWrapper(MeanSquaredError(), name='mse'),
            NamedMetricWrapper(MeanAbsoluteError(), name='mae'),
            NamedMetricWrapper(MeanSquaredError(squared=False), name='rmse'),
        ]

    def regression_weighted_metrics(self) -> List[NamedMetric]:
        return [WeightedMeanAbsoluteError(name='wmae'),
                WeightedMeanSquaredError(name='wmse'),
                NotImplementedMetric(name='wmse_by_var'), ]

    def categorical_crossentropy_loss(self):
        return WeightedCategoricalCrossEntropy(name='loss')

    def binary_crossentropy_loss(self):
        return BinaryMetricWrapper(WeightedCategoricalCrossEntropy(name='_loss'), name='loss')

    def mse_loss(self):
        return WeightedMeanSquaredError(name='loss')

    def gpu_error_classes(self) -> Tuple[Type[Exception], ...]:
        return (OutOfMemoryError,)

    def empty_model(self, loss):
        raise NotImplementedError('TODO')

    def clear_session(self):
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    def limit_memory_usage(self):
        import lib.memory_control
        lib.memory_control.MemoryLimiterTorch.limit_memory_usage()
        torch.set_float32_matmul_precision('medium')

    def reuse_training_workers(self) -> bool:
        return False
