from abc import abstractmethod
from typing import List, Any

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import Metric, Accuracy, MeanMetric, MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.utilities.data import dim_zero_cat


class NamedMetric(Metric):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def update(self, *_, **__) -> None:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def compute(self) -> Any:
        raise NotImplementedError('Abstract method')

    @staticmethod
    def reverse_broadcast_sample_weights(sample_weights: Tensor, ndim) -> Tensor:
        """
        There often is the issue that sample weights of shape (N,) need to be multiplied with some higher-dimensional Tensor of shape (N, ...).
        Usual broadcasting in pytorch would start from the right, but we want to start from the left, so that each sample weight is multiplied with the corresponding row.
        """
        assert len(sample_weights.shape) == 1
        while len(sample_weights.shape) < ndim:
            sample_weights = sample_weights.unsqueeze(-1)
        return sample_weights

    def reset(self) -> None:
        super().reset()

        def r(m):
            if hasattr(m, 'reset'):
                m.reset()

        for c in self.children():
            c.apply(r)


class NamedMetricWrapper(NamedMetric):
    def __init__(self, base_metric: Metric, name: str, wrapped_requires_class_idx_targets=False):
        super().__init__(name=name)
        self.base_metric = base_metric
        self.wrapped_requires_class_idx_targets = wrapped_requires_class_idx_targets

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.wrapped_requires_class_idx_targets:
            target = torch.argmax(target, dim=-1)
        self.base_metric.update(preds, target)

    def compute(self) -> Any:
        return self.base_metric.compute()


class ProbToClassIdxWrapper(NamedMetric):
    def __init__(self, base_metric: Metric, name: str):
        super().__init__(name=name)
        self.base_metric = base_metric

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = torch.argmax(preds, dim=1, keepdim=True)
        target = torch.argmax(target, dim=1, keepdim=True)
        self.base_metric.update(preds, target)

    def compute(self) -> Any:
        return self.base_metric.compute()


class WeightedMetric(NamedMetric):
    def __init__(self, unweighted_unaveraged_metric: Metric, name: str):
        super().__init__(name=name)
        self.wrapped = unweighted_unaveraged_metric
        self.add_state("sample_weights", [], dist_reduce_fx='cat')
        self.sample_weights: List[float]

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor) -> None:
        self.wrapped.update(preds, target)
        self.sample_weights.append(sample_weights)

    def compute(self) -> Any:
        unweighted_unaveraged_results: Tensor = self.wrapped.compute()
        flat_sample_weights = dim_zero_cat(self.sample_weights)
        unaveraged_results: Tensor = unweighted_unaveraged_results * flat_sample_weights
        return unaveraged_results.mean()


class WeightedCategoricalAccuracy(WeightedMetric):
    def __init__(self, num_classes: int, name: str):
        super().__init__(ProbToClassIdxWrapper(Accuracy(task="multiclass", num_classes=num_classes,
                                                        average='macro', multidim_average='samplewise'), name=f'_{name}'),
                         name=name)


class BinaryMetricWrapper(NamedMetric):
    def __init__(self, base_metric: Metric, name: str):
        super().__init__(name=name)
        self.base_metric = base_metric

    def update(self, preds: Tensor, target: Tensor, sample_weights=None) -> None:
        preds = torch.cat([1 - preds, preds], dim=1)
        target = torch.cat([1 - target, target], dim=1)
        if sample_weights is None:
            self.base_metric.update(preds, target)
        else:
            self.base_metric.update(preds, target, sample_weights)

    def compute(self) -> Any:
        return self.base_metric.compute()

    @property
    def higher_is_better(self):
        return self.base_metric.higher_is_better

    @property
    def is_differentiable(self):
        return self.base_metric.is_differentiable


class WeightedCategoricalCrossEntropy(NamedMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, name):
        super().__init__(name)
        self.m = MeanMetric()
        self.ce = CrossEntropyLoss(reduction='none')

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor) -> None:
        loss = self.ce(preds, target.float())
        self.m.update(loss, sample_weights)

    def compute(self) -> Any:
        return self.m.compute()


class CategoricalCrossEntropy(WeightedCategoricalCrossEntropy):
    def update(self, preds: Tensor, target: Tensor, sample_weights: float = 1.) -> None:
        if sample_weights != 1.:
            raise ValueError('CategoricalCrossEntropy does not support sample weights')
        loss = self.ce(preds, target.float())
        self.m.update(loss, weight=1.0)


class GenantScoreRegressionAccuracy(NamedMetric):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.m = MeanMetric()

    def update(self, preds: Tensor, target: Tensor) -> None:
        equals = self._per_sample_acc(preds, target)
        self.m.update(equals)

    @staticmethod
    def _per_sample_acc(preds, target):
        rounded_preds = torch.round(preds)
        zero = torch.zeros_like(rounded_preds)
        three = torch.ones_like(rounded_preds) * 3
        rounded_preds = torch.max(zero, rounded_preds)
        rounded_preds = torch.min(three, rounded_preds)
        equals = torch.eq(target, rounded_preds).float()
        assert equals.shape[-1] == 1
        equals = equals.squeeze(-1)
        return equals

    def compute(self) -> Any:
        return self.m.compute()


class WeightedGenantScoreRegressionAccuracy(NamedMetric):
    higher_is_better = True

    def __init__(self, name: str):
        super().__init__(name=name)
        self.m = MeanMetric()

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor) -> None:
        equals = GenantScoreRegressionAccuracy._per_sample_acc(preds, target)
        self.m.update(equals, sample_weights)

    def compute(self) -> Any:
        return self.m.compute()


class GenantScoreCustomLoss(NamedMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ce = CategoricalCrossEntropy(name='cce')
        self.mse = MeanSquaredError()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.ce.update(preds, target)
        self.mse.update(preds, target)

    def compute(self) -> Any:
        return self.ce.compute() + self.mse.compute()


class WeightedMeanAbsoluteError(NamedMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, name: str):
        super().__init__(name=name)
        self.mae = MeanAbsoluteError()

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor) -> None:
        sample_weights = self.reverse_broadcast_sample_weights(sample_weights, len(preds.shape))
        self.mae.update(preds * sample_weights, target * sample_weights)

    def compute(self) -> Any:
        return self.mae.compute()


class WeightedMeanSquaredError(NamedMetric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, name: str):
        super().__init__(name=name)
        self.mse = MeanSquaredError()

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor) -> None:
        sample_weights = torch.sqrt(sample_weights)
        sample_weights = self.reverse_broadcast_sample_weights(sample_weights, len(preds.shape))
        self.mse.update(preds * sample_weights, target * sample_weights)

    def compute(self) -> Any:
        return self.mse.compute()


class NotImplementedMetric(NamedMetric):
    def __init__(self, name: str, *_args, **_kwargs):
        super().__init__(name=name)

    def update(self, preds: Tensor, target: Tensor, sample_weights: Tensor = None) -> None:
        pass

    def compute(self) -> Any:
        return torch.nan


class OneVsAllAUROC(NamedMetric):
    WRAPPED_METRIC_REQUIRES_CLASS_IDX_TARGET = True

    def __init__(self, name: str, num_classes: int, one_cls_idx: int):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.auroc = MulticlassAUROC(num_classes=num_classes, average="none")
        self.one_cls_idx = one_cls_idx

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.WRAPPED_METRIC_REQUIRES_CLASS_IDX_TARGET:
            target = torch.argmax(target, dim=-1)
        self.auroc.update(preds, target)

    def compute(self) -> Any:
        return self.auroc.compute()[self.one_cls_idx]
