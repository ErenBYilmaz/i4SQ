import json
import math
import pickle
import typing
import uuid
from abc import ABC, abstractmethod
from copy import copy
from math import log
from numbers import Real
from typing import List, Dict, Iterable, Type, Optional, Tuple, Union, Any

import cachetools
import numpy
from cachetools import LRUCache

from lib import dl_backend

try:
    import tensorflow

    tensorflow_available = True
except ImportError:
    tensorflow = None
    tensorflow_available = False
if tensorflow_available:
    from tensorflow.keras.layers import Conv3D
    import tensorflow.keras.metrics
    from tensorflow.python.keras.layers import Dense, Layer

try:
    import torch

    torch_available = True
except ImportError:
    torch = None
    torch_available = False
if torch_available:
    import torch.nn

import hiwi

try:
    from annotation_loader import Vertebra
    from fresh_status import FreshStatus
except ImportError as e:
    Vertebra = None
    FreshStatus = None
from lib.util import EBC, powerset, all_sets_and_disjoint
from load_data import VERTEBRAE


class GroupByClass:
    def __init__(self, groups_by_class: Dict[str, List[Tuple[str, str]]], task: 'VertebraTask'):
        self.task = task
        self.groups_by_class = groups_by_class

    def all_vertebrae(self):
        patient_id_and_vertebra_index = lambda x: (x[0], f'{VERTEBRAE.index(x[1])}' if x[1] in VERTEBRAE else f'_{x[1]}')
        return sorted(sum(self.groups(), []), key=patient_id_and_vertebra_index)

    def groups(self):
        return self.groups_by_class.values()

    def num_classes(self):
        return self.task.num_classes()

    def class_names(self):
        return self.groups_by_class.keys()

    def items(self):
        return self.groups_by_class.items()

    def check_all_classes_present(self):
        assert set(self.class_names()) == set(self.task.class_names())
        for class_name in self.class_names():
            assert self.class_size(class_name) > 0

    def group_for_class(self, class_name: str):
        return self.groups_by_class[class_name]

    def vertebrae_in_class(self, class_name: str):
        """ Just an alias for group_for_class """
        return self.group_for_class(class_name)

    def check_all_sets_and_disjoint(self):
        assert all_sets_and_disjoint(self.groups())

    def class_size(self, class_name: str):
        return len(self.group_for_class(class_name))

    def check_all_vertebrae_present(self, vertebra_names: List[Tuple[str, str]]):
        assert set(vertebra_names).issubset(self.all_vertebrae())

    def check_subset(self, vertebra_names: List[Tuple[str, str]]):
        assert set(self.all_vertebrae()).issubset(vertebra_names)

    def class_weights(self) -> Dict[str, float]:
        return {
            class_name: ((self.num_vertebrae()) / ((self.num_classes() * self.class_size(class_name)) if self.class_size(class_name) > 0 else 1))
            for class_name in self.class_names()
        }

    def num_vertebrae(self):
        return len(self.all_vertebrae())

    def summary(self):
        return f'"{self.task.output_layer_name()}": {json.dumps(self.class_sizes())}'

    def class_sizes(self):
        return {class_name: self.class_size(class_name) for class_name in self.class_names()}


class VertebraTask(EBC, ABC):
    class Unusable(ValueError):
        """if a vertebra is passed that can not be used for training or validation"""
        pass

    class UnlabeledVertebra(Unusable):
        """if a vertebra is passed that can not be used for training or validation because it has not ground truth"""

        class ThrownByClassIdxOfLabel(RuntimeError):
            def __init__(self):
                super().__init__(f'UnlabeledVertebra should not be thrown by `class_idx_of_label`. '
                                 'If this is due to a label that does not belong to a class, use LabelWithoutClass instead.')

    class ExcludedVertebra(UnlabeledVertebra):
        """if a vertebra is passed that must not be used for training or validation"""
        pass

    class VertebraNotInTheImage(UnlabeledVertebra):
        """if a vertebra is passed that can not be used for training or validation because it is not in the image"""
        pass

    class LabelWithoutClass(ValueError):
        """
        Happens, if a label is passed that is not one of the valid class_labels().
        For example, this could be a soft pseudo-label generated from a teacher model.
        """
        pass

    def smooth_label(self, non_smoothed_label, smoothing_alpha: float):
        raise NotImplementedError('Abstract method')

    def __init__(self, loss_weight: float):
        self._uid = None
        self.loss_weight = loss_weight

    def __eq__(self, other):
        return self is other

    def identifier(self):
        if not hasattr(self, '_uid') or self._uid is None:
            self._uid = str(uuid.uuid4())
        return self._uid

    @abstractmethod
    def output_layer_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError

    @abstractmethod
    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        raise NotImplementedError

    def y_true_or_default_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        try:
            return self.y_true_from_hiwi_image_and_vertebra(image, vertebra_name)
        except self.UnlabeledVertebra:
            return self.default_label()

    @abstractmethod
    def metrics(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def weighted_metrics(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def natural_metrics_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def natural_weighted_metrics_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self):
        raise NotImplementedError

    @abstractmethod
    def build_output_layers_tf(self, x: 'tensorflow.Tensor', **regularizers) -> 'tensorflow.Tensor':
        raise NotImplementedError

    def build_output_layer_torch(self, input_shape: Tuple[int, ...]) -> 'torch.nn.Module':
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError

    def num_classes(self) -> int:
        return len(self.class_names())

    def class_name_of_label(self, label):
        idx = numpy.array(self.class_idx_of_label(label)).item()
        return self.class_names()[idx]

    def class_names(self) -> List[str]:
        return ['whole_dataset']

    def class_idx_of_label(self, label):
        return 0

    def class_name_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name, on_invalid='raise') -> str:
        return 'whole_dataset'

    def nearest_class_idx_for_model_output(self, y_pred) -> numpy.array:
        return numpy.zeros(y_pred.shape[0], dtype=int)

    def default_class_idx(self):
        return self.class_idx_of_label(self.default_label())

    def class_probabilities(self, y_pred: numpy.ndarray) -> numpy.ndarray:
        # this should preserve the batch axis if called with multiple samples at once
        y_pred: numpy.ndarray = numpy.ones_like(y_pred)
        y_pred = y_pred.prod(axis=-1)
        return y_pred

    def group_by_class(self,
                       image_list: hiwi.ImageList,
                       exclude_vertebrae=None,
                       on_invisible_vertebra='raise',
                       exclude_unclassified=False) -> GroupByClass:
        if exclude_vertebrae is None:
            exclude_vertebrae = []
        groups = {name: [] for name in self.class_names()}
        for image in image_list:
            for vertebra in image.parts:
                patient_id = image['patient_id']
                if (patient_id, vertebra) in exclude_vertebrae:
                    continue
                try:
                    label = self.y_true_from_hiwi_image_and_vertebra(image, vertebra).tolist()
                except self.VertebraNotInTheImage:
                    if on_invisible_vertebra == 'raise':
                        raise
                    elif on_invisible_vertebra == 'ignore':
                        continue
                    elif on_invisible_vertebra == 'same_as_unlabeled':
                        label = self.UnlabeledVertebra
                    elif on_invisible_vertebra == 'use_default_label':
                        label = self.default_label()
                    else:
                        raise ValueError(on_invisible_vertebra)
                except VertebraTask.UnlabeledVertebra:
                    label = VertebraTask.UnlabeledVertebra
                try:
                    if label is VertebraTask.UnlabeledVertebra:
                        raise VertebraTask.LabelWithoutClass
                    name = self.class_name_of_label(label)
                except VertebraTask.UnlabeledVertebra as e:
                    raise e.ThrownByClassIdxOfLabel()
                except VertebraTask.LabelWithoutClass:
                    if exclude_unclassified == 'dummy_label':
                        name = None
                        if None not in groups:
                            groups[None] = []
                    elif exclude_unclassified == 'default_label':
                        name = self.class_name_of_label(self.default_label())
                    elif exclude_unclassified:
                        continue
                    else:
                        raise
                groups[name].append((patient_id, vertebra))
        return GroupByClass(groups, task=self)

    def natural_metric_name_by_metric_name(self, metric_name):
        metrics_indices = [m_idx for m_idx, m in enumerate(self.metrics_names()) if metric_name.endswith(m)]
        assert len(metrics_indices) == 1
        return self.natural_metrics_names()[metrics_indices[0]]

    def metrics_as_tuples(self, include_output_name):
        return [
            (self.output_layer_name() + '_' + n, nn, wv) if include_output_name else (n, nn, wv)
            for n, nn, wv in zip(self.metrics_names(), self.natural_metrics_names(), self.worst_metric_values())
        ] + [
            (self.output_layer_name() + '_' + n, nn, wv) if include_output_name else (n, nn, wv)
            for n, nn, wv in zip(self.weighted_metrics_names(), self.natural_weighted_metrics_names(), self.worst_weighted_metric_values())
        ]

    @staticmethod
    def with_additional_spatial_dimensions(result, n_spatial_dims):
        return numpy.expand_dims(result, axis=tuple(range(n_spatial_dims)))

    def y_true_with_additional_spatial_dimensions(self, image: hiwi.Image, vertebra_name, n_spatial_dims: int) -> numpy.ndarray:
        result = self.y_true_from_hiwi_image_and_vertebra(image, vertebra_name)
        result = self.with_additional_spatial_dimensions(result, n_spatial_dims)
        return result

    def default_label_with_additional_spatial_dimensions(self, n_spatial_dims: int) -> numpy.ndarray:
        result = self.default_label()
        result = self.with_additional_spatial_dimensions(result, n_spatial_dims)
        return result

    def is_excluded(self, image: hiwi.Image, vertebra_name) -> bool:
        try:
            self.y_true_from_hiwi_image_and_vertebra(image, vertebra_name)
        except self.ExcludedVertebra:
            return True
        except self.UnlabeledVertebra:
            return False
        except self.VertebraNotInTheImage:
            return False
        return False

    def metrics_names(self) -> List[str]:
        results = []
        for s in self.metrics():
            if isinstance(s, str):
                results.append(s)
            elif hasattr(s, 'name'):
                results.append(s.name)
            elif hasattr(s, '__name__'):
                results.append(s.__name__)
            else:
                raise RuntimeError(s)
        return results

    @abstractmethod
    def worst_metric_values(self) -> List[float]:
        raise NotImplementedError

    def weighted_metrics_names(self) -> List[str]:
        results = []
        for s in self.weighted_metrics():
            if isinstance(s, str):
                results.append(s)
            elif hasattr(s, 'name'):
                results.append(s.name)
            elif hasattr(s, '__name__'):
                results.append(s.__name__)
            else:
                raise RuntimeError(s)
        return results

    @abstractmethod
    def worst_weighted_metric_values(self) -> List[float]:
        raise NotImplementedError

    def image_label(self, image: hiwi.Image, exclude_unlabeled=True) -> numpy.ndarray:
        y_trues = []
        for v in image.parts:
            try:
                y_trues.append(self.y_true_from_hiwi_image_and_vertebra(image, v))
            except self.UnlabeledVertebra:
                if exclude_unlabeled:
                    continue
                else:
                    raise
        assert len(y_trues) > 0, image.path
        y_trues = numpy.stack(y_trues, axis=0)
        y_true = self.patient_level_aggregation(y_trues)
        return y_true

    def binarized_label(self, y_true_or_y_pred: numpy.ndarray) -> numpy.array:
        return (self.nearest_class_idx_for_model_output(y_true_or_y_pred) > 0).astype(int)

    def binarized_image_label(self, image: hiwi.Image, exclude_unlabeled=True) -> numpy.array:
        return self.binarized_label(self.image_label(image, exclude_unlabeled=exclude_unlabeled))

    def optimization_criterion(self):
        names = self.weighted_metrics_names() + self.metrics_names()
        for n in names:
            if n == 'ce':
                return n
            if n == 'wcce':
                return n
            if n == 'wmse':
                return n
        raise ValueError(f'No suitable metric found. Available metrics: {names}')


class VertebraClassificationTask(VertebraTask, ABC):

    @abstractmethod
    def class_labels(self) -> numpy.ndarray:
        raise NotImplementedError

    def default_label(self):
        return self.class_labels()[0]

    @abstractmethod
    def class_names(self) -> List[str]:
        raise NotImplementedError

    def class_labels_by_name(self) -> Dict[str, numpy.ndarray]:
        assert len(self.class_labels()) == len(self.class_names())
        return {
            k: v for k, v in zip(self.class_names(), self.class_labels())
        }

    def class_name_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name, on_invalid='raise') -> str:
        try:
            return self.class_name_of_label(self.y_true_from_hiwi_image_and_vertebra(image, vertebra_name))
        except (VertebraClassificationTask.LabelWithoutClass, VertebraTask.UnlabeledVertebra):
            if on_invalid == 'raise':
                raise
            elif on_invalid == 'return_empty_string':
                return ''
            else:
                raise ValueError(on_invalid)

    def smooth_label(self, non_smoothed_label, smoothing_alpha: float):
        """See https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06"""
        return non_smoothed_label * (1 - smoothing_alpha) + smoothing_alpha / self.num_classes()

    def class_idx_of_label(self, label):
        label = numpy.array(label)
        result = numpy.argmax(label)
        if label[..., result] == 1:
            return result
        else:
            raise self.LabelWithoutClass()

    def label_has_class(self, label):
        try:
            self.class_idx_of_label(label)
        except self.LabelWithoutClass:
            return False
        else:
            return True

    def nearest_class_idx_for_model_output(self, y_pred):
        raise NotImplementedError

    def summary(self, groups_by_class):
        return f'Classes for task {repr(self)}: ' + str({c_name: len(groups_by_class[c_name]) for c_name in self.class_names()})

    def class_probabilities(self, y_pred) -> numpy.ndarray:
        raise NotImplementedError

    def prediction_confidences(self, y_pred):
        return self.class_probabilities(y_pred).max(axis=-1)

    @staticmethod
    def load_trauma_vertebra(image: hiwi.Image, vertebra_name):
        if Vertebra is None:
            raise VertebraTask.UnlabeledVertebra('Trauma dataset not loaded')
        try:
            return Vertebra.from_dict(image.parts[vertebra_name])
        except Vertebra.InvalidDict:
            raise VertebraTask.UnlabeledVertebra()


class VertebraTasks(List[VertebraTask]):
    MAIN_TASK_IDX = 0

    def main_task(self):
        return self[self.MAIN_TASK_IDX]

    def main_task_name(self):
        return self[self.MAIN_TASK_IDX].output_layer_name()

    def classification_tasks(self):
        return VertebraTasks([self[task_idx] for task_idx in self.classification_task_indices()])

    def classification_task_indices(self):
        return [task_idx for task_idx, task in enumerate(self) if isinstance(task, VertebraClassificationTask)]

    class CallableString:
        def __init__(self, s: str):
            self.s = s

        def __call__(self):
            return self.s

    def serialize(self) -> str:
        for task in self:
            task._uid = None
        return repr(pickle.dumps(self))

    @staticmethod
    def deserialize(data: str) -> 'VertebraTasks':
        return pickle.loads(eval(data))

    @classmethod
    def load_from_config(cls, config: Dict[str, typing.Any]) -> 'VertebraTasks':
        results = cls.deserialize(config['tasks'])
        results.simple_consistency_checks()
        return results

    def main_task_twice(self, second_loss_weight=None):
        task1 = copy(self.main_task())
        task2 = copy(self.main_task())
        task2.output_layer_name = self.CallableString(task1.output_layer_name() + '2')
        if second_loss_weight is not None:
            task2.loss_weight = second_loss_weight
        return VertebraTasks([task1, task2])

    def output_layer_names(self):
        return [t.output_layer_name() for t in self]

    @staticmethod
    def tasks_for_which_diagnostik_bilanz_has_annotations(main_task_type: Optional[Type[VertebraTask]] = None):
        tasks = VertebraTasks([
            BinaryOsteoporoticFractureClassification(loss_weight=1 / log(2)),
            GenantScoreRegression(loss_weight=1, deformity_as_zero=True),
            CombinedDifferentialDiagnosisCategoryAndGenantScoreClassification(loss_weight=1 / log(6)),
            DifferentialDiagnosisCategoryClassification(loss_weight=1 / log(3)),
            DeformityRegression(loss_weight=100, deformity_as_zero=True),
            SpondylosisClassification(loss_weight=10),
            GenantScoreClassification(loss_weight=1 / log(4), deformity_as_zero=True),
            # TotalBMDRegression(loss_weight=.01),
            # TrabPeeledBMDRegression(loss_weight=.01),
            # CorticalBMDRegression(loss_weight=.01),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0], positive=[1, 2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1], positive=[2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1, 2], positive=[3], deformity_as_zero=True),
        ])
        if main_task_type is not None:
            tasks.change_main_task(main_task_type)
        return tasks

    @staticmethod
    def tasks_for_which_diagnostik_bilanz_and_mros_scouts_have_annotations(main_task_type: Optional[Type[VertebraTask]] = None):
        tasks = VertebraTasks([
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0], positive=[1, 2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1], positive=[2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1, 2], positive=[3], deformity_as_zero=True),
            GenantScoreClassification(loss_weight=1 / log(4), deformity_as_zero=True),
            GenantScoreRegression(loss_weight=1, deformity_as_zero=True),
        ])
        if main_task_type is not None:
            tasks.change_main_task(main_task_type)
        return tasks

    @staticmethod
    def tasks_for_which_mros_has_annotations(main_task_type: Optional[Type[VertebraTask]] = None):
        tasks = VertebraTasks([
            GenantScoreRegression(loss_weight=1, deformity_as_zero=True),
            GenantScoreClassification(loss_weight=1 / log(4), deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0], positive=[1, 2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1], positive=[2, 3], deformity_as_zero=True),
            BinaryGenantScoreClassification(loss_weight=1 / log(2), negative=[0, 1, 2], positive=[3], deformity_as_zero=True),
        ])
        if main_task_type is not None:
            tasks.change_main_task(main_task_type)
        return tasks

    @staticmethod
    def tasks_for_which_trauma_dataset_has_annotations(main_task_type: Optional[Type[VertebraTask]] = None):
        from trauma_tasks import IsFracturedClassification, IsUnstableClassification, StabilityClassification, FreshnessClassification, \
            CombinedFreshUnstableClassification
        tasks = VertebraTasks(VertebraTasks([
            IsUnstableClassification(loss_weight=1 / log(2)),
            IsFracturedClassification(loss_weight=1 / log(2)),
            CombinedFreshUnstableClassification(loss_weight=1 / log(4)),
            StabilityClassification(loss_weight=1 / log(3)),
            FreshnessClassification(loss_weight=1 / log(3)),
        ]))
        if main_task_type is not None:
            tasks.change_main_task(main_task_type)
        return tasks

    def change_main_task(self, main_task_type: Type[VertebraTask]):
        main_tasks = []
        secondary_tasks = []
        for task in self:
            if isinstance(task, main_task_type):
                main_tasks.append(task)
            else:
                secondary_tasks.append(task)
        assert len(main_tasks) == 1
        self[:] = main_tasks + secondary_tasks

    def optimization_criterion(self):
        main_task = self.main_task()
        if len(self) > 1:
            prefix = main_task.output_layer_name() + '_'
        else:
            prefix = ''
        return prefix + main_task.optimization_criterion()

    def simple_consistency_checks(self):
        for task in self:
            assert len(task.weighted_metrics()) == len(task.natural_weighted_metrics_names()), type(task)
            assert len(task.weighted_metrics()) == len(task.worst_weighted_metric_values()), type(task)
            assert len(task.metrics()) == len(task.natural_metrics_names()), (type(task), len(task.metrics()), len(task.natural_metrics_names()))
            assert len(task.metrics()) == len(task.worst_metric_values()), (type(task), len(task.metrics()), len(task.natural_metrics_names()))

    def metrics_dicts_for_fnet(self):
        self.simple_consistency_checks()
        result: List[Dict[str, typing.Any]] = [
            {
                'name': f'{d[1]}_{m[0]}',
                'worst': m[2],
                'natural_name': f'{d[0]} {m[1]}',
                'keras_metric_name': m[0]
            }
            for d in [('Val.', 'val')]
            # for d in [('Val.', 'val'), ('Test', 'test')]
            for task in self
            for m in task.metrics_as_tuples(include_output_name=len(self) != 1)
        ]
        result += [
            {
                'name': f'{d[1]}_loss',
                'worst': math.inf,
                'natural_name': f'{d[0]} Loss',
                'keras_metric_name': 'loss'
            }
            for d in [('Val.', 'val'), ('Test', 'test')]
        ]
        result += [
            {
                'name': 'num_params',
                'worst': math.nan,
                'natural_name': '#params',
            },
            {
                'name': 'early_stopped_after',
                'worst': math.nan,
                'natural_name': 'Early stopped after',
            },
            {
                'name': 'early_stopped_during_last_2_epochs',
                'worst': math.nan,
                'natural_name': 'Early stopped during last 2 epochs',
            },
        ]
        assert len(set(m['name'] for m in result)) == len(result), list((m['name'] for m in result))
        assert len(set(m['natural_name'] for m in result)) == len(result), [m['natural_name'] for m in result]
        return result

    def by_name(self, name: str):
        for task in self:
            if task.output_layer_name() == name:
                return task
        raise ValueError(f'No task with name {name}')

    @classmethod
    def filter_artificial_tasks_from_config(cls, config: Dict[str, Any]):
        config = copy(config)
        tasks = VertebraTasks.load_from_config(config)
        tasks = VertebraTasks([t for t in tasks if not isinstance(t, OutputCombination)])
        config['tasks'] = tasks.serialize()
        return config


class BinaryClassificationTask(VertebraClassificationTask, ABC):
    def class_labels(self):
        return numpy.array([[0], [1]])

    def class_idx_of_label(self, label):
        if isinstance(label, list):
            assert len(label) == 1
            if label[0] != int(label[0]):
                raise self.LabelWithoutClass()
            return int(label[0])
        return int(label)

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self, threshold=0.5: (self.identifier(), threshold))
    def metrics(self, threshold: Optional[float] = 0.5) -> list:
        return dl_backend.b().metrics_for_binary_task(threshold=threshold)

    def worst_metric_values(self) -> List[float]:
        return [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, ]

    def natural_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return [f'{o} Accuracy', f'{o} ROC-AUC', f'{o} AP', f'{o} Sensitivity', f'{o} Specificity', f'{o} PPV', f'{o} NPV']

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def weighted_metrics(self):
        return dl_backend.b().weighted_metrics_for_binary_task()

    def worst_weighted_metric_values(self) -> List[float]:
        return [math.inf, -math.inf]

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        return numpy.max(y_pred_or_y_true, axis=0, keepdims=True)

    def natural_weighted_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return [f'{o} Weighted Binary Crossentropy', f'{o} Weighted Accuracy']

    def loss_function(self):
        return dl_backend.b().binary_crossentropy_loss()

    def build_output_layers_tf(self, x: 'tensorflow.Tensor', **regularizers) -> 'tensorflow.Tensor':
        input_shape = x.shape.as_list()
        conv_layer_before = len(input_shape) == 5
        dense_layer_before = len(input_shape) == 2
        if conv_layer_before:
            return Conv3D(filters=1, kernel_size=x.shape.as_list()[1:4], activation='sigmoid', kernel_initializer='zeros', name=self.output_layer_name(),
                          **regularizers)(x)
        elif dense_layer_before:
            return Dense(units=1, activation='sigmoid', kernel_initializer='zeros', name=self.output_layer_name(), **regularizers)(x)
        else:
            raise ValueError(len(input_shape))

    def build_output_layer_torch(self, input_shape: Tuple[int, ...]) -> 'torch.nn.Module':
        linear = torch.nn.Linear(numpy.prod(input_shape[0]).item(), 1)
        torch.nn.init.zeros_(linear.weight)
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            linear,
            torch.nn.Sigmoid()
        )

    def nearest_class_idx_for_model_output(self, y_pred):
        return numpy.round(y_pred)[..., 0].astype('int')

    @classmethod
    def class_idx_for_model_output_using_threshold(cls, y_pred, classification_threshold: Real):
        above_threshold: numpy.ndarray = (numpy.array(y_pred)[..., 0] >= classification_threshold)
        return above_threshold.astype('int')

    def class_probabilities(self, y_pred) -> numpy.ndarray:
        return numpy.stack([1 - y_pred[..., 0], y_pred[..., 0]], axis=-1)

    # def optimization_criterion(self):
    #     return 'roc'


class BinaryOsteoporoticFractureClassification(BinaryClassificationTask):
    def output_layer_name(self) -> str:
        return 'ifo'

    def class_names(self) -> List[str]:
        return ['Not Osteoporotic', 'Osteoporotic Fracture']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        return self.osteoporosis_label(image, vertebra_name)

    @classmethod
    def osteoporosis_label(cls, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise cls.VertebraNotInTheImage()
        metadata = image.parts[vertebra_name]
        if 'Differential Diagnosis Category' in metadata:
            ddc = str(metadata['Differential Diagnosis Category'])
            if ddc == "Osteoporotic Fracture":
                return numpy.array([1])
            elif ddc in ['Deformity', 'Normal']:
                return numpy.array([0])
            elif ddc == 'Unevaluable':
                raise cls.ExcludedVertebra()
            else:
                raise ValueError()
        try:
            score: numpy.ndarray = GenantScoreRegression.extract_genant_score_as_number(image, vertebra_name, deformity_as_zero=False)
            # noinspection PyTypeChecker
            return score > 0
        except GenantScoreRegression.UnlabeledVertebra:
            pass
        v = cls.load_trauma_vertebra(image, vertebra_name)
        if not v.body_is_fractured():
            # all unfractured vertebrae in the trauma dataset are not osteoporotic
            return numpy.array([0])
        elif FreshStatus is not None:
            # all old fractures in the trauma dataset are osteoporotic
            if v.fresh_status() is FreshStatus.OLD:
                return numpy.array([1])
            # all fresh fractures in the trauma dataset are not osteoporotic
            if v.fresh_status() is FreshStatus.FRESH:
                return numpy.array([0])
        raise cls.UnlabeledVertebra


class DummyTask(BinaryClassificationTask):
    def __init__(self):
        super().__init__(loss_weight=1.0)

    def output_layer_name(self) -> str:
        return 'dummy'

    def class_names(self) -> List[str]:
        return ['Negative', 'Positive']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        return numpy.random.randint(0, 2, size=(1,))


class GenantZeroVsNonzeroClassification(BinaryClassificationTask):
    def output_layer_name(self) -> str:
        return 'gsfo'

    def class_names(self) -> List[str]:
        return ['0', 'Genant > 0']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        try:
            score = GenantScoreClassification.extract_genant_score_label(image, vertebra_name, deformity_as_zero=False)
            return score
        except GenantScoreClassification.UnlabeledVertebra:
            pass
        v = self.load_trauma_vertebra(image, vertebra_name)
        if v.fresh_status() is FreshStatus.OLD:
            # all old fractures in the trauma dataset are osteoporotic and all osteoporotic fractures have a grade > 0
            return numpy.array([1])
        raise self.UnlabeledVertebra()


class FNetRegressionTask(VertebraTask, ABC):
    OUTPUT_UNITS: int = None

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def weighted_metrics(self) -> list:
        return dl_backend.b().regression_weighted_metrics()

    def class_idx_of_label(self, label):
        if numpy.array_equal(label, self.default_label()):
            raise self.LabelWithoutClass()
        return 0

    def worst_weighted_metric_values(self) -> List[float]:
        return [math.inf, math.inf, math.inf]

    def smooth_label(self, non_smoothed_label, smoothing_alpha: float):
        return non_smoothed_label

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def metrics(self) -> list:
        return dl_backend.b().regression_metrics()

    def worst_metric_values(self) -> List[float]:
        return [math.inf, math.inf, math.inf]

    def natural_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return [
            f'{o} MSE',
            f'{o} MAE',
            f'{o} RMSE',
        ]

    def natural_weighted_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return [
            f'{o} WMAE',
            f'{o} WMSE',
            f'{o} WMSE/Varaiance'
        ]

    def loss_function(self):
        return dl_backend.b().mse_loss()

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        return numpy.sum(y_pred_or_y_true, axis=0, keepdims=True)

    def build_output_layers_tf(self, x: 'tensorflow.Tensor', **regularizers) -> 'tensorflow.Tensor':
        input_shape = x.shape.as_list()
        conv_layer_before = len(input_shape) == 5
        dense_layer_before = len(input_shape) == 2
        if conv_layer_before:
            return Conv3D(filters=self.OUTPUT_UNITS, kernel_size=x.shape.as_list()[1:4], activation='linear', kernel_initializer='zeros',
                          name=self.output_layer_name(),
                          **regularizers)(x)
        elif dense_layer_before:
            return Dense(units=self.OUTPUT_UNITS, activation='linear', kernel_initializer='zeros', name=self.output_layer_name(), **regularizers)(x)
        else:
            raise ValueError()

    def build_output_layer_torch(self, input_shape: Tuple[int, ...]) -> 'torch.nn.Module':
        linear = torch.nn.Linear(numpy.prod(input_shape[0]).item(), self.OUTPUT_UNITS)
        torch.nn.init.zeros_(linear.weight)
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            linear,
        )


class KeyValueBasedTask(VertebraTask):
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError('Abstract method')

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        vertebra_dict = image.parts[vertebra_name]
        if 'not_in_the_image' in vertebra_dict and vertebra_dict['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        k = self.key()
        if k in vertebra_dict:
            return numpy.array([float(vertebra_dict[k])])
        if k in image:
            return numpy.array([float(image[k])])
        raise self.UnlabeledVertebra()

    def output_layer_name(self) -> str:
        return self.key().lower()


class AgeRegressionTask(FNetRegressionTask):
    UPPER_AGE_BOUND = 150

    def output_layer_name(self) -> str:
        return 'age'

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TODO')

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        age: Optional[Union[str, float]] = None
        if 'GIAGE1' in image:
            age = image['GIAGE1']
        if 'dicom_metadata' in image and '0010|1010' in image['dicom_metadata'] and image['dicom_metadata']['0010|1010'] != '':
            age = image['dicom_metadata']['0010|1010']
        if '0010|1010' in image and image['0010|1010'] != '':
            age = image['0010|1010']
        if 'PatientAge' in image:
            age = image['PatientAge']
        if age is None:
            raise self.UnlabeledVertebra()
        if isinstance(age, str):
            if age.endswith('Y'):
                age = age[:-len('Y')]
            if age == 'Anonymized':
                raise self.UnlabeledVertebra()
        age = int(age)
        if age > self.UPPER_AGE_BOUND and age % 10 == 0:
            age /= 10  # probably a typo
        return numpy.array([age])


class BMIRegressionTask(FNetRegressionTask):
    DCM_TAG = '0010|1022'
    MROS_TAG = 'HWBMI'

    def output_layer_name(self) -> str:
        return 'bmi'

    def default_label(self) -> numpy.ndarray:
        return numpy.array([27.378324])  # from MrOs: V1FEB23_DISTRIBUTIONS.PDF, page 451, HWBMI mean

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        bmi: Optional[Union[str, float]] = None
        if self.MROS_TAG in image and not math.isnan(image[self.MROS_TAG]):
            bmi = image[self.MROS_TAG]
        if 'dicom_metadata' in image and self.DCM_TAG in image['dicom_metadata'] and image['dicom_metadata'][self.DCM_TAG] != '':
            bmi = image['dicom_metadata'][self.DCM_TAG]
        if self.DCM_TAG in image and image[self.DCM_TAG] != '':
            bmi = image[self.DCM_TAG]
        if bmi == 'Anonymized' or bmi is None:
            raise self.UnlabeledVertebra()
        return numpy.array([bmi])


class TotalBMDRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'TotBMD'

    def default_label(self) -> numpy.ndarray:
        return numpy.array([100.])

    def output_layer_name(self) -> str:
        return 'tbmd'


class TimeToFirstSpineFractureRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'FASPNFV1'

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TO DO')


class IncidentSpineFractureClassification(BinaryClassificationTask, KeyValueBasedTask):
    def class_names(self) -> List[str]:
        return ['No incident spine fracture', 'Incident spine fracture']

    def key(self) -> str:
        return 'FAANYSPN'


class TimeToFirstHipFractureRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'FAHIPFV1'

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TO DO')


class IncidentHipFractureClassification(BinaryClassificationTask, KeyValueBasedTask):
    def class_names(self) -> List[str]:
        return ['No incident hip fracture', 'Incident hip fracture']

    def key(self) -> str:
        return 'FAANYHIP'


class TimeToFirstFractureRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'FAFXFV1'

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TO DO')


class IncidentFractureClassification(BinaryClassificationTask, KeyValueBasedTask):
    def class_names(self) -> List[str]:
        return ['No incident fracture', 'Incident fracture']

    def key(self) -> str:
        return 'FAANYFX'


class TimeToFirstHipOrSpineFractureRegression(FNetRegressionTask):
    OUTPUT_UNITS = 1

    def output_layer_name(self) -> str:
        return 'FASPNHIPFV1'

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TO DO')

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        earliest = math.inf
        for key in ['FASPNFV1', 'FAHIPFV1']:
            if key not in image and key not in image.parts[vertebra_name]:
                raise self.UnlabeledVertebra()
            if key in image.parts[vertebra_name]:
                earliest = min(earliest, image.parts[vertebra_name][key])
            if key in image:
                earliest = min(earliest, image[key])
        assert math.isfinite(earliest)
        return numpy.array([earliest])


class IncidentHipOrSpineFractureClassification(BinaryClassificationTask):
    def output_layer_name(self) -> str:
        return 'FAANYSPNHIP'

    def class_names(self) -> List[str]:
        return ['No incident hip or spine fracture', 'Incident hip or spine fracture']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        for key in ['FAANYSPN', 'FAANYHIP']:
            if key not in image and key not in image.parts[vertebra_name]:
                raise self.UnlabeledVertebra()
            if key in image.parts[vertebra_name] and image.parts[vertebra_name][key]:
                return numpy.array([1.0])
            if key in image and image[key]:
                return numpy.array([1.0])
        return numpy.array([0.0])

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('TO DO')


class TrabPeeledBMDRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'TrabPeeledBMD'

    def default_label(self) -> numpy.ndarray:
        return numpy.array([80.])

    def output_layer_name(self) -> str:
        return 'tpbmd'


class CorticalBMDRegression(FNetRegressionTask, KeyValueBasedTask):
    OUTPUT_UNITS = 1

    def key(self) -> str:
        return 'CortBMD'

    def default_label(self) -> numpy.ndarray:
        return numpy.array([200.])

    def output_layer_name(self) -> str:
        return 'cbmd'


class MulticlassVertebraTask(VertebraClassificationTask, ABC):
    def __init__(self, loss_weight: float):
        super().__init__(loss_weight)

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def metrics(self) -> list:
        return dl_backend.b().multi_task_metrics(self.num_classes())

    def one_hot_encode(self, class_idx: int):
        result = [0] * self.num_classes()
        result[class_idx] = 1
        return result

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        result = numpy.max(y_pred_or_y_true, axis=0, keepdims=True)

        p = numpy.zeros_like(result)
        for cls_idx in reversed(range(self.num_classes())):  # backwards from most severe to least sever
            p[..., cls_idx] = numpy.minimum(
                result[..., cls_idx],  # either copy the probability
                (1 - numpy.sum(p, axis=-1))  # or clip it to not get the sum above 1
            )
        p[p < 0] = 0
        p[p > 1] = 1
        return p

    def worst_metric_values(self) -> List[float]:
        return [math.inf, -math.inf, -math.inf, -math.inf] + [-math.inf] * self.num_classes()

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def weighted_metrics(self) -> list:
        return dl_backend.b().multi_task_weighted_metrics(self.num_classes())

    def worst_weighted_metric_values(self) -> List[float]:
        return [math.inf, -math.inf]

    def natural_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return [f'{o} Categorical Crossentropy', f'{o} Accuracy', f'{o} Macro-F1', f'{o} Mean ROC'] + [f'{o} ROC for {c}' for c in self.class_names()]

    def natural_weighted_metrics_names(self) -> List[str]:
        return [f'{self.output_layer_name()} Weighted Categorical Crossentropy', f'{self.output_layer_name()} Weighted Accuracy']

    def loss_function(self):
        return dl_backend.b().categorical_crossentropy_loss()

    def nearest_class_idx_for_model_output(self, y_pred):
        return numpy.argmax(y_pred, axis=-1)

    def class_probabilities(self, y_pred) -> numpy.ndarray:
        return y_pred

    def build_output_layers_tf(self, x: 'tensorflow.Tensor', **regularizers) -> 'tensorflow.Tensor':
        input_shape = x.shape.as_list()
        conv_layer_before = len(input_shape) == 5
        dense_layer_before = len(input_shape) == 2
        if conv_layer_before:
            return Conv3D(filters=self.num_classes(), kernel_size=x.shape.as_list()[1:4], activation='softmax', kernel_initializer='zeros',
                          name=self.output_layer_name(), **regularizers)(x)
        elif dense_layer_before:
            return Dense(units=self.num_classes(), activation='softmax', kernel_initializer='zeros', name=self.output_layer_name(), **regularizers)(x)
        else:
            raise ValueError()

    def build_output_layer_torch(self, input_shape: Tuple[int, ...]) -> 'torch.nn.Module':
        linear = torch.nn.Linear(numpy.prod(input_shape[0]).item(), self.num_classes())
        torch.nn.init.zeros_(linear.weight)
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            linear,
            torch.nn.Softmax(dim=-1)
        )

    def split_to_pairwise_tasks(self, loss_weight=1.):
        class_indices = set(range(len(self.class_names())))
        for task_idx, class_1 in enumerate(powerset(class_indices)):
            if len(class_1) == 0:
                continue
            class_2 = class_indices.difference(class_1)
            if len(class_2) == 0:
                continue
            if min(class_2) < min(class_1):
                continue
            yield PairWiseClassificationTask(f'pw_{task_idx}', base_task=self, positive_class_indices=class_2, loss_weight=loss_weight)


class PairWiseClassificationTask(BinaryClassificationTask):
    def __init__(self, output_name: str, base_task: VertebraClassificationTask, loss_weight: float, positive_class_indices: Iterable[int]):
        super().__init__(loss_weight)
        self.positive_class_indices = positive_class_indices
        self.base_task = base_task
        self.output_name = output_name

    def output_layer_name(self) -> str:
        return self.output_name

    def class_names(self) -> List[str]:
        return [','.join(name for name_idx, name in enumerate(self.base_task.class_names()) if name_idx not in self.positive_class_indices),
                ','.join(name for name_idx, name in enumerate(self.base_task.class_names()) if name_idx in self.positive_class_indices), ]

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        cls_idx = self.base_task.class_idx_of_label(self.y_true_from_hiwi_image_and_vertebra(image, vertebra_name))
        return numpy.array([1. if cls_idx in self.positive_class_indices else 0.])


class GenantScoreClassification(MulticlassVertebraTask):
    def __init__(self, loss_weight: float, deformity_as_zero: bool):
        super().__init__(loss_weight)
        self.deformity_as_zero = deformity_as_zero

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def class_names(self) -> List[str]:
        return ["0", "1", "2", "3"]

    def output_layer_name(self) -> str:
        if self.deformity_as_zero:
            return f"gscdz"
        else:
            return "gsc"

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        return self.extract_genant_score_label(image, vertebra_name, self.deformity_as_zero)

    @classmethod
    def extract_genant_score_label(cls, image: hiwi.Image, vertebra_name, deformity_as_zero: bool):
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise cls.VertebraNotInTheImage()
        if 'Genant Score' not in image.parts[vertebra_name]:
            raise cls.UnlabeledVertebra()
        score = numpy.array([image.parts[vertebra_name]['Genant Score']]).item()
        if numpy.isnan(score).any():
            raise cls.UnlabeledVertebra()
        if deformity_as_zero:
            try:
                binary_score = BinaryOsteoporoticFractureClassification.osteoporosis_label(image, vertebra_name)
            except VertebraTask.UnlabeledVertebra:
                pass
            else:
                if binary_score.item() == 0:
                    return numpy.array([1, 0, 0, 0])
        if score == 0:
            return numpy.array([1, 0, 0, 0])
        elif score == 1:
            return numpy.array([0, 1, 0, 0])
        elif score == 2:
            return numpy.array([0, 0, 1, 0])
        elif score == 3:
            return numpy.array([0, 0, 0, 1])
        raise ValueError

    @classmethod
    def extract_genant_score_as_number(cls, image, vertebra_name, deformity_as_zero: bool):
        one_hot = cls.extract_genant_score_label(image, vertebra_name, deformity_as_zero)
        score = numpy.sum(one_hot * [0, 1, 2, 3], axis=-1, keepdims=True)
        assert len(score.shape) == 1
        assert score.shape[-1] == 1
        return score

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        above_1 = numpy.sum(y_pred_or_y_true * [0, 1, 2, 3], axis=(0, 1)) >= 2
        result = numpy.array([[1., 0, 0, 0], [0, 0, 1., 0]])[[above_1.astype('int')]]
        return result


class BinaryGenantScoreClassification(BinaryClassificationTask):
    def __init__(self, loss_weight: float, negative: List[int], positive: List[int], deformity_as_zero: bool):
        if not set(negative).isdisjoint(positive):
            raise ValueError
        if not set(negative).issubset({0, 1, 2, 3}):
            raise ValueError
        if not set(positive).issubset({0, 1, 2, 3}):
            raise ValueError
        if set(positive).union(negative) != {0, 1, 2, 3}:
            raise ValueError
        if not set(positive).isdisjoint(negative):
            raise ValueError

        super().__init__(loss_weight)
        self.negative = negative
        self.positive = positive
        self.deformity_as_zero = deformity_as_zero

    def output_layer_name(self) -> str:
        n = self.short_negative_class_names()
        p = self.short_positive_class_names()
        z = 'dz' if self.deformity_as_zero else ''
        return f'gs{z}{n}v{p}'

    def in_tasks(self, tasks: VertebraTasks):
        return any(isinstance(t, BinaryGenantScoreClassification) and t.negative == self.negative and t.positive == self.positive for t in tasks)

    def short_positive_class_names(self):
        return self.short_class_names_by_ids(self.positive)

    def short_negative_class_names(self):
        return self.short_class_names_by_ids(self.negative)

    @staticmethod
    def short_class_names_by_ids(class_ids: List[int]):
        return ''.join(str(s) for s in sorted(class_ids))

    def all_old_fractures_are_positive(self):
        # old fractures are osteoporotic and osteoporotic fractures have grade > 0
        old_fracture_grades = {1, 2, 3}
        return old_fracture_grades.issubset(self.positive)

    def all_non_fractures_are_negative(self):
        # old fractures are osteoporotic and osteoporotic fractures have grade > 0
        non_fracture_grades = {0, 1}
        return non_fracture_grades.issubset(self.negative)

    def one_hot_positives(self):
        as_list = []
        for i in range(4):
            if i in self.positive:
                as_list.append(1)
            elif i in self.negative:
                as_list.append(0)
            else:
                raise ValueError(i)
        return numpy.array(as_list)

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        try:
            one_hot = GenantScoreClassification.extract_genant_score_label(image, vertebra_name, self.deformity_as_zero)
        except self.UnlabeledVertebra:
            if self.all_old_fractures_are_positive():
                v = self.load_trauma_vertebra(image, vertebra_name)
                if v.fresh_status() is FreshStatus.OLD:
                    return numpy.array([1.])
            elif self.all_non_fractures_are_negative():
                v = self.load_trauma_vertebra(image, vertebra_name)
                if v.fresh_status() is FreshStatus.NON_FRACTURED:
                    return numpy.array([0.])
            raise
        return numpy.array([numpy.sum(one_hot * self.one_hot_positives())])

    def class_names(self) -> List[str]:
        return [
            '[' + self.short_negative_class_names() + ']',
            '[' + self.short_positive_class_names() + ']',
        ]

    @classmethod
    def possible_class_names(cls):
        result = []
        for cutoff in range(1, 4):
            negative = cls.short_class_names_by_ids(list(range(cutoff)))
            result.append(negative)
            positive = cls.short_class_names_by_ids(list(range(cutoff, 4)))
            result.append(positive)
        return result


class ArtificialTaskNotBuildable(RuntimeError):
    pass


class OutputCombination(VertebraTask):
    def output_layer_name(self) -> str:
        raise NotImplementedError('Abstract method')

    def default_label(self) -> numpy.ndarray:
        raise NotImplementedError('Abstract method')

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        raise NotImplementedError('Abstract method')

    def metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    def weighted_metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    def natural_metrics_names(self) -> List[str]:
        raise NotImplementedError('Abstract method')

    def natural_weighted_metrics_names(self) -> List[str]:
        raise NotImplementedError('Abstract method')

    def loss_function(self):
        raise NotImplementedError('Abstract method')

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError('Abstract method')

    def worst_metric_values(self) -> List[float]:
        raise NotImplementedError('Abstract method')

    def worst_weighted_metric_values(self) -> List[float]:
        raise NotImplementedError('Abstract method')

    def input_task_names(self) -> List[str]:
        raise NotImplementedError('Abstract method')

    def build_output_layer(self, inputs):
        raise NotImplementedError('Abstract method')

    def build_output_layers_tf(self, x: 'tensorflow.Tensor', **regularizers) -> 'tensorflow.Tensor':
        raise ArtificialTaskNotBuildable('This is an artificial task and cannot be trained on.')

    def build_output_layer_torch(self, input_shape: Tuple[int, ...]) -> 'torch.nn.Module':
        raise ArtificialTaskNotBuildable('This is an artificial task and cannot be trained on.')

    def smooth_label(self, non_smoothed_label, smoothing_alpha: float):
        raise NotImplementedError('Abstract method')


class BinaryOutputCombination(MulticlassVertebraTask, OutputCombination):
    class CombinedLayer(Layer):
        def __init__(self, combination, clauses, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.combination = combination
            self.clauses = clauses

        def call(self, inputs, *args, **kwargs):
            outputs_for_cls = []
            for cls_idx in range(self.combination.num_classes()):
                to_stack = []
                for tsk_idx in self.clauses[cls_idx]:
                    if self.combination.thresholds_for_binary_outputs is None:
                        threshold = 0.5
                    else:
                        task_name = self.combination.binary_tasks[tsk_idx].output_layer_name()

                        threshold = self.combination.thresholds_for_binary_outputs[task_name]
                        if threshold is None:
                            threshold = 0.5
                    to_stack.append((inputs[tsk_idx] >= threshold) == self.clauses[cls_idx][tsk_idx])

                stacked = tensorflow.keras.backend.stack(to_stack, axis=-1)
                outputs_for_cls.append(tensorflow.reduce_all(stacked, axis=-1))
            output = tensorflow.keras.backend.concatenate(outputs_for_cls, axis=-1)
            # assert_sum_to_1 = tensorflow.debugging.assert_near(1., tensorflow.reduce_sum(tensorflow.cast(output, tensorflow.keras.backend.floatx()), axis=-1))
            # with tensorflow.control_dependencies([assert_sum_to_1]):
            output = tensorflow.keras.backend.cast(output, tensorflow.keras.backend.floatx())
            return output

        def get_config(self):
            config = {attr: getattr(self, attr) for attr in ['combination', 'clauses']}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    def output_layer_name(self) -> str:
        return self.name

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        return self.y_true_from.y_true_from_hiwi_image_and_vertebra(image, vertebra_name)

    def class_labels(self) -> numpy.ndarray:
        return numpy.eye(self.num_classes())

    def __init__(self, binary_tasks: List[BinaryClassificationTask],
                 classes: Dict[str, List[str]], name: str,
                 y_true_from: MulticlassVertebraTask,
                 add_brackets_to_classes=False,
                 thresholds_for_binary_outputs: Dict[str, float] = None):
        super().__init__(loss_weight=1)
        self.thresholds_for_binary_outputs = thresholds_for_binary_outputs
        self.binary_tasks = binary_tasks
        if add_brackets_to_classes:
            for c in classes:
                classes[c] = [f'[{c2}]' for c2 in classes[c]]
        self.classes = classes
        self.name = name
        self.y_true_from = y_true_from

    def class_names(self) -> List[str]:
        return list(self.classes)

    def input_task_names(self):
        return VertebraTasks(self.binary_tasks).output_layer_names()

    def build_output_layer(self, inputs):
        combination = self

        clauses: Dict[int, List[int]] = {
            idx: {}
            for idx in range(combination.num_classes())
        }
        for task_idx in range(len(combination.binary_tasks)):
            for cls_idx in range(combination.num_classes()):
                class_name = combination.class_names()[cls_idx]
                if combination.binary_tasks[task_idx].class_names()[1] in combination.classes[class_name]:
                    clauses[cls_idx][task_idx] = True
                elif combination.binary_tasks[task_idx].class_names()[0] in combination.classes[class_name]:
                    clauses[cls_idx][task_idx] = False
                else:
                    pass
        for c in clauses.values():
            assert len(c) > 0, clauses

        if self.thresholds_for_binary_outputs is not None:
            for task in self.binary_tasks:
                assert task.output_layer_name() in self.thresholds_for_binary_outputs
        return self.CombinedLayer(name=self.output_layer_name(), combination=combination, clauses=clauses)(inputs)

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        above_1 = numpy.sum(y_pred_or_y_true * [0, 1, 2, 3], axis=(0, 1)) >= 2
        result = numpy.array([[1., 0, 0, 0], [0, 0, 1., 0]])[[above_1.astype('int')]]
        return result


class GenantScoreClassificationWithCustomLoss(GenantScoreClassification):
    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def metrics(self) -> list:
        return super().metrics() + dl_backend.b().genant_score_custom_loss_metrics()

    def worst_metric_values(self) -> List[float]:
        return super().metrics + [math.inf]

    def natural_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return super().natural_metrics_names() + [
            f'{o} CUSTOM',
        ]

    def loss_function(self):
        return dl_backend.b().genant_score_custom_loss()

    def output_layer_name(self) -> str:
        if self.deformity_as_zero:
            return f"cgscdz"
        else:
            return "cgsc"


class DeformityRegression(FNetRegressionTask, GenantScoreClassification):
    OUTPUT_UNITS = 3
    deformity_as_zero = False  # for backwards compatibility, in case this was not set at training time

    def nearest_class_idx_for_model_output(self, y_pred):
        if isinstance(y_pred, list):
            y_pred = numpy.array(y_pred)
        return (
                numpy.any(y_pred > 0.4, axis=-1).astype('int')
                + numpy.any(y_pred > 0.25, axis=-1).astype('int')
                + numpy.any(y_pred > 0.2, axis=-1).astype('int')
        )

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([[0 for _ in range(self.OUTPUT_UNITS)],
                            [0.225 for _ in range(self.OUTPUT_UNITS)],
                            [0.325 for _ in range(self.OUTPUT_UNITS)],
                            [0.45 for _ in range(self.OUTPUT_UNITS)]])

    def default_label(self) -> numpy.ndarray:
        return numpy.array([0. for _ in self.deformity_label_names()])

    def output_layer_name(self) -> str:
        if self.deformity_as_zero:
            return 'dfodz'
        else:
            return 'dfo'

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        metadata = image.parts[vertebra_name]
        try:
            return numpy.array([float(metadata[label])
                                for label in self.deformity_label_names()])
        except KeyError:
            raise self.UnlabeledVertebra

    def class_idx_of_label(self, label):
        return self.nearest_class_idx_for_model_output(label)

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        score = GenantScoreRegression.patient_level_aggregation(self, self.nearest_class_idx_for_model_output(y_pred_or_y_true)[..., numpy.newaxis])
        label = self.class_labels()[score.item()][numpy.newaxis, :]
        return label

    @staticmethod
    def deformity_label_names():
        return ['Deformity Wedge',
                'Deformity Biconcave',
                'Deformity Crush']

    def class_probabilities(self, y_pred) -> numpy.ndarray:
        cls_idx = self.nearest_class_idx_for_model_output(y_pred)
        return numpy.eye(4)[cls_idx]


class GenantScoreRegression(FNetRegressionTask, GenantScoreClassification):
    OUTPUT_UNITS = 1

    def nearest_class_idx_for_model_output(self, y_pred):
        rounded_y = numpy.round(y_pred)  # round all predictions of current batch to the nearest whole number

        rounded_y = numpy.maximum(0, rounded_y)  # if one prediction is less than zero, set it to zero
        rounded_y = numpy.minimum(3, rounded_y)  # if one prediction is more than three, set it to three

        return rounded_y.astype('int')

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def metrics(self) -> list:
        return super().metrics() + dl_backend.b().genant_score_custom_loss_metrics()

    def worst_metric_values(self) -> List[float]:
        return super().worst_weighted_metric_values() + [math.inf]

    @cachetools.cached(LRUCache(maxsize=50), key=lambda self: self.identifier())
    def weighted_metrics(self) -> list:
        return super().weighted_metrics() + dl_backend.b().weighted_genant_score_regression_accuracy_metrics()

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        above_1 = numpy.sum(y_pred_or_y_true, axis=(0, 1)) >= 2
        result = numpy.array([[[0]], [[2]]])[above_1.astype('int')]
        return result

    def worst_weighted_metric_values(self) -> List[float]:
        return super().worst_weighted_metric_values() + [-math.inf]

    def natural_weighted_metrics_names(self) -> List[str]:
        return super().natural_weighted_metrics_names() + [
            f'{self.output_layer_name()} Weighted Accuracy',
        ]

    def natural_metrics_names(self) -> List[str]:
        o = self.output_layer_name()
        return super().natural_metrics_names() + [
            f'{o} GSR_ACC',
        ]

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([[0], [1], [2], [3]])

    def class_idx_of_label(self, label):
        if isinstance(label, list):
            assert len(label) == 1
            if label[0] != int(label[0]):
                raise self.LabelWithoutClass()
            return int(label[0])
        return int(label)

    def output_layer_name(self) -> str:
        if self.deformity_as_zero:
            return f"gsdz"
        else:
            return "gs"

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        score = self.extract_genant_score_as_number(image, vertebra_name, self.deformity_as_zero)
        return score

    def class_probabilities(self, y_pred) -> numpy.ndarray:
        cls_idx = self.nearest_class_idx_for_model_output(y_pred)[..., 0]
        return numpy.eye(4)[cls_idx]


class SpondylosisClassification(MulticlassVertebraTask):
    def output_layer_name(self) -> str:
        return 'ddx'

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([[1, 0], [0, 1]])

    def class_names(self) -> List[str]:
        return ['No Spondylosis', 'Spondylosis']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        if 'Differential Diagnosis' in image.parts[vertebra_name]:
            if image.parts[vertebra_name]['Differential Diagnosis'] == 'Spondylosis':
                return numpy.array([0, 1])
            elif image.parts[vertebra_name]['Differential Diagnosis'] != 'Spondylosis':
                return numpy.array([1, 0])
            else:
                raise ValueError()
        v = self.load_trauma_vertebra(image, vertebra_name)
        if v.body_is_fractured():
            return numpy.array([0, 1])
        raise self.UnlabeledVertebra


class CombinedDifferentialDiagnosisCategoryAndGenantScoreClassification(MulticlassVertebraTask):
    def output_layer_name(self) -> str:
        return 'ddxgs'

    def class_names(self) -> List[str]:
        return ['Normal G0', 'Deformity G0', 'Deformity G1', 'OF G1', 'OF G2', 'OF G3']

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        if 'Genant Score' not in image.parts[vertebra_name]:
            raise self.UnlabeledVertebra()
        score = numpy.array([image.parts[vertebra_name]['Genant Score']])
        if 'Differential Diagnosis Category' not in image.parts[vertebra_name]:
            raise self.UnlabeledVertebra()
        ddx_cat = image.parts[vertebra_name]['Differential Diagnosis Category']
        if ddx_cat == 'Unevaluable':
            raise self.UnlabeledVertebra()
        if numpy.isnan(score).any():
            raise self.UnlabeledVertebra()
        if ddx_cat == 'Normal' and score == 0:
            return numpy.array([1, 0, 0, 0, 0, 0])
        if ddx_cat == 'Deformity' and score == 0:
            return numpy.array([0, 1, 0, 0, 0, 0])
        if ddx_cat == 'Deformity' and score == 1:
            return numpy.array([0, 0, 1, 0, 0, 0])
        if ddx_cat == 'Osteoporotic Fracture' and score == 1:
            return numpy.array([0, 0, 0, 1, 0, 0])
        if ddx_cat == 'Osteoporotic Fracture' and score == 2:
            return numpy.array([0, 0, 0, 0, 1, 0])
        if ddx_cat == 'Osteoporotic Fracture' and score == 3:
            return numpy.array([0, 0, 0, 0, 0, 1])
        raise ValueError((ddx_cat, score, vertebra_name, image['patient_id']))

    def patient_level_aggregation(self, y_pred_or_y_true: numpy.ndarray) -> numpy.ndarray:
        above_1 = numpy.sum(y_pred_or_y_true * [0, 0, 0, 1, 2, 3], axis=(0, 1)) >= 2
        result = numpy.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])[[above_1.astype('int')]]
        return result


class DifferentialDiagnosisCategoryClassification(MulticlassVertebraTask):
    def output_layer_name(self) -> str:
        return 'ddx_cat'

    def class_labels(self) -> numpy.ndarray:
        return numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def class_names(self) -> List[str]:
        return ['Normal', 'Deformity', 'Osteoporotic Fracture']

    def y_true_from_hiwi_image_and_vertebra(self, image: hiwi.Image, vertebra_name) -> numpy.ndarray:
        if 'not_in_the_image' in image.parts[vertebra_name] and image.parts[vertebra_name]['not_in_the_image']:
            raise self.VertebraNotInTheImage()
        if 'Differential Diagnosis Category' in image.parts[vertebra_name]:
            if image.parts[vertebra_name]['Differential Diagnosis Category'] == 'Normal':
                return numpy.array([1, 0, 0])
            if image.parts[vertebra_name]['Differential Diagnosis Category'] == 'Deformity':
                return numpy.array([0, 1, 0])
            if image.parts[vertebra_name]['Differential Diagnosis Category'] == 'Osteoporotic Fracture':
                return numpy.array([0, 0, 1])
            if image.parts[vertebra_name]['Differential Diagnosis Category'] == 'Unevaluable':
                raise self.UnlabeledVertebra()
            raise ValueError
        v = self.load_trauma_vertebra(image, vertebra_name)
        if v.fresh_status() == FreshStatus.OLD:
            # all old vertebrae are osteoporotic
            return numpy.array([0, 0, 1])
        raise self.UnlabeledVertebra


class GenantByDeformity(GenantScoreRegression, OutputCombination):
    class DeformityToGenantScore(Layer):
        def call(self, inputs, *args, training=False, **kwargs):
            if len(inputs) != 1:
                raise ValueError(inputs)
            deformity = inputs[0]
            above_thresholds = [
                tensorflow.keras.backend.max(deformity, axis=-1) >= 0.2,
                tensorflow.keras.backend.max(deformity, axis=-1) >= 0.25,
                tensorflow.keras.backend.max(deformity, axis=-1) >= 0.4,
            ]
            above_thresholds = tensorflow.keras.backend.cast(above_thresholds, tensorflow.keras.backend.floatx())
            along_thresholds = 0
            output = tensorflow.keras.backend.sum(above_thresholds, axis=along_thresholds)
            output = output[..., tensorflow.newaxis]
            return output

    def __init__(self, deformity_as_zero: bool):
        super().__init__(1.0, deformity_as_zero)

    def output_layer_name(self) -> str:
        if self.deformity_as_zero:
            return f"gbddz"
        else:
            return "gbd"

    def input_task_names(self):
        return [DeformityRegression(0.01, deformity_as_zero=self.deformity_as_zero).output_layer_name()]

    def build_output_layer(self, inputs):
        if len(inputs) != 1:
            raise ValueError(inputs)

        return self.DeformityToGenantScore(name=self.output_layer_name())(inputs)
