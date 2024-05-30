import itertools
import os
import random
from abc import abstractmethod
from collections import OrderedDict
from math import nan, inf
from typing import List, Dict, Tuple, Any, Optional

import cachetools
import numpy
from lifelines.utils import concordance_index

from lib.classification_report import plot_confusion_matrix_with_class_names
from lib.my_logger import logging
from lib.profiling_tools import dump_pstats_if_profiling
from lib.progress_bar import ProgressBar
from lib.util import EBC
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, results_cache
from tasks import VertebraTasks, VertebraClassificationTask, VertebraTask, BinaryClassificationTask

ModelName = str


class ModelOutput(EBC):
    def __init__(self, model_name: str, y_preds, names, tasks: VertebraTasks, classification_thresholds: Dict[str, float]):
        self.classification_thresholds = classification_thresholds
        self.tasks = tasks
        self.model_name = model_name
        self.y_preds = y_preds
        self.names = names
        self._cached_outputs_per_vertebra = None

    def get_outputs_per_vertebra(self) -> Dict[str, numpy.ndarray]:
        if self._cached_outputs_per_vertebra is None:
            self._cached_outputs_per_vertebra = {
                n: [self.y_preds[self.tasks.index(task)][self.names.index(n)].astype('float16')
                    for task in self.tasks]
                for n in self.names
            }
        return self._cached_outputs_per_vertebra

    def clear_cached_outputs_per_vertebra(self):
        self._cached_outputs_per_vertebra = None


class ModelsOutputs(ModelOutput):
    def __init__(self, raw_outputs: List[ModelOutput], model_name):
        self.model_names = [o.model_name for o in raw_outputs]
        self.tasks = self.all_tasks(raw_outputs)
        outputs_per_vertebra = self.compute_outputs_per_vertebra(raw_outputs)
        names = sorted(outputs_per_vertebra)
        y_preds = [
            numpy.array([outputs_per_vertebra[n][self.tasks.index(task)].astype('float16') for n in names])
            for task in self.tasks
        ]
        thresholds = {
            task_name: numpy.mean([o.classification_thresholds[task_name] for o in raw_outputs])
            for task_name in self.tasks.output_layer_names()
        }
        super().__init__(model_name, y_preds, names, self.tasks,
                         classification_thresholds=thresholds)

    def compute_outputs_per_vertebra(self, raw_outputs: List[ModelOutput]) -> Dict[str, numpy.ndarray]:
        all_vertebrae = set(n for o in raw_outputs for n in o.names)
        for v in all_vertebrae:
            for o in raw_outputs:
                assert sum(
                    1 for n in o.names if n == v) <= 1, f'One model or model group contains multiple predictions for a single vertebra: {o.model_name}, {v}'
        non_averaged = {n: {task.output_layer_name(): [] for task in self.tasks} for n in all_vertebrae}
        for o in raw_outputs:
            for n_idx, n in enumerate(o.names):
                for task_idx, task in enumerate(o.tasks):
                    non_averaged[n][task.output_layer_name()].append(o.y_preds[task_idx][n_idx])
        return {
            n: [numpy.mean(non_averaged[n][task.output_layer_name()],
                           # [o.y_preds[o.tasks.index(task)][o.names.index(n)]
                           #             for o in raw_outputs
                           #             if n in o.names
                           #             if task in o.tasks],
                           axis=0)
                for task in self.tasks]
            for n in all_vertebrae
        }

    @staticmethod
    def all_tasks(raw_outputs):
        tasks = VertebraTasks()
        for model_output in raw_outputs:
            for task in model_output.tasks:
                if task not in tasks:
                    tasks.append(task)
        return tasks


class NotApplicable(RuntimeError):
    pass


class CompareOutputs(ModelAnalyzer):
    class Metric(EBC):
        def __init__(self,
                     task: VertebraTask, ):
            self.task = task

        @abstractmethod
        def compare(self, output_1: ModelOutput, output_2: ModelOutput) -> float:
            raise NotImplementedError()

    class HarrellsC(Metric):
        def __init__(self,
                     task: VertebraTask, ):
            super().__init__(task)
            if not isinstance(self.task, BinaryClassificationTask):
                raise NotApplicable()

        def compare(self, output_1: ModelOutput, output_2: ModelOutput) -> float:
            if output_1 is output_2:
                return 1.
            if self.task not in output_1.tasks or self.task not in output_2.tasks:
                return nan
            names_1 = output_1.names
            names_2 = output_2.names

            if set(names_1).isdisjoint(names_2):
                return nan
            y_pred1 = []
            y_pred2 = []
            for n in names_1:
                if n in names_2:
                    p1 = output_1.get_outputs_per_vertebra()[n][output_1.tasks.index(self.task)].item()
                    p2 = output_2.get_outputs_per_vertebra()[n][output_2.tasks.index(self.task)].item()
                    y_pred1.append(p1)
                    y_pred2.append(p2)
            if numpy.unique(y_pred1).size <= 1 or numpy.unique(y_pred2).size <= 1:
                return nan
            return concordance_index(y_pred1, y_pred2)

    class Agreement(Metric):
        def __init__(self,
                     task: VertebraTask):
            super().__init__(task)
            if not isinstance(self.task, VertebraClassificationTask):
                raise NotApplicable()

        def compare(self, output_1: ModelOutput, output_2: ModelOutput) -> float:
            if self.task not in output_1.tasks or self.task not in output_2.tasks:
                return nan
            if set(output_1.names).isdisjoint(output_2.names):
                return nan
            agreements, disagreements = self.lists(output_1, output_2)
            return len(agreements) / (len(agreements) + len(disagreements))

        def _lists_cache_key(self, output_1: ModelOutput, output_2: ModelOutput):
            return (id(output_1), id(output_2), self.task.output_layer_name(), output_1.model_name, output_2.model_name,)

        _lists_cache = cachetools.LRUCache(maxsize=20000)

        @cachetools.cached(_lists_cache, key=lambda self, output_1, output_2: self._lists_cache_key(output_1, output_2))
        def lists(self, output_1: ModelOutput, output_2: ModelOutput) -> Tuple[list, list]:
            if output_1 is output_2:
                return output_1.names, []
            agreed = []
            disagreed = []
            output_2_vertebrae = output_2.get_outputs_per_vertebra()
            for n in output_1.names:
                if n in output_2_vertebrae:
                    if self.cls_idx(n, output_1) == self.cls_idx(n, output_2):
                        agreed.append(n)
                    else:
                        disagreed.append(n)
            return agreed, disagreed

        def _cls_cache_key(self, name, output: ModelOutput):
            return (
                name,
                id(output),
                self.task.output_layer_name(),
                output.model_name,
                tuple(output.classification_thresholds.values())
            )

        _cls_cache = cachetools.LRUCache(maxsize=30000)

        @cachetools.cached(_cls_cache, key=lambda self, name, output: self._cls_cache_key(name, output))
        def cls_idx(self, name, output:ModelOutput) -> int:
            y_pred_1 = output.get_outputs_per_vertebra()[name][output.tasks.index(self.task)]
            cls_1 = ModelAnalyzer.class_idx_for_model_output_using_threshold(
                model_output=y_pred_1,
                task=self.task,
                threshold=output.classification_thresholds[self.task.output_layer_name()]
            )
            return cls_1

    def __init__(self,
                 evaluator: FNetParameterEvaluator,
                 skip_existing=None):
        super().__init__(evaluator=evaluator, model_level_caching=True, analysis_needs_xs=False, skip_existing=skip_existing)
        self.model_outputs = None

    def to_subdir(self):
        return ''

    def after_model_directory(self, results: List[ModelOutput]):
        logging.info(f'Processing model outputs for {self.model_dir} ...')
        output = ModelsOutputs(results, model_name=os.path.basename(self.model_dir))
        self.serializer().create_directory_if_not_exists(self.to_dir())
        tasks = output.tasks
        if self.random_task_order():
            print('  Processing tasks in random order.')
            tasks = tasks.copy()
            random.shuffle(tasks)
        for task in tasks:
            # logging.info(f'  Processing outputs at {task.output_layer_name()}')
            results_by_model = {o.model_name: o for o in results}
            self.plot_comparison_metrics_as_matrices(results_by_model, task)
            self.store_agreements_as_json(results_by_model, task)
            dump_pstats_if_profiling(ModelAnalyzer)
        return super().after_model_directory(output)

    def plot_comparison_metrics_as_matrices(self, results_by_model, task):
        models = sorted(results_by_model)
        comparison_metrics = self.metrics(task)
        for metric_name, comparison_metric in comparison_metrics.items():
            basename = os.path.join(self.to_dir(), f'{task.output_layer_name()}_{metric_name}')
            to_files = [
                basename + '.png',
                basename + '.svg',
            ]
            if self.skip_existing and all(self.serializer().isfile(to_file) for to_file in to_files):
                continue

            results_matrix = numpy.full((len(results_by_model), len(results_by_model)), dtype='float32', fill_value=nan)
            for model_1, model_2 in ProgressBar(list(itertools.product(models, models)), suffix=f'Comparing {task.output_layer_name()}_{metric_name}...', ):
                output_1 = results_by_model[model_1]
                output_2 = results_by_model[model_2]
                results_matrix[models.index(model_1)][models.index(model_2)] = comparison_metric.compare(output_1, output_2)
            print(f'Writing {basename}.*')

            for to_file in to_files:
                # this is not a confusion matrix but looks the same, so we can use the same method
                plot_confusion_matrix_with_class_names(
                    results_matrix,
                    normalize=False,
                    float_format=True,
                    class_names=models,
                    to_file=to_file,
                    remove_axis_labels=True,
                )

    def store_agreements_as_json(self, results_by_model: Dict[str, ModelOutput], task):
        models = sorted(results_by_model)
        to_file = os.path.join(self.to_dir(), f'{task.output_layer_name()}_agreement.json')
        if self.skip_existing and self.serializer().isfile(to_file):
            return
        try:
            a = self.Agreement(task, classification_threshold=self.classification_thresholds[self.model_path])
        except NotApplicable:
            return
        agreement_counts_by_vertebra = {model_1: {} for model_1 in models}
        results_dict: Dict[str, Any] = {
            'agreement_counts_by_vertebra': agreement_counts_by_vertebra,
        }
        for model_1 in ProgressBar(models, suffix='Counting agreements'):
            if task not in results_by_model[model_1].tasks:
                continue
            agreement_counts = {}
            disagreement_counts = {}
            for vertebra in results_by_model[model_1].names:
                # start counting agreements at -1 because we want to ignore the always-present self-agreement
                agreement_counts[vertebra] = -1
                disagreement_counts[vertebra] = 0
            for model_2 in models:
                if task not in results_by_model[model_2].tasks:
                    continue
                agreements, disagreements = a.lists(output_1=results_by_model[model_1],
                                                    output_2=results_by_model[model_2])
                for v in agreements:
                    agreement_counts[v] += 1
                for v in disagreements:
                    disagreement_counts[v] += 1

            agreement_counts_by_vertebra[model_1] = OrderedDict()
            for v in sorted(results_by_model[model_1].names,
                            key=lambda d: -inf if agreement_counts[d] == 0 else agreement_counts[d] / (agreement_counts[d] + disagreement_counts[d])):
                agreement_counts_by_vertebra[model_1][v] = {'agreed': agreement_counts[v], 'disagreed': disagreement_counts[v]}

        self.serializer().save_json(to_file, results_dict)

    def metrics(self, task) -> Dict[str, Metric]:
        metric_types = {
            'harrells_c': self.HarrellsC,
            'agreement': self.Agreement,
        }
        results = {}

        for name, Metric in metric_types.items():
            try:
                results[name] = Metric(task)
            except NotApplicable:
                continue
        return results

    def model_level_cache_key(self, model_path:Optional[str] = None):
        return super().model_level_cache_key(model_path) + (len(self.metrics(None)), )

    def directory_level_cache_key(self, model_dir: str):
        return super().directory_level_cache_key(model_dir) + (len(self.metrics(None)), )

    def before_model(self, model_path: str, ):
        super().before_model(model_path)
        self.model_outputs: List[ModelOutput] = []

    def after_dataset(self) -> ModelOutput:
        super().after_dataset()
        assert len(self.model_outputs) == 1
        return self.model_outputs[0]

    def analyze_batch(self, batch, y_preds, names):
        self.model_outputs.append(ModelOutput(model_name=os.path.basename(self.model_path),
                                              tasks=self.tasks,
                                              y_preds=y_preds,
                                              names=names,
                                              classification_thresholds=self.classification_thresholds[self.model_path]))

    @results_cache.cache(ignore=['self'], verbose=0)
    def _cached_model_analysis(self, model_path, _cache_key):
        if self.serializer().isfile(model_path):
            result: ModelOutput = super()._cached_model_analysis.func(self, model_path=model_path, _cache_key=_cache_key)
        else:
            assert os.path.isdir(model_path)
            result: ModelOutput = self.analyze_directory(model_path)
        result.clear_cached_outputs_per_vertebra()
        return result

    @staticmethod
    def model_files(model_dir):
        return [os.path.join(model_dir, model_file)
                for model_file in os.listdir(model_dir)
                if model_file.endswith('.h5') or os.path.isdir(os.path.join(model_dir, model_file))]

    def analyze_subdirectories(self, model_dir) -> Dict[str, List[List[ModelOutput]]]:
        self.before_multiple_directories(model_dir)
        results = {}
        for dir_path, sub_dirs, files in os.walk(model_dir, topdown=True):
            if len(files) > 0 or len(sub_dirs) > 0:
                results[dir_path] = self.analyze_directory(dir_path)
        return self.after_multiple_directories(results)

    def random_task_order(self):
        return self.RANDOM_MODEL_ORDER
