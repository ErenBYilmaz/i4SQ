import os.path

import collections
import copy
import matplotlib.ticker
import numpy
import random
import scipy.special
import warnings
from math import inf, nan
from matplotlib import pyplot
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine.base_layer import Layer
from typing import List, Dict, Union, Optional, Any, Iterable, Tuple

from lib import dl_backend
from lib.callbacks import RecordBatchMetrics
from lib.my_logger import logging
from lib.parameter_search import mean_confidence_interval_size
from lib.progress_bar import ProgressBar
from lib.util import EBC, EBE
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator
from model.fnet.builder import FNetBuilder
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.fnet import FNet
from model.fnet.model_analysis.analyze_trained_models import results_cache, ModelAnalyzer, SingleModelEvaluationResult
from model.fnet.model_analysis.evaluation_result_serializer import EvaluationResultSerializer
from model.fnet.model_analysis.plot_augmentation_robustness import RandomFlipLRModifier, Modifier, ShiftModifier, RotationRangeModifier, CombinedModifier, \
    EmptyModifier, AugmentationRobustnessDrawer, ShiftModifier2D, RandomFlipUDModifier
from tasks import VertebraTask, VertebraClassificationTask

FileName = str
ConfidenceName = str
MetricValue = float
MetricValueList = List[Optional[float]]
MetricValueDict = Dict[ConfidenceName, MetricValueList]


def relative_distances_from_median(values):
    m = numpy.median(values)
    return [max((m - m * x) / (m - 1), 1 - x / m) for x in values]


def relative_distances_from_avg(values):
    m = numpy.mean(values)
    return [max((m - m * x) / (m - 1), 1 - x / m) for x in values]


class TTAAggregator(EBC):
    class NotApplicable(RuntimeError):
        pass

    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        pass


class AverageVoting(TTAAggregator):
    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        return augment_values.mean(axis=0)


class First(TTAAggregator):
    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        return augment_values[0]


class MedianVoting(TTAAggregator):
    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        return numpy.median(augment_values, axis=0)


class MajorityVoting(TTAAggregator):
    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        if not isinstance(task, VertebraClassificationTask):
            return augment_values.mean(axis=0)  # voting not possible for regression tasks, so just average
        class_indices = []
        for row in augment_values:
            class_indices.append(int(task.nearest_class_idx_for_model_output(row)))
        class_counts = collections.Counter(class_indices)
        return task.class_labels()[class_counts.most_common(1)[0][0]]


class StandardDeviation(TTAAggregator):
    def __init__(self, ddof=0):
        self.ddof = ddof

    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        if self.ddof >= len(augment_values):
            return numpy.inf  # numpy would have returned inf or nan anyways but this way we avoid a warning
        return numpy.std(augment_values, ddof=self.ddof, axis=0)


class LogitStandardDeviation(TTAAggregator):
    def __init__(self, ddof=0):
        self.ddof = ddof

    def aggregate(self, augment_values: numpy.ndarray, task: VertebraTask) -> numpy.ndarray:
        if self.ddof >= len(augment_values):
            return numpy.inf  # numpy would have returned inf or nan anyways but this way we avoid a warning
        logit = scipy.special.logit(augment_values)
        if numpy.isnan(logit).any():
            return numpy.nan
        if numpy.isinf(logit).any():
            return numpy.inf
        return numpy.std(logit, ddof=self.ddof, axis=0)


class MetricsContainer(EBC):
    def __init__(self, metrics: Dict[int, Dict[str, Dict[Any, Dict[str, Dict[str, List[float]]]]]] = None):
        """
        The ordering of keys in the metrics dict is:
        metrics[repetitions][modifier_name][modifier_value][aggregation_method][metric_name] = metrics_values
        """
        if metrics is None:
            metrics = {}
        self.metrics: Dict[int, Dict[str, Dict[Any, Dict[str, Dict[str, List[float]]]]]] = metrics

    def add_entry(self,
                  repetitions,
                  modifier_name,
                  modifier_value,
                  aggregation_method,
                  metrics: Dict[str, Union[float, List[float]]], ):
        """
        :param metrics: either a keras like dictionaly {metric -> result} or a dictionary {metric -> (result, std, conf, count)}
        """
        for metric_name in metrics:
            if isinstance(metrics[metric_name], float):
                metrics[metric_name] = [metrics[metric_name]]
            self.metrics. \
                setdefault(repetitions, {}). \
                setdefault(modifier_name, {}). \
                setdefault(modifier_value, {}). \
                setdefault(aggregation_method, {}). \
                setdefault(metric_name, []). \
                extend(metrics[metric_name])

    def metric_names(self):
        return list(set([metric_name
                         for repetitions in (self.metrics)
                         for modifier_name in self.metrics[repetitions]
                         for modifier_value in self.metrics[repetitions][modifier_name]
                         for aggregation_method in self.metrics[repetitions][modifier_name][modifier_value]
                         for metric_name in self.metrics[repetitions][modifier_name][modifier_value][aggregation_method]
                         ]))

    @classmethod
    def combine(cls, containers: Iterable['MetricsContainer']):
        combined = cls()
        for container in containers:
            for repetitions, modifier_name, modifier_value, aggregation_method, metric_name, values in container.walk():
                combined.add_entry(repetitions, modifier_name, modifier_value, aggregation_method, {metric_name: values})
        return combined

    def create_single_metrics_plot(self, aggregation_method, metric_name, repetitions):
        metrics = self.metrics
        xticks = []
        ys = []
        y_error = []
        for modifier_name in sorted(metrics[repetitions]):
            for modifier_value in sorted(metrics[repetitions][modifier_name]):
                y = metrics[repetitions][modifier_name][modifier_value][aggregation_method][metric_name]
                y_array = numpy.array(y)
                y_mean = y_array.mean()
                ys.append(y_mean)
                xticks.append(f'{modifier_name}({modifier_value}) = {y_mean:.4g}')
                y_error.append(mean_confidence_interval_size(y_array))
        xs = range(len(xticks))
        pyplot.figure(figsize=(12.4, 9.6), dpi=100)
        pyplot.bar(x=xs, height=ys, yerr=y_error, width=0.8)
        pyplot.title(f'{metric_name}, {repetitions} repetitions, aggregated via "{aggregation_method}"', fontsize=10)
        pyplot.xlim(-0.5, len(xticks) + 0.5)
        pyplot.xticks(labels=xticks, ticks=xs, rotation=90)
        pyplot.tick_params(axis='both', which='major', labelsize=8)
        pyplot.tight_layout()

    def plot_to_dir(self, aggregations, skip_existing, to_dir, serializer: EvaluationResultSerializer):
        json_path = os.path.join(to_dir, 'metrics.json')
        serializer.create_directory_if_not_exists(to_dir)
        serializer.save_json(json_path, data=self.to_json())
        metrics_names = self.metric_names()
        random.shuffle(metrics_names)
        for metric_name in metrics_names:
            for repetitions in self.metrics:
                for aggregation_method in aggregations:
                    to_files = [
                        os.path.join(to_dir, f'{metric_name}_{aggregation_method}_{repetitions}rep.png'),
                        os.path.join(to_dir, f'{metric_name}_{aggregation_method}_{repetitions}rep.svg')
                    ]
                    if skip_existing and all(os.path.exists(f) for f in to_files):
                        continue
                    logging.info('Writing ' + os.path.join(to_dir, f'{metric_name}_{aggregation_method}_{repetitions}rep.*'))
                    self.create_single_metrics_plot(aggregation_method, metric_name, repetitions)
                    if not os.path.exists(to_dir):
                        serializer.create_directory_if_not_exists(to_dir)
                    for to_file in to_files:
                        serializer.save_current_pyplot_figure(to_file)
                    pyplot.close()

    def walk(self):
        for repetitions in self.metrics:
            for modifier_name in self.metrics[repetitions]:
                for modifier_value in self.metrics[repetitions][modifier_name]:
                    for aggregation_method in self.metrics[repetitions][modifier_name][modifier_value]:
                        for metric_name in self.metrics[repetitions][modifier_name][modifier_value][aggregation_method]:
                            yield repetitions, modifier_name, modifier_value, aggregation_method, metric_name, \
                                self.metrics[repetitions][modifier_name][modifier_value][aggregation_method][metric_name]

    def get_entry(self, repetitions, modifier_name, modifier_value, aggregation_method, metric_name=None):
        if metric_name is None:
            return self.metrics[repetitions][modifier_name][modifier_value][aggregation_method]
        else:
            return self.metrics[repetitions][modifier_name][modifier_value][aggregation_method][metric_name]


class UseOutput(EBE):
    NON_AUGMENTED = 'NON_AUGMENTED'
    TTA_FIRST = 'TTA_FIRST'
    TTA_MEAN = 'TTA_MEAN'

    def natural_name(self):
        return {
            self.TTA_MEAN: 'mean from TTA',
            self.TTA_FIRST: 'single augmented output',
            self.NON_AUGMENTED: 'single model output',
        }[self]

    def uses_tta(self):
        return {
            self.TTA_MEAN: True,
            self.TTA_FIRST: True,
            self.NON_AUGMENTED: False,
        }[self]

    def uses_averaging(self):
        return {
            self.TTA_MEAN: True,
            self.TTA_FIRST: False,
            self.NON_AUGMENTED: False,
        }[self]


class RetainedDataPlottingConfiguration(EBC):
    def __init__(self,
                 modifier: Modifier,
                 metric_name: str,
                 repetitions: int,
                 use_output: UseOutput,
                 y_limits: Optional[Tuple[float, float]] = None,
                 use_oracle=False):
        self.y_limits = y_limits
        if modifier.neutral():
            raise ValueError((modifier.name(), modifier.value))
        self.metric_name = metric_name
        self.repetitions = repetitions
        self.use_output = use_output
        self.modifier = modifier
        self.use_oracle = use_oracle

        typical_weighted_metrics_names = ['acc', 'ce', 'wmae', 'wmse']
        for n in typical_weighted_metrics_names:
            if n in self.metric_name:
                raise NotImplementedError('Weighted metrics are not yet supported for retained data plots. '
                                          'The main issue is that weights would need to change whenever samples are excluded.')

    @classmethod
    def example_configurations(cls, metric_names=None, repetitionss: List[int] = None, modifiers: Iterable[Iterable[Modifier]] = None) \
            -> List['RetainedDataPlottingConfiguration']:
        if metric_names is None:
            metric_names = [f'{task}_{metric}'
                            for task in ['ifo', 'hufo', 'hafo', 'gsdz0v123', 'gsdz01v23', 'gsdz012v3']
                            for metric in ['roc', 'sens', 'spec', 'ppv', 'npv']]
        if repetitionss is None:
            repetitionss = PlotTestTimeAugmentation.default_repetitions()
        if modifiers is None:
            modifiers = PlotTestTimeAugmentation.best_known_modifiers().values()
        return [
            cls(
                mod,
                metric_name=metric_name,
                repetitions=repetitions,
                use_output=use_output,
                y_limits=(50, 102),  # percent
                use_oracle=oracle,
            )
            for mods in modifiers
            for mod in mods
            for metric_name in metric_names
            for use_output in [
                UseOutput.TTA_MEAN,
                UseOutput.TTA_FIRST,
                UseOutput.NON_AUGMENTED,
            ]
            for repetitions in repetitionss
            for oracle in [True, False]
        ]

    def filename(self):
        result = f'{self.metric_name}_{self.modifier.name()}_{self.use_output.name.lower()}_{self.repetitions}'
        if self.use_oracle:
            result = f'oracle/{result}'
        return result

    def title(self):
        return f'{self.modifier.name()}({self.modifier.value}), {self.repetitions} rep.,\n' \
               f'evaluation based on {self.use_output.natural_name()}'

    def identifier(self):
        return (type(self), self.y_limits, self.use_output, self.repetitions, self.metric_name, self.modifier.identifier(), self.use_oracle)

    def metric_name_without_prefix(self):
        output_name, metric_name = self.metric_name.split('_')
        return metric_name

    def metric_name_task_prefix(self):
        output_name, metric_name = self.metric_name.split('_')
        return output_name

    def remove_metric_name_from_string(self, base: str):
        return base.replace('_' + self.metric_name_without_prefix(), '')

    def x_label(self):
        if self.use_oracle:
            return 'Retained data (oracle)'
        else:
            return 'Retained data'


class PlotTestTimeAugmentation(ModelAnalyzer):
    BATCH_SIZE_FOR_EVALUATION = inf

    # constructor
    def __init__(self,
                 evaluator: FNetParameterEvaluator,
                 tta_aggregators: Dict[str, TTAAggregator] = None,
                 generator_modifiers: Dict[str, List[Modifier]] = None,
                 repetitionss=None,
                 model_level_plotting=True,
                 retained_data_plots: Optional[List[RetainedDataPlottingConfiguration]] = None,
                 retained_data_precision=0.01):
        super().__init__(evaluator, analysis_needs_xs=False, model_level_caching=True)
        self.retained_data_precision = retained_data_precision
        if generator_modifiers is None:
            generator_modifiers = self.default_modifiers()
        if retained_data_plots is None:
            retained_data_plots = []
        self.retained_data_plots = retained_data_plots
        assert len(set(c.filename() for c in retained_data_plots)) == len(retained_data_plots)
        self.model_level_plotting = model_level_plotting
        self.aggregations_to_evaluate = ['mean']
        self.generator_modifiers = generator_modifiers
        if repetitionss is None:
            repetitionss = self.default_repetitions()
        self.repetitionss = repetitionss
        self.model_stats = {}
        self.task_number = 0
        self.vert_number = 0
        if tta_aggregators is None:
            tta_aggregators = self.default_tta_aggregators()
        self.tta_aggregators = tta_aggregators
        for a in self.aggregations_to_evaluate:
            if a not in self.tta_aggregators:
                raise ValueError(a)
        if len(self.retained_data_plots) > 0 and not {'mean', 'first'}.issubset(self.tta_aggregators):
            raise ValueError('Retained data plots are based on "mean" and "first" outputs of TTA, '
                             'but those were not specified to be computed.')
        self._cached_single_step_generator: Optional[AnnotatedPatchesGenerator] = None
        self._cached_few_step_generator: Optional[AnnotatedPatchesGenerator] = None

    @staticmethod
    def default_repetitions():
        return [1, 4, 10]

    def to_subdir(self):
        return 'test_time_augmentation'

    @staticmethod
    def default_tta_aggregators() -> Dict[str, TTAAggregator]:
        return {
            'std': StandardDeviation(),
            'logit_std': LogitStandardDeviation(),
            'sample_std': StandardDeviation(ddof=1),
            'mean': AverageVoting(),
            'first': First(),
            # 'majority': MajorityVoting(),
            # 'median': MedianVoting(),
        }

    @classmethod
    def default_modifiers(cls) -> Dict[str, List[Modifier]]:
        result = AugmentationRobustnessDrawer.default_modifiers()
        assert set(cls.best_known_modifiers()).issubset(result.keys())
        return result

    @classmethod
    def default_modifiers_for_2d(cls):
        result = AugmentationRobustnessDrawer.default_modifiers_for_2d()
        assert set(cls.best_known_modifiers()).issubset(result.keys())
        return result

    @classmethod
    def best_known_modifiers(cls) -> Dict[str, List[Modifier]]:
        return {'rot_shift_fliplr': [CombinedModifier([RotationRangeModifier(24), ShiftModifier(4), RandomFlipLRModifier(True)])]}

    @classmethod
    def best_known_modifiers_2d(cls) -> Dict[str, List[Modifier]]:
        return {'shift_flipud': [CombinedModifier([ShiftModifier2D(2), RandomFlipUDModifier(True)])]}

    def before_model(self, model_path: str, ):
        super().before_model(model_path)

        self.model = self.load_trained_model_for_evaluation()
        dummy_inputs = [Input(shape=s[1:]) for s in self.model.output_shapes()]
        dummy_outputs = [Layer(name=task.output_layer_name())(i) for i, task in zip(dummy_inputs, self.tasks)]
        assert len(dummy_outputs) == len(dummy_inputs)
        self.dummy_model = Model(inputs=dummy_inputs, outputs=dummy_outputs)
        if not isinstance(self.model, FNet):
            raise NotImplementedError('TO DO')
        self.model: FNet
        FNetBuilder(self.config).compile_built_model(self.dummy_model,
                                                     classification_threshold_for_task=self.classification_thresholds[self.model_path])
        FNetBuilder(self.config).compile_built_model(self.model.model,
                                                     classification_threshold_for_task=self.classification_thresholds[self.model_path])

    def after_model_directory(self, results: List[SingleModelEvaluationResult]):
        logging.info(f'Combining {len(results)} containers...')
        combined_metrics = MetricsContainer.combine([r[0] for r in results])
        to_dir = os.path.join(self.to_dir(), 'summary')
        combined_metrics.plot_to_dir(self.aggregations_to_evaluate, self.skip_existing, to_dir, serializer=self.serializer())
        self.plot_combined_retained_data_plots([retained_data_plot_data for metrics, retained_data_plot_data in results])
        return super().after_model_directory(results)

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        self.clear_cached_generator()
        return super().after_model(results)

    def clear_cached_generator(self):
        self._cached_single_step_generator = None
        self._cached_few_step_generator = None
        dl_backend.b().memory_leak_cleanup()

    def after_dataset(self) -> SingleModelEvaluationResult:
        self.clear_cached_generator()
        return super().after_dataset()

    def analyze_batch(self, batch, y_predss, names):
        y_true = batch[1]
        sample_weights = batch[2]

        metrics = type(self).cached_metrics_computation(self, names, y_true,
                                                        sample_weights=sample_weights,
                                                        _cache_key=self.compute_metrics_cache_key())
        retained_data_plot_data = self.plot_retained_data_plots(names, y_true, sample_weights)
        if isinstance(metrics, tuple) and isinstance(metrics[0], MetricsContainer):
            metrics = metrics[0]  # legacy support
        if self.model_level_plotting:
            to_dir = os.path.join(self.to_dir(), self.model_name())
            metrics.plot_to_dir(self.aggregations_to_evaluate, self.skip_existing, to_dir, serializer=self.serializer())

        return metrics, retained_data_plot_data

    def compute_metrics_cache_key(self):
        return (*self.model_level_cache_key(), len(self.aggregations_to_evaluate))

    def model_level_cache_key(self, model_path: Optional[str] = None):
        return (*super().model_level_cache_key(model_path), *self.repetitionss, len(self.generator_modifiers),
                len(self.tta_aggregators), len(self.retained_data_plots))

    def directory_level_cache_key(self, model_dir: str):
        return (*super().directory_level_cache_key(model_dir), *self.repetitionss, len(self.generator_modifiers), len(self.tta_aggregators))

    @results_cache.cache(ignore=['self', 'names', 'y_true', 'sample_weights'], verbose=0)
    def cached_metrics_computation(self, names, y_true, sample_weights, _cache_key):
        results = {}
        to_do = [
            (repetitions, modifier_name, modifier)
            for repetitions in self.repetitionss
            for modifier_name, modifiers in self.generator_modifiers.items()
            for modifier in modifiers
        ]
        logging.info(f'{len(to_do)} TTA runs to do...')
        if self.RANDOM_MODEL_ORDER:
            random.shuffle(to_do)
        to_do = ProgressBar(to_do, prefix='TTA predictions', line_length=120, print_speed=True)
        for repetitions, modifier_name, modifier in to_do:
            # dbg = MemoryLeakDebugger()
            # dbg.start()
            results_for_modifier = self.predict_model_in_modified_setting_and_aggregate(modifier, repetitions=repetitions, )
            # results_for_modifier = pickle.dumps(results_for_modifier)
            # # for some reasons this is more memory efficient than the original data structure
            # dbg.stop()
            results.setdefault(repetitions, {}).setdefault(modifier_name, {})[modifier.value] = results_for_modifier
        metrics = self.calculate_metrics(results, y_true, names,
                                         sample_weights=sample_weights,
                                         only_aggregations=self.aggregations_to_evaluate)
        return metrics

    def plot_retained_data_plots(self, names: list, y_true, sample_weights):
        assert len(names) == len(y_true[0])
        with pyplot.rc_context({'mathtext.default': 'regular'}):
            to_dir = os.path.join(self.to_dir(), 'retained_data', self.model_name())
            results: Dict[str, Dict[str, MetricValueDict]] = {}

            for config in self.plotting_configurations_for_iteration():
                task_idx = self.task_idx_of_metric(config.metric_name)
                if task_idx is None:
                    continue  # model does not have this task
                self.initialize_dummy_model_metrics_names(y_true)
                assert config.metric_name in self.dummy_model.metrics_names, "Because we already checked a few lines ago that the model has the required task, " \
                                                                             "we expect that it also has the required metric"
                results_for_modifier = self.predict_model_in_modified_setting_and_aggregate(config.modifier,
                                                                                            repetitions=config.repetitions, )
                results_for_empty_modifier = self.predict_model_in_modified_setting_and_aggregate(EmptyModifier(),
                                                                                                  repetitions=1, )
                self._check_name_consistency(names, results_for_empty_modifier, results_for_modifier)

                confidence_valuess = self.confidence_values_dict(names, results_for_empty_modifier, results_for_modifier, task_idx)

                results_for_y_pred = results_for_modifier if config.use_output.uses_tta() else results_for_empty_modifier
                aggregation_key = 'mean' if config.use_output.uses_averaging() else 'first'
                y_pred = [numpy.stack([results_for_y_pred[v][aggregation_key][task_idx] for v in names])
                          for task_idx in range(len(self.tasks))]

                num_vertebrae = len(results_for_modifier)

                for confidence_values in confidence_valuess.values():
                    assert len(confidence_values) == len(names) == num_vertebrae
                arg_sorted_confidencess = {
                    confidence_name: sorted(range(len(names)),
                                            key=lambda v_idx: confidence_values[v_idx])
                    for confidence_name, confidence_values in confidence_valuess.items()
                }
                sorted_confidence_values = {
                    confidence_name: sorted(confidence_values, reverse=True)
                    for confidence_name, confidence_values in confidence_valuess.items()
                }
                full_dataset_value: Optional[float] = None
                assert config.filename() not in results
                results[config.filename()] = {
                    'metric_values': {},
                    'confidences': {},
                    'num_positives': {},
                    'num_negatives': {},
                }
                for confidence_name, arg_sorted_confidences in arg_sorted_confidencess.items():
                    r = type(self).retained_data_metric_values(self,
                                                               y_pred,
                                                               y_true,
                                                               sample_weights,
                                                               arg_sorted_confidences,
                                                               config.metric_name,
                                                               use_oracle=config.use_oracle,
                                                               _cache_key=(confidence_name, self.model_level_cache_key(), config.identifier(),
                                                                           self.retained_data_precision))
                    metric_values, num_negatives, num_positives = r
                    full_dataset_value = metric_values[-1]
                    results[config.filename()]['metric_values'][confidence_name] = metric_values
                    results[config.filename()]['num_positives'][confidence_name] = num_positives
                    results[config.filename()]['num_negatives'][confidence_name] = num_negatives
                if full_dataset_value is not None:
                    if config.use_oracle and any(config.metric_name.endswith(s) for s in ['sens', 'spec']):
                        confidence_name = 'random (theoretical)'
                        metric_values = [full_dataset_value * a + 1 * (1 - a) for a in self.retained_data_steps(num_vertebrae) / num_vertebrae]
                        results[config.filename()]['metric_values'][confidence_name] = metric_values
                    elif not config.use_oracle:
                        confidence_name = 'random (theoretical)'
                        metric_values = [full_dataset_value for _ in self.retained_data_steps(num_vertebrae)]
                        results[config.filename()]['metric_values'][confidence_name] = metric_values
                results[config.filename()]['sorted_confidence_values'] = sorted_confidence_values

                if self.model_level_plotting:
                    base = os.path.join(to_dir, config.filename())
                    self.serializer().create_directory_if_not_exists(to_dir)
                    n = num_vertebrae
                    self.plot_retained_data_dict_to_file(plot_xs=(self.retained_data_steps(n) / n) * 100,
                                                         y_dict=results[config.filename()]['metric_values'],
                                                         y_label=(self.tasks[task_idx].natural_metric_name_by_metric_name(config.metric_name)),
                                                         config=config,
                                                         base_file_name=base,
                                                         y_lim=config.y_limits)
                    self.plot_retained_data_dict_to_file(plot_xs=(numpy.arange(1, n + 1) / n * 100),
                                                         y_dict=results[config.filename()]['sorted_confidence_values'],
                                                         y_label='Confidence',
                                                         config=config,
                                                         base_file_name=config.remove_metric_name_from_string(base) + '_conf',
                                                         y_lim=(50, 102))
                    for num in ['posi', 'nega']:
                        self.plot_retained_data_dict_to_file(plot_xs=(self.retained_data_steps(n) / n) * 100,
                                                             y_dict=results[config.filename()][f'num_{num}tives'],
                                                             y_label=f'Number of {num}tives',
                                                             config=config,
                                                             base_file_name=config.remove_metric_name_from_string(base) + f'_{num[:3]}',
                                                             y_lim=None)
            return results

    def model_name(self):
        return os.path.basename(self.model_path)

    def plot_retained_data_dict_to_file(self,
                                        plot_xs: List[float],
                                        y_dict: MetricValueDict,
                                        y_label: str,
                                        config: RetainedDataPlottingConfiguration,
                                        base_file_name: str,
                                        y_lim=None):
        out_files = [f'{base_file_name}{ext}' for ext in ['.png', '.svg']]
        plotting = not (self.skip_existing and all(self.serializer().isfile(out_file) for out_file in out_files))
        if not plotting:
            return
        logging.info(f'Creating {base_file_name}.*')
        self.serializer().create_directory_if_not_exists(os.path.dirname(base_file_name))

        self.plot_retained_data_dict(plot_xs=plot_xs,
                                     y_dict=y_dict,
                                     y_label=y_label,
                                     y_lim=y_lim,
                                     config=config)
        for filename in out_files:
            self.serializer().save_current_pyplot_figure(filename)
        pyplot.close()

    def plot_retained_data_dict(self,
                                plot_xs: List[float],
                                y_dict: MetricValueDict,
                                y_label: str,
                                config: RetainedDataPlottingConfiguration,
                                mean_conf_dict: Optional[MetricValueDict] = None,
                                y_lim=None, ):
        pyplot.figure(figsize=(12.8, 9.6), dpi=100)
        if y_lim is not None:
            pyplot.ylim(y_lim)
        y_as_percent = y_lim is not None and 95 <= y_lim[1] <= 105
        y_format = (lambda x: self.as_percent(x)) if y_as_percent else (lambda x: x)
        for model_confidence_name in sorted(y_dict.keys()):
            metric_values = y_dict[model_confidence_name]
            assert len(plot_xs) == len(metric_values)
            p = pyplot.plot(plot_xs,
                            y_format(metric_values),
                            label=model_confidence_name)
            if mean_conf_dict is not None:
                mean_confidences = mean_conf_dict[model_confidence_name]
                pyplot.fill_between(plot_xs,
                                    y1=y_format(numpy.array(metric_values) - mean_confidences),
                                    y2=y_format(numpy.array(metric_values) + mean_confidences),
                                    color=p[0].get_color(),  # https://stackoverflow.com/q/36699155
                                    alpha=0.1)
        if y_as_percent:
            pyplot.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        pyplot.gca().xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        pyplot.title(config.title())
        pyplot.legend()
        pyplot.xlabel(config.x_label())
        pyplot.ylabel(y_label)
        pyplot.tight_layout()

    @staticmethod
    def _check_name_consistency(names, results_for_empty_modifier, results_for_modifier):
        assert set(results_for_modifier) == set(names), set(results_for_modifier).symmetric_difference(names)
        assert set(names) == set(results_for_empty_modifier), set(names).symmetric_difference(results_for_empty_modifier)
        assert set(results_for_modifier) == set(results_for_empty_modifier), set(results_for_modifier).symmetric_difference(results_for_empty_modifier)

    @classmethod
    def confidence_values_dict(cls, names, results_for_empty_modifier, results_for_modifier, task_idx) -> Dict[str, List[float]]:
        model_output_to_confidence = lambda y: 0.5 + abs(0.5 - y)
        med = lambda s: r'$\tilde{' + s + '}$'
        avg = lambda s: r'$\overline{' + s + '}$'
        extra_thresholds = False
        if extra_thresholds:
            single_output = "y"
            augmented_output = "y'"
            avg_augmented_output = "y''"
        else:
            single_output = augmented_output = avg_augmented_output = ""
        confidence_measures: Dict[str, List[float]] = {
            f'0.5 + | 0.5 - single model output {single_output}|': [model_output_to_confidence(results_for_empty_modifier[v]['first'][task_idx]) for v in
                                                                    names],
            f'0.5 + | 0.5 - single augmented output {augmented_output}|': [model_output_to_confidence(results_for_modifier[v]['first'][task_idx]) for v in
                                                                           names],
            f'0.5 + | 0.5 - avg from TTA {avg_augmented_output}|': [model_output_to_confidence(results_for_modifier[v]['mean'][task_idx]) for v in names],
            '1 - std from TTA': [1 - results_for_modifier[v]['std'][task_idx] for v in names],
            '1 - 0.1 * logit_std from TTA': [1 - 0.1 * results_for_modifier[v]['logit_std'][task_idx] for v in names],
        }
        if extra_thresholds:
            for y_preds, agg, var in [
                (results_for_empty_modifier, 'first', single_output),
                (results_for_modifier, 'first', augmented_output),
                (results_for_modifier, 'mean', avg_augmented_output),
            ]:
                for base_name, base_fn in [(med, relative_distances_from_median), (avg, relative_distances_from_avg)]:
                    m = base_name(var)
                    formula = f'max(({m}-{m}{var})/({m}-1)+{var},1-{var}/{m})'
                    assert formula not in confidence_measures
                    confidence_measures[formula] = base_fn([y_preds[v][agg][task_idx] for v in names])
        confidence_measures['random (empirical)'] = [1 - 0.5 * random.random() for _ in names]
        return confidence_measures

    @staticmethod
    def as_percent(metric_values):
        return [y * 100 if isinstance(y, (float, numpy.ndarray)) else y for y in metric_values]

    def task_idx_of_metric(self, metric_name):
        tasks_with_fitting_names = [t for t in self.tasks if metric_name.startswith(t.output_layer_name() + '_')]
        if len(tasks_with_fitting_names) == 0:
            return None
        assert len(tasks_with_fitting_names) == 1, tasks_with_fitting_names
        return self.tasks.index(tasks_with_fitting_names[0])

    def initialize_dummy_model_metrics_names(self, y_true):
        if len(self.dummy_model.metrics_names) == 0:
            self.dummy_model.evaluate([y[:5] for y in y_true], verbose=0)  # this initializes dummy_model.metrics_names
        assert len(self.dummy_model.metrics_names) > 0

    @results_cache.cache(ignore=['self', 'y_pred', 'y_true', 'sample_weights', 'arg_sorted_confidences', 'metric_name'], verbose=0)
    def retained_data_metric_values(self, y_pred, y_true, sample_weights, arg_sorted_confidences, metric_name: str, use_oracle: bool, _cache_key):
        confidence_name = _cache_key[0]
        logging.info(f'Computing retained data metric values for "{confidence_name}" (oracle={use_oracle})...')
        results: MetricValueList = []
        # metrics_names = self.dummy_model.metrics_names
        task_idx = self.task_idx_of_metric(metric_name)
        n = len(arg_sorted_confidences)
        batches = []
        next_possibly_usable_index = 0
        usable_step_indices = []
        num_positives: List[float] = []
        num_negatives: List[float] = []
        for step_idx, num_retained_samples in enumerate(self.retained_data_steps(n)):
            num_excluded_samples = n - num_retained_samples
            retained_vertebra_indices = arg_sorted_confidences[num_excluded_samples:]
            if use_oracle:
                ground_truth = y_true
                sws = sample_weights
                model_input_data = copy.deepcopy(y_true)
                assert len(y_true) == len(model_input_data) == len(y_pred)
                for y, i in zip(y_pred, model_input_data):
                    i[retained_vertebra_indices] = y[retained_vertebra_indices]
            else:
                ground_truth = [y[retained_vertebra_indices] for y in y_true]
                model_input_data = [x[retained_vertebra_indices] for x in y_pred]
                sws = [s[retained_vertebra_indices] for s in sample_weights]
            num_positives.append(numpy.sum(ground_truth[task_idx]).item())
            num_negatives.append(numpy.sum(1 - numpy.array(ground_truth[task_idx])).item())
            if numpy.unique(ground_truth[task_idx]).size <= 1:
                usable_step_indices.append(None)
                continue
            # metrics_values: List[float] = self.dummy_model.evaluate(x=model_input_data, y=ground_truth, verbose=0)
            # metric_value = metrics_values[metrics_names.index(metric_name)]
            batches.append((model_input_data, ground_truth, sws))
            usable_step_indices.append(next_possibly_usable_index)
            next_possibly_usable_index += 1
        recorder = RecordBatchMetrics(only_metrics=[metric_name], cumulative=False)
        if len(batches) > 0:
            all_metrics = recorder.compute_metrics_on_batches(batches, self.dummy_model)
        else:
            assert all(step_idx is None for step_idx in usable_step_indices)
            all_metrics = []
        for step_idx in usable_step_indices:
            if step_idx is None:
                results.append(None)
            else:
                results.append(all_metrics[step_idx][metric_name])
        assert len(results) == len(self.retained_data_steps(n)) == len(num_negatives) == len(num_positives)

        return results, num_negatives, num_positives

    def retained_data_steps(self, n):
        # return numpy.array(list(range(1, n, max(round(n * self.retained_data_precision), 1))) + [n])
        return numpy.round(numpy.linspace(0., 1., endpoint=True, num=round(1 / self.retained_data_precision)) * n).astype('int')

    def calculate_metrics(self, aggregated_predictions, y_true, names, sample_weights, only_aggregations: Optional[List[str]] = None):
        results = MetricsContainer()
        steps = sum(1 for repetitions in aggregated_predictions
                    for modifier_name in aggregated_predictions[repetitions]
                    for _modifier_value in aggregated_predictions[repetitions][modifier_name]
                    for aggregation_method in self.tta_aggregators
                    if only_aggregations is None or aggregation_method in only_aggregations)

        p = ProgressBar(steps, prefix='Calculating metrics', line_length=120)
        p.step(0)
        for repetitions in aggregated_predictions:
            for modifier_name in aggregated_predictions[repetitions]:
                for modifier_value in aggregated_predictions[repetitions][modifier_name]:
                    for aggregation_method in self.tta_aggregators:
                        if only_aggregations is not None and aggregation_method not in only_aggregations:
                            continue
                        xs = []
                        for task_idx in range(len(self.tasks)):
                            x = []
                            for name in names:
                                results_for_modifier = aggregated_predictions[repetitions][modifier_name][modifier_value]
                                # results_for_modifier = pickle.loads(results_for_modifier)
                                if name not in results_for_modifier:
                                    print(list(results_for_modifier))
                                x.append(results_for_modifier[name][aggregation_method][task_idx])
                            xs.append(numpy.array(x))

                        metrics_values: List[float] = self.dummy_model.evaluate(x=xs,
                                                                                y=y_true,
                                                                                sample_weight=sample_weights,
                                                                                verbose=0)

                        metric_names: List[str] = self.dummy_model.metrics_names

                        metrics = dict(zip(metric_names, metrics_values))
                        results.add_entry(repetitions, modifier_name, modifier_value, aggregation_method, metrics)
                        p.step(1)

        return results

    def predict_model_in_modified_setting_and_aggregate(self, modifier, repetitions: int, dont_cache=False):
        if modifier.neutral():
            modifier = EmptyModifier()
        f = type(self)._cached_prediction_of_model
        if dont_cache:
            f = f.func
        return f(self,
                 modifier=modifier,
                 repetitions=repetitions,
                 _cache_key=(modifier.value, modifier.name(), repetitions,
                             len(self.tta_aggregators),
                             super().model_level_cache_key()))

    @results_cache.cache(ignore=['self', 'modifier', 'repetitions'], verbose=0)
    def _cached_prediction_of_model(self, modifier: Modifier, repetitions, _cache_key) -> Dict[str, Dict[str, List[numpy.ndarray]]]:
        data_generator = self.few_step_prediction_generator(cache_vertebra_volumes_in_ram=True)
        assert data_generator.image_list is self.dataset
        with modifier.modified_context(data_generator, self.model) as (data_generator, modified_model):
            model_prediction_results = modified_model.predict_generator(data_generator,
                                                                        steps=data_generator.steps_per_epoch() * repetitions)
            del modified_model
            dataset_names = data_generator.all_vertebra_names()
            assert len(dataset_names) * repetitions == len(model_prediction_results[0]), (len(dataset_names), repetitions, model_prediction_results[0].shape[0])
            # if model.predict keeps the ordering, then the last few names should be the last ones of the dataset
            # this does not imply that the predictions are also in order, only that the generator produced them correctly
            # I hope this works...
            last_batch_vertebrae = data_generator.last_batch_names
            last_vertebrae_in_dataset = dataset_names[-len(last_batch_vertebrae):]
            assert last_vertebrae_in_dataset == last_batch_vertebrae, (last_vertebrae_in_dataset, last_batch_vertebrae)
            grouped_prediction_results = self.group_predictions_per_vertebra(model_prediction_results, names=dataset_names)
        assert len(data_generator.batch_cache) == 0

        aggregated_results = {}
        for v, results_for_vertebra in grouped_prediction_results.items():
            aggregated_results[v] = {
                agg_name: [agg.aggregate(numpy.stack(y_pred, axis=0), task)
                           for y_pred, task in zip(results_for_vertebra, self.tasks)]
                for agg_name, agg in self.tta_aggregators.items()
            }

        return aggregated_results

    def single_step_prediction_generator(self, cache_vertebra_volumes_in_ram=False):
        if self._cached_single_step_generator is None:
            self._cached_single_step_generator = self.evaluator.prediction_generator_from_config(self.dataset,
                                                                                                 self.config,
                                                                                                 batch_size=self.BATCH_SIZE_FOR_EVALUATION,
                                                                                                 cache_vertebra_volumes_in_ram=cache_vertebra_volumes_in_ram,
                                                                                                 exclude_unlabeled=True,
                                                                                                 ndim=self.evaluator.data_source.ndim())
        assert self._cached_single_step_generator.steps_per_epoch() == 1
        assert not self._cached_single_step_generator.random_mode
        assert self._cached_single_step_generator.cache_vertebra_volumes_in_ram == cache_vertebra_volumes_in_ram
        return self._cached_single_step_generator

    def few_step_prediction_generator(self, cache_vertebra_volumes_in_ram=False):
        if self._cached_few_step_generator is None:
            self._cached_few_step_generator = self.evaluator.prediction_generator_from_config(self.dataset,
                                                                                              self.config,
                                                                                              batch_size=self.model.validation_batch_size,
                                                                                              cache_vertebra_volumes_in_ram=cache_vertebra_volumes_in_ram,
                                                                                              exclude_unlabeled=True,
                                                                                              ndim=self.evaluator.data_source.ndim())
        assert self._cached_few_step_generator.steps_per_epoch() <= 50, self._cached_few_step_generator.steps_per_epoch()
        assert not self._cached_few_step_generator.random_mode
        assert self._cached_few_step_generator.cache_vertebra_volumes_in_ram == cache_vertebra_volumes_in_ram
        return self._cached_few_step_generator

    def group_predictions_per_vertebra(self,
                                       y_preds: List[numpy.ndarray],
                                       names: List[str]) -> Dict[str, List[numpy.ndarray]]:
        """
        groups augmentation results per vertebra by looking for the n-th result in each augmentation for the
        n-th vertebra.
        """
        results = {}
        for task_idx, task in enumerate(self.tasks):
            y_pred = y_preds[task_idx]
            for sample_idx in range(len(y_pred)):
                results.setdefault(names[sample_idx % len(names)], [[] for _ in self.tasks])[task_idx].append(y_pred[sample_idx])

        return results

    def plot_retained_data_dicts_to_file(self,
                                         plot_xs: List[float],
                                         y_dicts: List[MetricValueDict],
                                         y_label: str,
                                         config: RetainedDataPlottingConfiguration,
                                         base_file_name: str,
                                         y_lim=None):
        out_files = [f'{base_file_name}{ext}' for ext in ['.png', '.svg']]
        plotting = not (self.skip_existing and all(self.serializer().isfile(out_file) for out_file in out_files))
        if not plotting:
            return
        logging.info(f'Creating {base_file_name}.*')
        self.serializer().create_directory_if_not_exists(os.path.dirname(base_file_name))

        means, confidences = self.aggregate_retained_data_curves(y_dicts)

        self.plot_retained_data_dict(plot_xs=plot_xs,
                                     y_dict=means,
                                     mean_conf_dict=confidences,
                                     y_label=y_label,
                                     y_lim=y_lim,
                                     config=config)

        for filename in out_files:
            self.serializer().save_current_pyplot_figure(filename)
        pyplot.close()

    @staticmethod
    def aggregate_retained_data_curves(y_dicts: List[MetricValueDict]) -> Tuple[MetricValueDict, MetricValueDict]:
        curve_names = set(curve_name for y_dict in y_dicts for curve_name in y_dict)
        means = {}
        confidences = {}
        for curve_name in curve_names:
            none_to_nan = lambda xs: [nan if x is None else x for x in xs]
            individual_plots = numpy.stack(
                [none_to_nan(y_dict[curve_name])
                 for y_dict in y_dicts],
                axis=0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means[curve_name] = list(numpy.nanmean(individual_plots, axis=0))  # list() to make type checker happy...
            confidences[curve_name] = []
            for i in range(individual_plots.shape[1]):
                y_values = individual_plots[:, i]
                y_values = y_values[~numpy.isnan(y_values)]
                confidences[curve_name].append(mean_confidence_interval_size(y_values))
        return means, confidences

    def plot_combined_retained_data_plots(self, results: List[Dict[FileName, Dict[str, MetricValueDict]]]):
        to_dir = os.path.join(self.to_dir(), 'retained_data', 'summary')
        available_filenames = set(filename
                                  for individual_model_results in results
                                  for filename in individual_model_results)
        for config in self.plotting_configurations_for_iteration():
            filename = config.filename()
            if filename not in available_filenames:
                continue
            some_large_n = 1000
            xs = self.retained_data_steps(some_large_n) / some_large_n * 100

            base = os.path.join(to_dir, config.filename())
            self.serializer().create_directory_if_not_exists(to_dir)
            self.plot_retained_data_dicts_to_file(plot_xs=xs,
                                                  y_dicts=[r[config.filename()]['metric_values'] for r in results],
                                                  y_label=config.metric_name,
                                                  config=config,
                                                  base_file_name=base,
                                                  y_lim=config.y_limits)
            for num in ['posi', 'nega']:
                self.plot_retained_data_dicts_to_file(plot_xs=xs,
                                                      y_dicts=[r[config.filename()][f'num_{num}tives'] for r in results],
                                                      y_label=f'Number of {num}tives',
                                                      config=config,
                                                      base_file_name=config.remove_metric_name_from_string(base) + f'_{num[:3]}',
                                                      y_lim=None)

    def plotting_configurations_for_iteration(self):
        plotting_configurations: List[RetainedDataPlottingConfiguration] = copy.copy(self.retained_data_plots)
        if self.RANDOM_MODEL_ORDER:
            random.shuffle(plotting_configurations)
        return plotting_configurations
