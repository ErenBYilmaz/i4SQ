import math
import os
import random
from typing import Dict, Set, Optional, List, Union, Sized, Tuple

import matplotlib.colors
import numpy
from matplotlib import pyplot
from matplotlib.axes import Axes
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer

from lib.callbacks import RecordBatchMetrics
from lib.my_logger import logging
from lib.parameter_search import mean_confidence_interval_size
from lib.util import my_tabulate, num_reasonable_decimals_by_confidence
from load_data.load_image import image_by_patient_id
from load_data.vertebra_group import AgeRange, VertebraRange, VertebraLevel, PatientClass, VertebraClass, Sex, DiagBilanzCVFold, TraumaCVFold, Group, MetricName, GroupName
from model.fnet.builder import FNetBuilder
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import split_name, ModelAnalyzer
from tasks import VertebraClassificationTask


class PlotGroupedMetrics(ModelAnalyzer):
    NON_FLOAT_METRICS = ['vertebrae']

    def __init__(self, evaluator: FNetParameterEvaluator,
                 confusion_task: Optional[VertebraClassificationTask] = None,
                 bar_width=None,
                 skip_existing=None,
                 exclude_vertebrae: List[Group] = None,
                 only_json=False):
        super().__init__(evaluator=evaluator, model_level_caching=True, analysis_needs_xs=False, skip_existing=skip_existing)
        self.only_json = only_json
        if exclude_vertebrae is None:
            exclude_vertebrae: List[Group] = []
        self.exclude_vertebrae = exclude_vertebrae
        self.confusion_task = confusion_task
        if bar_width is None:
            bar_width = 0.35
            if confusion_task is None:
                bar_width *= 2
        self.bar_width = bar_width

    def before_model(self, model_path: str):
        super().before_model(model_path)
        self.model = self.load_trained_model_for_evaluation()
        dummy_inputs = [Input(shape=output_shape[1:]) for output_shape in self.model.output_shapes()]
        dummy_outputs = [Layer(name=task.output_layer_name())(i) for i, task in zip(dummy_inputs, self.tasks)]
        assert len(dummy_outputs) == len(dummy_inputs)
        self.dummy_model = Model(inputs=dummy_inputs, outputs=dummy_outputs)
        FNetBuilder(self.config).compile_built_model(self.dummy_model,
                                                     classification_threshold_for_task=self.classification_thresholds[self.model_path])

        self.groups = self.create_evaluation_groups()
        self.grouped_metrics: Dict[MetricName, Dict[GroupName, Union[float, Sized]]] = {}

    def to_subdir(self):
        if len(self.exclude_vertebrae) == 0:
            return 'grouped_metrics'
        else:
            exclusions = "_".join(g.name for g in self.exclude_vertebrae)
            return f'grouped_metrics/excl_{exclusions}'

    def after_multiple_models(self, results: List[Dict[MetricName, Dict[GroupName, Union[str, float]]]]):
        self.serializer().create_directory_if_not_exists(self.to_dir())
        self.before_model(self.example_model_path())
        self.analyzing_model = False
        metrics = list(results[0])

        vertebrae = self.vertebra_groups(self.group_names(), results)
        group_names_to_plot = []
        for g in self.group_names():
            if len(vertebrae[g]) > 0:
                if g not in group_names_to_plot:
                    group_names_to_plot.append(g)
        mean_results = {}
        logging.info(f'Plotting to {self.to_dir()}')

        if self.random_metric_order():
            metrics = metrics.copy()
            logging.info('Plotting metrics in random order.')
            random.shuffle(metrics)

        json_path = os.path.join(self.to_dir(), f'grouped_results.json')
        markdown_path_mean_ci = os.path.join(self.to_dir(), f'grouped_results_mean_ci.md')
        markdown_path_std = os.path.join(self.to_dir(), f'grouped_results_std.md')
        text_output_paths = [
            json_path,
            markdown_path_mean_ci,
            markdown_path_std,
        ]
        for metric in metrics:
            if metric in self.NON_FLOAT_METRICS:
                continue

            out_paths = [os.path.join(self.to_dir(), f'{metric}.png'),
                         os.path.join(self.to_dir(), f'{metric}.svg')]
            create_plots = not (self.skip_existing and all(self.serializer().isfile(p) for p in out_paths)) and not self.only_json
            if all(self.serializer().isfile(p) for p in text_output_paths) and not create_plots:
                continue
            means, stds, mean_cis, values = self.means_for_groups(group_names_to_plot, metric, results)
            mean_results[metric] = {
                group: {
                    'mean': means[group_idx],
                    'std': stds[group_idx],
                    'mean_ci': mean_cis[group_idx],
                    'values': values[group_idx],
                }
                for group_idx, group in enumerate(group_names_to_plot)
            }

            if not create_plots:
                continue
            logging.info(f'Plotting {metric}...')
            fig, ax1 = pyplot.subplots()
            self.plot_metric_as_bars(ax1, metric=metric, results=results, group_names=group_names_to_plot)

            if self.confusion_task is not None:
                ax2: Axes = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                self.plot_classes_as_bars(ax2, group_names=group_names_to_plot, vertebrae=vertebrae)

                ax2.set_xticks(range(len(group_names_to_plot)))
                ax2.set_xticklabels([f'{group} ({self.group_size(vertebrae[group])})' for group in group_names_to_plot])
            else:
                ax1.set_xticks(range(len(group_names_to_plot)))
                ax1.set_xticklabels([f'{group} ({self.group_size(vertebrae[group])})' for group in group_names_to_plot])

            fig.set_size_inches([16.97, 9.89])
            pyplot.tight_layout()
            for p in out_paths:
                self.serializer().save_current_pyplot_figure(p)

            pyplot.close()

        mean_results = {k: mean_results[k] for k in sorted(mean_results)}

        if not self.skip_existing or not self.serializer().isfile(json_path):
            self.serializer().save_json(json_path, mean_results)

        if not self.skip_existing or not self.serializer().isfile(markdown_path_mean_ci):
            results_table = self.markdown_results_table(mean_results, confidence_key='mean_ci')
            self.serializer().save_text(markdown_path_mean_ci, results_table)

        if not self.skip_existing or not self.serializer().isfile(markdown_path_std):
            results_table = self.markdown_results_table(mean_results, confidence_key='std')
            self.serializer().save_text(markdown_path_std, results_table)

        return super().after_multiple_models(results)

    @staticmethod
    def markdown_results_table(mean_results, confidence_key, confidence_symbol='±'):
        example_dataset_name = next(iter(mean_results))
        for metric_name in mean_results:
            assert list(mean_results[metric_name]) == list(mean_results[example_dataset_name])
        header = ['task_metric'] + [dataset for dataset in mean_results[example_dataset_name]]
        rows = []
        for metric_name in mean_results:
            new_row = [metric_name]
            for dataset in mean_results[metric_name]:
                mean = mean_results[metric_name][dataset]["mean"]
                conf = mean_results[metric_name][dataset][confidence_key]
                num_digits = num_reasonable_decimals_by_confidence(conf)
                new_row.append(f'{mean:.{num_digits}f}{confidence_symbol}{conf:.2g}')
            rows.append(new_row)
        results_table = my_tabulate(rows, headers=header)
        return results_table

    def group_names(self):
        return list(g.name for g in self.groups)

    def plot_metric_as_bars(self, ax, metric, results, group_names):
        means, stds, mean_cis, values = self.means_for_groups(group_names, metric, results)
        assert len(means) == len(stds) == len(mean_cis) == len(group_names) == len(values)
        pyplot.xticks(rotation=90)
        pyplot.tick_params(axis='both', which='major', labelsize=12)
        pyplot.tick_params(axis='both', which='minor', labelsize=10)
        orange = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
        # other_color = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
        x = numpy.arange(len(group_names))
        ax.set_xlim(-0.5, len(group_names) + 0.5)

        if self.confusion_task is None:
            ax.bar(x=x, height=means, width=self.bar_width, color=orange)
            ax.errorbar(fmt='none', x=x, y=means, yerr=mean_cis)
        else:
            ax.bar(x=x - self.bar_width / 2, height=means, width=self.bar_width, color=orange)
            ax.errorbar(fmt='none', x=x - 0.2, y=means, yerr=mean_cis)

        ax.set_ylabel(metric, color=orange)
        # ax1.set_ylim(0, log(2))
        ax.grid(False)

    @staticmethod
    def means_for_groups(groups, metric, results: List[Dict[MetricName, Dict[GroupName, Union[str, float]]]]):
        means = []
        stds = []
        mean_cis = []
        valuess = []
        for group in groups:
            values_this_group = list(r[metric][group]
                                     for r in results
                                     if group in r[metric]
                                     if not math.isnan(r[metric][group]) and r[metric][group] != 0)  # probably doe to missing examples in a class
            means.append(numpy.mean(values_this_group))
            stds.append(numpy.std(values_this_group))
            mean_cis.append(mean_confidence_interval_size(values_this_group))
            valuess.append(values_this_group)
        return means, stds, mean_cis, valuess

    def plot_classes_as_bars(self, ax, group_names, vertebrae: Dict[GroupName, Set[str]]):
        x = numpy.arange(len(group_names))
        cmap = pyplot.cm.get_cmap('bwr')
        assert self.confusion_task.output_layer_name() in self.tasks.output_layer_names(), (self.confusion_task.output_layer_name(), self.tasks.output_layer_names())
        num_classes = self.confusion_task.num_classes()
        class_sizes = self.class_sizes(group_names, vertebrae)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=num_classes - 1)
        colors = [cmap(norm(cls_idx)) for cls_idx in range(num_classes)]
        ax.set_ylabel('Class distribution (%)', color=cmap(0))  # TODO legend about colors
        distribution = numpy.array(class_sizes).astype('float32')
        distribution /= distribution.sum(axis=-1, dtype='float32')[:, numpy.newaxis]
        distribution = numpy.nan_to_num(distribution, nan=0)
        distribution *= 100
        bars = [
            ax.bar(x=x + self.bar_width / 2, height=distribution[:, i], width=self.bar_width, bottom=distribution[:, :i].sum(axis=-1),
                   color=colors[i], )
            for i in range(distribution.shape[-1])
        ]
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.legend(bars[::-1], self.confusion_task.class_names()[::-1], loc='upper right',
                  fontsize='x-small', handlelength=1.5)
        ax.grid(False)

    @staticmethod
    def group_size(vertebrae: Set[str]):
        return len([v for v in vertebrae])

    def class_sizes(self, groups, vertebrae: Dict[GroupName, Set[str]]):
        return [[sum(1 for v in vertebrae[group_name]
                     if group in vertebrae and v in vertebrae[group])
                 for class_name in self.confusion_task.class_names()
                 for group_name in (self.group_name_for_class(class_name),)]
                for group in groups]

    def group_name_for_class(self, class_name: str):
        return VertebraClass.group_name(class_name)

    @staticmethod
    def vertebra_groups(groups: List[GroupName], results: List[Dict[MetricName, Dict[GroupName, Union[str, float]]]]):
        vertebrae = {}
        for group in groups:
            try:
                vertebrae[group] = set(v for r in results
                                       if group in r['vertebrae']
                                       for v in r['vertebrae'][group])
            except TypeError as _e:
                breakpoint()
        return vertebrae

    def model_level_cache_key(self, model_path: Optional[str] = None):
        return super().model_level_cache_key(model_path) + tuple(excl.name for excl in self.exclude_vertebrae)

    def directory_level_cache_key(self, model_dir: str):
        return super().directory_level_cache_key(model_dir) + tuple(excl.name for excl in self.exclude_vertebrae)

    def create_evaluation_groups(self):
        groups = Group.defaults()
        groups += TraumaCVFold.all_folds()
        groups += DiagBilanzCVFold.all_folds()
        groups += Sex.all_sexes()
        groups += VertebraClass.from_tasks(self.tasks)
        groups += PatientClass.from_tasks(self.tasks)
        groups += VertebraLevel.all_groups()
        groups += VertebraRange.relevant_groups()
        groups += AgeRange.groups_of_size(5)
        return groups

    def after_dataset(self) -> Dict[MetricName, Dict[GroupName, float]]:
        super().after_dataset()
        return self.grouped_metrics

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()

        vertebra_names = [split_name(name) for name in names]
        batches_for_groups = {}
        indices_for_groups = {}
        for group in self.groups:
            if group.name in batches_for_groups:
                continue
            group_indices = self.group_indices(group, vertebra_names)
            assert group_indices == sorted(group_indices)
            if group.name in indices_for_groups:
                assert indices_for_groups[group.name] == group_indices
                continue
            indices_for_groups[group.name] = group_indices
            if len(group_indices) == 0:
                continue
            # logging.info(f'Loading batch for group {group.name.replace("∕", "/")} ({len(group_indices)} vertebrae this fold)...')
            batches_for_groups[group.name] = self.restricted_batch(group_indices, batch, y_preds=y_preds, names=names)
            self.grouped_metrics.setdefault('vertebrae', {}).setdefault(group.name, set()).update(names[name_idx] for name_idx in group_indices)
            self.grouped_metrics.setdefault('count', {})[group.name] = len(self.grouped_metrics['vertebrae'][group.name])
        metrics_record = RecordBatchMetrics(cumulative=False)
        group_names = []
        batches = []
        for group_name, batch in batches_for_groups.items():
            group_names.append(group_name)
            batches.append(batch)
        print(f'Evaluating groups with model {os.path.basename(self.model_path)}...')
        metrics_record.compute_metrics_on_batches(batches=batches, model=self.dummy_model)
        assert len(batches) == len(group_names) == len(metrics_record.metrics)
        for group_name, batch, metrics_values in zip(group_names, batches, metrics_record.metrics):
            assert f'loss' in metrics_values
            for metric_name, metric_value in metrics_values.items():
                self.grouped_metrics.setdefault(metric_name, {})[group_name] = metric_value

    def group_indices(self, group: Group, vertebra_names: List[Tuple[str, str]]):
        return [sample_idx
                for sample_idx, (patient_id, vertebra) in enumerate(vertebra_names)
                for img in [image_by_patient_id(patient_id, self.dataset)]
                if group.contains_vertebra(img, vertebra_name=vertebra)
                and not any(excluded_group.contains_vertebra(img, vertebra_name=vertebra)
                            for excluded_group in self.exclude_vertebrae)]

    def restricted_batch(self, group_indices, batch, y_preds, names):
        xs, ys, sample_weights = batch
        restricted_batch = (
            ...,
            [y[group_indices] for y in ys],
            [w[group_indices] for w in sample_weights],
        )
        y_pred = [y[group_indices] for y in y_preds]
        sample_weight = restricted_batch[2]
        y_true = restricted_batch[1]
        return y_pred, y_true, sample_weight

    def random_metric_order(self) -> bool:
        return self.RANDOM_MODEL_ORDER


class PlotPatientLevelGroupedMetrics(PlotGroupedMetrics):
    def restricted_batch(self, group_indices, batch, y_preds, names):
        xs, ys, sample_weights = batch
        restricted_batch = (
            ...,
            [y[group_indices] for y in ys],
            [w[group_indices] for w in sample_weights],
        )
        y_pred = [y[group_indices] for y in y_preds]
        y_true = restricted_batch[1]
        restricted_names = [names[group_idx] for group_idx in group_indices]
        y_pred_patient_level, y_true_patient_level, names = self.patient_level_aggregation(restricted_names, y_pred, y_true)
        return y_pred_patient_level, y_true_patient_level

    def to_subdir(self):
        if len(self.exclude_vertebrae) == 0:
            return 'grouped_metrics/patient_level'
        else:
            exclusions = "_".join(g.name for g in self.exclude_vertebrae)
            return f'grouped_metrics/patient_level_excl_{exclusions}'

    def create_evaluation_groups(self):
        return [g for g in super().create_evaluation_groups() if g.above_patient_level]

    def group_name_for_class(self, class_name: str):
        return PatientClass.group_name(class_name)

    @staticmethod
    def group_size(vertebrae: Set[str]):
        return len(set(split_name(v)[0] for v in vertebrae))

    def class_sizes(self, groups, vertebrae: Dict[GroupName, Set[str]]):
        return [[len(set(split_name(v)[0]
                         for v in vertebrae[group_name]
                         if v in vertebrae[group]))
                 for class_name in self.confusion_task.class_names()
                 for group_name in (self.group_name_for_class(class_name),)]
                for group in groups]
