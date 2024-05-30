import os
import random
from itertools import product
from typing import List, Dict, Tuple

import numpy
from matplotlib import pyplot
from matplotlib.axes import Axes

from lib.my_logger import logging
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, SingleModelEvaluationResult, split_name
from tasks import VertebraClassificationTask

CLASS_NAME_PLACEHOLDER = 'HCFNBFTK227BD5OD9PMOF974WO9DF4'


class OutputsPlotter(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator, skip_existing=None, model_level_plotting=True):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, model_level_caching=False, skip_existing=skip_existing,
                         binary_threshold_or_method=0.5)
        self.model_level_plotting = model_level_plotting
        self.tasks_by_name: Dict[str, VertebraClassificationTask] = {}
        self.y_trues_and_preds: Dict[str, List[Tuple[numpy.ndarray, numpy.ndarray, List[str]]]] = {}

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)
        self.y_trues_and_preds: Dict[str, List[Tuple[numpy.ndarray, numpy.ndarray, List[str]]]] = {}
        self.tasks_by_name: Dict[str, VertebraClassificationTask] = {}

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        to_dir = os.path.join(self.to_dir(), 'summary')

        plots = list(product(self.y_trues_and_preds.items(),
                             self.plots().items()))
        if self.RANDOM_MODEL_ORDER:
            random.shuffle(plots)
        for (output_layer_name, results), (suffix, method) in plots:
            task = self.tasks_by_name[output_layer_name]
            out_paths = [
                os.path.join(to_dir, suffix, f'{task.output_layer_name()}_{CLASS_NAME_PLACEHOLDER}.png'),
                # os.path.join(to_dir, suffix, f'{task.output_layer_name()}_{CLASS_NAME_PLACEHOLDER}.svg'),
            ]
            if self.skip_existing and self.output_paths_exist(out_paths, task):
                continue
            self.serializer().create_directory_if_not_exists(os.path.join(to_dir, suffix))

            y_trues = []
            y_preds = []
            namess = []
            for (y_true, y_pred, names) in results:
                y_trues.append(y_true)
                y_preds.append(y_pred)
                assert y_true.shape == y_pred.shape
                namess.append(names)
            y_true_agg = numpy.concatenate(y_trues, axis=0)
            y_pred_agg = numpy.concatenate(y_preds, axis=0)
            names = numpy.concatenate(namess, axis=0)
            assert isinstance(task, VertebraClassificationTask)
            logging.info(f'Plotting to {os.path.join(to_dir, suffix, f"{task.output_layer_name()}_*.*")}...')
            method(y_true_agg, y_pred_agg, out_paths, task, names=names)
            assert self.output_paths_exist(out_paths, task)

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()
        _, ys, sample_weights = batch
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))

        task_zip = list(zip(self.tasks, y_preds, ys))
        if self.RANDOM_MODEL_ORDER:
            random.shuffle(task_zip)
        for task, y_pred, y_true in task_zip:
            if not isinstance(task, VertebraClassificationTask):
                continue
            self.tasks_by_name[task.output_layer_name()] = task
            self.y_trues_and_preds.setdefault(task.output_layer_name(), []).append((y_true, y_pred, names))
            if not self.model_level_plotting:
                continue
            for suffix, method in self.plots().items():
                out_paths = [
                    os.path.join(to_dir, suffix, f'{task.output_layer_name()}_{CLASS_NAME_PLACEHOLDER}.png'),
                    # os.path.join(to_dir, suffix, f'{task.output_layer_name()}_{CLASS_NAME_PLACEHOLDER}.svg'),
                ]
                if self.skip_existing and self.output_paths_exist(out_paths, task):
                    continue
                self.serializer().create_directory_if_not_exists(os.path.join(to_dir, suffix))
                logging.info(f'Plotting to {os.path.join(to_dir, suffix, f"{task.output_layer_name()}_*.*")}...')
                # noinspection PyArgumentList
                method(y_true, y_pred, out_paths, task, names=names)
                assert self.output_paths_exist(out_paths, task)

    def output_paths_exist(self, out_paths, task: VertebraClassificationTask):
        return all(self.serializer().isfile(self.format_output_path(out_path, class_name))
                   for out_path in out_paths
                   for class_name in task.class_names())

    @staticmethod
    def format_output_path(out_path: str, class_name):
        assert CLASS_NAME_PLACEHOLDER in out_path
        return out_path.replace(CLASS_NAME_PLACEHOLDER, class_name)

    def plots(self):
        return {
            '': self.plot_outputs_overview_sorted_by_sum,
            # 'name': self.plot_outputs_overview_sorted_by_name,
            'name_grouped': self.plot_outputs_overview_grouped_by_name,
        }

    def plot_outputs_overview_sorted_by_name(self, y_true, y_pred, out_paths, task: VertebraClassificationTask, names: List[str]):
        y_indices = numpy.argsort(names)
        y_true: numpy.ndarray = y_true[y_indices]
        y_pred: numpy.ndarray = y_pred[y_indices]
        names = [names[y_idx] for y_idx in y_indices]
        y_pred_all_classes: numpy.ndarray = task.class_probabilities(y_pred)
        y_true_all_classes: numpy.ndarray = task.class_probabilities(y_true)
        assert y_pred_all_classes.shape == y_true_all_classes.shape
        assert y_true_all_classes.shape[0] == y_true.shape[0]

        valid_labels = numpy.isclose(y_true_all_classes.max(axis=1), 1.)
        y_pred_all_classes = y_pred_all_classes[valid_labels]
        y_true_all_classes = y_true_all_classes[valid_labels]

        assert y_pred_all_classes.shape == y_true_all_classes.shape
        assert len(y_true_all_classes.shape) > 1

        for class_idx, class_name in enumerate(task.class_names()):
            thresholds = self.one_vs_all_classification_thresholds(class_idx, y_pred_all_classes, y_true_all_classes,
                                                                   human_readable=True)
            y_pred = y_pred_all_classes[..., class_idx]
            y_true = y_true_all_classes[..., class_idx]
            xs = numpy.arange(y_true.shape[0])
            # patients = sorted(set(split_name(name)[0] for name in names))
            fig = pyplot.figure(figsize=(20, 4), dpi=202)
            ax: Axes = fig.add_subplot(111)
            ax.set_xlim(-0.5, len(y_true) - 0.5)
            ax.set_xlabel(self.xlabel(y_true))
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(f'Output for {class_name} ({task.output_layer_name()})')
            positive_xs = xs[y_true.astype('bool')]
            positive_ys = y_pred[y_true.astype('bool')]
            negative_xs = xs[~y_true.astype('bool')]
            negative_ys = y_pred[~y_true.astype('bool')]
            ax.scatter(positive_xs, positive_ys, facecolors='black', edgecolors='grey', label=task.class_names()[1])
            ax.scatter(negative_xs, negative_ys, facecolors='white', edgecolors='grey', label=task.class_names()[0])
            ratios = numpy.linspace(0, 1, num=11)
            ax.set_yticks([ratio for ratio in ratios])
            ax.set_yticklabels([f'{ratio:.0%}' for ratio in ratios])
            for t_name, t_value in thresholds.items():
                ax.plot([-0.5, len(y_true) - 0.5],
                        [t_value, t_value],
                        label=t_name)
            ax.legend(prop={'size': 6})

            xticks = range(len(names))
            xticklabels = []
            for n in names:
                p, v = split_name(n)
                xticklabels.append(f'{p}\n{v}')
            previous = None
            for idx, xtick in enumerate(xticks):
                if previous is not None:
                    if 0 < (xtick - previous) < 0.08 * len(y_true):
                        xticklabels[idx] = ''
                        continue
                previous = xtick
            ax.grid(False)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            pyplot.tight_layout()
            for out_path in out_paths:
                self.serializer().save_current_pyplot_figure(self.format_output_path(out_path, class_name))
            pyplot.close()

    def plot_outputs_overview_grouped_by_name(self, y_true, y_pred, out_paths, task: VertebraClassificationTask, names: List[str]):
        grouped_names = sorted(set(names))

        y_true = numpy.array([
            y_true[[idx for idx in range(len(names)) if names[idx] == name]].mean(axis=0)
            for name in grouped_names
        ])
        y_pred = numpy.array([
            y_pred[[idx for idx in range(len(names)) if names[idx] == name]].mean(axis=0)
            for name in grouped_names
        ])
        return self.plot_outputs_overview_sorted_by_name(
            y_true,
            y_pred,
            out_paths,
            task,
            names=grouped_names
        )

    def plot_outputs_overview_sorted_by_sum(self, y_true, y_pred, out_paths, task: VertebraClassificationTask, **kwargs):
        for k in kwargs:
            if k != 'names':
                raise ValueError(k)
        y_pred_all_classes = task.class_probabilities(y_pred)
        y_true_all_classes = task.class_probabilities(y_true)

        valid_labels = numpy.isclose(y_true_all_classes.max(axis=1), 1.)
        y_pred_all_classes = y_pred_all_classes[valid_labels]
        y_true_all_classes = y_true_all_classes[valid_labels]

        for class_idx, class_name in enumerate(task.class_names()):
            thresholds = self.one_vs_all_classification_thresholds(class_idx, y_pred_all_classes, y_true_all_classes,
                                                                   human_readable=True)
            y_pred = y_pred_all_classes[..., class_idx]
            y_true = y_true_all_classes[..., class_idx]
            y_indices = numpy.argsort(y_true * 2 + y_pred)
            y_true = y_true[y_indices]
            y_pred = y_pred[y_indices]
            xs = numpy.arange(y_true.shape[0])
            fig = pyplot.figure(figsize=(20, 4), dpi=202)
            ax: Axes = fig.add_subplot(111)
            ax.set_xlim(-0.5, len(y_true) - 0.5)
            ax.set_xlabel(self.xlabel(y_true))
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(f'Model output ({task.output_layer_name()})')
            positive_xs = xs[y_true.astype('bool')]
            positive_ys = y_pred[y_true.astype('bool')]
            negative_xs = xs[~y_true.astype('bool')]
            negative_ys = y_pred[~y_true.astype('bool')]
            ax.axvline(x=(1 - y_true).sum() - 0.5)
            ratios = numpy.linspace(0, 1, num=11)
            xticks = [ratio * len(negative_xs) - 0.5 for ratio in ratios] + [len(negative_xs) + (1 - ratio) * len(positive_xs) - 0.5 for ratio in
                                                                             reversed(ratios)]
            xticklabels = [f'{ratio:.0%}' for ratio in ratios] + [f'{ratio:.0%}' for ratio in reversed(ratios)]
            previous = None
            for idx, xtick in enumerate(xticks):
                if previous is not None:
                    if 0 < (xtick - previous) < 0.03 * len(y_true):
                        xticklabels[idx] = ''
                        continue
                previous = xtick

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            ax.scatter(positive_xs, positive_ys, facecolors='black', edgecolors='grey', label=task.class_names()[1])
            ax.scatter(negative_xs, negative_ys, facecolors='white', edgecolors='grey', label=task.class_names()[0])
            ax.set_yticks([ratio for ratio in ratios])
            ax.set_yticklabels([f'{ratio:.0%}' for ratio in ratios])
            # ax.set_aspect('equal', adjustable='box')

            for t_name, t_value in thresholds.items():
                ax.plot([-0.5, len(y_true) - 0.5],
                        [t_value, t_value],
                        label=t_name)

            ax.legend(prop={'size': 6})
            pyplot.tight_layout()
            for out_path in out_paths:
                self.serializer().save_current_pyplot_figure(self.format_output_path(out_path, class_name))
            pyplot.close()

    def xlabel(self, y_true):
        xlabel = f'{len(y_true)} vertebrae'
        return xlabel

    def to_subdir(self):
        return 'outputs_overview'


class PlotOutputsOnPatientLevel(OutputsPlotter):

    def to_subdir(self):
        return f'{super().to_subdir()}/patient_level'

    def xlabel(self, y_true):
        xlabel = f'{len(y_true)} patients'
        return xlabel

    def plots(self):
        return {
            '': self.plot_outputs_overview_sorted_by_sum,
            # 'name': self.plot_outputs_overview_sorted_by_name,
            'name_grouped': self.plot_outputs_overview_grouped_by_name,
        }

    def analyze_batch(self, batch, y_preds, names):
        y_pred_patient_level, y_true_patient_level, names = self.patient_level_aggregation(names, y_preds, batch[1])
        patient_level_batch = (
            None,
            y_true_patient_level,
            None,
        )
        return super().analyze_batch(patient_level_batch, y_pred_patient_level, names)
