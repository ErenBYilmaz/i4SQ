import logging
import math
import os.path
from math import nan
from numbers import Real
from typing import List, Any, Dict, Set

import numpy
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score

from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer
from tasks import VertebraClassificationTask

X = Y = Z = int

TaskName = str
CurveName = str


class PlotROCCurves(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator, model_level_plotting=True):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, model_level_caching=True)
        self.model_level_plotting = model_level_plotting

    def to_subdir(self):
        return 'roc_curves'

    def before_model(self, model_path: str):
        super().before_model(model_path)
        self.curves: Dict[TaskName, Dict[CurveName, Dict[str, Any]]] = {}

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()
        xs, ys, sample_weights = batch

        for task_idx, task in enumerate(self.tasks):
            if isinstance(task, VertebraClassificationTask):
                for pos_label in range(len(task.class_names())):
                    y_true = []
                    y_score = []
                    ys_for_task = ys[task_idx]
                    y_preds_for_task = y_preds[task_idx]
                    for y, y_pred in zip(ys_for_task, y_preds_for_task):
                        try:
                            true_cls_idx = task.class_idx_of_label(y)
                        except task.LabelWithoutClass:
                            continue
                        y_true.append(float(true_cls_idx == pos_label))
                        y_score.append(float(task.class_probabilities(y_pred)[pos_label]))
                    auc, fpr, thresholds, tpr = self.roc_stats(y_true, y_score, num_patients=len(self.dataset))
                    self.curves.setdefault(task.output_layer_name(), {})[task.class_names()[pos_label]] = {
                        'auc': float(auc),
                        'thresholds': thresholds.tolist(),
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'y_true': y_true,
                        'y_score': y_score,
                        'num_patients': len(self.dataset),
                    }

    @staticmethod
    def roc_stats(y_true, y_score, num_patients: int):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError as e:
            if 'Only one class present' in str(e):
                auc = nan
            else:
                raise
        return auc, fpr, thresholds, tpr

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
        self.serializer().create_directory_if_not_exists(to_dir)
        for task_name, curves in self.curves.items():
            for curve_name, curve in curves.items():
                out_paths = [
                    os.path.join(to_dir, f'{task_name}_{curve_name}.png'),
                    os.path.join(to_dir, f'{task_name}_{curve_name}.svg')
                ] if self.model_level_plotting else []
                if self.skip_existing and all(self.serializer().isfile(out_path) for out_path in out_paths):
                    continue

                self.plot_curve(curve_name, task_name, curve)
                logging.info('Writing ' + os.path.abspath(os.path.join(to_dir, f'{task_name}_{curve_name}.*')))
                for out_path in out_paths:
                    self.serializer().save_current_pyplot_figure(out_path)
                pyplot.close()

        self.serializer().save_json(os.path.join(to_dir, 'roc.json'), data=self.curves)
        return super().after_model(self.curves)

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        results: List[Dict[TaskName, Dict[CurveName, Dict[str, Any]]]]
        to_dir = os.path.join(self.to_dir(), 'summary')
        self.serializer().create_directory_if_not_exists(to_dir)
        task_names: Set[TaskName] = {
            task_name
            for curves_for_model in results
            for task_name in curves_for_model
        }

        summarized_results: Dict[TaskName, Dict[CurveName, Dict[str, Any]]] = {}
        for task_name in task_names:
            curve_names: Set[CurveName] = {
                curve_name
                for curves_for_model in results
                for curve_name in curves_for_model[task_name]
            }
            for curve_name in curve_names:
                curves = [
                    curves_for_model[task_name][curve_name]
                    for curves_for_model in results
                ]
                y_score = sum([curve['y_score'] for curve in curves], [])
                y_true = sum([curve['y_true'] for curve in curves], [])
                num_patients = sum([curve['num_patients'] for curve in curves])
                auc, fpr, thresholds, tpr = self.roc_stats(y_true, y_score, num_patients=num_patients)
                summarized_results.setdefault(task_name, {})[curve_name] = {
                    'auc': float(auc),
                    'thresholds': thresholds.tolist(),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'y_true': y_true,
                    'y_score': y_score,
                    'num_patients': num_patients,
                }
        for task_name, curves in summarized_results.items():
            for curve_name, curve in curves.items():
                out_paths = [
                    os.path.join(to_dir, f'{task_name}_{curve_name}.png'),
                    os.path.join(to_dir, f'{task_name}_{curve_name}.svg')
                ]
                if self.skip_existing and all(self.serializer().isfile(out_path) for out_path in out_paths):
                    continue
                self.plot_curve(curve_name, task_name, curve)
                logging.info('Writing ' + os.path.abspath(os.path.join(to_dir, f'{task_name}_{curve_name}.*')))
                for out_path in out_paths:
                    self.serializer().save_current_pyplot_figure(out_path)
                pyplot.close()

        if not self.skip_existing or not self.serializer().isfile(os.path.join(to_dir, 'roc.json')):
            self.serializer().save_json(os.path.join(to_dir, 'roc.json'), data=summarized_results)
        return super().after_multiple_models(results)

    def plot_curve(self, curve_name, task_name, curve: Dict[str, Any]):
        pyplot.plot(
            curve['fpr'],
            curve['tpr'],
            label=f"ROC curve (area = {curve['auc']:0.2f})",
        )
        loosely_dashed = (0, (5, 10))
        pyplot.plot([0, 1], [0, 1], color="black", linestyle=loosely_dashed)
        pyplot.xlim([-0.03, 1.0])
        pyplot.ylim([0.0, 1.03])
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.title(f"{self.plot_type()} for class {curve_name}")
        t = self.use_threshold(task_name)
        half_threshold_idx = numpy.argmin(numpy.abs(numpy.array(curve['thresholds']) - t)).item()
        if curve['thresholds'][half_threshold_idx] < t:
            half_threshold_idx -= 1
        pyplot.scatter(curve['fpr'][half_threshold_idx], curve['tpr'][half_threshold_idx], color='red', marker='x', s=100, label=f'Operating point θ = {t:.2f}')
        pyplot.legend(loc="lower right")
        pyplot.tight_layout()

    def use_threshold(self, task_name):
        if self.classification_thresholds is None:
            return 0.5
        if self.model_path is None and isinstance(self.binary_threshold_or_method, Real):
            assert math.isfinite(self.binary_threshold_or_method)
            return self.binary_threshold_or_method
        if self.classification_thresholds[self.model_path] is None:
            return 0.5
        if task_name not in self.classification_thresholds[self.model_path]:
            return 0.5
        t = self.classification_thresholds[self.model_path][task_name]
        if t is None:
            return 0.5
        assert math.isfinite(t)
        return t

    def plot_type(self):
        return 'ROC curve'


class PlotROCCurvesOnPatientLevel(PlotROCCurves):

    def to_subdir(self):
        return f'{super().to_subdir()}/patient_level'

    def analyze_batch(self, batch, y_preds, names):
        y_pred_patient_level, y_true_patient_level, names = self.patient_level_aggregation(names, y_preds, batch[1])
        patient_level_batch = (
            None,
            y_true_patient_level,
            None,
        )
        return super().analyze_batch(patient_level_batch, y_pred_patient_level, names)

    def plot_type(self):
        return 'Patient level ROC curve'


class PlotFROCCurves(PlotROCCurves):
    def to_subdir(self):
        return f'froc_curves'

    def roc_stats(self, y_true, y_score, num_patients: int):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fpr = fpr * len(y_true) / num_patients
        auc = nan
        return auc, fpr, thresholds, tpr

    def plot_type(self):
        return 'FROC curve'

    def plot_curve(self, curve_name, task_name, curve: Dict[str, Any]):
        pyplot.plot(
            curve['fpr'],
            curve['tpr'],
        )
        pyplot.plot([0, len(curve['y_true']) / curve['num_patients']], [0, 1], color="black", linestyle="--")
        pyplot.xlim(left=-0.03)
        pyplot.ylim([0.0, 1.03])
        pyplot.xlabel("False Positives per Scan")
        pyplot.ylabel("True Positive Rate")
        pyplot.title(f"{self.plot_type()} for class {curve_name}")
        t = self.use_threshold(task_name)
        half_threshold_idx = numpy.argmin(numpy.abs(numpy.array(curve['thresholds']) - t)).item()
        if curve['thresholds'][half_threshold_idx] < t:
            half_threshold_idx -= 1
        pyplot.scatter(curve['fpr'][half_threshold_idx], curve['tpr'][half_threshold_idx], color='red', marker='x', s=100, label=f'Operating point θ = {t:.2f}')
        pyplot.legend(loc="lower right")
        pyplot.tight_layout()
