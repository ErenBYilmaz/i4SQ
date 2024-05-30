import logging
import os.path
from typing import List, Union

import numpy
from matplotlib import pyplot

from lib.classification_report import plot_confusion_matrix_with_class_names
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer, tasks_of_any_model
from tasks import VertebraClassificationTask

X = Y = Z = int


class PlotConfusionMatrices(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator,
                 skip_existing=None,
                 binary_threshold_or_method: Union[str, float] = None,
                 model_level_plotting=True):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False,
                         skip_existing=skip_existing,
                         binary_threshold_or_method=binary_threshold_or_method)
        self.model_level_caching = self.skip_existing
        self.model_level_plotting = model_level_plotting

    def to_subdir(self):
        return 'confusion_matrices'

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)
        self.matrices = None

    def before_model(self, model_path: str):
        super().before_model(model_path)
        self.matrices = [self.empty_matrix_for_task(task) for task in self.classification_tasks()]

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()
        xs, ys, sample_weights = batch
        for task_idx, task, matrix in self.task_matrices():
            task: VertebraClassificationTask
            y_true = ys[task_idx]
            y_pred = y_preds[task_idx]
            for sample_idx in range(len(y_true)):
                try:
                    true_cls_idx = task.class_idx_of_label(y_true[sample_idx])
                except task.LabelWithoutClass:
                    continue
                except task.UnlabeledVertebra as e:
                    raise e.ThrownByClassIdxOfLabel()
                pred_cls_idx = self.class_idx_for_model_output_using_precomputed_threshold(y_pred[sample_idx], task)
                if isinstance(pred_cls_idx, numpy.ndarray):
                    pred_cls_idx = pred_cls_idx.item()
                if isinstance(true_cls_idx, numpy.ndarray):
                    true_cls_idx = true_cls_idx.item()
                matrix[true_cls_idx][pred_cls_idx] += 1

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
        self.serializer().create_directory_if_not_exists(to_dir)
        for _, task, matrix in self.task_matrices():
            for normalized in [False, True]:
                base_path = os.path.join(to_dir, f'{"normalized_" if normalized else ""}{task.output_layer_name()}')
                to_files = [base_path + '.png', base_path + '.svg'] if self.model_level_plotting else []
                if self.skip_existing and all(self.serializer().isfile(to_file) for to_file in to_files):
                    continue
                logging.info(f'Writing confusion matrix to {base_path}.*')
                plot_confusion_matrix_with_class_names(
                    matrix,
                    normalize=normalized,
                    class_names=task.class_names(),
                )
                for to_file in to_files:
                    self.serializer().save_current_pyplot_figure(to_file)
                    pyplot.close()
        return super().after_model(self.matrices)

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        matrices_per_task_per_model = results
        tasks = tasks_of_any_model(self.model_files_being_analyzed, self.evaluator, only_tasks=self.ONLY_TASKS).classification_tasks()
        to_dir = os.path.join(self.to_dir(), 'summary')
        self.serializer().create_directory_if_not_exists(to_dir)
        transpose_list_of_lists = lambda l: list(map(lambda *a: list(a), *l))
        matrices_per_model_per_task = transpose_list_of_lists(matrices_per_task_per_model)
        assert len(matrices_per_model_per_task) == len(tasks)
        for task, matrices_per_model in zip(tasks, matrices_per_model_per_task):
            for normalized in [False, True]:
                to_files = [os.path.join(to_dir, f'{"normalized_" if normalized else ""}{task.output_layer_name()}.png'),
                            os.path.join(to_dir, f'{"normalized_" if normalized else ""}{task.output_layer_name()}.svg')]
                for to_file in to_files:
                    if self.skip_existing and self.serializer().isfile(to_file):
                        continue
                    matrix = numpy.sum(matrices_per_model, axis=0)
                    plot_confusion_matrix_with_class_names(
                        matrix,
                        normalize=normalized,
                        class_names=task.class_names(),
                        to_file=to_file,
                    )

        return super().after_multiple_models(results)

    def task_matrices(self):
        tasks = self.classification_tasks()
        task_indices = self.classification_task_indices()
        if len(self.matrices) != len(tasks):
            raise RuntimeError
        if len(self.matrices) != len(task_indices):
            raise RuntimeError
        for matrix, task, task_idx in zip(self.matrices, tasks, task_indices):
            yield task_idx, task, matrix

    def classification_task_indices(self):
        self.check_if_analyzing_model()
        return self.tasks.classification_task_indices()

    def classification_tasks(self):
        self.check_if_analyzing_model()
        return self.tasks.classification_tasks()

    @staticmethod
    def empty_matrix_for_task(task: VertebraClassificationTask):
        n = task.num_classes()
        return numpy.zeros(shape=(n, n), dtype='int')


class PlotConfusionOnPatientLevel(PlotConfusionMatrices):

    def to_subdir(self):
        return f'{super().to_subdir()}/patient_level'

    def analyze_batch(self, batch, y_preds, names):
        y_trues = batch[1]
        y_pred_patient_level, y_true_patient_level, names = self.patient_level_aggregation(names, y_preds, y_trues)
        patient_level_batch = (
            None,
            y_true_patient_level,
            None,
        )
        return super().analyze_batch(patient_level_batch, y_pred_patient_level, names)
