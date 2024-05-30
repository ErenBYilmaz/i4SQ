import os.path
import random
import shutil
from math import log
from numbers import Number, Real
from typing import List, Iterable, Any, Optional, Dict, Union, Type, Tuple

import cachetools
import numpy
import sklearn
from cachetools import LRUCache
from sklearn.metrics import roc_curve, precision_recall_curve

from hiwi import ImageList
from lib import dl_backend
from lib.image_processing_tool import TrainableImageProcessingTool
from lib.my_logger import logging
from lib.profiling_tools import dump_pstats_if_profiling
from lib.tuned_cache import TunedMemory
from lib.util import EBC, compute_sample_weights, in_debug_mode
from load_data.update_iml_coordinates import UsingCSVCoordinates
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.evaluation_result_serializer import EvaluationResultSerializer, FileSystemSerializer
from tasks import VertebraClassificationTask, VertebraTasks, VertebraTask, BinaryClassificationTask

results_cache = TunedMemory('.cache')

FAST_MODE = False
if FAST_MODE:
    print('Warning: Invalid results due to FAST_MODE')

SingleModelEvaluationResult = Any


class UseDataSet(EBC):
    def analyze(self, evaluator) -> SingleModelEvaluationResult:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class UseValidationSet(UseDataSet):
    def name(self) -> str:
        return ''

    def analyze(self, evaluator: 'ModelAnalyzer') -> SingleModelEvaluationResult:
        return evaluator.analyze_validation_dataset()


class UseTrainSet(UseDataSet):
    def name(self) -> str:
        return 'train'

    def analyze(self, evaluator: 'ModelAnalyzer') -> SingleModelEvaluationResult:
        return evaluator.analyze_train_dataset()


class UseTestSet(UseDataSet):
    def name(self) -> str:
        return 'test'

    def analyze(self, evaluator: 'ModelAnalyzer') -> SingleModelEvaluationResult:
        return evaluator.analyze_test_dataset()


class UseWholeSet(UseDataSet):
    def __init__(self, dataset_name='whole_dataset'):
        self.dataset_name = dataset_name

    def name(self) -> str:
        return self.dataset_name

    def analyze(self, evaluator: 'ModelAnalyzer') -> SingleModelEvaluationResult:
        return evaluator.analyze_whole_dataset()


class UseSpecificDataset(UseDataSet):
    def name(self):
        return self._name

    def __init__(self, dataset: Union[ImageList, str], name: Optional[str] = None, ):
        if name is None:
            assert isinstance(dataset, str)
            name = os.path.basename(dataset)
        if isinstance(dataset, str):
            dataset = ImageList.load(dataset)
        self.dataset = dataset
        self._name = name

    def analyze(self, evaluator):
        return evaluator.analyze_datasets([self.dataset])


def tasks_of_first_model(model_files, evaluator: FNetParameterEvaluator) -> VertebraTasks:
    config = evaluator.load_config_for_model_evaluation(model_files[0])
    return VertebraTasks.load_from_config(config)


def tasks_of_any_model(model_files, evaluator, only_tasks: Optional[List[str]] = None) -> VertebraTasks:
    tasks = tasks_of_first_model(model_files, evaluator)
    return VertebraTasks([task for task in tasks if only_tasks is None or task.output_layer_name() in only_tasks])


class ModelAnalyzer(EBC):
    use_dataset: UseDataSet = UseValidationSet()
    BASE_IMG_DIR = 'img/generated'
    DISABLE_INTERMEDIATE_ANALYSES: bool = False
    RANDOM_MODEL_ORDER = False
    RANDOM_SUBDIRECTORY_ORDER = False
    SKIP_EXISTING_BY_DEFAULT = False
    EXCLUDE_SITE_6 = False
    DEFAULT_THRESHOLD: Union[str, Number] = 0.5
    IGNORE_EXCEPTIONS: Tuple[Type[Exception], ...] = ()
    ONLY_TASKS: Optional[List[str]] = None
    RESULT_SERIALIZER: EvaluationResultSerializer = FileSystemSerializer()

    def __init__(self, evaluator: FNetParameterEvaluator, model_level_caching=False,
                 analysis_needs_xs=True, coordinate_csv_path: Optional[str] = None,
                 coordinate_csv_name=None,
                 analysis_needs_ground_truth_ys=True,
                 skip_existing=None, binary_threshold_or_method: Union[str, float] = None):
        if binary_threshold_or_method is None:
            binary_threshold_or_method = self.DEFAULT_THRESHOLD
        self.binary_threshold_or_method = binary_threshold_or_method
        self.coordinate_csv_path = coordinate_csv_path
        self.coordinate_csv_name = coordinate_csv_name
        self.analysis_needs_xs = analysis_needs_xs
        self.analysis_needs_ground_truth_ys = analysis_needs_ground_truth_ys
        self.model_level_caching = model_level_caching
        self.analyzing_dataset = False
        self.analyzing_model = False
        self.analyzing_model_directory = False
        self.root_directory = None
        if skip_existing is None:
            skip_existing = self.SKIP_EXISTING_BY_DEFAULT
        self.skip_existing = skip_existing
        self.evaluator = evaluator
        self.classification_thresholds: Dict[str, Dict[str, float]] = {}
        self._subdir_info_printed = False
        self.model_path: Optional[str] = None
        self.model: Optional[TrainableImageProcessingTool] = None

    # noinspection PyAttributeOutsideInit
    def before_model(self, model_path: str, ):
        self.analyzing_model = True
        self.model_path = model_path
        self.config = self.load_model_config_for_evaluation()
        self.process_config(self.config)
        self.evaluator.configure(self.config)
        self.tasks: VertebraTasks = VertebraTasks.load_from_config(self.config)
        self.main_task: VertebraClassificationTask = self.tasks.main_task()
        self.last_analysis_results = None
        self.classification_thresholds[model_path] = {}
        for task_idx, task in enumerate(self.tasks):
            key = self.model_level_cache_key()
            threshold = type(self).compute_classification_threshold_for_model(self, task_idx, _cache_key=key)
            self.classification_thresholds[model_path][task.output_layer_name()] = threshold

    def process_config(self, config):
        config['exclude_site_6'] = config['exclude_site_6'] or self.EXCLUDE_SITE_6

    def to_dir(self, short=False, subdir: str = None) -> str:
        if subdir is None:
            if hasattr(self, 'model_dir'):
                subdir = os.path.relpath(self.model_dir, "models")
            else:
                subdir = ''
        return os.path.join(
            '' if short else self.BASE_IMG_DIR,
            '' if short else subdir,
            self.evaluator.data_source.name()
            + ('_exclude_6' if self.EXCLUDE_SITE_6 else '')
            + (f'_threshold_{self.binary_threshold_or_method}' if self.binary_threshold_or_method != 0.5 else '')
            + ('_' + self.coordinate_csv_name if self.coordinate_csv_name is not None else ''),
            self.use_dataset.name(),
            self.to_subdir()
        )

    def to_subdir(self):
        raise NotImplementedError

    def test_set(self):
        self.check_if_analyzing_model()
        return self.evaluator.test_set_as_iml()

    def train_set(self):
        self.check_if_analyzing_model()
        return self.evaluator.training_set_as_iml()

    def validation_set(self):
        self.check_if_analyzing_model()
        return self.evaluator.validation_set_as_iml()

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        self.analyzing_model = False
        self.model_path = None
        self.model = None
        dl_backend.b().memory_leak_cleanup()
        self.serializer().upsert()
        return results

    # noinspection PyAttributeOutsideInit
    def before_dataset(self, dataset: ImageList):
        self.analyzing_dataset = True
        self.dataset = dataset
        if self.coordinate_csv_path is not None:
            self.using_other_coordinates = UsingCSVCoordinates(iml=self.dataset,
                                                               csv_path=self.coordinate_csv_path,
                                                               on_new_coordinates='add',
                                                               on_missing_coordinates='invalidate',
                                                               coordinate_sources='all', )
            self.using_other_coordinates.__enter__()

    def after_dataset(self) -> SingleModelEvaluationResult:
        if self.coordinate_csv_path is not None:
            self.using_other_coordinates.__exit__(None, None, None)
        self.analyzing_dataset = False

    def before_model_directory(self, model_dir: str):
        logging.info(f'{self.name()}: Analyzing models in {model_dir} ...')
        self.analyzing_model_directory = True
        self.model_dir = model_dir

    def after_model_directory(self, results: List[SingleModelEvaluationResult]):
        self.analyzing_model_directory = False
        del self.model_dir
        self.serializer().upsert()
        return results

    def before_multiple_models(self, model_files):
        self.model_files_being_analyzed = model_files

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        dump_pstats_if_profiling(ModelAnalyzer)
        return results

    def serializer(self):
        return self.RESULT_SERIALIZER

    def analyze_batch(self, batch, y_preds, names):
        raise NotImplementedError

    def analyze_validation_dataset(self) -> SingleModelEvaluationResult:
        return self.analyze_datasets([self.validation_set()])[0]

    def analyze_test_dataset(self) -> SingleModelEvaluationResult:
        return self.analyze_datasets([self.test_set()])[0]

    def analyze_train_dataset(self) -> SingleModelEvaluationResult:
        return self.analyze_datasets([self.train_set()])[0]

    def analyze_dataset(self, dataset: ImageList) -> SingleModelEvaluationResult:
        self.check_if_analyzing_model()
        self.before_dataset(dataset)
        if len(dataset) > 0:
            batch, names, y_preds = self.get_batch_and_apply_model()
            batch_result = self.analyze_batch(batch, y_preds, names)
        else:
            batch_result = None
        dataset_result = self.after_dataset()
        if dataset_result is not None:
            return dataset_result
        elif batch_result is not None:
            return batch_result
        else:
            return None

    def get_batch_and_apply_model(self):
        self.check_if_analyzing_dataset()
        return type(self)._cached_model_application(self,
                                                    self.dataset,
                                                    classification_thresholds=self.classification_thresholds[
                                                        self.model_path],
                                                    _cache_key=self.model_application_cache_key())

    def model_application_cache_key(self,
                                    dataset: ImageList = None,
                                    data_source_name: str = None,
                                    use_dataset_name: str = None,
                                    binary_threshold_or_method: Union[str, float] = None):
        if dataset is None:
            dataset = self.dataset
        if binary_threshold_or_method is None:
            binary_threshold_or_method = self.binary_threshold_or_method
        if data_source_name is None:
            data_source_name = self.evaluator.data_source.name()
        if use_dataset_name is None:
            use_dataset_name = self.use_dataset.name()
        return (self.model_path,
                [v for image in dataset for v in image.parts],
                dataset.name if hasattr(dataset, 'name') else None,
                self.analysis_needs_xs,
                self.analysis_needs_ground_truth_ys,
                use_dataset_name,
                data_source_name,
                binary_threshold_or_method,
                self.EXCLUDE_SITE_6)

    @results_cache.cache(ignore=['self'], verbose=0)
    def compute_classification_threshold_for_model(self, task_idx: int, _cache_key) -> float:
        dataset = self.validation_set()
        task = self.tasks[task_idx]
        if len(dataset) == 0:
            # raise RuntimeError('Can not estimate classification threshold on empty validation dataset.')
            y_pred = None
            y_true = None
        else:
            k = self.model_application_cache_key(dataset=dataset,
                                                 use_dataset_name=UseValidationSet().name(),
                                                 binary_threshold_or_method=0.5)
            batch, names, y_preds = type(self)._cached_model_application(self,
                                                                         dataset,
                                                                         classification_thresholds=None,
                                                                         _cache_key=k)
            xs, ys, sample_weights = batch

            y_pred = y_preds[task_idx]
            y_true = ys[task_idx]
        return self.compute_classification_threshold(task, y_pred, y_true)

    def class_idx_for_model_output_using_precomputed_threshold(self, model_output, task: VertebraTask):
        threshold = self.classification_thresholds[self.model_path][task.output_layer_name()]
        return self.class_idx_for_model_output_using_threshold(model_output, task, threshold)

    def class_name_for_model_output_using_precomputed_threshold(self, model_output, task: VertebraTask) -> str:
        cls_idx = self.class_idx_for_model_output_using_precomputed_threshold(model_output, task)
        cls_idx = int(cls_idx)
        return task.class_names()[cls_idx]

    @staticmethod
    def class_idx_for_model_output_using_threshold(model_output, task: VertebraTask, threshold: Real):
        if isinstance(task, BinaryClassificationTask):
            assert isinstance(threshold, Real)
            pred_cls_idx = task.class_idx_for_model_output_using_threshold(model_output, threshold)
        else:
            pred_cls_idx = task.nearest_class_idx_for_model_output(model_output)
        return pred_cls_idx

    def load_trained_model_for_evaluation(self,
                                          include_artificial_outputs=True,
                                          thresholds_for_artificial_outputs: Dict[
                                              str, float] = None) -> TrainableImageProcessingTool:
        model = self.evaluator.load_trained_model_for_evaluation(self.model_path,
                                                                 include_artificial_outputs=include_artificial_outputs,
                                                                 thresholds_for_artificial_outputs=thresholds_for_artificial_outputs)
        model = model.copy_with_filtered_outputs(self.ONLY_TASKS)
        return model

    def load_model_config_for_evaluation(self,
                                         include_artificial_outputs=True,
                                         thresholds_for_artificial_outputs: Dict[str, float] = None) -> Dict[str, Any]:
        config = self.evaluator.load_config_for_model_evaluation(self.model_path,
                                                                 include_artificial_outputs=include_artificial_outputs,
                                                                 thresholds_for_artificial_outputs=thresholds_for_artificial_outputs)
        return self.evaluator.model_type_from_file(self.model_path).filter_tasks(config, self.ONLY_TASKS)

    @results_cache.cache(ignore=['self', 'dataset', 'classification_thresholds'], verbose=0)
    def _cached_model_application(self, dataset, classification_thresholds: Optional[Dict[str, float]], _cache_key):
        self.model = self.load_trained_model_for_evaluation(include_artificial_outputs=True,
                                                            thresholds_for_artificial_outputs=classification_thresholds)
        config = self.model.config
        self.process_config(config)
        if config['tasks'] != self.config['tasks']:
            # probably because other classification thresholds were used for output combinations
            self.config['tasks'] = config['tasks']
        assert config == self.config
        if self.analysis_needs_ground_truth_ys:
            generator_generator = self.evaluator.validation_generator_from_config
        else:
            generator_generator = self.evaluator.prediction_generator_from_config
        data_generator = generator_generator(dataset,
                                             config=self.config,
                                             cache_batches=False,
                                             batch_size=self.model.validation_batch_size,
                                             ndim=self.evaluator.data_source.ndim(), )
        logging.info(f'Applying model {self.model_path} ...')
        y_preds, names, xs, ys, sample_weights = self.model.predict_generator_returning_inputs(data_generator,
                                                                                               steps=data_generator.steps_per_epoch())
        assert names == data_generator.all_vertebra_names()
        assert len(xs[0]) == len(names)
        if isinstance(y_preds, numpy.ndarray):
            assert len(data_generator.tasks) == 1
            y_preds = [y_preds]
        if self.analysis_needs_xs:
            batch = (xs, ys, sample_weights)
        else:
            batch = (None, ys, sample_weights)
        assert len(y_preds) == len(data_generator.tasks) == self.model.num_outputs() == len(self.tasks)
        for y_pred, task in zip(y_preds, data_generator.tasks):
            task: VertebraTask
            for sample_idx in range(y_pred.shape[0]):
                if numpy.isnan(y_pred[sample_idx]).any():
                    y_pred[sample_idx] = task.default_label()
        return batch, names, y_preds

    def name(self):
        return type(self).__name__

    @results_cache.cache(ignore=['self'], verbose=0)
    def _cached_model_analysis(self, model_path, _cache_key):
        self.before_model(model_path)
        results = self.use_dataset.analyze(self)
        return self.after_model(results)

    def analyze_whole_dataset(self):
        return self.analyze_datasets([self.evaluator.whole_dataset()])[0]

    def main_task_idx(self):
        self.check_if_analyzing_model()
        return self.tasks.index(self.main_task)

    def example_model_path(self):
        return self.model_files(self.model_dir)[0]

    def analyze_datasets(self, datasets: Iterable[ImageList]) -> List[SingleModelEvaluationResult]:
        results: List[SingleModelEvaluationResult] = []
        for dataset in datasets:
            results.append(self.analyze_dataset(dataset))
        return results

    def check_if_analyzing_model(self):
        if not self.analyzing_model:
            raise RuntimeError

    def check_if_analyzing_dataset(self):
        if not self.analyzing_dataset:
            raise RuntimeError

    def analyze_test_and_validation_datasets(self) -> List[SingleModelEvaluationResult]:
        return self.analyze_datasets([self.validation_set(), self.test_set()])

    def analyze_test_plus_validation_dataset(self) -> List[SingleModelEvaluationResult]:
        return self.analyze_datasets([self.validation_set() + self.test_set()])

    def analyze_multiple_models(self, model_files: List[str]) -> List[SingleModelEvaluationResult]:
        self.before_multiple_models(model_files)
        results = []
        if self.RANDOM_MODEL_ORDER:
            model_files = model_files.copy()
            logging.info('Analyzing models in random order.')
            random.shuffle(model_files)
        for model_idx, model_path in enumerate(model_files):
            # logging.info(f'Analyzing model {model_idx + 1} of {len(model_files)}: {model_path} ...')
            results.append(self.analyze_single_model(model_path))
            logging.info(f'({self.name()}): Done analyzing model {model_idx}: {model_path}')
            if not self.DISABLE_INTERMEDIATE_ANALYSES and round(log(model_idx + 1, 2)) == log(model_idx + 1,
                                                                                              2) and model_idx > 20:
                # [31,   63,  127,  255,  511, 1023, 2047, 4095, 8191, ...]
                self.after_multiple_models(results)

        return self.after_multiple_models(results)

    def analyze_single_model(self, model_path: str, ) -> SingleModelEvaluationResult:
        if self.model_level_caching:
            return type(self)._cached_model_analysis(self, model_path,
                                                     _cache_key=self.model_level_cache_key(model_path))
        else:
            return type(self)._cached_model_analysis.func(self, model_path,
                                                          _cache_key=self.model_level_cache_key(model_path))

    def model_level_cache_key(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = self.model_path
        return (
            type(self),
            type(self.evaluator),
            self.use_dataset.name(),
            self.evaluator.data_source.name(),
            self.coordinate_csv_path,
            self.EXCLUDE_SITE_6,
            self.binary_threshold_or_method,
            model_path
        )

    def directory_level_cache_key(self, model_dir: str):
        return (
            type(self).__name__,
            self.to_dir(short=False, subdir=model_dir)
        )

    @staticmethod
    def model_files(model_dir):
        return [os.path.join(model_dir, model_file) for model_file in os.listdir(model_dir) if
                model_file.endswith('.h5')]

    def analyze_directory(self, model_dir: str) -> Any:
        if self.skip_existing:
            analysis_method = type(self)._cached_directory_analysis
        else:
            analysis_method = type(self)._cached_directory_analysis.func

        return analysis_method(self,
                               model_dir,
                               _cache_key=self.directory_level_cache_key(model_dir))

    @results_cache.cache(ignore=['self', 'model_dir'], verbose=0)
    def _cached_directory_analysis(self, model_dir, _cache_key):
        try:
            self.before_model_directory(model_dir)
            results = self.analyze_multiple_models(self.model_files(model_dir))
            results = self.after_model_directory(results)
            logging.info(f'Done with directory analysis: {self.directory_level_cache_key(model_dir)}')
            return results
        except self.IGNORE_EXCEPTIONS:
            logging.info(f'Skipping this evaluation due to one of {[e.__name__ for e in self.IGNORE_EXCEPTIONS]}')
            return

    def before_multiple_directories(self, model_dir: str):
        self.root_directory = model_dir

    def after_multiple_directories(self, results: Dict[str, List[List[SingleModelEvaluationResult]]]) -> Dict[
        str, List[List[SingleModelEvaluationResult]]]:
        self.root_directory = None
        return results

    def analyze_subdirectories(self, model_dir) -> Dict[str, List[List[SingleModelEvaluationResult]]]:
        if (self.coordinate_csv_path is None) != (self.coordinate_csv_name is None):
            raise RuntimeError(
                f'Only one of coordinate_csv_path and coordinate_csv_name was set: Either both or none are needed: '
                f'name={self.coordinate_csv_name} vs path={self.coordinate_csv_path}')
        results = {}
        os_walk_model_dir = list(os.walk(model_dir))
        if self.RANDOM_SUBDIRECTORY_ORDER and not self._subdir_info_printed:
            if len(os_walk_model_dir) > 1:
                self._subdir_info_printed = True
                logging.info('Analyzing subdirectories in random order.')
        for dir_path, sub_dirs, files in os_walk_model_dir:
            if self.RANDOM_SUBDIRECTORY_ORDER:
                random.shuffle(sub_dirs)
            if len(files) > 0:
                results[dir_path] = self.analyze_directory(dir_path)
        return self.after_multiple_directories(results)

    def patient_level_aggregation(self, names, y_pred, y_true):
        assert len(y_pred) == len(y_true)
        for y1, y2 in zip(y_true, y_pred):
            assert len(y1) == len(y2)
        patient_ids = list(split_name(name)[0] for name in names)
        patient_indices_in_restricted_batch = {
            p: [sample_idx for sample_idx, patient_id in enumerate(patient_ids)
                if patient_id == p]
            for p in set(patient_ids)
        }
        patient_indicess = list(patient_indices_in_restricted_batch.values())
        assert len(y_pred) == len(self.tasks), len(self.tasks)
        assert len(y_true) == len(self.tasks), len(self.tasks)
        task: VertebraTask
        y_pred_patient_level = [numpy.concatenate([task.patient_level_aggregation(y[patient_indices])
                                                   for patient_indices in patient_indicess], axis=0)
                                for y, task in zip(y_pred, self.tasks)]
        y_true_patient_level = [numpy.concatenate([task.patient_level_aggregation(y[patient_indices])
                                                   for patient_indices in patient_indicess], axis=0)
                                for y, task in zip(y_true, self.tasks)]
        patient_names = []
        for patient_indices in patient_indicess:
            p, _ = split_name(names[patient_indices[0]])
            patient_names.append(str((p, '')))
        assert len(y_pred_patient_level[0]) == len(y_true_patient_level[0]) == len(patient_names)
        assert len(patient_names) == len(set(patient_names))
        return y_pred_patient_level, y_true_patient_level, patient_names

    @classmethod
    @cachetools.cached(cachetools.LRUCache(500),
                       key=lambda cls, y_true, y_pred, human_readable: (
                       str(y_true), str(y_pred), y_pred.shape, y_pred.sum(), human_readable))
    def binary_classification_thresholds(cls, y_true, y_pred, human_readable) -> Dict[str, float]:
        if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
            y_true = y_true[:, 0]
        if len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
            y_pred = y_pred[:, 0]
        if len(numpy.unique(y_true, axis=0)) <= 1 or len(numpy.unique(y_pred, axis=0)) <= 1:
            return {}
        fprs, tprs, roc_thresholds = roc_curve(y_true, y_pred)
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred)

        f1_scores = 2 * precisions * recalls / (precisions + recalls)
        unweighted_accuracies = (y_true == (y_pred >= roc_thresholds[:, numpy.newaxis])).mean(axis=1)

        sample_weights = compute_sample_weights(y_true)
        assert numpy.count_nonzero(sample_weights) == sample_weights.size
        svm = sklearn.svm.LinearSVC(C=1)
        svm.fit(y_pred[..., numpy.newaxis], y_true, sample_weight=sample_weights)
        svm_based_threshold = -svm.intercept_.item() / svm.coef_.item()

        results = {}
        for term_name, human_readable_name, term, base_array in [
            ('f1', 'F1-score', f1_scores, pr_thresholds),
            ('acc', 'Unweighted Accuracy', unweighted_accuracies, roc_thresholds),
            ('sens_spec_prod', 'Sens-Spec-Product', (1 - fprs) * tprs, roc_thresholds),
            ('youden', 'Youden\'s J', (1 - fprs) + tprs - 1, roc_thresholds),
            ('svm', 'SVM "score"', [svm.score(y_pred[..., numpy.newaxis], y_true, sample_weights)],
             [svm_based_threshold]),
        ]:
            if human_readable:
                results[
                    f'{human_readable_name} of {numpy.max(term):.3g} at θ = {base_array[numpy.argmax(term)]:.3g}'] = float(
                    base_array[numpy.argmax(term)])
            else:
                results[term_name] = float(base_array[numpy.argmax(term)])
        return results

    @classmethod
    def one_vs_all_classification_thresholds(cls, class_idx, y_pred_all_classes, y_true_all_classes,
                                             human_readable=False):
        y_pred = y_pred_all_classes[..., class_idx]
        y_true = y_true_all_classes[..., class_idx]
        thresholds = cls.binary_classification_thresholds(y_true, y_pred, human_readable=human_readable)
        return thresholds

    def compute_classification_threshold(self, task, y_pred, y_true) -> Optional[Number]:
        if not isinstance(task, BinaryClassificationTask):
            return None
        if isinstance(self.binary_threshold_or_method, Number):
            return self.binary_threshold_or_method
        return self.binary_classification_thresholds(y_true, y_pred, human_readable=False)[
            self.binary_threshold_or_method]

    @classmethod
    def clear_caches(cls):
        cache_path = os.path.join(results_cache.location, 'joblib', 'model', 'fnet', 'model_analysis')
        assert os.path.isdir(cache_path)
        logging.warning(f'Flushing cache in {cache_path}')
        shutil.rmtree(cache_path)

    @classmethod
    def clear_directory_level_cache(cls):
        cache_path = os.path.join(results_cache.location, 'joblib', 'model', 'fnet', 'model_analysis',
                                  'analyze_trained_models', '_cached_directory_analysis')
        assert os.path.isdir(cache_path)
        logging.warning(f'Flushing cache in {cache_path}')
        shutil.rmtree(cache_path)

    @classmethod
    def clear_directory_level_cache_unless_debugging(cls):
        if in_debug_mode():
            logging.info(
                'Not clearing directory level analysis cache: In debug mode we probably want to step into the code '
                'and it would be annoying to have to wait for the stuff that already ran through')
        else:
            cls.clear_directory_level_cache()


@cachetools.cached(cache=LRUCache(maxsize=30000), key=lambda name: name)
def split_name(name):
    patient_id, name = eval(name)
    return patient_id, name
