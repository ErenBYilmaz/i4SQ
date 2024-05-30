import logging
import math
import os
from abc import abstractmethod
from contextlib import redirect_stdout
from typing import Optional, List, Dict, TYPE_CHECKING, Tuple

from lib import dl_backend
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator

if TYPE_CHECKING:
    from typing import Literal
    Split = Literal['train', 'test', 'val']
else:
    Literal = Split = None
import dill


import h5py
import lifelines.exceptions
import numpy
import pandas
from lifelines import CoxPHFitter

from load_data.load_image import image_by_patient_id

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, split_name, UseValidationSet, UseTestSet, SingleModelEvaluationResult, UseTrainSet, UseDataSet, results_cache
from tasks import BinaryClassificationTask, FNetRegressionTask, VertebraTask, OutputCombination, MulticlassVertebraTask


class VertebraTaskFilter:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def include_task(self, task: VertebraTask):
        raise NotImplementedError('Abstract method')


class AllTasksFilter(VertebraTaskFilter):
    def __init__(self):
        super().__init__(name='all')

    def include_task(self, task: VertebraTask):
        return True


class NonArtificialTasksFilter(VertebraTaskFilter):
    def __init__(self):
        super().__init__(name='nonart')

    def include_task(self, task: VertebraTask):
        return not isinstance(task, OutputCombination)


class NoTasksFilter(VertebraTaskFilter):
    def __init__(self):
        super().__init__(name='no_outputs')

    def include_task(self, task: VertebraTask):
        return False


class BinaryClassificationTasksFilter(VertebraTaskFilter):
    def __init__(self):
        super().__init__(name='binary_classification')

    def include_task(self, task: VertebraTask):
        return isinstance(task, BinaryClassificationTask)


class SpecificTaskFilter(VertebraTaskFilter):
    def __init__(self, output_name: str):
        super().__init__(name=output_name)

    def include_task(self, task: VertebraTask):
        return task.output_layer_name() == self.name


class AndFilter(VertebraTaskFilter):
    def __init__(self, filters: List[VertebraTaskFilter]):
        super().__init__(name='_'.join(f.name for f in filters))
        self.filters = filters

    def include_task(self, task: VertebraTask):
        return all(f.include_task(task) for f in self.filters)


class ExportOutputsForSurvivalAnalysis(ModelAnalyzer):
    class NoPredictors(ValueError):
        pass

    IGNORE_EXCEPTIONS = (NoPredictors,)

    def __init__(self, evaluator: FNetParameterEvaluator,
                 patient_level_binary_prognosis_task_for_event_flag: BinaryClassificationTask,
                 patient_level_tte_task: FNetRegressionTask,
                 task_exclusion_filter: VertebraTaskFilter = None,
                 extra_input_tasks: List[VertebraTask] = None,
                 skip_existing=None,
                 drop_first_class_values=True,
                 standardize_features=True,
                 group_by_patient=False,
                 patient_level_aggregation_method='mean'):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, skip_existing=skip_existing,
                         binary_threshold_or_method=0.5)
        if task_exclusion_filter is None:
            task_exclusion_filter = AllTasksFilter()
        if extra_input_tasks is None:
            extra_input_tasks = []
        self.model_level_caching = self.skip_existing
        self.patient_level_binary_prognosis_task_for_event_flag = patient_level_binary_prognosis_task_for_event_flag
        self.patient_level_tte_task = patient_level_tte_task
        self.task_exclusion_filter = task_exclusion_filter
        self.extra_input_tasks = extra_input_tasks
        self.drop_first_class_values = drop_first_class_values
        self.standardize_features = standardize_features
        self.group_by_patient = group_by_patient
        self.patient_level_aggregation_method = patient_level_aggregation_method
        self._cached_few_step_generator: Optional[AnnotatedPatchesGenerator] = None

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)

    def before_model(self, model_path: str, ):
        super().before_model(model_path)
        self.model = self.load_trained_model_for_evaluation()

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        del self.model
        self.clear_cached_generator()
        return super().after_model(results)

    def update_h5_file(self, h5_file: str, y_preds: List[numpy.ndarray], names: List[str]):
        kilo_bytes = 1024
        if self.skip_existing and os.path.isfile(h5_file) and os.path.getsize(h5_file) >= 4 * kilo_bytes:
            with h5py.File(h5_file, 'r') as f:
                if self.h5_subset() in f:
                    return
        if len(names) != len(y_preds[0]):
            raise ValueError(f'len(names)={len(names)} != len(y_pred[0])={len(y_preds[0])}')

        x_components = self.x_component_names(y_preds)
        if len(x_components) == 0:
            raise self.NoPredictors()
        df = self.df_with_model_outputs_and_extra_inputs(y_preds, names, x_components)
        try:
            self.save_survival_dataset_to_h5_file(df, h5_file, x_components)
        except OSError as e:
            if 'Resource temporarily unavailable' in str(e):
                # probably some other process is writing
                assert os.path.isfile(h5_file)
            else:
                raise

    def save_survival_dataset_to_h5_file(self, df, h5_file_path, x_components):
        split = self.h5_subset()
        h5_file_path = os.path.abspath(h5_file_path)
        with h5py.File(h5_file_path, 'a') as f:
            if split not in f:
                f.create_group(split)
            assert split in f
            self.create_or_replace_dataset(f[split], 'x', numpy.array(df[x_components]))
            self.create_or_replace_dataset(f[split], 'e', df['e'])
            self.create_or_replace_dataset(f[split], 't', df['t'])
            self.create_or_replace_dataset(f[split], 'names', numpy.array(df['patient_id'], dtype='S') if self.group_by_patient else numpy.array(df['name'], dtype='S'))
            self.create_or_replace_dataset(f[split], 'x_components', numpy.array(x_components, dtype='S'))
            assert not numpy.isnan(f[split]['x']).any()
            assert not numpy.isnan(f[split]['e']).any()
            assert not numpy.isnan(f[split]['t']).any()
            assert len(x_components) == f[split]['x'].shape[-1]
            df_for_cox = self.df_for_cox_from_h5_file(f, split)
        logging.info(f'Updated {h5_file_path}')
        # self.analyze_with_cox(df_for_cox, f'{h5_file_path}_{split}')

    def df_with_model_outputs_and_extra_inputs(self, y_preds, names, x_components):
        df_data_as_list = []
        model_outputs = []
        for sample_idx in range(len(names)):
            name = names[sample_idx]
            patient_id, vertebra_name = split_name(name)
            img = image_by_patient_id(patient_id, self.dataset)
            try:
                tte = self.patient_level_tte_task.y_true_from_hiwi_image_and_vertebra(img, vertebra_name).item()
            except VertebraTask.UnlabeledVertebra:
                tte = math.nan
            try:
                event_flag = self.patient_level_binary_prognosis_task_for_event_flag.y_true_from_hiwi_image_and_vertebra(img, vertebra_name).item()
            except VertebraTask.UnlabeledVertebra:
                tte = math.nan
            flat_outputs = [y_preds[task_idx][sample_idx].flat for task_idx in range(len(self.tasks_to_export()))]
            if math.isnan(tte) or math.isnan(event_flag):
                continue
            if self.drop_first_class_values:
                flat_outputs = [flat_outputs[task_idx][1 if self.drop_first_class_of_task(task_idx) else 0:] for task_idx
                                in range(len(self.tasks_to_export()))]
            for extra_input in self.extra_input_tasks:
                flat_outputs.append(extra_input.y_true_or_default_from_hiwi_image_and_vertebra(img, vertebra_name).flat)
            flat_outputs = numpy.concatenate(flat_outputs)
            assert len(x_components) == len(flat_outputs), (len(x_components), len(flat_outputs))
            df_data_as_list.append({'name': name, 'e': event_flag, 't': tte, 'patient_id': patient_id,
                                    **{x_components[i]: flat_outputs[i] for i in range(max(len(x_components), len(flat_outputs)))}})
            model_outputs.append(flat_outputs)
        df = pandas.DataFrame(df_data_as_list, columns=['name', 'e', 't', 'patient_id', *x_components])
        if self.group_by_patient:
            df = df.groupby('patient_id').agg(self.patient_level_aggregation_method).reset_index()
            assert 1 <= len(df['e'].unique()) <= 2, df['e'].unique()
            if len(df['e'].unique()) == 1:
                logging.warning(f'Only one class in split: {df["e"].unique()}')
            assert (df['e'].round() == df['e']).all(), df['e'].unique()
        if self.standardize_features:
            for feature in x_components:
                # https://www.researchgate.net/post/What-is-the-difference-between-hazard-ratio-standard-deviation-andHRs-Can-we-convert-HR-SD-to-HR-Can-we-combine-HR-SD-and-Hazard-Ratios
                df[feature] = (df[feature] - df[feature].mean(axis=0)) / df[feature].std(axis=0)
        return df

    def x_component_names(self, y_preds) -> List[bytes]:
        x_components = []
        for task_idx, task in enumerate(self.tasks_to_export()):
            for output_idx in range(len(y_preds[task_idx][0])):
                if output_idx == 0 and self.drop_first_class_of_task(task_idx):
                    continue
                suffix = f'_{output_idx}' if len(y_preds[task_idx][0]) > 1 else ''
                x_components.append(f'{task.output_layer_name()}{suffix}'.encode('ascii'))
        for extra_input in self.extra_input_tasks:
            x_components.append(f'{extra_input.output_layer_name()}'.encode('ascii'))
        return x_components

    def analyze_with_cox(self, df, output_name):
        if df['e'].sum() > 1:
            cph = CoxPHFitter()
            try:
                cph.fit(df, duration_col='t', event_col='e', show_progress=True)
            except (numpy.linalg.LinAlgError, lifelines.exceptions.ConvergenceError) as e:
                logging.warning(f'{type(e).__name__} when trying to fit cox model for {output_name}: {e}')
                return
            with open(f'{output_name}_cox.log', 'a') as f:
                with redirect_stdout(f):
                    cph.print_summary()
            # cph.print_summary()
            summary: pandas.DataFrame = cph.summary
            summary.to_csv(f'{output_name}_cox.csv')
            with open(f'{output_name}_cox.md', 'w') as f:
                summary.to_markdown(f)
            with open(f'{output_name}_cox.dill', 'wb') as f:
                dill.dump(cph, f)
        else:
            logging.warning(f'Not enough data for CoxPHFitter. Skipping analysis for {output_name}')

    def name(self):
        return f'{super().name()} -> {self.base_file_name()}'

    def df_for_cox_from_h5_file(self, f: h5py.File, split: Split):
        return pandas.DataFrame(
            {
                'e': f[split]['e'][:],
                't': f[split]['t'][:],
                **{
                    column_name.decode('ascii'):
                        (((column_data - column_data.mean()) / column_data.std())
                         if self.standardize_features
                         else column_data)
                    for column_idx, column_name in enumerate(f[split]['x_components'])
                    for column_data in [f[split]['x'][..., column_idx]]
                    if numpy.unique(column_data).size > 1  # drop constant columns
                },
            }
        )

    def drop_first_class_of_task(self, task_idx):
        return (self.drop_first_class_values and isinstance(self.tasks[task_idx], MulticlassVertebraTask))

    def tasks_to_export(self) -> List[VertebraTask]:
        return [task for task in self.tasks if self.task_exclusion_filter.include_task(task)]

    def create_or_replace_dataset(self, container, key, data):
        if key in container:
            del container[key]
        container.create_dataset(key, data=data)

    def concatenate_with_existing_data(self, container, key, new_data):
        if key in container:
            data = numpy.concatenate([(container[key][()]), new_data], axis=0)
        else:
            data = new_data
        return data

    def h5_subset(self):
        subset_map = {
            UseTrainSet().name(): 'train',
            UseValidationSet().name(): 'val',
            UseTestSet().name(): 'test'
        }
        subset = subset_map[self.use_dataset.name()]
        return subset

    def clear_cached_generator(self):
        self._cached_few_step_generator = None
        dl_backend.b().memory_leak_cleanup()

    def few_step_prediction_generator_including_unlabeled_examples(self, cache_vertebra_volumes_in_ram=False):
        if self._cached_few_step_generator is None:
            self._cached_few_step_generator = self.evaluator.prediction_generator_from_config(self.dataset,
                                                                                              self.config,
                                                                                              batch_size=self.model.validation_batch_size,
                                                                                              cache_vertebra_volumes_in_ram=cache_vertebra_volumes_in_ram,
                                                                                              exclude_unlabeled=False,
                                                                                              ndim=self.evaluator.data_source.ndim())
        assert self._cached_few_step_generator.steps_per_epoch() <= 100, self._cached_few_step_generator.steps_per_epoch()
        assert not self._cached_few_step_generator.random_mode
        assert self._cached_few_step_generator.cache_vertebra_volumes_in_ram == cache_vertebra_volumes_in_ram
        return self._cached_few_step_generator


    def analyze_batch(self, batch, y_preds, names):
        # before = (y_preds, names)
        del batch
        y_preds, names = self.prediction_results()

        self.check_if_analyzing_dataset()
        model_name: str = os.path.basename(self.model_path)
        to_file = self.to_file(model_name)
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        self.update_h5_file(h5_file=to_file, y_preds=y_preds, names=names)
        return to_file

    def prediction_results(self, dont_cache=False):
        f = type(self)._cached_prediction_of_model
        if dont_cache:
            f = f.func
        return f(self, _cache_key=(super().model_level_cache_key()))

    @results_cache.cache(ignore=['self'], verbose=0)
    def _cached_prediction_of_model(self, _cache_key) -> Tuple[List[numpy.ndarray], List[str]]:
        data_generator = self.few_step_prediction_generator_including_unlabeled_examples()
        assert data_generator.image_list is self.dataset
        model_prediction_results = self.model.predict_generator(data_generator, steps=data_generator.steps_per_epoch())
        dataset_names = data_generator.all_vertebra_names()
        assert len(dataset_names) == len(model_prediction_results[0]), (len(dataset_names), model_prediction_results[0].shape[0])
        # if model.predict keeps the ordering, then the last few names should be the last ones of the dataset
        # this does not imply that the predictions are also in order, only that the generator produced them correctly
        # I hope this works...
        last_batch_vertebrae = data_generator.last_batch_names
        last_vertebrae_in_dataset = dataset_names[-len(last_batch_vertebrae):]
        assert last_vertebrae_in_dataset == last_batch_vertebrae, (last_vertebrae_in_dataset, last_batch_vertebrae)
        assert len(data_generator.batch_cache) == 0

        return model_prediction_results, dataset_names

    def after_model_directory(self, results: List[SingleModelEvaluationResult]):
        split = self.h5_subset()
        x_components = None

        x = []
        e = []
        t = []
        names = []
        try:
            for h5_path in results:
                with h5py.File(h5_path, 'r+') as f:
                    x.extend(f[split]['x'][()])
                    e.extend(f[split]['e'][()])
                    t.extend(f[split]['t'][()])
                    names.extend(f[split]['names'][()])
                    if x_components is not None:
                        assert numpy.array_equal(x_components, f[split]['x_components'][()])
                    x_components = f[split]['x_components'][()]
            h5_path = os.path.abspath(self.to_file('merged'))
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)
            with h5py.File(h5_path, 'a') as f:
                self.create_or_replace_dataset(f, split + '/x', numpy.array(x))
                self.create_or_replace_dataset(f, split + '/e', numpy.array(e))
                self.create_or_replace_dataset(f, split + '/t', numpy.array(t))
                self.create_or_replace_dataset(f, split + '/names', numpy.array(names))
                self.create_or_replace_dataset(f, split + '/x_components', numpy.array(x_components))
                df_for_cox = self.df_for_cox_from_h5_file(f, split)
        except OSError as e:
            if 'Resource temporarily unavailable' in str(e):
                # probably some other process is reading/writing
                assert os.path.isfile(h5_path)
            else:
                raise

        # self.analyze_with_cox(df_for_cox, f'{h5_path}_{split}')

    def to_file(self, model_name: Optional[str]):
        return os.path.join(self.to_dir(),
                            '' if model_name is None else model_name,
                            f'patient_level_{self.patient_level_aggregation_method}' if self.group_by_patient else '',
                            self.base_file_name())

    def base_file_name(self):
        base_file_name = f'{self.task_exclusion_filter.name}{self.input_tasks_suffix()}{self.standardization_suffix()}_{self.prognosis_task_name()}_{self.tte_task_name()}.h5'
        return base_file_name

    def input_tasks_suffix(self):
        return '_' + self.extra_input_names() if len(self.extra_input_tasks) > 0 else ''

    def standardization_suffix(self):
        return '' if self.standardize_features else '_nonstd'

    def extra_input_names(self):
        return '_'.join([task.output_layer_name() for task in self.extra_input_tasks])

    def to_subdir(self):
        return 'outputs_as_survival_dataset'

    def to_dir(self, short=False, subdir: str = None) -> str:
        """test and validation to the same file"""
        return super().to_dir(short, subdir).replace(self.use_dataset.name(), '')

    def model_level_cache_key(self, model_path: Optional[str] = None):
        model_name = None if model_path is None else os.path.basename(model_path)
        return super().model_level_cache_key(model_path) + (self.to_file(model_name),)

    def prognosis_task_name(self):
        return self.patient_level_binary_prognosis_task_for_event_flag.output_layer_name()

    def directory_level_cache_key(self, model_dir: str):
        return super().directory_level_cache_key(model_dir) + (self.to_file('model_name_placeholder_string'),)

    def tte_task_name(self):
        return self.patient_level_tte_task.output_layer_name()
