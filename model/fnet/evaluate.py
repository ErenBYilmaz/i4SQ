import gc
import json
import math
import random
from copy import copy
from math import inf, nan, isnan
from typing import List, Optional, Dict, Tuple, Any, Type

import wandb.wandb_run

import hiwi
import lib.util
from hiwi import ImageList
from lib import swa, dl_backend
from lib.additional_validation_sets import AdditionalValidationSets
from lib.callbacks import AnyCallback, TerminateOnNaN, CallbackList
from lib.image_processing_tool import TrainableImageProcessingTool
from lib.my_logger import logging
from lib.parameter_search import EvaluationResult, InvalidParametersError, Prediction
from lib.print_time_after_each_epoch import PrintTimeAfterEachEpoch
from lib.util import Z, Y, X, EBC, all_sets_and_disjoint
from lib.wandb_interface import WandBRunInterface
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator, AnnotatedPatchesGeneratorPseudo3D, AnnotatedPatchesGenerator2D
from load_data.patch_request import PatchRequest
from load_data.spaced_ct_volume import HUWindow
from model.fnet.const import RECORD_PREDICTIONS, \
    FIXED_TRAIN_EPOCHS, LARGER_RESULT_IS_BETTER, DATA_SOURCES
from model.fnet.data_source import FNetDataSource, DataSourceCombination
from model.fnet.hyperparameter_set import HyperparameterSet
from tasks import VertebraTasks

DatasetName = str


class CustomWandbMetricsLogger(AnyCallback):
    def __init__(self, run: wandb.wandb_run.Run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        self.run.log({"epoch": epoch}, commit=False)
        self.run.log(logs or {}, commit=True)


class FNetParameterEvaluator(EBC):
    CLEANUP_MEMORY_AFTERWARDS = True
    CLEANUP_MEMORY_BEFORE = True

    def evaluate_parameters_on_all_folds(self, parameters) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        for cv_fold in HyperparameterSet.cv_fold_parameter().possible_values:
            config = parameters.copy()
            config[HyperparameterSet.cv_fold_parameter().name] = cv_fold

            # m = MemoryLeakDebugger()
            # m.start()

            results += copy(self).evaluate(config)

            # m.stop()

            gc.collect()
        return results

    def evaluate_parameters_on_random_fold(self, parameters) -> List[EvaluationResult]:
        parameters = parameters.copy()
        parameters[HyperparameterSet.cv_fold_parameter().name] = random.choice(HyperparameterSet.cv_fold_parameter().possible_values)
        results = copy(self).evaluate(parameters)
        return results

    def evaluate_parameters_on_specific_fold(self, parameters, fold_id: int, beep=False) -> List[EvaluationResult]:
        parameters = parameters.copy()
        logging.info(f'Starting evaluating on fold {fold_id}.')
        parameters[HyperparameterSet.cv_fold_parameter().name] = fold_id
        results = copy(self).evaluate(parameters)
        if beep:
            import lib.util
            lib.util.beep(1000, 500)
        logging.info(f'Finished evaluating on fold {fold_id}: {parameters}.')
        return results

    def evaluate(self, parameters):
        return self(parameters)

    def __init__(self, data_source: FNetDataSource, parameter_searching=False):
        assert isinstance(parameter_searching, bool), type(parameter_searching)
        self.parameter_searching = parameter_searching
        self.called = False
        self.termination_callbacks = self.create_termination_callbacks()
        self.data_source = data_source
        self.wandb_interface: Optional[WandBRunInterface] = None

    def __call__(self, parameters):
        if self.called:
            raise RuntimeError(f'Please do not re-use instances of {type(self)}. '
                               f'However, you can copy unused instances with deepcopy.')
        if self.CLEANUP_MEMORY_AFTERWARDS:
            dl_backend.b().memory_leak_cleanup()
        self.called = True
        self.configure(parameters)
        self.set_swa_parameters_to_none(self.config)

        test_train_and_validation_dataset = self.load_dataset_split(verbose=True)
        test_dataset, train_dataset, validation_dataset = test_train_and_validation_dataset
        assert all_sets_and_disjoint([[(img['patient_id']) for img in test_dataset],
                                      [(img['patient_id']) for img in train_dataset],
                                      [(img['patient_id']) for img in validation_dataset]])

        self.import_known_model_types()
        self.image_processing_tool = TrainableImageProcessingTool.from_config(self.config)
        if self.log_to_wandb():
            self.wandb_interface = WandBRunInterface(
                run_name=self.image_processing_tool.name(),
                config={'data_source_name': self.data_source.name(), **self.config},
                project='FNet',
                group=self.experiment_name(),
            )
            assert self.wandb_interface.api_key is not None
        self.image_processing_tool.plot_model()
        logging.info(f'After training, model will be saved to {self.image_processing_tool.model_path}')
        logging.info(f'Model has {self.image_processing_tool.count_parameters():,} trainable parameters.')

        training_data_generator = self.training_data_generator(train_dataset)

        if not self.parameter_searching:
            self.add_training_info('training_samples', training_data_generator.all_vertebrae)

        val_datasets = self.pick_sets_to_use_for_validation(test_train_and_validation_dataset)
        validation_sets = self.validation_sets_for_callbacks(val_datasets)
        data_generators = [training_data_generator] + [
            validation_data_generator
            for (validation_data_generator, _), _ in validation_sets
        ]

        self.swa_callbacks = self.prepare_swa_callbacks(validation_sets)
        self.main_validation_callback = self.prepare_main_validation_callback(validation_sets)

        if not self.parameter_searching:
            self.add_training_info_about_validation_sets()

        (val_generator, val_steps), val_set_name = self.main_validation_callback.validation_sets[0]
        assert self.main_validation_callback.validation_sets[0][1] == 'val'
        try:
            self.image_processing_tool.train_some_time(training_data_generator,
                                                       validation_data_generator=val_generator,
                                                       validation_steps=val_steps,
                                                       callbacks=self.prepare_callbacks(verbose=0 if self.log_to_wandb() else 2))
        except (MemoryError, *self.backend().gpu_error_classes()) as e:
            if not self.parameter_searching:
                if self.log_to_wandb():
                    self.wandb_interface.mark_as_crashed()
                # raise
            # print_exc_plus(print=logging.info)
            # raise DontSaveResultsError()
            logging.warning('Error during training, cancelling this training run.')
            logging.warning('Since a re-run probably will lead to the same error, this is saved as invalid parameters in the database.')
            self.backend().clear_session()
            raise InvalidParametersError() from e
        evaluation_results = self.load_evaluation_results_from_callbacks()

        if self.log_to_wandb():
            self.wandb_interface.commit()
        if self.save_original_model():
            self.check_if_model_was_actually_saved()

        if self.log_to_wandb():
            self.wandb_interface.finish()

            # print_exc_plus(print=logging.info)
        del training_data_generator
        for gen in data_generators:
            gen.invalidate()
        self.backend().clear_session()
        if self.CLEANUP_MEMORY_AFTERWARDS:
            dl_backend.b().memory_leak_cleanup()
        # raise lib.util.DontSaveResultsError('I do not want to save results atm.')
        return evaluation_results

    @staticmethod
    def backend():
        return dl_backend.b()

    def save_original_model(self):
        return not self.parameter_searching

    def add_training_info_about_validation_sets(self):
        for dataset in self.main_validation_callback.validation_sets:
            data_gen = dataset[0][0]
            assert isinstance(data_gen, AnnotatedPatchesGenerator)
            dataset_name = dataset[-1]
            self.add_training_info(f'{dataset_name}_samples', data_gen.all_vertebrae)

    def add_training_info(self, key, value, allow_overwrite=False):
        for config in [self.original_parameters, self.config]:
            # this training_info parameter is not used in training but only to be saved in the config-json of the model
            if 'training_info' not in config:  # we need to update original parameters here otherwise it won't be saved
                config['training_info'] = json.dumps({})
            if key in config['training_info'] and not allow_overwrite:
                raise RuntimeError
            config['training_info'] = json.dumps({
                **json.loads(config['training_info']),
                key: value
            })

    # noinspection PyAttributeOutsideInit
    def configure(self, config: Dict[str, Any]):
        self.original_parameters = config.copy()
        self.config = config.copy()
        if 'conv_ndim' not in self.config:
            self.config['conv_ndim'] = 3  # backwards compatibility
        self.tasks = VertebraTasks.load_from_config(config)
        self.hu_window = self.hu_window_from_config(config)
        HyperparameterSet.validate_swa_parameters(self.config)
        self.swa_start_batches = self.compute_swa_start_batches(self.config)
        self.check_that_swa_start_batch_will_be_used_for_training(self.original_parameters, self.swa_start_batches)
        self.load_num_epochs_from_constant(self.config)

    @staticmethod
    def steps_per_epoch(config):
        return config['steps_per_epoch']

    @staticmethod
    def load_trained_model_for_evaluation(model_path,
                                          include_artificial_outputs=True,
                                          thresholds_for_artificial_outputs: Dict[str, float] = None) -> TrainableImageProcessingTool:
        model_type = FNetParameterEvaluator.model_type_from_file(model_path)
        return model_type.load_trained_model_for_evaluation(model_path=model_path,
                                                            include_artificial_outputs=include_artificial_outputs,
                                                            thresholds_for_artificial_outputs=thresholds_for_artificial_outputs)

    @staticmethod
    def load_config_for_model_evaluation(model_path,
                                         include_artificial_outputs=True,
                                         thresholds_for_artificial_outputs: Dict[str, float] = None) -> Dict[str, Any]:
        model_type = FNetParameterEvaluator.model_type_from_file(model_path)
        return model_type.load_config_for_model_evaluation(model_path=model_path,
                                                           include_artificial_outputs=include_artificial_outputs,
                                                           thresholds_for_artificial_outputs=thresholds_for_artificial_outputs)

    @staticmethod
    def model_type_from_file(model_path) -> Type[TrainableImageProcessingTool]:
        with open(model_path.replace('.h5', '.json')) as json_file:
            model_config = json.load(json_file)
        if 'model_type' not in model_config:
            model_type = 'FNet'  # legacy support
        else:
            model_type = model_config['model_type']
        model_type = TrainableImageProcessingTool.SUBCLASSES_BY_NAME[model_type]
        assert issubclass(model_type, TrainableImageProcessingTool)
        return model_type

    def check_if_model_was_actually_saved(self):
        if not self.image_processing_tool.model_saved():
            raise RuntimeError('You wanted to save the original model but it was not saved for some unknown reason')

    def load_evaluation_results_from_callbacks(self) -> List[EvaluationResult]:
        evaluation_results = []
        for history in [self.main_validation_callback, *self.swa_callbacks]:
            evaluation_results += self.results_for_history(history)
        parameters_not_evaluated = all(result.parameters != self.original_parameters for result in evaluation_results)
        if parameters_not_evaluated:
            if self.save_original_model():
                raise RuntimeError('You wanted to save the original model but it was not even trained')
            if self.training_terminated_early():
                evaluation_results.append(self.worst_possible_result_for_parameters(self.original_parameters))
        return evaluation_results

    def training_terminated_early(self):
        return self.image_processing_tool.stop_training

    def worst_possible_result_for_parameters(self, params):
        return EvaluationResult(
            parameters=params,
            results=self.worst_metrics_dict(),
        )

    def worst_metrics_dict(self):
        return {metric['name']: metric['worst'] for metric in self.tasks.metrics_dicts_for_fnet()}

    def results_for_history(self, history: AdditionalValidationSets):
        results_for_history = []
        currently_lowest: Optional[EvaluationResult] = None
        for epoch_end in range(self.num_epochs()):

            result = self.evaluation_result_after_specific_epoch(epoch_end, history)

            val_metric = 'val_' + self.early_stopping_metric()
            if currently_lowest is None or isnan(currently_lowest.results[val_metric]):
                currently_lowest = result
            # the following lines emulate early stopping with keeping only the best model
            if not LARGER_RESULT_IS_BETTER and result.results[val_metric] < currently_lowest.results[val_metric]:
                currently_lowest = result
            elif LARGER_RESULT_IS_BETTER and result.results[val_metric] > currently_lowest.results[val_metric]:
                currently_lowest = result
            else:
                result.results = currently_lowest.results.copy()
                result.predictions = currently_lowest.predictions.copy()
            results_for_history.append(result)

            if self.save_original_model():
                newest_result = results_for_history[-1]
                if newest_result.parameters == self.original_parameters:
                    # because we used early stopping, we can access the best model from the history
                    assert isinstance(history.best_model, TrainableImageProcessingTool), type(history.best_model)
                    history.best_model.save_model_and_config()
                    logging.info(f'early_stopped_after: {newest_result.results["early_stopped_after"]}')
                    assert self.image_processing_tool.model_saved()
        return results_for_history

    def evaluation_result_after_specific_epoch(self, epoch, history):
        if epoch not in range(self.num_epochs_in_history(history)):
            current_parameters = self.config_with_adjusted_swa_parameters(epoch, history)
            result = self.worst_possible_result_for_parameters(current_parameters)
        else:
            result = self.compute_evaluation_result_after_specific_epoch_from_history(epoch, history)
        return result

    def compute_evaluation_result_after_specific_epoch_from_history(self, epoch, history):
        current_parameters = self.config_with_adjusted_swa_parameters(epoch, history)
        h = history.history
        prefix = history.prefix()
        mean_metric = lambda m: h[prefix + m][epoch] if prefix + m in h else nan
        early_stopped_during_last_2_epochs = current_parameters['epochs'] >= self.num_epochs() - 2
        # if early_stopped_during_last_2_epochs:
        #     logging.warning(f'Early stopped during epoch {current_parameters["epochs"]} of {self.num_epochs()}. You may need to increase the number of epochs')
        result = EvaluationResult(
            parameters=current_parameters,
            results={
                **{
                    metric['name']: mean_metric(metric['name'])
                    for metric in self.tasks.metrics_dicts_for_fnet()
                    if 'keras_metric_name' in metric
                },
                'num_params': self.image_processing_tool.count_parameters(),
                'early_stopped_after': current_parameters['epochs'],
                'early_stopped_during_last_2_epochs': early_stopped_during_last_2_epochs,
            },
            predictions=[
                Prediction(dataset=k[len(prefix):-len('_predictions')],
                           y_true=last_h_entry['y_true'][i],
                           y_pred=last_h_entry['y_pred'][i],
                           name=last_h_entry['names'][i], )
                for k in h.keys()
                if k.endswith('_predictions')
                for last_h_entry in [h[k][epoch]]
                for i in range(len(last_h_entry['y_true']))
            ]
        )
        return result

    def config_with_adjusted_swa_parameters(self, epoch_end, history):
        current_parameters = self.config.copy()
        self.set_swa_parameters_according_to_history(current_parameters, history)
        current_parameters['epochs'] = epoch_end + 1
        return current_parameters

    @staticmethod
    def num_epochs_in_history(history):
        h = history.history
        epochs_trained = len(h[next(iter(h))])
        return epochs_trained

    def set_swa_parameters_according_to_history(self, current_parameters, history):
        if isinstance(history, swa.SWA):
            self.set_parameters_from_swa_callback(current_parameters, history)
        else:
            self.set_swa_parameters_to_none(current_parameters)

    @staticmethod
    def set_parameters_from_swa_callback(current_parameters, swa_callback: swa.SWA):
        current_parameters['swa_start_batch'] = swa_callback.start_batch
        current_parameters['swa_cycle_length'] = swa_callback.c

    def validation_sets_for_callbacks(self, val_datasets) -> List[Tuple[Tuple[AnnotatedPatchesGenerator, int], str]]:
        validation_sets = []
        for name, dataset in val_datasets.items():
            if len(dataset) == 0:
                logging.warning(f'No images in dataset `{name}`')
                continue
            logging.info(f'Setting up validation data generator ({name})...')
            data_generator = self.validation_generator_from_config(dataset, self.config,
                                                                   batch_size=self.image_processing_tool.validation_batch_size,
                                                                   ndim=self.data_source.ndim())
            validation_sets.append(((data_generator, data_generator.steps_per_epoch()), name))
        assert len(set(map(lambda x: x[1], validation_sets))) == len(list(map(lambda x: x[1], validation_sets)))
        return validation_sets

    def plot_whole_dataset(self):
        iml = self.data_source.whole_dataset()
        data_generator = self.prediction_generator_from_config(iml, self.config, ndim=self.data_source.ndim())
        assert data_generator.steps_per_epoch() == 1
        batch = data_generator.get_batch(0)
        batch.serialize(
            generate_images=True,
            save_as_nifti=True,
            generate_histograms=True,
            relative_dir_name=self.data_source.name(),
            progress_bar=True,
        )

    def num_epochs(self):
        return self.config['epochs']

    def prepare_callbacks(self, verbose):
        callbacks: List[AnyCallback] = []
        callbacks += self.termination_callbacks
        callbacks += self.swa_callbacks
        callbacks += [self.main_validation_callback]
        callbacks += [PrintTimeAfterEachEpoch()]

        # callbacks += [HeapSummaryAfterFirstEpoch()]
        callback_list = CallbackList(callbacks,
                                     add_history=False,
                                     add_progbar=verbose != 0,
                                     model=self.image_processing_tool,
                                     verbose=verbose,
                                     epochs=self.num_epochs(),
                                     steps=self.steps_per_epoch(self.config))
        return callback_list

    def log_to_wandb(self):
        return not lib.util.in_debug_mode()

    def prepare_swa_callbacks(self, validation_sets) -> List[AnyCallback]:
        return [
            swa.SWAWithoutLrChange(
                start_batch=swa_start_batch,
                cycle_length=self.steps_per_epoch(self.config),
                validation_sets=validation_sets,
                record_original_history=False,
                record_predictions=RECORD_PREDICTIONS,
                keep_best_model_by_metric='val_' + self.early_stopping_metric(),
                larger_result_is_better=LARGER_RESULT_IS_BETTER,
                log_using_wandb_run=self.wandb_interface.get_run() if self.log_to_wandb() else None,
                verbose=self.callback_verbosity(),
            ) for swa_start_batch in self.swa_start_batches
        ]

    def prepare_main_validation_callback(self, validation_sets):
        return AdditionalValidationSets(validation_sets=validation_sets,
                                        record_original_history=False,
                                        record_predictions=RECORD_PREDICTIONS,
                                        keep_best_model_by_metric='val_' + self.early_stopping_metric(),
                                        larger_result_is_better=LARGER_RESULT_IS_BETTER,
                                        verbose=self.callback_verbosity(),
                                        log_using_wandb_run=self.wandb_interface.get_run() if self.log_to_wandb() else None, )

    def callback_verbosity(self):
        return 0

    @staticmethod
    def create_termination_callbacks() -> List[AnyCallback]:
        return [TerminateOnNaN()]

    def early_stopping_metric(self) -> str:
        if self.config['early_stopping_metric'] is not None:
            return self.config['early_stopping_metric']
        return self.tasks.optimization_criterion()

    @classmethod
    def validation_generator_from_config(cls, dataset: ImageList, config, cache_batches=True, batch_size=inf, ndim=None):
        if ndim is None:
            raise ValueError(ndim)
        return cls.pick_gen(ndim, config['conv_ndim'])(
            image_list=dataset,
            patch_requests=FNetParameterEvaluator.load_patch_requests_from_config(config, adjust_for_rotation=True),
            batch_size=batch_size,
            auto_tune_batch_size=True,
            tasks=VertebraTasks.load_from_config(config),
            random_mode=None,
            weighted=True,
            hu_window=FNetParameterEvaluator.hu_window_from_config(config),
            cache_batches=cache_batches,
            cache_vertebra_volumes_in_ram=not cache_batches,
            ignore_border_vertebrae=config['ignore_border_vertebrae'],
            exclude_site_6=config['exclude_site_6'],
            pixel_scaling=config['pixel_scaling'] if 'pixel_scaling' in config else 'divide_by_2k',
            spatial_dims=3 if config['fully_convolutional'] and not config['use_gap'] else 0,
            only_vertebra_levels=json.loads(config['restrict_to_vertebra_levels']),
            exclude_patients=[],
            vertebra_size_adjustment=config['vertebra_size_adjustment'],
        )

    @classmethod
    def prediction_generator_from_config(cls, dataset: ImageList, config, cache_batches=False, cache_vertebra_volumes_in_ram=False, batch_size=inf,
                                         tasks=None, ndim: int = None, exclude_unlabeled=False):
        if tasks is None:
            tasks = VertebraTasks.load_from_config(config)
        if ndim is None:
            raise ValueError(ndim)
        return cls.pick_gen(ndim, config['conv_ndim'])(
            image_list=dataset,
            patch_requests=FNetParameterEvaluator.load_patch_requests_from_config(config, adjust_for_rotation=True),
            batch_size=batch_size,
            auto_tune_batch_size=True,
            tasks=tasks,
            random_mode=None,
            weighted=False,
            hu_window=FNetParameterEvaluator.hu_window_from_config(config),
            cache_batches=cache_batches,
            cache_vertebra_volumes_in_ram=cache_vertebra_volumes_in_ram,
            cache_vertebra_volumes_on_disk=True,
            ignore_border_vertebrae=config['ignore_border_vertebrae'],
            exclude_site_6=config['exclude_site_6'],
            pixel_scaling=config['pixel_scaling'] if 'pixel_scaling' in config else 'divide_by_2k',
            spatial_dims=3 if config['fully_convolutional'] else 0,
            only_vertebra_levels=json.loads(config['restrict_to_vertebra_levels']),
            exclude_patients=[],
            vertebra_size_adjustment=config['vertebra_size_adjustment'],
            exclude_unlabeled=exclude_unlabeled,
            verbose=False)

    def training_data_generator(self, dataset: hiwi.ImageList):
        logging.info('Setting up training data generator...')
        return self.gen_cls()(image_list=dataset,
                              patch_requests=self.patch_requests(adjust_for_rotation=True),
                              batch_size=self.config['batch_size'],
                              random_mode=self.random_mode_for_training_generator(),
                              weighted=self.weighting_mode_for_training_generator(),
                              tasks=self.tasks,
                              hu_window=self.hu_window,
                              random_flip_lr=self.config['random_flip_lr'],
                              random_flip_ud=self.config['random_flip_ud'],
                              random_flip_fb=self.config['random_flip_fb'],
                              random_shift_mm=(self.config['random_shift_px_x'],
                                               self.config['random_shift_px_y'],
                                               self.config['random_shift_px_z'],),
                              data_amount=self.config['training_data_amount'],
                              label_smoothing=self.config['label_smoothing'],
                              cache_batches=False,
                              cache_vertebra_volumes_in_ram=True,
                              additional_random_crops=self.config['additional_random_crops'],
                              pad_augment_probabilities=(self.config['pad_augment_probability_z'],
                                                         self.config['pad_augment_probability_y'],
                                                         self.config['pad_augment_probability_x']),
                              random_project_x=self.config['random_project_x'],
                              random_project_y=self.config['random_project_y'],
                              random_project_z=self.config['random_project_z'],
                              pad_augment_ratios=(self.config['pad_augment_ratio_z'],
                                                  self.config['pad_augment_ratio_y'],
                                                  self.config['pad_augment_ratio_x']),
                              shuffle_labels=self.config['shuffle_labels'],
                              ignore_border_vertebrae=self.config['ignore_border_vertebrae'],
                              exclude_site_6=self.config['exclude_site_6'],
                              pixel_scaling=self.config['pixel_scaling'],
                              spatial_dims=3 if self.config['fully_convolutional'] and not self.config['use_gap'] else 0,
                              only_vertebra_levels=json.loads(self.config['restrict_to_vertebra_levels']),
                              exclude_patients=json.loads(self.config['exclude_patients_from_training']),
                              vertebra_size_adjustment=self.config['vertebra_size_adjustment'],
                              exclude_unlabeled=True, )

    def gen_cls(self):
        ndim = self.data_source.ndim()
        return self.pick_gen(ndim, self.conv_ndim())

    def conv_ndim(self):
        return self.config['conv_ndim']

    @staticmethod
    def pick_gen(input_ndim, conv_ndim) -> Type[AnnotatedPatchesGenerator]:
        return {
            2: AnnotatedPatchesGeneratorPseudo3D if conv_ndim == 3 else AnnotatedPatchesGenerator2D,
            3: AnnotatedPatchesGenerator,
        }[input_ndim]

    def random_mode_for_training_generator(self):
        if self.config['class_weighting'] is True:
            logging.warning('Deprecated: Use class_weighting = "oversampling" instead')
            return 'random_class'  # legacy code, True was used back then to indicate oversampling at training time and class weights at test time
        elif self.config['class_weighting'] == 'oversampling':
            return 'random_class'
        elif self.config['class_weighting'] == 'relative':
            return 'random_class'
        elif self.config['class_weighting'] == 'per_output':
            return 'random_vertebra'
        elif self.config['class_weighting'] in [False, None]:
            return 'random_vertebra'
        else:
            raise ValueError(self.config['class_weighting'])

    def weighting_mode_for_training_generator(self):
        if self.config['class_weighting'] is True:
            logging.warning('Deprecated: Use class_weighting = "oversampling" instead')
            return False  # legacy code, True was used back then to indicate oversampling at training time and class weights at test time
        elif self.config['class_weighting'] == 'oversampling':
            return False
        elif self.config['class_weighting'] == 'per_output':
            return True
        elif self.config['class_weighting'] == 'relative':
            return True
        elif self.config['class_weighting'] in [False, None]:
            return False
        else:
            raise ValueError(self.config['class_weighting'])

    @classmethod
    def input_shape_px(cls, config) -> Tuple[Z, Y, X]:
        shapes = [p.size_px() for p in cls.load_patch_requests_from_config(config)]
        assert len(set(shapes)) == 1
        return shapes[0]

    def cv_test_fold(self):
        test_fold_parameter = self.config['cv_test_fold']
        if isinstance(test_fold_parameter, int):
            return test_fold_parameter
        if isinstance(test_fold_parameter, dict):
            return test_fold_parameter['test']
        raise ValueError(f'Invalid test fold {test_fold_parameter}')

    def force_validation_fold_id(self) -> Optional[int]:
        test_fold_parameter = self.config['cv_test_fold']
        if isinstance(test_fold_parameter, int):
            return None
        if isinstance(test_fold_parameter, dict):
            return test_fold_parameter['val']
        raise ValueError(f'Invalid fold specification {test_fold_parameter}')

    def experiment_name(self):
        return self.config['experiment_name']

    @staticmethod
    def pick_sets_to_use_for_validation(test_train_and_validation_dataset: Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]
                                        ) -> Dict[DatasetName, ImageList]:
        test_dataset, train_dataset, validation_dataset = test_train_and_validation_dataset
        val_datasets = {
            'val': validation_dataset,
            # 'train': train_dataset,
            # 'test': test_dataset,
        }
        return val_datasets

    def load_dataset_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        random_validation_set = 'random_validation_set' in self.config and self.config['random_validation_set']
        site_validation_set = 'site_validation_set' in self.config and self.config['site_validation_set']
        official_split = 'official_split' in self.config and self.config['official_split']
        assert sum([random_validation_set, site_validation_set, official_split]) <= 1
        if random_validation_set:
            test_dataset = self.load_test_fold()
            train_dataset, validation_dataset = self.randomly_split_training_and_validation_data_into_new_training_and_validation_sets(verbose=verbose)
        elif site_validation_set:
            test_dataset, train_dataset, validation_dataset = self.split_by_site(verbose=verbose)
        elif official_split:
            test_dataset, train_dataset, validation_dataset = self.official_test_train_validation_split(verbose=verbose)
        else:  # default case
            test_dataset = self.load_test_fold()
            train_dataset, validation_dataset = self.cv_training_and_validation_folds(verbose=verbose)
        result = (test_dataset, train_dataset, validation_dataset)
        for dataset_idx, dataset in enumerate(result):
            assert hasattr(dataset, 'name'), (dataset_idx, self.data_source.name())
        return result

    def load_test_fold(self):
        return self.data_source.load_test_dataset_by_fold_id(self.cv_test_fold())

    def cv_training_and_validation_folds(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        force_val_fold_id = self.force_validation_fold_id()
        if force_val_fold_id is None:
            return self.data_source.cv_training_and_validation_folds(verbose=verbose, test_fold_id=self.cv_test_fold())
        else:
            return self.data_source.cv_training_and_validation_folds(verbose=verbose, test_fold_id=self.cv_test_fold(),
                                                                     val_fold_id=force_val_fold_id)

    def test_set_as_iml(self):
        return self.load_dataset_split()[0]

    def training_set_as_iml(self):
        return self.load_dataset_split()[1]

    def validation_set_as_iml(self):
        return self.load_dataset_split()[2]

    def whole_dataset(self):
        return self.data_source.whole_dataset()

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        return self.data_source.split_by_site(verbose=verbose)

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        return self.data_source.official_test_train_validation_split(verbose=verbose)

    def randomly_split_training_and_validation_data_into_new_training_and_validation_sets(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        try:
            training_samples = json.loads(self.config['training_info'])['training_samples']
        except KeyError:
            training_samples = None
        return self.data_source.randomly_split_training_and_validation_data_into_new_training_and_validation_sets(cv_test_fold=self.cv_test_fold(),
                                                                                                                  verbose=verbose,
                                                                                                                  training_samples=training_samples)

    @staticmethod
    def compute_swa_start_batches(parameters):
        swa_start_batches = []
        if parameters['swa_start_batch'] is not None and parameters['swa_start_batch'] not in swa_start_batches:
            swa_start_batches.append(parameters['swa_start_batch'])
        return swa_start_batches

    def load_num_epochs_from_constant(self, parameters):
        if self.parameter_searching:
            assert FIXED_TRAIN_EPOCHS is not None
        if FIXED_TRAIN_EPOCHS is not None:
            if parameters['epochs'] != FIXED_TRAIN_EPOCHS:
                parameters['epochs'] = FIXED_TRAIN_EPOCHS

    @staticmethod
    def check_that_swa_start_batch_will_be_used_for_training(original_parameters, swa_start_batches):
        assert original_parameters['swa_start_batch'] is None or original_parameters['swa_start_batch'] in swa_start_batches

    @staticmethod
    def set_swa_parameters_to_none(parameters):
        parameters['swa_start_batch'] = None
        parameters['swa_cycle_length'] = None

    @classmethod
    def default_evaluator(cls, data_source: FNetDataSource = None, parameter_searching=False):
        if data_source is None:
            data_source = DataSourceCombination(DATA_SOURCES)
        return cls(data_source, parameter_searching=parameter_searching)

    @classmethod
    def load_patch_requests_from_config(cls, config, adjust_for_rotation=False) -> List[PatchRequest]:
        if 'input_patches' not in config:
            input_patches = cls._default_legacy_input_patch_size(config)
        else:
            input_patches = [PatchRequest.from_json(r) for r in json.loads(config['input_patches'])]
        if adjust_for_rotation:
            for p in input_patches:
                p.set_size_px_by_changing_size_mm(size_px=lib.util.required_size_for_safe_rotation_zyx(p.size_px(), config['rotate_range_deg']))
        return input_patches

    @classmethod
    def _default_legacy_input_patch_size(cls, config):
        if 'num_adjacent_vertebra_inputs' in config and config['num_adjacent_vertebra_inputs'] > 0:
            raise NotImplementedError('TODO implement compatibility')
        return [PatchRequest(
            size_mm=(config['input_shape_mm_x'], config['input_shape_mm_y'], config['input_shape_mm_z']),
            spacing=(config['spacing_x'], config['spacing_y'], config['spacing_z'])
        )]

    def patch_requests(self, adjust_for_rotation=False):
        return self.load_patch_requests_from_config(self.config, adjust_for_rotation=adjust_for_rotation)

    @classmethod
    def hu_window_from_config(cls, config):
        if 'hu_window_min' in config:
            minimum = config['hu_window_min']
        else:
            minimum = -math.inf
        if 'hu_window_max' in config:
            maximum = config['hu_window_max']
        else:
            maximum = math.inf
        return HUWindow(minimum=minimum, maximum=maximum)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def import_known_model_types():
        try:
            from model.fnet.fnet import FNet
        except ImportError:
            pass
        try:
            from model.perceiver.perceiver import FracturePerceiver
            from model.pytorch_model.mnist_model import BasicTorchModel
            from model.swin_transformer.swin_transformer import FractureSwinTransformer
        except ImportError:
            pass
