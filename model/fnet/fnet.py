import contextlib
import json
import os
import random
from copy import deepcopy
from typing import List, Union, ContextManager, Callable, Dict, Any, Optional, Tuple, Type

import cachetools
import numpy
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import CallbackList

from hiwi import ImageList, Image
from lib.clone_compiled_model import clone_compiled_model
from lib.count_trainable_params import count_trainable_params
from lib.image_processing_tool import TrainableImageProcessingTool
from lib.my_logger import logging
from lib.parameter_search import run_name
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator
from model.fnet.builder import FNetBuilder, KerasModelBuilder
from model.fnet.data_source import ImageListDataSource
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, split_name, UseWholeSet
from model.fnet.model_analysis.list_difficult_vertebrae import DifficultVertebraePlotter
from model.fnet.model_analysis.plot_models import PlotModels
from model.fnet.model_analysis.plot_output_as_text_on_sagittal_slice import PlotOutputAsTextOnSagittalSlice
from tasks import VertebraTask, VertebraTasks, OutputCombination


class FNet(TrainableImageProcessingTool):
    def __init__(self, model_path: str,
                 dataset_name_for_evaluation='unnamed_dataset',
                 use_coordinates: Union[str, Callable[[ImageList], ContextManager]] = 'default',
                 model: Model = None,
                 config: Dict[str, Any] = None,
                 tasks: VertebraTasks = None, ):
        if model_path.endswith('.h5'):
            model_path = model_path[:-len('.h5')]
        assert (model is None) == (config is None)
        if model is None:
            model, config = self.load_tf_model_and_config(model_path + '.h5', include_artificial_outputs=False)
        super().__init__(config=config, tasks=tasks, model_path=model_path, validation_batch_size=1800)
        if use_coordinates == 'default':
            use_coordinates = lambda iml: contextlib.suppress()
        self.use_coordinates: Callable[[ImageList], ContextManager] = use_coordinates
        self.dataset_name_for_evaluation = dataset_name_for_evaluation
        self.model: Model = model
        self.ensure_model_can_be_serialized_to_json()

    def name(self) -> str:
        return os.path.basename(self.model_path)

    @classmethod
    def from_config(cls, config: dict) -> 'FNet':
        tasks_contain_artificial_tasks = any(isinstance(t, OutputCombination) for t in VertebraTasks.load_from_config(config))
        if tasks_contain_artificial_tasks:
            config_without_artificial_tasks = VertebraTasks.filter_artificial_tasks_from_config(config)
            builder = cls.model_builder(config_without_artificial_tasks)
            model = builder.build()
            model, config = builder.add_artificial_outputs_to_trained_model(model)
            cls.model_builder(config).compile_built_model(model)
        else:
            model = cls.model_builder(config).build()
        cv_test_fold = config['cv_test_fold']
        experiment_name = config['experiment_name']
        model_subdir = config['model_subdir']
        model_name = f'{model_subdir}/{run_name()}_fold{cv_test_fold}'
        return FNet(model=model,
                    config=config,
                    model_path=os.path.join('models', experiment_name, model_name))

    @classmethod
    def model_builder(cls, config: Dict[str, Any]) -> KerasModelBuilder:
        return FNetBuilder(config)

    def ensure_model_can_be_serialized_to_json(self):
        self.model.to_json()

    def plot_model(self, to_file: Optional[str] = None):
        self.plot_model_with_shapes(to_file)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_model(self.model)
        self.save_config(self.config)
        self.plot_model_with_shapes()
        logging.info(f'Saved model to {self.model_path}')
        assert self.model_saved()

    def plot_model_with_shapes(self, to_file: Optional[str] = None):
        if to_file is None:
            to_file = self.model_path + '.png'
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        try:
            os.remove(to_file)
        except FileNotFoundError:
            pass
        plot_model(self.model, show_shapes=True, show_layer_names=True, expand_nested=True, to_file=to_file)
        print(f'Wrote {os.path.abspath(to_file)}')

    def model_saved(self):
        return os.path.isfile(self.model_path + '.h5') and os.path.isfile(self.model_path + '.json')

    def save_model_and_config(self):
        self.save_model(self.model)
        self.save_config(self.config)

    def clone(self) -> 'TrainableImageProcessingTool':
        return FNet(model_path=self.model_path,
                    dataset_name_for_evaluation=self.dataset_name_for_evaluation,
                    use_coordinates=self.use_coordinates,
                    model=clone_compiled_model(self.model),
                    config=deepcopy(self.config),
                    tasks=self.tasks)

    def count_parameters(self) -> int:
        return count_trainable_params(self.model)

    def save_model(self, current_model: Model):
        save_path = self.model_path
        assert isinstance(current_model, Model)
        try:
            current_model.save(save_path + '.h5',
                               overwrite=False,
                               include_optimizer=True)
        except RuntimeError as e:
            if 'input(): lost sys.stdin' not in e:
                raise
            logging.info(f'Tried asking about overwriting {save_path} despite passing overwrite=False')
        logging.info('Wrote ' + save_path + '.h5')

    def save_config(self, config):
        save_path = self.model_path
        with open(save_path + '.json', 'w') as outfile:
            json.dump(config, outfile, indent=2)

    @cachetools.cached(cachetools.LRUCache(maxsize=1), key=lambda self: self.model_path)
    def load_tasks_for_evaluation(self):
        model, config = self.load_tf_model_and_config(self.model_path + '.h5')
        tasks = VertebraTasks.load_from_config(config)
        return tasks

    def predict_on_single_image(self, img: Image) -> dict:
        iml = ImageList([img])
        with self.use_coordinates(iml):
            tasks = self.tasks
            model, config = self.model, self.config
            evaluator = FNetParameterEvaluator(ImageListDataSource(iml, self.dataset_subdir_name(iml), num_dims=3))
            evaluator.configure(config)

            try:
                data_generator = evaluator.prediction_generator_from_config(iml, evaluator.config, tasks=tasks, ndim=evaluator.data_source.ndim())
            except AnnotatedPatchesGenerator.EmptyDataset:
                assert len([v for v in img.parts if img.parts[v].position is not None]) == 0
                result = {task.output_layer_name(): [] for task in tasks}
                result['names'] = []
                return result
            assert data_generator.steps_per_epoch() == 1
            assert data_generator.num_samples() == len([v for v in img.parts if img.parts[v].position is not None])
            y_preds = model.predict(data_generator, steps=data_generator.steps_per_epoch())
            names = data_generator.last_batch_names
            result = {}
            for sample_idx, name in enumerate(names):
                patient_id, vertebra = split_name(name)
                assert patient_id == img['patient_id']
                for task_idx, task in enumerate(tasks):
                    task: VertebraTask
                    assert len(y_preds[task_idx]) == len(names)
                    label: numpy.ndarray = y_preds[task_idx][sample_idx]
                    result.setdefault(task.output_layer_name(), []).append(label.tolist())
            result['names'] = names
            return result

    def dataset_subdir_name(self, iml: ImageList = None):
        result = self.dataset_name_for_evaluation
        if iml is not None and hasattr(iml, 'name'):
            result += '_' + iml.name
        return result

    def evaluate_on_iml(self, iml: ImageList):
        with self.use_coordinates(iml):
            evaluator = FNetParameterEvaluator(ImageListDataSource(iml, self.dataset_subdir_name(iml), num_dims=3))
            ModelAnalyzer.DISABLE_INTERMEDIATE_ANALYSES = False
            random_analysis_order = True
            ModelAnalyzer.RANDOM_MODEL_ORDER = random_analysis_order
            ModelAnalyzer.RANDOM_SUBDIRECTORY_ORDER = random_analysis_order
            ModelAnalyzer.SKIP_EXISTING_BY_DEFAULT = True
            ModelAnalyzer.use_dataset = UseWholeSet(dataset_name=self.dataset_subdir_name(iml))
            analyses: List[ModelAnalyzer] = [
                PlotModels(evaluator=evaluator),
                DifficultVertebraePlotter(evaluator,
                                          to_subdir='all_vertebrae',
                                          save_as_nifti=False,
                                          difficult_vertebrae_table=None, ),
                PlotOutputAsTextOnSagittalSlice(evaluator=evaluator, model_level_plotting=False),
            ]
            if random_analysis_order:
                analyses = analyses.copy()
                random.shuffle(analyses)
            for a in analyses:
                logging.info(f'Running {a.name()}')
                a.analyze_multiple_models([self.model_path + '.h5'])

    def train_some_time(self, training_data_generator, validation_data_generator=None, validation_steps=None,
                        callbacks: CallbackList = None):
        verbose = 2
        self.model.fit(
            training_data_generator,
            validation_data=validation_data_generator,
            validation_steps=validation_steps,
            steps_per_epoch=self.steps_per_epoch(self.config),
            verbose=verbose,
            callbacks=callbacks,
            workers=1,
            epochs=self.num_epochs(),
        )

    @property
    def metrics(self) -> List[tensorflow.keras.metrics.Metric]:
        return self.model.metrics

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    @property
    def stop_training(self):
        return self.model.stop_training

    @stop_training.setter
    def stop_training(self, value):
        self.model.stop_training = value

    @property
    def metrics_names(self):
        return self.model.metrics_names

    def evaluate(self,
                 x: AnnotatedPatchesGenerator = None,
                 verbose=1,
                 steps=None,
                 callbacks=None,
                 return_dict=False):
        return self.model.evaluate(
            x=x,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            return_dict=return_dict
        )

    @classmethod
    def load_trained_model_for_evaluation(cls,
                                          model_path,
                                          include_artificial_outputs=True,
                                          thresholds_for_artificial_outputs: Dict[str, float] = None) -> 'FNet':
        m, model_config = cls.load_tf_model_and_config(model_path, include_artificial_outputs, thresholds_for_artificial_outputs)
        return FNet(
            model=m,
            config=model_config,
            model_path=model_path,
        )

    @classmethod
    def load_tf_model_and_config(cls,
                                 model_path,
                                 include_artificial_outputs=True,
                                 thresholds_for_artificial_outputs: Dict[str, float] = None):
        with open(model_path.replace('.h5', '.json')) as json_file:
            model_config = json.load(json_file)
        builder = cls.model_builder(model_config)
        custom_layers = builder.custom_layers()
        m = tensorflow.keras.models.load_model(os.path.abspath(model_path),
                                               custom_objects=custom_layers,
                                               compile=False)
        if include_artificial_outputs:
            m, model_config = builder.add_artificial_outputs_to_trained_model(m, thresholds_for_artificial_outputs)
        return m, model_config

    # def add_artificial_outputs(self, thresholds: Dict[str, float]):
    #     self.model, self.config = FNetBuilder(self.config).add_artificial_outputs_to_trained_model(self.model, thresholds)

    @classmethod
    def load_config_for_model_evaluation(cls,
                                         model_path,
                                         include_artificial_outputs=True,
                                         thresholds_for_artificial_outputs: Dict[str, float] = None) -> Dict[str, Any]:
        with open(model_path.replace('.h5', '.json')) as json_file:
            model_config = json.load(json_file)
        ts = thresholds_for_artificial_outputs
        if include_artificial_outputs:
            _, model_config = cls.model_builder(model_config).add_artificial_outputs_to_trained_model(None, ts)
        return model_config

    def initialize_metrics_names(self):
        fill_nan = lambda xs: tuple(x if x is not None else 1 for x in xs)
        self.model.evaluate(
            x=[numpy.zeros(fill_nan(i.shape)) for i in self.model.inputs],
            y=[numpy.zeros(fill_nan(o.shape)) for o in self.model.outputs],
            verbose=0,
        )

    def copy_with_filtered_outputs(self, only_tasks: List[str]) -> 'TrainableImageProcessingTool':
        new_config = self.filter_tasks(self.config, only_tasks)
        filtered_model_outputs = [o for o, n in zip(self.model.outputs, self.model.output_names)
                                  if only_tasks is None or n in only_tasks]
        model = Model(inputs=self.model.inputs, outputs=filtered_model_outputs)
        return FNet(
            model=model,
            config=new_config,
            model_path=self.model_path,
        )

    def predict(self,
                x: numpy.ndarray,
                batch_size=None,
                verbose=0,
                steps=None, ) -> List[numpy.ndarray]:
        return self.model.predict(x,
                                  batch_size=batch_size,
                                  steps=steps,
                                  verbose=verbose,
                                  max_queue_size=1,  # hopefully this fixes the ordering of the outputs
                                  )

    def predict_generator(self, gen: 'AnnotatedPatchesGenerator', steps: int) -> List[numpy.ndarray]:
        gen.step_count = 0
        results = []
        names = []
        for step_idx in range(steps):
            x, _, _ = next(gen)
            results.append(self.model.predict_on_batch(x))
            names.extend(gen.last_batch_names)
        results = [numpy.concatenate(r, axis=0) for r in zip(*results)]  # one array per output
        assert len(results[0]) == len(names)
        assert names[:gen.num_samples()] == gen.all_vertebra_names()
        return results

    def predict_generator_returning_inputs(self, gen: 'AnnotatedPatchesGenerator', steps: int) -> tuple:
        results = []
        names = []
        xs = []
        ys = []
        sample_weights = []
        for step_idx in range(steps):
            x, y, w = next(gen)
            results.append(self.model.predict_on_batch(x))
            xs.append(x)
            ys.append(y)
            sample_weights.append(w)
            names.extend(gen.last_batch_names)
        results = [numpy.concatenate(r, axis=0) for r in zip(*results)]  # one array per output
        xs = [numpy.concatenate(x, axis=0) for x in zip(*xs)]  # one array per input
        ys = [numpy.concatenate(y, axis=0) for y in zip(*ys)]  # one array per output
        sample_weights = [numpy.concatenate(w, axis=0) for w in zip(*sample_weights)]  # one array per output
        assert len(results[0]) == len(names)
        assert names[:gen.num_samples()] == gen.all_vertebra_names()
        return results, names, xs, ys, sample_weights

    def output_shapes(self):
        return [o.shape for o in self.model.outputs]
