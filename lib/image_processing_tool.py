import copy
import math
from abc import abstractmethod

import os
import typing
from copy import deepcopy
from typing import List, NoReturn, Dict, Any, Optional

import numpy

from tasks import VertebraTasks

if typing.TYPE_CHECKING:
    from tensorflow.python.keras.callbacks import CallbackList
    from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator

from hiwi import ImageList, Image
from lib.my_logger import logging
from lib.parameter_search import EvaluationResult
from lib.progress_bar import ProgressBar
from lib.util import EBC


class ImageProcessingTool:
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError('Abstract method')

    def predict_on_iml(self, iml: ImageList) -> ImageList:
        iml = deepcopy(iml)
        logging.info(f'Predicting with {self.name()}...')
        for img in ProgressBar(iml, line_length=78):
            self.predict_on_image_and_store_outputs(img)
        return iml

    @abstractmethod
    def evaluate_on_iml(self, iml: ImageList):
        raise NotImplementedError('Abstract method')

    def predict_on_image_and_store_outputs(self, img) -> NoReturn:
        outputs = self.predict_on_single_image(img)
        self.store_outputs(img, outputs)

    @abstractmethod
    def predict_on_single_image(self, img: Image) -> dict:
        raise NotImplementedError('Abstract method')

    def store_outputs(self, store_in, outputs):
        store_in.setdefault('tool_outputs', {})[self.name()] = outputs

    def store_outputs_for_img_in_iml(self, iml: ImageList, patient_id, outputs):
        for img in iml:
            if img['patient_id'] == patient_id:
                self.store_outputs(img, outputs)
                return

    def store_outputs_for_vertebra_in_iml(self, iml: ImageList, patient_id, vertebra_name, outputs):
        for img in iml:
            if img['patient_id'] == patient_id:
                self.store_outputs(img.parts[vertebra_name], outputs)
                return

    @staticmethod
    def infer_nii_path_if_necessary(input_file):
        if not input_file.endswith('.nii.gz'):
            if '2_dcm_series' in input_file:
                # possibly a dicom files passed from CLs framework
                input_file = os.path.abspath(input_file)
                input_file = input_file.replace('2_dcm_series', '3_processing')
                input_file = os.path.dirname(input_file)
                input_file = os.path.join(input_file, '2_processed', 'imagesTs', 'image.nii.gz')
                assert os.path.isfile(input_file), input_file
        return input_file


class ImageProcessingPipeline(ImageProcessingTool):
    def __init__(self, tools: List[ImageProcessingTool]):
        self.tools = tools

    def name(self) -> str:
        return '->'.join([tool.name() for tool in self.tools])

    def predict_on_single_image(self, img: Image) -> dict:
        results = {}
        for tool in self.tools:
            results[tool.name()] = tool.predict_on_single_image(img)
            tool.store_outputs(img, results[tool.name()])
        return results

    def evaluate_on_iml(self, iml: ImageList):
        iml = self.predict_on_iml(iml)
        self.evaluate_whole_pipeline(iml)

    @abstractmethod
    def evaluate_whole_pipeline(self, iml_with_intermediate_predictions_and_evaluations: ImageList):
        raise NotImplementedError('Abstract method')


class TrainableImageProcessingTool(EBC, ImageProcessingTool):
    def __init__(self,
                 model_path: str,
                 config: Dict[str, Any],
                 tasks: VertebraTasks,
                 validation_batch_size = math.inf):
        super().__init__()
        self.config = config
        if tasks is None:
            tasks = VertebraTasks.load_from_config(self.config)
        self.tasks = tasks
        self.validation_batch_size = validation_batch_size
        if model_path.endswith('.h5'):
            model_path = model_path[:-len('.h5')]
        self.model_path = model_path

    @staticmethod
    def steps_per_epoch(config):
        return config['steps_per_epoch']

    def num_epochs(self):
        return self.config['epochs']

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> 'TrainableImageProcessingTool':
        subclass = TrainableImageProcessingTool.SUBCLASSES_BY_NAME[config['model_type']]
        assert issubclass(subclass, TrainableImageProcessingTool)
        if subclass.from_config is cls.from_config:
            raise NotImplementedError('Abstract method')
        return subclass.from_config(config)

    @abstractmethod
    def plot_model(self, to_file: Optional[str] = None):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def model_saved(self) -> bool:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def train_some_time(self, training_data_generator: 'AnnotatedPatchesGenerator',
                        validation_data_generator: 'AnnotatedPatchesGenerator' = None,
                        validation_steps: int = None,
                        callbacks: 'CallbackList' = None) -> List[EvaluationResult]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def count_parameters(self) -> int:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def save_model_and_config(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def clone(self) -> 'TrainableImageProcessingTool':
        raise NotImplementedError('Abstract method')

    @property
    @abstractmethod
    def metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def set_weights(self, weights):
        raise NotImplementedError('Abstract method')

    @property
    @abstractmethod
    def stop_training(self):
        raise NotImplementedError('Abstract method')

    @stop_training.setter
    @abstractmethod
    def stop_training(self, value):
        raise NotImplementedError('Abstract method')

    @property
    @abstractmethod
    def metrics_names(self):
        raise NotImplementedError('Abstract method')

    def evaluate(self,
                 x: 'AnnotatedPatchesGenerator' = None,
                 verbose=1,
                 steps=None,
                 callbacks=None,
                 return_dict=False):
        raise NotImplementedError('Abstract method')

    @classmethod
    @abstractmethod
    def load_trained_model_for_evaluation(cls,
                                          model_path,
                                          include_artificial_outputs=True,
                                          thresholds_for_artificial_outputs: Dict[str, float] = None) -> 'TrainableImageProcessingTool':
        raise NotImplementedError('Abstract method')

    @classmethod
    @abstractmethod
    def load_config_for_model_evaluation(cls,
                                         model_path,
                                         include_artificial_outputs=True,
                                         thresholds_for_artificial_outputs: Dict[str, float] = None) -> Dict[str, Any]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    @abstractmethod
    def initialize_metrics_names(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    @abstractmethod
    def copy_with_filtered_outputs(self, only_tasks: List[str]) -> 'TrainableImageProcessingTool':
        raise NotImplementedError('Abstract method')

    @classmethod
    def filter_tasks(cls, config: Dict[str, Any], only_tasks: List[str]) -> Dict[str, Any]:
        new_config = deepcopy(config)
        if only_tasks is None:
            return new_config
        from tasks import VertebraTasks
        tasks = VertebraTasks.deserialize(config['tasks'])
        tasks = VertebraTasks([t for t in tasks if t.output_layer_name() in only_tasks])
        new_config['tasks'] = tasks.serialize()
        return new_config

    @abstractmethod
    def predict_generator(self, gen: 'AnnotatedPatchesGenerator', steps: int) -> List[numpy.ndarray]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def predict_generator_returning_inputs(self, gen: 'AnnotatedPatchesGenerator', steps: int) -> tuple:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def predict(self,
                x: numpy.ndarray,
                batch_size=None,
                verbose=0,
                steps=None, ) -> List[numpy.ndarray]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def output_shapes(self) -> List[typing.Tuple[int, ...]]:
        """
        returns output shapes including the batch dimension
        """
        raise NotImplementedError('Abstract method')

    def num_outputs(self):
        return len(self.tasks)

    def to_mcdo_model(self):
        assert self.mcdo_applicable() and not self.mcdo_applied(), 'Please check if mcdo is applicable or even already enabled before calling this method.'
        # mcdo_model = self.clone()
        mcdo_model = self.mcdo_model_from_config(self.config)
        # non_mcdo_model = self.non_mcdo_model_from_config(config)
        # assert len(mcdo_model.get_weights()) == len(non_mcdo_model.get_weights())
        assert len(self.tasks) == len(mcdo_model.tasks), (self.tasks.output_layer_names(), mcdo_model.tasks.output_layer_names())
        assert len(self.tasks) == mcdo_model.num_outputs()
        if len(mcdo_model.get_weights()) != len(self.get_weights()):
            print([layer.name for layer in mcdo_model.model.layers])
            print([layer.name for layer in self.model.layers])
            print([w.shape for w in mcdo_model.get_weights()])
            print([w.shape for w in self.get_weights()])
            raise AssertionError(len(mcdo_model.get_weights()), len(self.get_weights()))
        mcdo_model.set_weights(self.get_weights())
        mcdo_model.validation_batch_size //= 3  # for some reason dropout leads to memory error on larger batch sizes
        return mcdo_model

    def mcdo_applicable(self):
        return 'dropout_rate' in self.config and self.config['dropout_rate'] > 0

    def mcdo_applied(self):
        return 'monte_carlo_dropout' in self.config and self.config['monte_carlo_dropout']

    @staticmethod
    def mcdo_model_from_config(config):
        config = copy.copy(config)
        config['monte_carlo_dropout'] = True
        mcdo_model = TrainableImageProcessingTool.from_config(config)
        return mcdo_model

    @staticmethod
    def non_mcdo_model_from_config(config):
        config = copy.copy(config)
        if 'monte_carlo_dropout' in config:
            del config['monte_carlo_dropout']
        mcdo_model = TrainableImageProcessingTool.from_config(config)
        return mcdo_model
