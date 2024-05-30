import json
import math
from abc import ABCMeta, abstractmethod
from math import inf
from typing import Tuple, Optional, List, Any, Dict, Type, Union

import tensorflow
from tensorflow import Tensor
from tensorflow.keras.layers import Conv3D, GlobalAveragePooling3D, Flatten, Dropout, Activation, Concatenate, Dense, Add, Subtract, \
    LSTM, BatchNormalization, Conv2D
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation as RandomRotation2D
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2

from lib.count_trainable_params import try_counting_params
from lib.custom_layers import ShapedMultiplicativeGaussianNoise, ShapedGaussianNoise, RandomChoice, RandomBlur3D, PointWiseMultiplication, SpecificChannel, \
    Stack, MyInstanceNormalization, RandomBlur2D
from lib.my_logger import logging
from lib.parameter_search import InvalidParametersError, BadParametersError
from lib.random_rotate import RandomRotation3D, CropLayer
from lib.squeeze_and_excitation import SqueezeExciteBlock
from lib.util import LogicError, X, Y, Z, remove_duplicates_using_identity, required_size_for_safe_rotation_zyx
from load_data.patch_request import PatchRequest
from tasks import VertebraTasks, BinaryGenantScoreClassification, BinaryOutputCombination, GenantScoreClassification, GenantByDeformity, OutputCombination


class LayerCounter:
    class EmptyCounter(ValueError):
        pass

    def __init__(self, start, stop):
        if stop <= start:
            raise self.EmptyCounter(f'stop <= start ({stop} <= {start})')
        self.start = start
        self.stop = stop
        self.c = 0

    def increment(self):
        self.c += 1

    def layer_finished(self):
        self.increment()
        if self.c >= self.stop:
            raise StopIteration


class InvalidLayerNumber(InvalidParametersError):
    pass


class KerasModelBuilder(metaclass=ABCMeta):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if 'input_patches' in self.config:
            self.input_patches: List[PatchRequest] = [PatchRequest.from_json(r) for r in json.loads(self.config['input_patches'])]
        else:
            self.input_patches = self._default_legacy_input_patch_size(self.config)
        self.tasks = VertebraTasks.deserialize(self.config['tasks'])
        self.optimizer = Adam
        self.learn_rate = self.config['learn_rate']
        self.sigma_a_1 = self.config['sigma_a_1']
        self.sigma_a_2 = self.config['sigma_a_2']
        self.sigma_m_1 = self.config['sigma_m_1']
        self.sigma_m_2 = self.config['sigma_m_2']
        self.sigma_blur = self.config['sigma_blur']
        self.rotate_range_deg = self.config['rotate_range_deg']

    def _default_legacy_input_patch_size(self, config):
        if 'num_adjacent_vertebra_inputs' in config and config['num_adjacent_vertebra_inputs'] > 0:
            raise NotImplementedError('TODO implement compatibility')
        return [PatchRequest(
            size_mm=(config['input_shape_mm_x'], config['input_shape_mm_y'], config['input_shape_mm_z']),
            spacing=(config['spacing_x'], config['spacing_y'], config['spacing_z'])
        )]

    def input_shapes_px(self):
        return [p.size_px() for p in self.input_patches]

    def input_shapes_mm(self):
        return [p.size_mm for p in self.input_patches]

    def input_shape_px(self):
        shapes = set(self.input_shapes_px())
        assert len(shapes) == 1
        return shapes.pop()

    def input_shapes_px_after_rotation(self):
        return self.input_shapes_px()

    def input_shape_px_after_rotation(self):
        return self.input_shape_px()

    def input_shape_px_after_rotation_2d(self):
        shape_3d = self.input_shape_px()
        assert len(shape_3d) == 3
        assert shape_3d[2] == 1
        return shape_3d[:2]

    def input_shape_px_before_rotation(self) -> Union[Tuple[Z, Y, X], Tuple[Z, Y]]:
        rotate_range_deg = self.rotate_range_deg
        input_shape_px_after_rotation = self.input_shape_px_after_rotation()
        result = required_size_for_safe_rotation_zyx(input_shape_px_after_rotation, rotate_range_deg, ceil=True)
        if self.ndim() == 2:
            assert result[-1] == 1
            result = result[:-1]
        assert len(result) == self.ndim()
        return result

    def rotation_as_fraction_of_2_pi(self):
        # return self.rotate_range_deg / 180 * math.pi / 2 / math.pi
        return self.rotate_range_deg / 360

    def default_input(self, input_=None):
        if input_ is not None:
            return input_
        return Input(shape=(*self.input_shape_px_before_rotation(), 1))

    def num_vertebra_inputs(self):
        return len(self.input_patches)

    def build(self) -> Model:
        if self.num_vertebra_inputs() < 1:
            raise ValueError

        model_psp, volume_inputs = self.backbone()
        outputs = self.create_outputs(model_psp)
        model = Model(inputs=volume_inputs, outputs=outputs)
        self.compile_built_model(model)

        try_counting_params(model)
        # model.summary(line_length=150)
        return model

    @abstractmethod
    def backbone(self) -> Tuple[Tensor, List[Tensor]]:
        pass

    def compile_built_model(self, model, classification_threshold_for_task: Dict[str, float] = None):
        run_eagerly = False
        if run_eagerly:
            logging.warning('Running eagerly. Disable this for better performance. Only for debugging.')
        model.compile(self.optimizer(learning_rate=self.learn_rate),
                      loss=(self.loss_dict()),
                      loss_weights=(self.loss_weights_dict()),
                      metrics=(self.metrics_dict(classification_threshold_for_task)),
                      weighted_metrics=self.weighted_metrics_dict(),
                      run_eagerly=run_eagerly
                      )

    def loss_weights_dict(self):
        return {
            task.output_layer_name(): task.loss_weight
            for task in self.tasks
        }

    def metrics_dict(self, classification_threshold_for_task: Dict[str, float] = None) -> Dict[str, List[Union[str, Metric]]]:
        return {
            task.output_layer_name(): (
                task.metrics(threshold=classification_threshold_for_task[task.output_layer_name()])
                if classification_threshold_for_task is not None
                   and task.output_layer_name() in classification_threshold_for_task
                   and classification_threshold_for_task[task.output_layer_name()] is not None
                else task.metrics()
            )
            for task in self.tasks
        }

    def weighted_metrics_dict(self):
        return {
            task.output_layer_name(): task.weighted_metrics()
            for task in self.tasks
        }

    def loss_dict(self):
        return {
            task.output_layer_name(): task.loss_function()
            for task in self.tasks
        }

    def create_outputs(self, model_psp):
        return [
            task.build_output_layers_tf(model_psp)
            for task in self.tasks
        ]

    def add_artificial_outputs_to_trained_model(self,
                                                model: Optional[Model],
                                                thresholds_for_binary_outputs: Dict[str, float] = None) -> Tuple[Model, Dict[str, Any]]:
        """
        :param model: if this is None only the config is returned
        """
        new_outputs = []
        new_tasks: List[OutputCombination] = self.genant_score_binary_combinations(thresholds_for_binary_outputs)
        new_tasks += self.genant_by_deformity()
        if len(new_tasks) == 0:
            return model, self.config

        if model is not None:
            original_outputs = model.outputs
            for new_task in new_tasks:
                assert len(model.outputs) == len(model.output_names)
                new_outputs.append(new_task.build_output_layer([o for o, n in zip(model.outputs, model.output_names)
                                                                if n in new_task.input_task_names()]))
            model = Model(inputs=model.inputs, outputs=original_outputs + new_outputs)
        config = self.config.copy()
        assert len(set(t.output_layer_name() for t in self.tasks + new_tasks)) == len(self.tasks) + len(new_tasks)
        config['tasks'] = VertebraTasks(self.tasks + new_tasks).serialize()
        return model, config

    def genant_by_deformity(self) -> List[GenantByDeformity]:
        results = []
        for deformity_as_zero in [True, False]:
            gbd = GenantByDeformity(deformity_as_zero=deformity_as_zero)

            if set(gbd.input_task_names()).issubset(self.tasks.output_layer_names()):
                assert gbd.output_layer_name() not in self.tasks.output_layer_names()
                results.append(gbd)
        return results

    def genant_score_binary_combinations(self, thresholds_for_binary_outputs: Dict[str, float] = None) -> List[BinaryOutputCombination]:
        gs_combined_tasks = []
        ts = thresholds_for_binary_outputs
        for deformity_as_zero in [True, False]:
            based_on_output_names = [
                BinaryGenantScoreClassification(math.nan, [0, ], [1, 2, 3], deformity_as_zero).output_layer_name(),
                BinaryGenantScoreClassification(math.nan, [0, 1, ], [2, 3], deformity_as_zero).output_layer_name(),
                BinaryGenantScoreClassification(math.nan, [0, 1, 2, ], [3], deformity_as_zero).output_layer_name(),
            ]
            if all(n in self.tasks.output_layer_names() for n in based_on_output_names):
                binary_tasks = [t for t in self.tasks if t.output_layer_name() in based_on_output_names]
                assert len(binary_tasks) == len(based_on_output_names)
                kwargs = {
                    'binary_tasks': binary_tasks,
                    'y_true_from': GenantScoreClassification(
                        deformity_as_zero=deformity_as_zero,
                        loss_weight=1.0,
                    ),
                    'add_brackets_to_classes': True,
                    'thresholds_for_binary_outputs': ts
                }
                # gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['0'],
                #                                                           '1': ['123', '01', ],
                #                                                           '2': ['123', '23', '012', ],
                #                                                           '3': ['123', '23', '3', ], },
                #                                                  name=f'0v123_01v23_012v3',
                #                                                  **kwargs), )  # 012
                # gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['0', '01', '012'],
                #                                                           '1': ['123', '01', '012', ],
                #                                                           '2': ['23', '012', ],
                #                                                           '3': ['3', ], },
                #                                                  name='012v3_01v23_0v123',
                #                                                  **kwargs), )  # 210
                # gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['0'],
                #                                                           '1': ['123', '012', '01'],
                #                                                           '2': ['123', '012', '23', ],
                #                                                           '3': ['123', '3', ], },
                #                                                  name='0v123_012v3_01v23',
                #                                                  **kwargs))  # 021
                # gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['012', '0'],
                #                                                           '1': ['123', '01', '012', ],
                #                                                           '2': ['123', '23', '012', ],
                #                                                           '3': ['3', ], },
                #                                                  name='012v3_0v123_01v23',
                #                                                  **kwargs))  # 201
                gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['01', '012', '0'],
                                                                          '1': ['01', '012', '123'],
                                                                          '2': ['23', '012', '123'],
                                                                          '3': ['23', '3', ], },
                                                                 name='01v23_012v3_0v123',
                                                                 **kwargs))  # 120
                # gs_combined_tasks.append(BinaryOutputCombination(classes={'0': ['01', '0'],
                #                                                           '1': ['01', '123', '012'],
                #                                                           '2': ['23', '123', '012'],
                #                                                           '3': ['23', '123', '3', ], },
                #                                                  name='01v23_0v123_012v3',
                #                                                  **kwargs))  # 102
        return gs_combined_tasks

    def rotation_layer(self, x):
        if self.ndim() == 3:
            x = RandomRotation3D(axis=2,
                                 max_abs_angle_deg=self.rotate_range_deg,
                                 crop_to_size=self.input_shape_px_after_rotation(), )(x)
        elif self.ndim() == 2:
            x = RandomRotation2D(self.rotation_as_fraction_of_2_pi(),
                                 fill_mode='constant',
                                 interpolation='bilinear',
                                 fill_value=0)(x)
            x = CropLayer(self.input_shape_px_after_rotation_2d())(x)
        else:
            raise RuntimeError(self.ndim())
        return x

    def data_augmentation(self, x):
        before_noise_and_blur = x
        if self.sigma_m_1 != 0:
            x = ShapedMultiplicativeGaussianNoise(std=self.sigma_m_1)(before_noise_and_blur)
        if self.sigma_m_2 != 0:
            x = ShapedMultiplicativeGaussianNoise(std=self.sigma_m_2, size=(1,) * (self.ndim() + 1))(x)
        if self.sigma_a_1 != 0:
            x = ShapedGaussianNoise(std=self.sigma_a_1)(x)
        if self.sigma_a_2 != 0:
            x = ShapedGaussianNoise(std=self.sigma_a_2, size=(1,) * (self.ndim() + 1))(x)
        noised = x
        x = before_noise_and_blur
        if self.sigma_blur is not None and self.sigma_blur != 0:
            assert self.sigma_blur > 0
            if self.ndim() == 3:
                blur = RandomBlur3D
            elif self.ndim() == 2:
                blur = RandomBlur2D
            else:
                raise RuntimeError
            x = RandomChoice()(
                [x, blur(self.sigma_blur, blur_probability=1, min_filter_size=3, filters=x.shape.as_list()[-1])(before_noise_and_blur)])
        blurred = x

        xs = [before_noise_and_blur, noised, blurred]
        xs = remove_duplicates_using_identity(xs)
        if len(xs) > 1:
            x = RandomChoice()(xs)
        elif len(xs) == 1:
            x = xs[0]
        else:
            raise LogicError('list of set of some values can not be empty')
        return x

    def ndim(self):
        return self.config['conv_ndim']

    @staticmethod
    def custom_layers():
        custom_layers = {
            klass.__name__: klass
            # If your model uses any custom layers, add them here
            for klass in [ShapedGaussianNoise,
                          ShapedMultiplicativeGaussianNoise,
                          MyInstanceNormalization,
                          RandomRotation3D,
                          PointWiseMultiplication,
                          RandomBlur3D,
                          RandomBlur2D,
                          SqueezeExciteBlock, ]
        }
        custom_layers['RandomRotation'] = RandomRotation3D # backwards compatibility
        return custom_layers


class FNetBuilder(KerasModelBuilder):
    NUM_DATA_AUGMENTATION_LAYERS = 3

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dropout_rate = self.config['dropout_rate']
        if 'monte_carlo_dropout' in config:
            self.monte_carlo_dropout = config['monte_carlo_dropout']
            assert self.dropout_rate > 0
        else:
            self.monte_carlo_dropout = False
        self.num_conv_blocks = self.config['num_conv_blocks']
        self.num_dense_layers = self.config['num_dense_layers']
        self.num_hidden_units = self.config['num_hidden_units']
        self.batch_normalization = self.config['batch_normalization']
        self.instance_normalization = self.config['instance_normalization']
        self.validate_normalization_parameters()
        self.kernel_size = self.config['kernel_size']
        self.conv_block_size = self.config['conv_block_size']
        self.num_filters = self.config['num_filters']
        self.num_filters_exponent = self.config['num_filters_exponent']
        if 'use_squeeze_excitation' in self.config:
            self.use_squeeze_excitation = self.config['use_squeeze_excitation']
        else:
            self.use_squeeze_excitation = False
        if 's_e_placement' in self.config:
            self.s_e_placement = self.config['s_e_placement']
        else:
            self.s_e_placement = False
        self.initial_conv_layer_size = self.config['initial_conv_layer_size']
        self.use_gap = self.config['use_gap']
        self.l2_regularizer = self.config['l2_regularizer']
        self.regularize_bias = self.config['regularize_bias']
        self.merge_after_layer = self.config['merge_after_layer']
        self.merge_paths_method = self.config['merge_paths_method']
        if 'independent_augmentation' in self.config:
            self.independent_augmentation: bool = self.config['independent_augmentation']
        else:
            assume = self.merge_after_layer >= self.NUM_DATA_AUGMENTATION_LAYERS
            logging.warning(f'You forgot to specify `independent_augmentation`. Assuming `independent_augmentation = {assume}`')
            self.independent_augmentation = assume
        if 'merge_paths_shared_weights' in self.config:
            self.merge_paths_shared_weights: bool = self.config['merge_paths_shared_weights']
        else:
            logging.warning('You forgot to specify `merge_paths_shared_weights`. Assuming `merge_paths_shared_weights = False`')
            self.merge_paths_shared_weights = False
        if 'initialize_extra_paths_with_zero_weights' in self.config:
            self.initialize_extra_paths_with_zero_weights = self.config['initialize_extra_paths_with_zero_weights']
        elif 'adjacent_init_zero_weights' in self.config:  # deprecated, backwards compatibility
            self.initialize_extra_paths_with_zero_weights = self.config['adjacent_init_zero_weights']
        if 'fully_convolutional' not in self.config:
            self.config['fully_convolutional'] = False
        self.fully_convolutional = self.config['fully_convolutional']

        if self.independent_augmentation and self.merge_after_layer < self.NUM_DATA_AUGMENTATION_LAYERS:
            # Cannot merge before independent augmentations
            raise InvalidLayerNumber()

        if self.regularize_bias:
            self.regularizers = {
                "kernel_regularizer": l2(self.l2_regularizer),
                "bias_regularizer": l2(self.l2_regularizer)
            }
        else:
            self.regularizers = {
                "kernel_regularizer": l2(self.l2_regularizer),
                "bias_regularizer": None
            }

    def squeeze_excite_block_to_use(self) -> Type[Layer]:
        if isinstance(self.use_squeeze_excitation, str):
            from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
            cls = _GLOBAL_CUSTOM_OBJECTS[self.use_squeeze_excitation]
            assert issubclass(cls, Layer)
            return cls
        raise ValueError(self.use_squeeze_excitation)

    @staticmethod
    def last_layer_of_tensor(x: Tensor):
        assert list(x._keras_history[0].output_shape) == x.shape.as_list()
        return x._keras_history[0]

    def ConvND(self) -> Type[Conv]:
        if self.ndim() == 2:
            return Conv2D
        elif self.ndim() == 3:
            return Conv3D

    # the following disables inpsection of unbound local variables, because pycharm can not know how I bind them but I want to do it this way
    # noinspection PyUnboundLocalVariable
    def single_vertebra_path(self, start, stop, input_to_path=None):
        fwl: Optional[self.ConvND()] = None  # first layer with weights

        if start > 0:
            if input_to_path is None:
                raise ValueError
            x = input_to_path

        if start == stop:
            return input_to_path, x, fwl
        counter = LayerCounter(start=start, stop=stop)

        try:
            if counter.c >= start:
                x = self.default_input(input_to_path)
                input_to_path = x
            counter.layer_finished()
            if self.rotate_range_deg != 0:
                if counter.c >= start:
                    x = self.rotation_layer(x)
            counter.layer_finished()

            if counter.c >= start:
                x = self.data_augmentation(x)
            counter.layer_finished()
            assert self.NUM_DATA_AUGMENTATION_LAYERS == counter.c

            if self.initial_conv_layer_size is None:
                pass
            elif self.initial_conv_layer_size == 'default':
                self.num_conv_blocks += 1
            else:
                if counter.c >= start:
                    x = self.ConvND()(self.initial_conv_layer_size,
                                      kernel_size=self.kernel_size,
                                      padding=self.config['conv_padding'],
                                      strides=self.config['conv_block_strides'],
                                      **self.regularizers)(x)
                    if fwl is None:
                        fwl = self.last_layer_of_tensor(x)
                    x = self.normalization_activation_and_dropout_layers(x)
                counter.layer_finished()

            if isinstance(self.num_conv_blocks, float):
                self.num_conv_blocks = round(self.num_conv_blocks)
            for idx in range(self.num_conv_blocks):
                num_filters_this_block = self.num_filters_in_block(idx)
                num_filters_next_block = self.num_filters_in_block((idx + 1))
                self.check_if_layer_has_no_filters(num_filters_this_block)
                self.check_if_layer_has_no_filters(num_filters_next_block)
                if isinstance(self.conv_block_size, float):
                    self.conv_block_size = round(self.conv_block_size)

                for _ in range(self.conv_block_size - 1):
                    if counter.c >= start:
                        x = self.ConvND()(num_filters_this_block,
                                          kernel_size=self.kernel_size,
                                          padding=self.config['conv_padding'],
                                          **self.regularizers)(x)
                        if fwl is None:
                            fwl = self.last_layer_of_tensor(x)
                        x = self.normalization_activation_and_dropout_layers(x)
                    counter.layer_finished()
                    place_se_layer_here = self.use_squeeze_excitation and (
                            self.s_e_placement in ['after_conv_layer'] or
                            self.s_e_placement == 'every_second_conv_layer' and counter.c % (2 + 1) == 0
                    )
                    if place_se_layer_here:
                        if counter.c >= start:
                            x = self.squeeze_excite_block_to_use()()(x)
                            if fwl is None:
                                fwl = self.last_layer_of_tensor(x)
                        counter.layer_finished()
                if counter.c >= start:
                    x = self.ConvND()(num_filters_next_block,
                                      kernel_size=self.kernel_size,
                                      padding=self.config['conv_padding'],
                                      strides=self.config['conv_block_strides'],
                                      **self.regularizers)(x)
                    if fwl is None:
                        fwl = self.last_layer_of_tensor(x)
                    x = self.normalization_activation_and_dropout_layers(x)
                counter.layer_finished()
                if self.use_squeeze_excitation and self.s_e_placement in ['after_conv_block', 'after_conv_layer']:
                    if counter.c >= start:
                        x = self.squeeze_excite_block_to_use()()(x)
                        if fwl is None:
                            fwl = self.last_layer_of_tensor(x)
                    counter.layer_finished()
            if counter.c >= start:
                x = self.gap_or_flatten(x)
            counter.layer_finished()

            for idx in range(self.num_dense_layers):
                if counter.c >= start:
                    x = self.dense_or_1_by_1_conv(x, self.num_hidden_units)(x)
                    if self.dropout_rate != 0:
                        x = self.dropout_type()(rate=self.dropout_rate)(x)
                    if fwl is None:
                        fwl = self.last_layer_of_tensor(x)
                counter.layer_finished()
        except StopIteration:
            pass
        except ValueError as e:
            if 'Negative dimension size caused by subtracting' in str(e):
                raise InvalidParametersError
            else:
                raise
        if counter.c < start:
            logging.error(f'Not enough layers: Only {counter.c} layers but you want to merge after layer {start}')
            raise InvalidLayerNumber()
        return input_to_path, x, fwl

    def dropout_type(self):
        if self.monte_carlo_dropout:
            from lib.custom_layers import MCDropout
            return MCDropout
        else:
            return Dropout

    def gap_or_flatten(self, x):
        if self.use_gap:
            x = GlobalAveragePooling3D()(x)
        elif self.fully_convolutional:
            pass  # no need to flatten
        else:
            x = Flatten()(x)
        return x

    @staticmethod
    def check_if_layer_has_no_filters(num_filters_this_block):
        if num_filters_this_block == 0:
            # 'found empty layer'
            raise InvalidParametersError()

    def num_filters_in_block(self, idx):
        return round(self.num_filters / self.num_filters_exponent ** (self.num_conv_blocks - idx))

    def normalization_activation_and_dropout_layers(self, x):
        if self.batch_normalization == 'before_activation':
            x = BatchNormalization()(x)
        if self.instance_normalization == 'before_activation':
            x = MyInstanceNormalization()(x)
        x = Activation('relu')(x)
        if self.batch_normalization == 'after_activation':
            x = BatchNormalization()(x)
        if self.instance_normalization == 'after_activation':
            x = MyInstanceNormalization()(x)
        if self.dropout_rate != 0:
            x = self.dropout_type()(rate=self.dropout_rate)(x)
        return x

    def backbone(self) -> Tuple[Tensor, List[Tensor]]:
        vertebra_path_outputs, volume_inputs = self.build_parallel_paths()
        merged = self.merge_parallel_paths(vertebra_path_outputs)
        if self.merge_after_layer != 0 and self.num_vertebra_inputs() == 1:
            raise InvalidLayerNumber
        _, model_psp, first_weight_layer = self.single_vertebra_path(start=self.merge_after_layer + 1, stop=inf, input_to_path=merged)
        if self.initialize_extra_paths_with_zero_weights and first_weight_layer is not None and self.merge_paths_method in ['concatenate']:
            self.initialize_weights_to_adjacent_vertebra_paths_with_zeros(first_weight_layer)
        return model_psp, volume_inputs

    def initialize_weights_to_adjacent_vertebra_paths_with_zeros(self, first_weight_layer):
        assert isinstance(first_weight_layer, Dense) or isinstance(first_weight_layer, self.ConvND())
        initial_weights = first_weight_layer.get_weights()
        assert initial_weights[0].shape[-2] % self.num_vertebra_inputs() == 0  # input channels
        input_channels = initial_weights[0].shape[-2] // self.num_vertebra_inputs()
        center_start_idx = input_channels * (self.num_vertebra_inputs() // 2)
        center_end_idx = input_channels * (self.num_vertebra_inputs() // 2 + 1)
        initial_weights[0][..., :center_start_idx, :] = 0
        initial_weights[0][..., center_end_idx:, :] = 0
        first_weight_layer.set_weights(initial_weights)
        del initial_weights

    def merge_parallel_paths(self, vertebra_path_outputs, method=None):
        if method is None:
            method = self.merge_paths_method
        if len(vertebra_path_outputs) == 1:
            merged = vertebra_path_outputs[0]
        elif len(vertebra_path_outputs) > 1:
            if method == 'concatenate':
                merged = Concatenate(axis=-1)(vertebra_path_outputs)
                assert not self.same_shape(merged, vertebra_path_outputs[0])
            elif method == 'sum':
                merged = Add()(vertebra_path_outputs)
                assert self.same_shape(merged, vertebra_path_outputs[0])
            elif method == 'bilstm':
                units = vertebra_path_outputs[0].shape[
                    -1 if tensorflow.keras.backend.image_data_format == 'channels_last' else 1
                ]
                stacked = Stack(axis=1)(vertebra_path_outputs)
                if len(vertebra_path_outputs[0].shape) == 2:
                    lstm = LSTM(units=units, return_sequences=False)
                elif len(vertebra_path_outputs[0].shape) == self.ndim() + 2:
                    try:
                        from tensorflow.python.keras.layers import ConvLSTM3D
                    except ImportError:
                        # bilstm merging is not possible after a convolutional layer in tensorflow 2.5 or below
                        raise InvalidLayerNumber()
                    lstm = ConvLSTM3D(filters=units,
                                      kernel_size=(1, 1, 1),
                                      padding=self.config['conv_padding'],
                                      return_sequences=False)
                else:
                    raise InvalidParametersError(len(vertebra_path_outputs[0].shape))
                merged = Bidirectional(lstm, merge_mode='ave')(stacked)
                assert self.same_shape(merged, vertebra_path_outputs[0])
            elif method == 'difference_to_mean':
                if len(vertebra_path_outputs) % 2 != 1:
                    raise NotImplementedError
                main_path_idx = math.floor(len(vertebra_path_outputs) / 2)
                other_path_sum = Add()([
                    path for path_idx, path in enumerate(vertebra_path_outputs)
                    if path_idx != main_path_idx
                ])
                other_path_mean = PointWiseMultiplication(1 / (len(vertebra_path_outputs) - 1), name=f'divide_by_{(len(vertebra_path_outputs) - 1)}')(
                    other_path_sum)
                merged = Subtract()([vertebra_path_outputs[main_path_idx], other_path_mean])
                assert self.same_shape(merged, vertebra_path_outputs[0])
            else:
                raise InvalidParametersError
        else:
            raise LogicError
        return merged

    @staticmethod
    def same_shape(tensor1, tensor2):
        return tensor1.shape.as_list() == tensor2.shape.as_list()

    def build_parallel_paths(self) -> Tuple[List[Tensor], List[Tensor]]:
        volume_inputs = []
        vertebra_path_outputs = []
        augmented_inputs = []

        if not self.independent_augmentation:
            first_layer_outputs = []
            for _ in self.input_patches:
                i, o, _ = self.single_vertebra_path(start=0, stop=1)
                volume_inputs.append(i)
                first_layer_outputs.append(o)
            all_inputs_as_one_tensor = self.merge_parallel_paths(first_layer_outputs, method='concatenate')
            _, after_data_augmentation, _ = self.single_vertebra_path(start=1,
                                                                      stop=min(self.NUM_DATA_AUGMENTATION_LAYERS, self.merge_after_layer + 1),
                                                                      input_to_path=all_inputs_as_one_tensor)
            for channel_idx in range(self.num_vertebra_inputs()):
                if self.num_vertebra_inputs() > 1:
                    augmented_inputs.append(SpecificChannel(channel_idx=channel_idx)(after_data_augmentation))
                else:
                    augmented_inputs.append(after_data_augmentation)

        if self.merge_paths_shared_weights:
            if self.independent_augmentation:
                nested_input, nested_output, _ = self.single_vertebra_path(start=0, stop=self.merge_after_layer + 1)
            else:
                nested_input, nested_output, _ = self.single_vertebra_path(start=min(self.NUM_DATA_AUGMENTATION_LAYERS, self.merge_after_layer + 1),
                                                                           stop=self.merge_after_layer + 1,
                                                                           input_to_path=self.input_like(augmented_inputs[0]))
            nested_model = Model(inputs=nested_input, outputs=nested_output)
            for input_idx in range(self.num_vertebra_inputs()):
                if self.independent_augmentation:
                    i = self.default_input()
                    volume_inputs.append(i)
                else:
                    i = augmented_inputs[input_idx]
                o = nested_model(i)
                vertebra_path_outputs.append(o)
        else:
            for input_idx in range(self.num_vertebra_inputs()):
                if self.independent_augmentation:
                    i, o, _ = self.single_vertebra_path(start=0, stop=self.merge_after_layer + 1)
                    volume_inputs.append(i)
                else:
                    i, o, _ = self.single_vertebra_path(start=min(self.NUM_DATA_AUGMENTATION_LAYERS, self.merge_after_layer + 1),
                                                        stop=self.merge_after_layer + 1,
                                                        input_to_path=augmented_inputs[input_idx])
                vertebra_path_outputs.append(o)
        return vertebra_path_outputs, volume_inputs

    def validate_normalization_parameters(self):
        if self.batch_normalization not in ['before_activation', 'after_activation', 'not']:
            raise NotImplementedError()
        if self.instance_normalization not in ['before_activation', 'after_activation', 'not']:
            raise NotImplementedError()
        if self.batch_normalization != 'not' and self.instance_normalization != 'not':
            raise BadParametersError()

    def dense_or_1_by_1_conv(self, x, units, **kwargs) -> Layer:
        input_shape = x.shape.as_list()
        conv_layer_before = len(input_shape) == 2 + self.ndim()
        dense_layer_or_gap_before = len(input_shape) == 2
        if not self.fully_convolutional:
            return Dense(units=units, **kwargs, **self.regularizers)
        elif conv_layer_before:
            return self.ConvND()(filters=units, kernel_size=x.shape.as_list()[1:1 + self.ndim()], **kwargs, **self.regularizers)
        elif dense_layer_or_gap_before:
            return Dense(units=units, **kwargs, **self.regularizers)
        else:
            raise NotImplementedError

    @staticmethod
    def input_like(i: Input):
        return Input(shape=i.shape[1:], batch_size=i.shape[0])

    def trauma_binary_combinations(self, thresholds_for_binary_outputs: Dict[str, float] = None) -> List[BinaryOutputCombination]:
        from trauma_tasks import IsFracturedClassification, IsFreshClassification, IsUnstableClassification, \
            CombinedFreshUnstableClassification
        combined_tasks = []
        ts = thresholds_for_binary_outputs
        based_on_output_names = [
            IsFracturedClassification().output_layer_name(),
            IsFreshClassification().output_layer_name(),
            IsUnstableClassification().output_layer_name(),
        ]
        if all(n in self.tasks.output_layer_names() for n in based_on_output_names):
            binary_tasks = [t for t in self.tasks if t.output_layer_name() in based_on_output_names]
            assert len(binary_tasks) == len(based_on_output_names)
            kwargs = {
                'binary_tasks': binary_tasks,
                'y_true_from': CombinedFreshUnstableClassification(),
                'add_brackets_to_classes': True,
                'thresholds_for_binary_outputs': ts
            }
            cls0 = 'Unfractured'
            cls1 = 'Old∕Subacute+Stable'
            cls2 = 'Fresh+Stable'
            cls3 = 'Unstable'
            cls123 = 'Fractured'
            cls01 = 'Not Fresh'
            cls23 = 'Fresh'
            cls012 = 'Stable'
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls0],
                                                                   cls1: [cls123, cls01, ],
                                                                   cls2: [cls123, cls23, cls012, ],
                                                                   cls3: [cls123, cls23, cls3, ], },
                                                          name=f'hafo_hffo_hufo',
                                                          **kwargs), )  # 012
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls0, cls01, cls012],
                                                                   cls1: [cls123, cls01, cls012, ],
                                                                   cls2: [cls23, cls012, ],
                                                                   cls3: [cls3, ], },
                                                          name='hufo_hffo_hafo',
                                                          **kwargs), )  # 210
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls0],
                                                                   cls1: [cls123, cls012, cls01],
                                                                   cls2: [cls123, cls012, cls23, ],
                                                                   cls3: [cls123, cls3, ], },
                                                          name='hafo_hufo_hffo',
                                                          **kwargs))  # 021
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls012, cls0],
                                                                   cls1: [cls123, cls01, cls012, ],
                                                                   cls2: [cls123, cls23, cls012, ],
                                                                   cls3: [cls3, ], },
                                                          name='hufo_hafo_hffo',
                                                          **kwargs))  # 201
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls01, cls012, cls0],
                                                                   cls1: [cls01, cls012, cls123],
                                                                   cls2: [cls23, cls012, cls123],
                                                                   cls3: [cls23, cls3, ], },
                                                          name='hffo_hufo_hafo',
                                                          **kwargs))  # 120
            combined_tasks.append(BinaryOutputCombination(classes={cls0: [cls01, cls0],
                                                                   cls1: [cls01, cls123, cls012],
                                                                   cls2: [cls23, cls123, cls012],
                                                                   cls3: [cls23, cls123, cls3, ], },
                                                          name='hffo_hafo_hufo',
                                                          **kwargs))  # 102
        return combined_tasks


def required_size_for_model_in_mm(model, spacing: Tuple[X, Y, Z]):
    if isinstance(model.input_shape, list):
        for model_input_shape in model.input_shape:
            assert model_input_shape == model.input_shape[0]
        model_input_shape = model.input_shape[0]
    else:
        model_input_shape = model.input_shape
    required_size_mm: Tuple[X, Y, Z] = required_size_for_input_shape_in_mm(model_input_shape, spacing)
    return required_size_mm


def required_size_for_input_shape_in_mm(model_input_shape, spacing: Tuple[X, Y, Z]) -> Tuple[int, int, int]:
    return tuple(m * s for m, s in zip(model_input_shape[-2:0:-1], spacing))
