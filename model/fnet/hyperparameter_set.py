import json
import pickle
from abc import abstractmethod
from math import inf
from typing import List, Dict, Any

import cachetools

from lib import parameter_search
from lib.parameter_search import Parameter, InvalidParametersError, ConstantParameter, ExponentialParameter, ExponentialIntegerParameter
from lib.util import EBC
from load_data.patch_request import PatchRequest
from model.fnet import const
from model.fnet.const import TASKS, EXPERIMENT_NAME
from tasks import VertebraTasks


class HyperparameterSet(EBC):
    def __init__(self):
        self.tasks_parameter = parameter_search.BinaryParameter('tasks', TASKS.serialize(), repr(pickle.dumps(TASKS.main_task_twice(second_loss_weight=0.01))))
        self.swa_start_batch_parameter = parameter_search.ConstantParameter('swa_start_batch', self.steps_per_epoch() - 1)
        self.swa_cycle_length_parameter = parameter_search.ConstantParameter('swa_cycle_length', self.steps_per_epoch())
        self.steps_per_epoch_parameter = parameter_search.ConstantParameter('steps_per_epoch', self.steps_per_epoch())
        self.experiment_name_parameter = self._get_experiment_name_parameter()
        self.model_subdir_parameter = self._get_model_subdir_parameter()

        self.sigma_a_1_parameter = parameter_search.ExponentialParameter('sigma_a_1', 0.1, 2, minimum=0.05, maximum=0.2)
        self.sigma_a_2_parameter = parameter_search.ExponentialParameter('sigma_a_2', 0.05, 2, minimum=0.05, maximum=0.2)
        self.sigma_m_2_parameter = parameter_search.ExponentialParameter('sigma_m_2', 0.16, 2, maximum=0.32)
        self.sigma_blur_parameter = parameter_search.ExponentialParameter('sigma_blur', 1.5, 2, maximum=1.5, minimum=0.375)
        self.random_flip_lr_parameter = parameter_search.ConstantParameter('random_flip_lr', True)
        self.random_flip_ud_parameter = parameter_search.BooleanParameter('random_flip_ud', False)
        self.random_flip_fb_parameter = parameter_search.BooleanParameter('random_flip_fb', False)
        self.pad_augment_ratio_z_parameter = parameter_search.LinearParameter('pad_augment_ratio_z', 0.4, 0.2, minimum=0.2, maximum=0.6)
        self.pad_augment_ratio_y_parameter = parameter_search.LinearParameter('pad_augment_ratio_y', 0.2, 0.2, minimum=0.2, maximum=0.6)
        self.pad_augment_ratio_x_parameter = parameter_search.LinearParameter('pad_augment_ratio_x', 0., 0.2, minimum=0., maximum=0.6)
        self.rotate_range_deg_parameter = parameter_search.LinearParameter('rotate_range_deg', 18, 6, minimum=0, maximum=42)
        self.random_shift_px_x_parameter = parameter_search.LinearIntegerParameter('random_shift_px_x', 4, 2, minimum=2, maximum=16)
        self.random_shift_px_y_parameter = parameter_search.LinearIntegerParameter('random_shift_px_y', 4, 2, minimum=2, maximum=16)
        self.random_shift_px_z_parameter = parameter_search.LinearIntegerParameter('random_shift_px_z', 4, 2, minimum=2, maximum=16)
        self.random_project_x_parameter = parameter_search.LinearParameter('random_project_x', 0.1, 0.1, minimum=0, maximum=1.)
        self.random_project_y_parameter = parameter_search.LinearParameter('random_project_y', 0.1, 0.1, minimum=0, maximum=1.)
        self.random_project_z_parameter = parameter_search.LinearParameter('random_project_z', 0.1, 0.1, minimum=0, maximum=1.)
        self.input_patches_parameter = parameter_search.ConstantParameter('input_patches', json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 1, 1),
                                                        input_size_px=(31, 47, 47)).to_json()
        ]))
        self.rotate_range_deg_parameter = parameter_search.LinearParameter('rotate_range_deg', 18, 6, minimum=0, maximum=42)
        self.pixel_scaling_parameter = parameter_search.TernaryParameter('pixel_scaling', 'divide_by_2k', 'range01_prescale', 'range01')
        self.class_weighting_parameter = parameter_search.ConstantParameter('class_weighting', 'per_output')
        self.vertebra_levels_parameter = parameter_search.ConstantParameter('restrict_to_vertebra_levels', json.dumps(None))
        self.size_adjustment_parameter = parameter_search.ConstantParameter('vertebra_size_adjustment', None)
        self.training_data_amount_parameter = parameter_search.ConstantParameter('training_data_amount', 1)
        self.exclude_patients_parameter = parameter_search.ConstantParameter('exclude_patients_from_training',
                                                                             json.dumps([]))
        self.batch_size_parameter = parameter_search.ConstantParameter('batch_size', 64)
        self.hu_window_max_parameter = parameter_search.ConstantParameter('hu_window_max', inf)
        self.hu_window_min_parameter = parameter_search.ConstantParameter('hu_window_min', 0.)
        self.epochs_parameter = parameter_search.ConstantParameter('epochs',
                                                                   const.FIXED_TRAIN_EPOCHS if const.FIXED_TRAIN_EPOCHS is not None
                                                                   else self.default_num_epochs())
        self.label_smoothing_parameter = parameter_search.ListParameter('label_smoothing', 0., [0., 1e-6, 1e-5, 1e-4, 0.001,
                                                                                                # 0.01, 0.02, 0.05, 0.1, 0.5
                                                                                                ], circle=True)
        self.fully_convolutional_parameter = parameter_search.ConstantParameter('fully_convolutional', False)
        self.early_stopping_metric_parameter = parameter_search.BinaryParameter('early_stopping_metric', None, 'loss')  # None means default criterion

        self.conv_ndim_parameter = parameter_search.ConstantParameter('conv_ndim', 3)

    def data_loader_parameters(self):
        return [
            ConstantParameter('shuffle_labels', False),
            *self.augmentation_parameters(),
            self.input_patches_parameter,
            self.rotate_range_deg_parameter,
            self.class_weighting_parameter,
            self.vertebra_levels_parameter,
            self.size_adjustment_parameter,
            self.training_data_amount_parameter,
            self.exclude_patients_parameter,
            self.batch_size_parameter,
            self.hu_window_min_parameter,
            self.hu_window_max_parameter,
            self.label_smoothing_parameter,
            ConstantParameter('ignore_border_vertebrae', False),
            ConstantParameter('exclude_site_6', False),
            ConstantParameter('additional_random_crops', 0.),
            self.pixel_scaling_parameter,
            self.conv_ndim_parameter
        ]

    def augmentation_parameters(self):
        return [
            self.random_flip_lr_parameter,
            self.random_flip_ud_parameter,
            self.random_flip_fb_parameter,
            self.random_shift_px_x_parameter,
            self.random_shift_px_y_parameter,
            self.random_shift_px_z_parameter,
            self.random_project_x_parameter,
            self.random_project_y_parameter,
            self.random_project_z_parameter,
            self.sigma_a_1_parameter,
            self.sigma_a_2_parameter,
            parameter_search.ExponentialParameter('sigma_m_1', 0, 2),
            self.sigma_m_2_parameter,
            self.sigma_blur_parameter,
            self.pad_augment_ratio_z_parameter,
            self.pad_augment_ratio_y_parameter,
            self.pad_augment_ratio_x_parameter,
            parameter_search.ConstantParameter('pad_augment_probability_z', 0.25),
            parameter_search.ConstantParameter('pad_augment_probability_y', 0.25),
            parameter_search.ConstantParameter('pad_augment_probability_x', 0.),
            self.rotate_range_deg_parameter,
        ]

    def hyper_parameters(self) -> List[Parameter]:
        return [
            ConstantParameter('model_type', self.model_type()),
            ConstantParameter('parameter_set_type', type(self).__name__),
            self.tasks_parameter,
            self.steps_per_epoch_parameter,
            self.experiment_name_parameter,
            self.model_subdir_parameter,
            self.epochs_parameter,
            self.fully_convolutional_parameter,
            *self.data_loader_parameters(),
            *self.callback_parameters()
        ]

    def callback_parameters(self):
        return [
            self.swa_start_batch_parameter,
            self.swa_cycle_length_parameter,
            self.early_stopping_metric_parameter,
        ]

    @abstractmethod
    def steps_per_epoch(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def default_num_epochs(self) -> int:
        raise NotImplementedError('Abstract method')

    def default_values(self):
        return {
            p.name: p.initial_value
            for p in self.hyper_parameters()
        }

    def parameters_by_name(self, name: str):
        return [p for p in self.hyper_parameters() if p.name == name]

    @staticmethod
    def validate_swa_parameters(parameters: Dict[str, Any]):
        if (parameters['swa_start_batch'] is None) != (parameters['swa_cycle_length'] is None):
            raise InvalidParametersError
        if parameters['swa_cycle_length'] not in [None, parameters['steps_per_epoch']]:
            raise AssertionError(parameters['swa_cycle_length'], parameters['steps_per_epoch'])

    @staticmethod
    def cv_fold_parameter():
        return parameter_search.ListParameter('cv_test_fold', 0, [0, 1, 2, 3], circle=True)

    def parameter_names(self):
        return sorted(set([p.name for p in self.hyper_parameters()]))

    @classmethod
    def _get_experiment_name_parameter(cls):
        return parameter_search.ConstantParameter('experiment_name', EXPERIMENT_NAME)

    @classmethod
    def _get_model_subdir_parameter(cls):
        return parameter_search.ConstantParameter('model_subdir', EXPERIMENT_NAME)

    @abstractmethod
    def model_type(self):
        raise NotImplementedError('Abstract method')

    def initial_tasks(self) -> VertebraTasks:
        return VertebraTasks.deserialize(self.tasks_parameter.initial_value)

    @cachetools.cached(cachetools.LRUCache(maxsize=200), key=lambda self: self.tasks_parameter.initial_value)
    def optimization_criterion(self):
        tasks = self.initial_tasks()
        return tasks.optimization_criterion()

    @cachetools.cached(cachetools.LRUCache(maxsize=200), key=lambda self: self.tasks_parameter.initial_value)
    def metrics_dicts(self):
        return self.initial_tasks().metrics_dicts_for_fnet()

    def metrics_names(self):
        return [m['name'] for m in self.metrics_dicts()]

    def num_epochs(self):
        return self.epochs_parameter.initial_value


class SimplePytorchModelConfig(HyperparameterSet):
    def model_type(self):
        from model.pytorch_model.mnist_model import BasicTorchModel
        model_type = BasicTorchModel.__name__
        return model_type

    def steps_per_epoch(self):
        return 160

    def default_num_epochs(self) -> int:
        return 40

    def __init__(self):
        super().__init__()
        self.hidden_size_parameter = parameter_search.ExponentialParameter('hidden_size', 128, 2, minimum=4)
        self.learn_rate_parameter = parameter_search.ExponentialParameter('learn_rate', 5e-4, 10)

    def hyper_parameters(self) -> List[Parameter]:
        return [
            *super().hyper_parameters(),
            self.hidden_size_parameter,
            self.learn_rate_parameter,
        ]


class PerceiverConfigPretrained(HyperparameterSet):
    def model_type(self):
        from model.perceiver.perceiver import FracturePerceiver
        model_type = FracturePerceiver.__name__
        return model_type

    def default_num_epochs(self) -> int:
        return 160

    def __init__(self):
        super().__init__()
        self.learn_rate_parameter = parameter_search.ExponentialParameter('learn_rate', 5e-4, 10)
        self.pretrained_parameter = parameter_search.ConstantParameter('pretrained', True)
        self.backbone_output_size_parameter = parameter_search.ExponentialIntegerParameter('backbone_output_size', 8, 2, minimum=4)
        self.fourier_feature_bands_parameter = parameter_search.ConstantParameter('fourier_feature_bands', 32)
        self.batch_size_parameter.initial_value = self.default_batch_size()  # reduced batch size due to increased memory usage
        self.steps_per_epoch_parameter.initial_value = self.steps_per_epoch()
        self.swa_cycle_length_parameter.initial_value = self.steps_per_epoch()
        self.swa_start_batch_parameter.initial_value = self.steps_per_epoch() - 1
        self.downsample_factor_parameter = parameter_search.ConstantParameter('downsample_factor', 1)
        self.min_modality_specific_padding = parameter_search.ConstantParameter('min_modality_specific_padding', 65)  # to get 32 * 3 * 2 + 3 + 1 + p = 261 channels

    @classmethod
    def default_batch_size(cls):
        return 2

    def steps_per_epoch(self) -> int:
        return 16 * (64 // self.default_batch_size())

    def hyper_parameters(self) -> List[Parameter]:
        return [
            *super().hyper_parameters(),
            self.learn_rate_parameter,
            self.pretrained_parameter,
            self.backbone_output_size_parameter,
            self.downsample_factor_parameter,
            self.min_modality_specific_padding,
            self.fourier_feature_bands_parameter,
        ]


class SwinConfig(HyperparameterSet):
    def model_type(self):
        from model.swin_transformer.swin_transformer import FractureSwinTransformer
        model_type = FractureSwinTransformer.__name__
        return model_type

    def default_num_epochs(self) -> int:
        return 160

    def __init__(self):
        super().__init__()
        self.learn_rate_parameter = parameter_search.ExponentialParameter('learn_rate', 5e-4, 10)
        self.feature_size_parameter = parameter_search.LinearIntegerParameter('feature_size', 48, 2, minimum=12)
        self.backbone_output_size_parameter = parameter_search.ExponentialParameter('backbone_output_size', 128, 2, minimum=4)
        self.num_stages_parameter = parameter_search.LinearIntegerParameter('num_stages', 4, 1)
        self.layers_per_stage_parameter = parameter_search.LinearIntegerParameter('layers_per_stage', 2, 1)
        self.num_heads_first_stage_parameter = parameter_search.ExponentialIntegerParameter('num_heads_first_stage', 4, 2)
        self.num_heads_exponent_parameter = parameter_search.LinearIntegerParameter('num_heads_exponent', 2, 1)
        self.batch_size_parameter.initial_value = self.default_batch_size()
        self.steps_per_epoch_parameter.initial_value = self.steps_per_epoch()
        self.swa_cycle_length_parameter.initial_value = self.steps_per_epoch()
        self.swa_start_batch_parameter.initial_value = self.steps_per_epoch() - 1
        self.drop_rate_parameter = parameter_search.BinaryParameter('drop_rate', 0, 0.5)
        self.attn_drop_rate_parameter = parameter_search.BinaryParameter('attn_drop_rate', 0, 0.5)
        self.dropout_path_rate_parameter = parameter_search.BinaryParameter('dropout_path_rate', 0, 0.5)
        self.downsample_parameter = parameter_search.BinaryParameter('downsample', 'mergingv2', 'merging')
        self.patch_size_parameter = parameter_search.LinearIntegerParameter('patch_size', 2, 1, minimum=2)
        self.use_v2_parameter = parameter_search.BooleanParameter('use_v2', False)
        self.input_patches_parameter = ConstantParameter(
            'input_patches',
            json.dumps([PatchRequest.from_input_size_px_and_spacing(spacing=(0.75, 0.75, 0.75), input_size_px=(64, 64, 64)).to_json()]),
            # json.dumps([PatchRequest(spacing=(3, 1, 1), size_mm=(60, 50, 40)).to_json()]),
        )
        self.monte_carlo_dropout_parameter = parameter_search.ConstantParameter('monte_carlo_dropout', False)

    def add_monte_carlo_dropout(self):
        assert not self.monte_carlo_dropout_parameter.initial_value
        assert self.drop_rate_parameter.initial_value != 0
        self.monte_carlo_dropout_parameter.initial_value = True

    @classmethod
    def default_batch_size(cls):
        return 64  # TODO prüfen

    def steps_per_epoch(self) -> int:
        return 160 * (64 // self.default_batch_size())

    def hyper_parameters(self) -> List[Parameter]:
        return [
            *super().hyper_parameters(),
            self.learn_rate_parameter,
            self.feature_size_parameter,
            self.backbone_output_size_parameter,
            self.num_stages_parameter,
            self.layers_per_stage_parameter,
            self.num_heads_first_stage_parameter,
            self.num_heads_exponent_parameter,
            self.batch_size_parameter,
            self.steps_per_epoch_parameter,
            self.swa_cycle_length_parameter,
            self.swa_start_batch_parameter,
            self.drop_rate_parameter,
            self.attn_drop_rate_parameter,
            self.dropout_path_rate_parameter,
            self.downsample_parameter,
            self.patch_size_parameter,
            self.use_v2_parameter,
        ]


class PerceiverConfigFromScratch(PerceiverConfigPretrained):
    def __init__(self):
        super().__init__()
        self.pretrained_parameter = parameter_search.ConstantParameter('pretrained', False)
        self.num_latents_parameter = parameter_search.ExponentialIntegerParameter('num_latents', 1024, 2, minimum=8)
        self.d_latents_parameter = parameter_search.ExponentialIntegerParameter('d_latents', 32, 2, minimum=16)
        self.num_blocks_parameters = [parameter_search.LinearIntegerParameter('num_blocks', 1, step_size, minimum=1) for step_size in [1, 2]]
        self.num_self_attends_per_block_parameter = parameter_search.LinearIntegerParameter('num_self_attends_per_block', 2, 2, minimum=1)
        self.num_self_attention_heads_parameter = parameter_search.ExponentialIntegerParameter('num_self_attention_heads', 4, 2, minimum=1, maximum=16)
        self.num_cross_attention_heads_parameter = parameter_search.ConstantParameter('num_cross_attention_heads', 1)
        self.attention_probs_dropout_prob_parameter = parameter_search.ListParameter('attention_probs_dropout_prob', 0.5, [0, 0.1, 0.5])
        self.fourier_feature_bands_parameter = parameter_search.ExponentialIntegerParameter('fourier_feature_bands', 8, 2, minimum=8)
        self.downsample_factor_parameter = parameter_search.ExponentialIntegerParameter('downsample_factor', 8, 2, minimum=1, maximum=16)
        self.input_patches_parameter = ConstantParameter(
            'input_patches',
            json.dumps([PatchRequest.from_input_size_px_and_spacing(spacing=(2, 1, 1), input_size_px=(32, 48, 48)).to_json()]),
            # json.dumps([PatchRequest(spacing=(3, 1, 1), size_mm=(60, 50, 40)).to_json()]),
        )
        self.min_modality_specific_padding = ExponentialIntegerParameter('min_modality_specific_padding', 1, 2)
        self.tasks_parameter.initial_value = self.tasks_parameter.possible_values[-1]
        self.early_stopping_metric_parameter.initial_value = None
        self.learn_rate_parameter.initial_value = 0.0005
        self.pad_augment_ratio_x_parameter.initial_value = 0.0
        self.pad_augment_ratio_y_parameter.initial_value = 0.4
        self.pad_augment_ratio_z_parameter.initial_value = 0.2
        self.pixel_scaling_parameter.initial_value = 'divide_by_2k'
        self.random_flip_fb_parameter.initial_value = True
        self.random_project_x_parameter.initial_value = 0.2
        self.random_project_y_parameter.initial_value = 0.0
        self.random_project_z_parameter.initial_value = 0.0
        self.random_shift_px_x_parameter.initial_value = 4
        self.random_shift_px_y_parameter.initial_value = 4
        self.random_shift_px_z_parameter.initial_value = 2
        self.rotate_range_deg_parameter.initial_value = 18
        self.sigma_a_1_parameter.initial_value = 0.1
        self.sigma_a_2_parameter.initial_value = 0.1
        self.sigma_m_2_parameter.initial_value = 0.08
        self.sigma_blur_parameter.initial_value = 0.75

    @staticmethod
    def memory_consumption_notes():
        """
        Notes are from training the beginning of the first epoch regarding what are good parameters to utilize the available memory and time.
        """
        return [
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 32, 'num_blocks': 2, 'fourier_feature_bands': 16, 'memory_during_training': '3.9 GB', 'it/s': 1.06},
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 16, 'num_blocks': 2, 'fourier_feature_bands': 16, 'memory_during_training': '2.7 GB', 'it/s': 3.0},
            # around 3.5 minutes per epoch
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 8, 'num_blocks': 2, 'fourier_feature_bands': 16, 'memory_during_training': '1.5 GB', 'it/s': 4.8},
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 16, 'num_blocks': 4, 'fourier_feature_bands': 16, 'memory_during_training': '3.7 GB', 'it/s': 1.9},
            # around 5.5 minutes per epoch
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 8, 'num_blocks': 4, 'fourier_feature_bands': 16, 'memory_during_training': '2.0 GB', 'it/s': 3.6},
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 4, 'num_blocks': 4, 'fourier_feature_bands': 16, 'memory_during_training': '1.4 GB', 'it/s': 4.7},
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 2, 'num_blocks': 4, 'fourier_feature_bands': 16, 'memory_during_training': '1.0 GB', 'it/s': 4.9},
            {'num_latents': 8, 'd_latents': 8, 'downsample': 2, 'batch_size': 2, 'num_blocks': 8, 'fourier_feature_bands': 32, 'memory_during_training': '0.9 GB', 'it/s': 2.8},
            # invalid due to missing caches???
            {'num_latents': 256, 'd_latents': 128, 'downsample': 2, 'batch_size': 2, 'num_blocks': 8, 'fourier_feature_bands': 32, 'memory_during_training': '1.3 GB', 'it/s': 4.1},
            {'num_latents': 256, 'd_latents': 256, 'downsample': 2, 'batch_size': 2, 'num_blocks': 8, 'fourier_feature_bands': 32, 'memory_during_training': '1.4 GB', 'it/s': 4.07},
            {'num_latents': 256, 'd_latents': 256, 'downsample': 2, 'batch_size': 2, 'num_blocks': 4, 'fourier_feature_bands': 32, 'memory_during_training': '1.2 GB', 'it/s': 5.36},
            {'num_latents': 256, 'd_latents': 256, 'downsample': 2, 'batch_size': 2, 'num_blocks': 4, 'fourier_feature_bands': 16, 'memory_during_training': '1.1 GB', 'it/s': 6.64},
        ]

    def model_type(self):
        from model.perceiver.perceiver import FracturePerceiver
        model_type = FracturePerceiver.__name__
        return model_type

    def default_num_epochs(self) -> int:
        return 50

    @classmethod
    def default_batch_size(cls):
        return 16

    def steps_per_epoch(self):
        return 32 * (64 // self.default_batch_size())

    def hyper_parameters(self) -> List[Parameter]:
        return [
            *super().hyper_parameters(),
            self.num_latents_parameter,
            self.d_latents_parameter,
            *self.num_blocks_parameters,
            self.num_self_attends_per_block_parameter,
            self.num_self_attention_heads_parameter,
            self.num_cross_attention_heads_parameter,
            self.attention_probs_dropout_prob_parameter,
        ]


class FNetDefaultHyperparameterSet(HyperparameterSet):
    def __init__(self):
        super().__init__()
        self.merge_after_parameter = parameter_search.ConstantParameter('merge_after_layer', 0)
        self.merge_paths_method_parameter = parameter_search.ListParameter('merge_paths_method', 'concatenate',
                                                                           ['concatenate', 'sum', 'difference_to_mean'])  # 'bilstm',
        self.merge_paths_shared_weights_parameter = parameter_search.ConstantParameter('merge_paths_shared_weights', False)
        self.independent_augmentation_parameter = parameter_search.ConstantParameter('independent_augmentation', False)
        self.conv_padding_parameter = parameter_search.ConstantParameter('conv_padding', 'valid')
        self.conv_block_strides_parameter = parameter_search.ConstantParameter('conv_block_strides', 2)
        self.num_conv_blocks_parameter = parameter_search.LinearParameter('num_conv_blocks', 3, 1, minimum=0, maximum=4)
        self.num_filters_parameter = parameter_search.ExponentialIntegerParameter('num_filters', 256, 2, minimum=4, maximum=512)
        self.learn_rate_parameter = parameter_search.ExponentialParameter('learn_rate', 5e-4, 10)
        self.initial_conv_layer_size_parameter = parameter_search.TernaryParameter('initial_conv_layer_size', 3, None, 'default')
        self.conv_block_size_parameter = parameter_search.LinearParameter('conv_block_size', 1, 1, minimum=1, maximum=4)
        self.num_dense_layers_parameter = parameter_search.LinearIntegerParameter('num_dense_layers', 1, 1, minimum=0, maximum=4)
        self.num_hidden_units_parameter = parameter_search.ExponentialIntegerParameter('num_hidden_units', 128, 2, minimum=4, maximum=256)
        self.dropout_parameter = parameter_search.TernaryParameter('dropout_rate', 0.0, 0.2, 0.5)
        self.batch_normalization_parameter = parameter_search.TernaryParameter('batch_normalization', 'not', 'before_activation', 'after_activation')
        self.instance_normalization_parameter = parameter_search.TernaryParameter('instance_normalization', 'not', 'before_activation', 'after_activation')
        self.num_filters_exponent_parameter = parameter_search.ConstantParameter('num_filters_exponent', 2)
        self.use_gap_parameter = parameter_search.ConstantParameter('use_gap', False)
        self.use_squeeze_excitation_parameter = parameter_search.ListParameter('use_squeeze_excitation', False, [False,
                                                                                                                 'SqueezeExciteBlock',
                                                                                                                 'SpatialSqueezeExciteBlock',
                                                                                                                 'ChannelSpatialSqueezeExciteBlock'])
        self.squeeze_excitation_placement_parameter = parameter_search.ListParameter('s_e_placement',
                                                                                     'after_conv_block',
                                                                                     ['after_conv_block', 'after_conv_layer', 'every_second_conv_layer'])

    def model_type(self):
        # noinspection PyUnresolvedReferences
        from model.fnet.fnet import FNet
        model_type = FNet.__name__
        return model_type

    def default_num_epochs(self) -> int:
        return 100

    def steps_per_epoch(self):
        return 160

    def hyper_parameters(self) -> List[Parameter]:
        return [
            *super().hyper_parameters(),
            self.num_conv_blocks_parameter,
            self.conv_block_strides_parameter,
            self.conv_block_size_parameter,
            self.num_dense_layers_parameter,
            ConstantParameter('kernel_size', 3, ),
            self.initial_conv_layer_size_parameter,
            self.use_gap_parameter,
            self.num_hidden_units_parameter,
            self.num_filters_parameter,
            self.learn_rate_parameter,
            self.dropout_parameter,
            self.batch_normalization_parameter,
            self.instance_normalization_parameter,
            self.num_filters_exponent_parameter,
            self.label_smoothing_parameter,
            self.training_data_amount_parameter,
            ExponentialParameter('l2_regularizer', 0., 2),
            ConstantParameter('regularize_bias', False),
            # parameter_search.TernaryParameter('class_weighting', 'per_output', 'relative', 'oversampling'),
            self.class_weighting_parameter,
            self.exclude_patients_parameter,
            ConstantParameter('random_validation_set', False),
            self.conv_padding_parameter,
            self.merge_after_parameter,
            self.merge_paths_method_parameter,
            self.merge_paths_shared_weights_parameter,
            self.independent_augmentation_parameter,
            ConstantParameter('initialize_extra_paths_with_zero_weights', False),
            self.vertebra_levels_parameter,
            self.size_adjustment_parameter,
            self.input_patches_parameter,
            self.use_squeeze_excitation_parameter,
            self.squeeze_excitation_placement_parameter,
        ]


class FNetWithBatchNorm(FNetDefaultHyperparameterSet):
    def __init__(self):
        super().__init__()
        self.batch_normalization_parameter.initial_value = 'after_activation'


class FNetHyperparameterSetForScoutScans(FNetDefaultHyperparameterSet):
    def __init__(self):
        super().__init__()
        self.random_project_x_parameter = ConstantParameter('random_project_x', 1.0)
        self.random_project_y_parameter.initial_value = 0.0
        self.random_project_z_parameter.initial_value = 0.0
        self.pixel_scaling_parameter.initial_value = 'range01'
        self.pad_augment_ratio_x_parameter = ConstantParameter.from_param(self.pad_augment_ratio_x_parameter)
        self.sigma_a_1_parameter.initial_value = 0.005
        self.sigma_a_1_parameter.minimum = 0.005
        self.hu_window_min_parameter.initial_value = -inf
        self.hu_window_max_parameter.initial_value = inf
        self.epochs_parameter.initial_value = 45
        self.merge_paths_method_parameter = ConstantParameter.from_param(self.merge_paths_method_parameter)
        self.instance_normalization_parameter.initial_value = 'after_activation'
        self.learn_rate_parameter.initial_value = 1e-4
        self.num_conv_blocks_parameter.initial_value = 2
        self.conv_block_size_parameter.initial_value = 2
        self.rotate_range_deg_parameter.initial_value = 18
        self.random_shift_px_x_parameter.initial_value = 0.
        self.random_shift_px_y_parameter.initial_value = 4.0
        self.random_shift_px_z_parameter.initial_value = 2.0
        self.pad_augment_ratio_y_parameter.initial_value = 0.4
        self.use_squeeze_excitation_parameter.initial_value = 'SqueezeExciteBlock'


class FNet2DForScoutScans(FNetHyperparameterSetForScoutScans):
    def __init__(self):
        super().__init__()
        self.conv_ndim_parameter.initial_value = 2
        self.input_patches_parameter = parameter_search.ConstantParameter('input_patches', json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 1, 1),
                                                        input_size_px=(1, 47, 47)).to_json()
        ]))
        self.learn_rate_parameter.initial_value *= 3  # going from 3x3x3 with constant values across third dimension to 3x3 without duplicates

        self.sigma_a_1_parameter.initial_value = 0.005
        self.sigma_a_2_parameter.initial_value = 0.2
        self.sigma_m_2_parameter.initial_value = 0.0025

        self.tasks_parameter = ConstantParameter.from_param(self.tasks_parameter)

        self.random_shift_px_y_parameter.initial_value = 8
        self.instance_normalization_parameter.initial_value = 'before_activation'
        self.random_flip_ud_parameter.initial_value = 1
        self.num_filters_parameter.initial_value = 256
        self.use_squeeze_excitation_parameter.initial_value = 'SpatialSqueezeExciteBlock'
        self.rotate_range_deg_parameter.initial_value = 18
        self.num_dense_layers_parameter.initial_value = 1
        self.pad_augment_ratio_z_parameter.initial_value = 0.6
        self.early_stopping_metric_parameter.initial_value = 'loss'
        self.conv_block_size_parameter.initial_value = 3
        self.initial_conv_layer_size_parameter.initial_value = None
        self.random_project_y_parameter.initial_value = 0.1
        self.pad_augment_ratio_y_parameter.initial_value = 0.2


class FNetParametersAsUSedForSPIE(FNetDefaultHyperparameterSet):
    def __init__(self):
        super().__init__()
        self.num_conv_blocks_parameter.initial_value = 4
        self.conv_block_strides_parameter.initial_value = 2
        self.num_filters_parameter.initial_value = 512
        self.epochs_parameter.initial_value = 45
        self.steps_per_epoch_parameter.initial_value = 160
        self.batch_size_parameter.initial_value = 64
        self.learn_rate_parameter.initial_value = 5e-4
        self.dropout_parameter.initial_value = 0.0
        self.batch_normalization_parameter.initial_value = 'not'
        self.instance_normalization_parameter.initial_value = 'not'
        self.num_filters_exponent_parameter.initial_value = 2
        self.use_gap_parameter.initial_value = False
        self.hu_window_max_parameter.initial_value = inf
        self.hu_window_min_parameter.initial_value = -inf
        self.use_squeeze_excitation_parameter.initial_value = False
        self.conv_padding_parameter.initial_value = 'same'
        self.merge_after_parameter.initial_value = 0
        self.input_patches_parameter.initial_value = json.dumps([
            PatchRequest(spacing=(3, 1, 1), size_mm=(60, 50, 40)).to_json(),
        ])
        self.random_flip_lr_parameter.initial_value = True
        self.random_flip_ud_parameter.initial_value = False
        self.random_flip_fb_parameter.initial_value = False
        self.random_shift_px_x_parameter.initial_value = 4
        self.random_shift_px_y_parameter.initial_value = 4
        self.random_shift_px_z_parameter.initial_value = 4
        self.sigma_a_1_parameter.initial_value = 0.1
        self.sigma_a_2_parameter.initial_value = 0.05
        self.sigma_m_2_parameter.initial_value = 0.16
        self.initial_conv_layer_size_parameter.initial_value = 3
        self.class_weighting_parameter.initial_value = 'relative'


class NicolaesHyperparameters(FNetDefaultHyperparameterSet):
    """
    This parameter set aims to reproduce the architecture of Nicolaes et al. (2019).
    """

    def __init__(self):
        super().__init__()
        self.num_filters_parameter = parameter_search.ExponentialIntegerParameter('num_filters', 25, 2, minimum=25, maximum=200)
        self.num_filters_exponent_parameter.initial_value = 1
        self.conv_block_strides_parameter.initial_value = 1
        self.num_dense_layers_parameter = parameter_search.LinearIntegerParameter('num_dense_layers', 2, 1, minimum=0, maximum=4)
        self.num_hidden_units_parameter = parameter_search.ExponentialIntegerParameter('num_hidden_units', 50, 2, minimum=25, maximum=400)
        self.initial_conv_layer_size_parameter.initial_value = None
        self.conv_padding_parameter.initial_value = 'valid'
        self.merge_paths_method_parameter.initial_value = 'concatenate'
        self.merge_after_parameter.initial_value = 10
        self.merge_paths_shared_weights_parameter.initial_value = False
        nicolaes_input_patches = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
                                                        input_size_px=(19, 19, 19)).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(3, 3, 3),
                                                        input_size_px=(19, 19, 19)).to_json(),
        ])
        one_input_patch = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 2, 2),
                                                        input_size_px=(19, 19, 19)).to_json()
        ])
        three_input_patches = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
                                                        input_size_px=(19, 19, 19)).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 2, 2),
                                                        input_size_px=(19, 19, 19)).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(3, 3, 3),
                                                        input_size_px=(19, 19, 19)).to_json(),
        ])
        self.input_patches_parameter = parameter_search.ListParameter('input_patches',
                                                                      nicolaes_input_patches,
                                                                      [nicolaes_input_patches, one_input_patch, three_input_patches])
        self.conv_block_size_parameter = parameter_search.LinearParameter('conv_block_size', 8, 1, minimum=1, maximum=12)
        self.num_conv_blocks_parameter = parameter_search.ConstantParameter('num_conv_blocks', 1)
        self.another_conv_block_size_parameter = parameter_search.LinearParameter('conv_block_size', 8, 2, minimum=1, maximum=12)

    def hyper_parameters(self) -> List[Parameter]:
        result = super().hyper_parameters()
        result.append(self.another_conv_block_size_parameter)
        return result


class FullyConvolutionalNicolaes(NicolaesHyperparameters):
    def __init__(self):
        super().__init__()
        self.fully_convolutional_parameter.initial_value = True
        self.conv_block_size_parameter.initial_value = 9
        self.num_dense_layers_parameter.initial_value = 1
        self.use_gap_parameter.initial_value = True
        self.num_filters_exponent_parameter.initial_value = 2
        self.num_filters_parameter.initial_value = 50
        additional_padding_each_size = 4
        input_size_px = (19 + 2 * additional_padding_each_size,
                         19 + 2 * additional_padding_each_size,
                         19 + 2 * additional_padding_each_size)
        self.input_patches_parameter.initial_value = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
                                                        input_size_px=input_size_px).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(3, 3, 3),
                                                        input_size_px=input_size_px).to_json(),
        ])


class HyperparametersOptimizedForTraumaDataset(FNetDefaultHyperparameterSet):
    """
    As a result of the experiments from 220209
    """

    def __init__(self):
        super().__init__()
        self.input_patches_parameter = parameter_search.ConstantParameter('input_patches', json.dumps([
            PatchRequest(spacing=(1, 1, 1),
                         size_mm=(60, 50, 40)).to_json(),
        ]))

        self.random_flip_fb_parameter.initial_value = True
        self.random_flip_lr_parameter.initial_value = True
        self.random_flip_ud_parameter.initial_value = True
        self.random_shift_px_x_parameter.initial_value = 2
        self.random_shift_px_y_parameter.initial_value = 2
        self.random_shift_px_z_parameter.initial_value = 2
        self.sigma_a_1_parameter.initial_value = 0.05
        self.sigma_a_2_parameter.initial_value = 0.05
        self.sigma_m_2_parameter.initial_value = 0.32
        self.pad_augment_ratio_x_parameter.initial_value = 0.0  # after the hyperparameter search of 220209, this was set to 0.4 but the probability was 0.
        self.pad_augment_ratio_y_parameter.initial_value = 0.4
        self.pad_augment_ratio_z_parameter.initial_value = 0.4
        self.rotate_range_deg_parameter.initial_value = 30

        self.num_conv_blocks_parameter.initial_value = 4
        self.conv_block_size_parameter.initial_value = 1
        self.num_dense_layers_parameter.initial_value = 4
        self.initial_conv_layer_size_parameter.initial_value = 24
        self.num_hidden_units_parameter.initial_value = 64
        self.num_filters_parameter.initial_value = 512
        self.dropout_parameter.initial_value = 0.5
        self.learn_rate_parameter.initial_value = 1e-3
        self.batch_normalization_parameter.initial_value = 'before_activation'
        self.class_weighting_parameter.initial_value = 'oversampling'

        self.random_project_x_parameter = parameter_search.LinearParameter('random_project_x', 0., 0.1, minimum=0, maximum=1.)
        self.random_project_y_parameter = parameter_search.LinearParameter('random_project_y', 0., 0.1, minimum=0, maximum=1.)
        self.random_project_z_parameter = parameter_search.LinearParameter('random_project_z', 0., 0.1, minimum=0, maximum=1.)

        self.conv_padding_parameter.initial_value = 'same'

        # TODO check these values, maybe copy from defaults
        self.hu_window_min_parameter = parameter_search.ConstantParameter('hu_window_min', -inf)

    def default_num_epochs(self) -> int:
        return 20

    def steps_per_epoch(self):
        return 320


class FNetHyperparametersWithMoreBranches(NicolaesHyperparameters):
    """
    Experiment 7a from experiments.odp
    """

    def __init__(self):
        super().__init__()
        self.input_patches_parameter.initial_value = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
                                                        input_size_px=(19, 19, 19)).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 2, 2),
                                                        input_size_px=(19, 19, 19)).to_json(),
            PatchRequest.from_input_size_px_and_spacing(spacing=(3, 3, 3),
                                                        input_size_px=(19, 19, 19)).to_json(),
        ])
        self.num_filters_parameter.initial_value = 64
        self.merge_after_parameter.initial_value = 5


class FNetConfigForSendingToFaraz20231123(SimplePytorchModelConfig):
    def __init__(self):
        super().__init__()
        self.input_patches_parameter.initial_value = json.dumps([
            PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
                                                        input_size_px=(60, 50, 40)).to_json(),
        ])


class FNetFor3Epochs(SimplePytorchModelConfig):
    def __init__(self):
        super().__init__()
        self.epochs_parameter.initial_value = 3


class FNetFor45Epochs(FNetDefaultHyperparameterSet):
    def default_num_epochs(self) -> int:
        return 45
