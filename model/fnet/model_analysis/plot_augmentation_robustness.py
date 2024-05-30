import math
import os
from abc import abstractmethod
from typing import List, Dict, Optional

import tensorflow.keras.backend
from matplotlib import pyplot

from lib.image_processing_tool import TrainableImageProcessingTool
from lib.my_logger import logging
from lib.util import EBC
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator
from model.fnet.builder import FNetBuilder
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.fnet import FNet
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, results_cache

ModificationName = str
ModifierValue = int


class Modifier(EBC):
    def name(self):
        return type(self).__name__

    def apply(self, g: AnnotatedPatchesGenerator):
        """
        In-place modification of the data generator.
        For example this could enable data augmentation so that the generator can be used for test time augmentation.
        """
        assert self._value_before_modification is None, self._value_before_modification
        assert self._applied_to_generator is None, self._applied_to_generator
        self._value_before_modification = self.get_value_from_generator(g)
        self._applied_to_generator = g
        self.set_value(g, self.value)
        assert self.get_value_from_generator(g) == self.value

    def undo(self):
        """
        This reverts any changes done to the generator in the `apply` method.
        It does not revert the model to its original state (but the model was copied anyways, see `prediction_model` method documentation).
        """
        assert self.get_value_from_generator(self._applied_to_generator) == self.value
        self.set_value(self._applied_to_generator,
                       self._value_before_modification)
        assert self.get_value_from_generator(self._applied_to_generator) == self._value_before_modification
        self._applied_to_generator = None
        self._value_before_modification = None

    def prediction_model(self, m: TrainableImageProcessingTool):
        """
        Creates a copy of the model and applies the modifier to it so that it can be used for prediction afterwards (for example enabling Monte-Carlo-Dropout).
        The only case where the original model itself is returned, is when there are no changes done to it.
        This method never modifies the passed model in-place.
        """
        return m

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        raise NotImplementedError('Abstract method')

    def neutral(self):
        raise NotImplementedError('Abstract method')

    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        raise NotImplementedError('Abstract method')

    def __call__(self, g: AnnotatedPatchesGenerator):
        return self.apply(g=g)

    def __init__(self, value):
        self.value = value
        self._value_before_modification = None
        self._applied_to_generator: Optional[AnnotatedPatchesGenerator] = None

    class ModifiedContext:
        """
        Important note: The data generator is modified in-place, while the model is either left unchanged or copied before making any changes.
        """

        def __init__(self, modifier: 'Modifier', generator: AnnotatedPatchesGenerator, model: TrainableImageProcessingTool):
            self.modifier = modifier
            self.generator = generator
            self.model = model

        def __enter__(self):
            self.modifier.apply(self.generator)
            return self.generator, self.modifier.prediction_model(self.model)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.modifier.undo()

    def modified_context(self, generator: AnnotatedPatchesGenerator, model: TrainableImageProcessingTool):
        return self.ModifiedContext(self, generator, model)

    def identifier(self):
        return (type(self), self.value, self.name())


class ShiftModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        assert len(set(s for s in g.random_shift_mm)) == 1
        return g.random_shift_mm[0]

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_shift_mm = (value, value, value)

    def neutral(self):
        return self.value == 0


class ModelModifier(Modifier):
    @abstractmethod
    def prediction_model(self, m: TrainableImageProcessingTool):
        """See documentation of super class."""
        raise NotImplementedError('Abstract method')

    def apply(self, g: AnnotatedPatchesGenerator):
        pass

    def undo(self):
        pass

    def set_value(self, g: AnnotatedPatchesGenerator):
        pass

    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        raise NotImplementedError('Not needed for this modifier. TO DO refactor')


class MCDOModifier(ModelModifier):
    def __init__(self, value):
        self.batch_size_before: Optional[int] = None
        self.batch_size_updated: Optional[int] = None
        super().__init__(value)

    def prediction_model(self, m: TrainableImageProcessingTool):
        if self.value:
            assert m.mcdo_applicable(), f'Can not enable Monte Carlo Dropout on model {m.model_path}. Maybe it has no Dropout layers?'
            assert not m.mcdo_applied(), f'Can not enable Monte Carlo Dropout on model {m.model_path}. Monte Carlo Dropout seems to be already enabled.'
            mcdo_model = m.to_mcdo_model()
            assert mcdo_model.mcdo_applied()
            return mcdo_model
        else:
            return m

    def apply(self, g: AnnotatedPatchesGenerator):
        assert self._applied_to_generator is None
        self.batch_size_before = g.batch_size
        max_batch_size = math.ceil(g.batch_size / 4)
        g.tune_batch_size(max_batch_size)
        self.batch_size_updated = g.batch_size
        logging.info(f'Changed batch size from {self.batch_size_before} to {self.batch_size_updated} (<{max_batch_size}) temporarily')
        self._applied_to_generator = g

    def undo(self):
        g = self._applied_to_generator
        assert g.batch_size == self.batch_size_updated
        g.batch_size = self.batch_size_before
        self._applied_to_generator = None

    def neutral(self):
        return not self.value


class ShiftModifier2D(ShiftModifier):
    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_shift_mm = (0, value, value)

    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        assert len(set(s for s in g.random_shift_mm[1:])) == 1
        return g.random_shift_mm[1]


class AdditivePixelWiseNoiseModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_noise_sigma_a_1

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_noise_sigma_a_1 = value

    def neutral(self):
        return self.value == 0


class MultiplicativePixelWiseNoiseModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_noise_sigma_m_1

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_noise_sigma_m_1 = value

    def neutral(self):
        return self.value == 0


class AdditiveImageWiseNoiseModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_noise_sigma_a_2

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_noise_sigma_a_2 = value

    def neutral(self):
        return self.value == 0


class MultiplicativeImageWiseNoiseModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_noise_sigma_m_2

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_noise_sigma_m_2 = value

    def neutral(self):
        return self.value == 0


class GaussianBlurModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.sigma_blur

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.sigma_blur = value

    def neutral(self):
        return self.value == 0


class RandomProjectXModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_x

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_x = value

    def neutral(self):
        return not self.value


class RandomProjectYModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_y

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_y = value

    def neutral(self):
        return not self.value


class RandomProjectZModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_z

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_z = value

    def neutral(self):
        return not self.value


class RotationRangeModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.rotate_range_deg

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.rotate_range_deg = value

    def neutral(self):
        return self.value == 0


class RandomFlipLRModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_flip_lr

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_flip_lr = value

    def neutral(self):
        return not self.value


class RandomFlipUDModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_flip_ud

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_flip_ud = value

    def neutral(self):
        return not self.value


class RandomFlipFBModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_flip_fb

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_flip_fb = value

    def neutral(self):
        return not self.value


class RandomProjectXModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_x

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_x = value

    def neutral(self):
        return self.value == 0


class RandomProjectYModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_y

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_y = value

    def neutral(self):
        return self.value == 0


class RandomProjectZModifier(Modifier):
    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return g.random_project_z

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        g.random_project_z = value

    def neutral(self):
        return self.value == 0


class EmptyModifier(Modifier):
    def __init__(self, value=None):
        super().__init__(value)

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        if value is not None:
            raise ValueError(value)

    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        return None

    def neutral(self):
        return True


class CombinedModifier(Modifier):
    def __init__(self, modifiers: List[Modifier], name: str = None):
        if name is None:
            name = repr(tuple(modifier.value for modifier in modifiers))
        super().__init__(value=name)
        self.modifiers = modifiers

    @staticmethod
    def get_value_from_generator(g: AnnotatedPatchesGenerator):
        raise NotImplementedError('Not available. Use apply and undo methods directly instead.')

    @staticmethod
    def set_value(g: AnnotatedPatchesGenerator, value):
        raise NotImplementedError('Not available. Use apply and undo methods directly instead.')

    def apply(self, g: AnnotatedPatchesGenerator):
        for modifier in self.modifiers:
            modifier.apply(g)

    def undo(self):
        for modifier in reversed(self.modifiers):
            modifier.undo()

    def neutral(self):
        return all(modifier.neutral() for modifier in self.modifiers)


class AugmentationRobustnessDrawer(ModelAnalyzer):

    def to_subdir(self):
        return 'augmentation_robustness'

    def __init__(self,
                 evaluator: FNetParameterEvaluator,
                 generator_modifiers: Dict[str, List[Modifier]] = None,
                 steps_per_vertebra: int = 8):
        super().__init__(evaluator, analysis_needs_xs=False, model_level_caching=False)
        self.steps_per_vertebra = steps_per_vertebra
        if generator_modifiers is None:
            generator_modifiers = self.default_modifiers()
        self.generator_modifiers = generator_modifiers

    @staticmethod
    def default_modifiers() -> Dict[str, List[Modifier]]:
        return {
            'rotate': [RotationRangeModifier(v) for v in [0, 6, 12, 18, 24, 30]],
            'shift': [ShiftModifier(v) for v in [0, 2, 4, 6, 8, 10]],
            'flip_lr': [RandomFlipLRModifier(v) for v in [False, True]],
            'flip_ud': [RandomFlipUDModifier(v) for v in [False, True]],
            'flip_fb': [RandomFlipFBModifier(v) for v in [False, True]],
            'mcdo': [MCDOModifier(v) for v in [False, True]],
            'project_x': [RandomProjectXModifier(v) for v in [0, 0.05, 0.1, 0.5, 1]],
            'project_y': [RandomProjectYModifier(v) for v in [0, 0.05, 0.1, 0.5, 1]],
            'project_z': [RandomProjectZModifier(v) for v in [0, 0.05, 0.1, 0.5, 1]],
            'random_project_xyz': [CombinedModifier([RandomProjectXModifier(True), RandomProjectYModifier(True), RandomProjectZModifier(True)])],
            'add_px_noise': [AdditivePixelWiseNoiseModifier(round(v * 0.05, 10)) for v in range(4)],
            'mult_px_noise': [MultiplicativePixelWiseNoiseModifier(round(v * 0.05, 10)) for v in range(4)],
            'add_img_noise': [AdditiveImageWiseNoiseModifier(round(v * 0.05, 10)) for v in range(4)],
            'mult_img_noise': [MultiplicativeImageWiseNoiseModifier(round(v * 0.05, 10)) for v in range(4)],
            'blur': [GaussianBlurModifier(round(v * 0.5, 10)) for v in range(6)],
            'rot_shift_fliplr': [CombinedModifier([RotationRangeModifier(24), ShiftModifier(4), RandomFlipLRModifier(True)])],
            'shift_flipud': [CombinedModifier([ShiftModifier(2), RandomFlipUDModifier(True)])],
        }

    @staticmethod
    def default_modifiers_for_2d():
        result = AugmentationRobustnessDrawer.default_modifiers()
        result['shift'] = [ShiftModifier2D(m.value) for m in result['shift']]
        del result['flip_lr']
        del result['project_x']
        del result['random_project_xyz']
        result['random_project_yz'] = [CombinedModifier([RandomProjectYModifier(True), RandomProjectZModifier(True)])]
        result['rot_shift_fliplr'] = [CombinedModifier([RotationRangeModifier(24), ShiftModifier2D(4), RandomFlipLRModifier(True)])]
        result['shift_flipud'] = [CombinedModifier([ShiftModifier2D(2), RandomFlipUDModifier(True)])]
        return result


    @staticmethod
    def mcdo_modifiers_for_2d():
        result = AugmentationRobustnessDrawer.default_modifiers_for_2d()
        result = {'mcdo': result['mcdo']}
        return result

    def analyze_batch(self, batch, y_preds, names):
        resultss = {}
        del batch
        for modifier_name, modifier_group in self.generator_modifiers.items():
            self.prepare_model()

            for modifier in modifier_group:
                resultss.setdefault(modifier_name, {})[modifier.value] = self.evaluate_model_in_modified_setting(modifier, modifier_name)

            tensorflow.keras.backend.clear_session()

        self.draw_results(resultss)
        return resultss

    def prepare_model(self):
        self.model = self.load_trained_model_for_evaluation()
        assert self.model.config == self.config
        if not isinstance(self.model, FNet):
            raise NotImplementedError('TO DO')
        FNetBuilder(self.config).compile_built_model(self.model.model,
                                                     classification_threshold_for_task=self.classification_thresholds[self.model_path])

    def evaluate_model_in_modified_setting(self, modifier, modifier_name, dont_cache=False, verbose=True):
        f = type(self)._cached_evaluation_of_model
        if dont_cache:
            f = f.func
        if modifier.neutral():  # more cache hits!
            modifier = EmptyModifier()
            modifier_name = None

        return f(self,
                 modifier=modifier,
                 verbose=verbose,
                 _cache_key=(modifier.value, modifier_name, self.steps_per_vertebra, self.model_level_cache_key()))

    @results_cache.cache(ignore=['self', 'modifier', 'verbose'], verbose=0)
    def _cached_evaluation_of_model(self, modifier: Modifier, verbose, _cache_key):
        if verbose:
            logging.info(f'Preparing robustness evaluation with "{modifier.name()}({modifier.value})"...')
            logging.info(f'  Data generator setup...')
        data_generator = self.evaluator.validation_generator_from_config(self.dataset, self.config, cache_batches=False,
                                                                         batch_size=self.model.validation_batch_size,
                                                                         ndim=self.evaluator.data_source.ndim())
        if verbose:
            logging.info('  Model evaluation...')
        with modifier.modified_context(data_generator, self.model) as (data_generator, modified_model):
            model_evaluation_results = modified_model.evaluate(data_generator,
                                                               steps=self.steps_per_vertebra * data_generator.steps_per_epoch(),
                                                               verbose=2, )
        return model_evaluation_results

    def draw_results(self, resultss: Dict[ModificationName, Dict[ModifierValue, List[float]]]):
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
        self.serializer().create_directory_if_not_exists(to_dir)
        self.serializer().save_text(os.path.join(to_dir, 'WARNING_sample_weights_not_used.txt'), '')
        if len(self.model.metrics_names) == 0:
            self.model.initialize_metrics_names()
        for metric_idx, metric in enumerate(self.model.metrics_names):
            ys: List[float] = []
            xticks = []
            for modification_name, results in sorted(resultss.items()):
                for modification_value, result in results.items():
                    assert len(result) == len(self.model.metrics_names)
                    ys.append(result[metric_idx])
                    xticks.append(f'{modification_name}({modification_value}) = {result[metric_idx]:.4g}')
            num_results = len(xticks)
            xs = range(num_results)
            pyplot.figure(figsize=(12.4, 9.6), dpi=100)
            pyplot.bar(x=xs, height=ys, width=0.8)
            pyplot.title(f'Augmentation robustness, {metric}, {self.steps_per_vertebra} augmentations per vertebra')
            pyplot.xlim(-0.5, num_results + 0.5)
            pyplot.xticks(labels=xticks, ticks=xs, rotation=90)
            pyplot.tick_params(axis='both', which='major', labelsize=8)
            pyplot.tight_layout()
            self.serializer().save_current_pyplot_figure(os.path.join(to_dir, f'{metric}' + '.png'))
            self.serializer().save_current_pyplot_figure(os.path.join(to_dir, f'{metric}' + '.svg'))
            pyplot.close()
