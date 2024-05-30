import os
import os.path
import random
import re
from typing import List, Dict, Optional

import matplotlib.cm
import numpy
from matplotlib import pyplot

import hiwi
from lib.my_logger import logging
from lib.util import EBC
from load_data.load_image import coordinate_transformer_from_file
from load_data.spaced_ct_volume import spaced_ct_volume
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer, tasks_of_any_model
from tasks import VertebraClassificationTask, BinaryGenantScoreClassification, VertebraTasks


def rainbow_text(x, y, strings, colors, horizontal: bool = True, separate_lines: List[bool] = None, **kw):
    """
    source: https://stackoverflow.com/a/9185851
    """
    if separate_lines is None:
        separate_lines = [False for _ in strings]
    t = pyplot.gca().transData
    fig = pyplot.gcf()
    from matplotlib import transforms
    additional_kwargs = {} if horizontal else {
        'rotation': 90,
        'va': 'bottom',
        'ha': 'center',
    }

    for s, c, l in zip(strings, colors, separate_lines):
        text = pyplot.text(x, y, s + " ", color=c, transform=t, **additional_kwargs, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        y_offset = horizontal == l
        if y_offset:
            t = transforms.offset_copy(text._transform, y=-ex.height, units='dots')
        else:
            t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


class PlottingConfig(EBC):
    def __init__(self,
                 ignore_classes: List[str] = None,
                 ignore_outputs: List[str] = None,
                 plot_text: bool = None,
                 name: str = None,
                 write_output_name: bool = None,
                 newlines_per_output: bool = None,
                 marker_color_task_name: str = None,
                 require_task_names: List[str] = None,
                 no_probability_for_outputs: List[str] = None,
                 marker_size_factor: float = None,
                 text_color: str = None):
        self.text_color = text_color
        self.write_output_name = write_output_name
        self.ignore_classes = ignore_classes
        self.ignore_outputs = ignore_outputs
        self.marker_size_factor = marker_size_factor
        self.plot_text = plot_text
        self.marker_color_task_name = marker_color_task_name
        self.require_task_names = require_task_names
        self.newlines_per_output = newlines_per_output
        self.no_probability_for_outputs = no_probability_for_outputs
        self.name = name
        for k, v in self.defaults().items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    @staticmethod
    def defaults():
        return {
            'ignore_classes': ['whole_dataset'] + BinaryGenantScoreClassification.possible_class_names(),
            'ignore_outputs': ['cfso', 'sto', 'fro', 'gs', 'ddxgs', 'ddx', 'gs0v123', 'gs01v23', 'gs012v3'],
            'plot_text': True,
            'name': 'outputs_on_sagittal_slices',
            'marker_size_factor': 1,
            'marker_color_task_name': None,
            'newlines_per_output': False,
            'write_output_name': False,
            'require_task_names': [],
            'text_color': 'correctness',
            'no_probability_for_outputs': [
                '0v123_01v23_012v3',
                '0v123_012v3_01v23',
                '01v23_0v123_012v3',
                '01v23_012v3_0v123',
                '012v3_0v123_01v23',
                '012v3_01v23_0v123',
                'gbd',
                'gbddz',
            ]
        }

    @classmethod
    def markers_only(cls):
        return PlottingConfig(plot_text=False, name='markers_on_sagittal_slices',
                              marker_size_factor=2)

    @classmethod
    def for_spie(cls):
        return PlottingConfig(
            ignore_outputs=[
                               t.output_layer_name()
                               for t in VertebraTasks.tasks_for_which_diagnostik_bilanz_has_annotations()
                               if t.output_layer_name() not in ['gsdz0v123', 'gsdz01v23', 'gsdz012v3', ]
                           ] + [
                               '0v123_01v23_012v3',
                               '0v123_012v3_01v23',
                               # '01v23_0v123_012v3',
                               '01v23_012v3_0v123',
                               '012v3_0v123_01v23',
                               '012v3_01v23_0v123',
                               'gbd',
                               'gbddz',
                           ],
            write_output_name=True,
            newlines_per_output=True,
            name='slices_for_spie',
            require_task_names=['01v23_0v123_012v3', 'gsdz01v23'],
            # text_color='output_class',
        )

    @classmethod
    def for_asbmr_2023(cls):
        return PlottingConfig(
            ignore_outputs=[
                               t.output_layer_name()
                               for t in VertebraTasks.tasks_for_which_diagnostik_bilanz_has_annotations()
                               if t.output_layer_name() not in ['gsdz0v123', 'gsdz01v23', 'gsdz012v3', ]
                           ] + [
                               '0v123_01v23_012v3',
                               '0v123_012v3_01v23',
                               # '01v23_0v123_012v3',
                               '01v23_012v3_0v123',
                               '012v3_0v123_01v23',
                               '012v3_01v23_0v123',
                               'gbd',
                               'gbddz',
                           ],
            write_output_name=True,
            newlines_per_output=True,
            name='slices_for_asbmr',
            # require_task_names=['01v23_0v123_012v3', 'gsdz01v23'],
            # text_color='output_class',
        )

    @classmethod
    def for_ages_artemis_workshop(cls):
        result = cls.for_asbmr_2023()
        result.name = 'slices_for_ages_artemis_workshop'
        result.text_color = 'output_class'
        return result

    @classmethod
    def binary_combination_markers_only(cls):
        return PlottingConfig.task_markers_only('01v23_0v123_012v3')

    @classmethod
    def task_markers_only(cls, task_name):
        return PlottingConfig(plot_text=False, name=task_name + '_on_sagittal_slices',
                              marker_size_factor=2,
                              marker_color_task_name=task_name)


class PlotOutputAsTextOnSagittalSlice(ModelAnalyzer):
    class MissingTask(ValueError):
        pass

    IGNORE_EXCEPTIONS = (MissingTask,)

    def __init__(self,
                 evaluator: FNetParameterEvaluator,
                 plotting_config=PlottingConfig(),
                 model_level_plotting=True,
                 skip_existing=None):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False,
                         model_level_caching=not model_level_plotting,
                         skip_existing=skip_existing,
                         binary_threshold_or_method=0.5,
                         analysis_needs_ground_truth_ys=plotting_config.text_color == 'correctness')
        self.model_level_plotting = model_level_plotting
        self.plotting_config = plotting_config

    def model_level_cache_key(self, model_path: Optional[str] = None):
        return super().model_level_cache_key(model_path) + (self.plotting_config,)

    def directory_level_cache_key(self, model_dir):
        return super().directory_level_cache_key(model_dir) + (self.plotting_config,)

    def to_subdir(self):
        return self.plotting_config.name

    def before_model(self, model_path: str):
        super().before_model(model_path)
        self.per_vertebra_outputs: Dict[str, list] = {}

    def analyze_batch(self, batch, y_preds, names):
        if self.plotting_config.marker_color_task_name is not None:
            if self.plotting_config.marker_color_task_name not in self.tasks.output_layer_names():
                raise self.MissingTask(self.plotting_config.marker_color_task_name)
        for task_name in self.plotting_config.require_task_names:
            if task_name not in self.tasks.output_layer_names():
                raise self.MissingTask(self.plotting_config.marker_color_task_name)
        self.check_if_analyzing_dataset()
        for sample_idx, name in enumerate(names):
            self.per_vertebra_outputs[name] = [y_pred[sample_idx] for y_pred in y_preds]

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        if self.model_level_plotting:
            per_vertebra_outputs = self.per_vertebra_outputs
            to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
            self.serializer().create_directory_if_not_exists(to_dir)
            self.plot_outputs_on_slice(per_vertebra_outputs, to_dir)
        return super().after_model(self.per_vertebra_outputs)

    def plot_outputs_on_slice(self,
                              per_vertebra_outputs: Dict[str, list],
                              to_dir,
                              iml: hiwi.ImageList = None,
                              use_model_threshold=True):
        if iml is None:
            iml = self.dataset
        imgs = [img for img in iml]
        if self.RANDOM_MODEL_ORDER:
            random.shuffle(imgs)
        for hiwi_img in imgs:
            hiwi_img: hiwi.Image
            patient_id = hiwi_img['patient_id']
            base = os.path.join(to_dir, f'{patient_id}')
            out_paths = [
                base + '.png',
                # base + '.svg'
            ]
            if self.skip_existing and all(self.serializer().isfile(p) for p in out_paths):
                continue
            logging.info(f'Writing {base}.*')
            volume = spaced_ct_volume(str(hiwi_img.path), desired_spacing=(1, 1, 1), dtype='float16')
            c_orig = coordinate_transformer_from_file(os.path.abspath(hiwi_img.path), patient_id)
            c_1_1_1 = coordinate_transformer_from_file(os.path.abspath(hiwi_img.path), patient_id, spacing=(1, 1, 1))

            vertebra_positions = [v.position for v in hiwi_img.parts.values() if v.position is not None]
            if len(vertebra_positions) == 0:
                use_slice_idx = volume.shape[2] // 2
            else:
                mean_pos = numpy.mean(vertebra_positions, axis=0).tolist()
                example_world_coords = c_orig.TransformContinuousIndexToPhysicalPoint(mean_pos)
                example_voxel_coords = c_1_1_1.TransformPhysicalPointToContinuousIndex(example_world_coords)
                use_slice_idx = round(example_voxel_coords[0])
            slice_img_data = volume[:, :, use_slice_idx]
            pyplot.figure(figsize=(12, 22), dpi=150)
            pyplot.grid(False)
            pyplot.imshow(slice_img_data.astype('float32'), origin='lower', cmap='gray')

            any_outputs = False
            for vertebra in hiwi_img.parts:
                if hiwi_img.parts[vertebra].position is None:
                    continue
                world_coords = c_orig.TransformContinuousIndexToPhysicalPoint(hiwi_img.parts[vertebra].position)
                _, y, z = c_1_1_1.TransformPhysicalPointToContinuousIndex(world_coords)
                try:
                    outputs = per_vertebra_outputs[str((patient_id, vertebra))]
                except KeyError:
                    outputs = None

                clss = []
                output_colors = []
                if outputs is not None:
                    any_outputs = True
                    task: VertebraClassificationTask
                    for y_pred, task in zip(outputs, self.tasks):
                        if task.output_layer_name() not in self.plotting_config.ignore_outputs:
                            true_class_name = task.class_name_from_hiwi_image_and_vertebra(hiwi_img, vertebra, on_invalid='return_empty_string')
                            if use_model_threshold:
                                cls_idx = self.class_idx_for_model_output_using_precomputed_threshold(y_pred, task)
                                cls_idx = numpy.array(cls_idx).item()
                            else:
                                cls_idx = numpy.array(task.nearest_class_idx_for_model_output(y_pred)).item()
                            cls_name = task.class_names()[cls_idx]
                            probability_for_class = float(task.class_probabilities(y_pred)[cls_idx])
                            p = probability_for_class
                            output_summary = ''
                            if self.plotting_config.write_output_name:
                                output_summary += task.output_layer_name() + ':'
                                output_summary = re.sub(r'^gsdz(\d+vs?\d+)', r'\1', output_summary)
                            if numpy.max(p).item() != 1 and task.output_layer_name() not in self.plotting_config.no_probability_for_outputs:
                                output_summary += f' {numpy.max(p).item():.1%}'
                            output_summary += ' ' + cls_name
                            correct_class = cls_name == true_class_name
                            if self.plotting_config.text_color == 'correctness':
                                output_color = 'cyan' if correct_class else 'orange'
                            elif self.plotting_config.text_color == 'output_class':
                                cmap = matplotlib.cm.get_cmap('hsv')
                                output_color = cmap(cls_idx / task.num_classes() + 0.3333)
                            else:
                                output_color = self.plotting_config.text_color
                            output_colors.append(output_color)
                            clss.append(output_summary)
                        assert len(output_colors) == len(clss)
                if self.plotting_config.plot_text:
                    rainbow_text(x=y,
                                 y=z,
                                 strings=[vertebra] + clss,
                                 colors=['lime'] + output_colors,
                                 separate_lines=[False] + [self.plotting_config.newlines_per_output for _ in clss],
                                 fontsize=6, )
                pyplot.scatter(x=y, y=z, marker='x', color='magenta',
                               linewidths=1 * self.plotting_config.marker_size_factor,
                               s=8 * self.plotting_config.marker_size_factor)
            if any_outputs:
                pyplot.tight_layout()
                for out_path in out_paths:
                    self.serializer().save_current_pyplot_figure(out_path)
            pyplot.close()

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        self.tasks = tasks_of_any_model(self.model_files_being_analyzed, self.evaluator, only_tasks=self.ONLY_TASKS)
        outputs_per_vertebra_per_model: List[Dict[str, list]] = results
        to_dir = os.path.join(self.to_dir(), 'average')
        self.serializer().create_directory_if_not_exists(to_dir)
        outputs_by_vertebra_and_task = {}
        task_indices = set()
        vertebrae = set()
        for outputs_per_vertebra in outputs_per_vertebra_per_model:
            for vertebra, output in outputs_per_vertebra.items():
                for task_idx, task_output in enumerate(output):
                    outputs_by_vertebra_and_task.setdefault((vertebra, task_idx), []).append(task_output)
                    task_indices.add(task_idx)
                    vertebrae.add(vertebra)
        task_indices = sorted(task_indices)
        vertebrae = sorted(vertebrae)
        mean_outputs: Dict[str, list] = {
            vertebra: [numpy.mean(outputs_by_vertebra_and_task[vertebra, task_idx], axis=0)
                       for task_idx in task_indices]
            for vertebra in vertebrae
        }
        self.plot_outputs_on_slice(mean_outputs, to_dir,
                                   iml=self.evaluator.whole_dataset(),
                                   use_model_threshold=False)

        return super().after_multiple_models(results)
