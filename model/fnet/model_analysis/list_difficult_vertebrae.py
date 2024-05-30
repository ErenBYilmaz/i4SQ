import itertools
import os.path
import re
from typing import List, Optional, Tuple, Callable, Union

import cachetools
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import numpy
import pandas
import scipy.interpolate
from matplotlib import pyplot
from matplotlib.artist import Artist
from matplotlib.axes import Axes

import lib.util
from lib import dl_backend
from load_data import VERTEBRAE
from model.fnet.model_analysis.evaluation_result_serializer import EvaluationResultSerializer

try:
    from annotation_supplement_ey import LIKELY_DIFFICULTIES
except ImportError:
    LIKELY_DIFFICULTIES = {}
from data.plot_patches import plot_patches_in_batch, DontSave
from lib.callbacks import RecordBatchLosses
from lib.my_logger import logging
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer, tasks_of_any_model, results_cache, split_name
from tasks import VertebraClassificationTask, BinaryOutputCombination, GenantScoreRegression

X = Y = Z = float



def color_fader_2d(c00, c01, c10, c11) -> Callable[[float, float], str]:
    """
    fade (linear interpolate) between
    colors:
     - cij (at mix1=i, mix2=j), for example c10 at mix1=1, mix2=0
    adjusted from: https://stackoverflow.com/a/50784012
    """
    c00 = numpy.array(matplotlib.colors.to_rgb(c00))
    c01 = numpy.array(matplotlib.colors.to_rgb(c01))
    c10 = numpy.array(matplotlib.colors.to_rgb(c10))
    c11 = numpy.array(matplotlib.colors.to_rgb(c11))
    interpolation_functions = [
        scipy.interpolate.interp2d(x=[0, 0, 1, 1],
                                   y=[0, 1, 0, 1],
                                   z=[c00[color_idx], c01[color_idx], c10[color_idx], c11[color_idx]],
                                   kind='linear',
                                   bounds_error=True)
        for color_idx in range(3)
    ]
    coloring_function = lambda x, y: matplotlib.colors.to_hex([interpolation_function(x, y).item() for interpolation_function in interpolation_functions])
    return coloring_function


class DifficultVertebraTable:
    def __init__(self, df: pandas.DataFrame, task: VertebraClassificationTask, model_dir: str, skip_existing: bool):
        self.skip_existing = skip_existing
        self.model_dir = model_dir
        self.task = task
        self.df = df

    def only_mistakes(self):
        return self.df[~self.df['best guess'].str.contains(r'\*')]

    def as_dict(self):
        return {
            (row['study'], row['vertebra']): row
            for row_idx, row in self.df.iterrows()
        }

    def plot_classification_arrays(self, to_dir: str, serializer: EvaluationResultSerializer):
        serializer.create_directory_if_not_exists(to_dir)

        out_paths = [
            os.path.join(to_dir, f'classification_overview_{self.task.output_layer_name()}.png'),
            os.path.join(to_dir, f'classification_overview_{self.task.output_layer_name()}.svg')
        ]
        if self.skip_existing and all(serializer.isfile(out_path) for out_path in out_paths):
            return

        logging.info('Plotting squares as classification overview...')
        patients = sorted(self.df['study'].unique())
        num_patients = len(patients)
        diagbilanz_size = 159
        x_axis_size_factor = num_patients / diagbilanz_size
        fig = pyplot.figure(figsize=(15 * x_axis_size_factor, # 15 is known to look good for diagbilanz
                                     4), dpi=202)
        ax = fig.add_subplot(111)
        ax: Axes
        ax.set_xlim(-0.5, len(patients) - 0.5)

        ax.set_xlabel(f'{len(patients)} patients')
        x_ticks = range(0, len(patients), 1)
        pyplot.xticks(rotation=90)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([patients[x] for x in x_ticks])

        vertebra_dict = self.as_dict()
        vertebrae = self.vertebrae_to_plot()
        y_ticks = range(1, len(vertebrae), 3)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([vertebrae[y] for y in y_ticks])
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(-0.5, len(vertebrae) - 0.5)
        ax.set_ylabel(f'{len(vertebrae)} vertebrae')

        class_to_color = color_fader_2d('palegreen', 'darkorange', 'red', 'blue')
        class_names = self.task.class_names()
        for c in class_names:
            assert '*' not in c
            assert '%' not in c

        def class_idx_by_name(c_str):
            assert '%' not in c_str, c_str
            assert '*' not in c_str, c_str
            indices = [name_idx for name_idx, name in enumerate(class_names) if name == c_str]
            assert len(indices) == 1, (c_str, class_names, indices)
            return indices[0]

        num_classes = len(class_names)
        if num_classes == 2:
            class_names_to_colors = lambda y_true_name, y_pred_name: [class_to_color(class_idx_by_name(y_true_name) / (num_classes - 1),
                                                                                     class_idx_by_name(y_pred_name) / (num_classes - 1))] * 2
        else:
            cmap = matplotlib.cm.get_cmap('hsv')
            class_names_to_colors = lambda y_true_name, y_pred_name: [cmap(class_idx_by_name(y_true_name) / num_classes + 0.3333),
                                                                      cmap(class_idx_by_name(y_pred_name) / num_classes + 0.3333)]

        w1 = 10 / 18
        if num_classes > 2:
            w1 *= 1.4

        for patient_idx, p in enumerate(patients):
            for vertebra_idx, v in enumerate(vertebrae):
                assert vertebra_idx == vertebrae.index(v)
                assert patient_idx == patients.index(p), (patient_idx, patients.index(p))
                if (p, v) not in vertebra_dict:
                    color1 = color2 = 'white'
                else:
                    row = vertebra_dict[p, v]
                    y_true = row['ground truth']
                    y_pred = row['best guess'].replace('*', '')
                    y_pred = re.sub(r'^\s*[\d.]+% ', '', y_pred)
                    color1, color2 = class_names_to_colors(y_true, y_pred)
                assert color1 is not None
                assert color2 is not None
                if len(class_names) == 2:
                    if color1 != 'white':
                        rect = matplotlib.patches.Rectangle((patient_idx - w1 / 2, vertebra_idx - w1 / 2), w1, w1, color=color1)
                        ax.add_patch(rect)
                else:
                    if color1 != 'white':
                        ax.add_patch(self.upper_left_right_triangle(color1, patient_idx, vertebra_idx, w1))
                    if color2 != 'white':
                        ax.add_patch(self.lower_right_right_triangle(color2, patient_idx, vertebra_idx, w1))

        if len(class_names) == 2:
            legend_elements = [matplotlib.patches.Rectangle((0, w1), w1, w1,
                                                            color=class_names_to_colors(y_true, y_pred)[0],
                                                            label=f'true: {y_true} | pred: {y_pred}')
                               for y_pred in class_names
                               for y_true in class_names]
            extra_legend_rows = 1
            extra_legend_columns = 0
        else:
            legend_elements: List[Artist] = [
                matplotlib.patches.Rectangle((0, w1), w1, w1,
                                             color=class_names_to_colors(class_name, class_name)[0],
                                             label=f'{class_name}')
                for class_name in class_names
            ]
            legend_elements += [
                matplotlib.lines.Line2D([], [], color='black', marker=[(-1, -1), (-1, 1), (1, 1)],
                                        linestyle='None', label='Ground truth'),
                matplotlib.lines.Line2D([], [], color='black', marker=[(1, 1), (1, -1), (-1, -1)],
                                        linestyle='None', label='Model output'),
            ]
            extra_legend_rows = 0
            extra_legend_columns = 2
        font_size_1 = 14
        font_size_2 = 5
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_yticklabels()):
            item.set_fontsize(font_size_1)
        for item in ax.get_xticklabels():
            item.set_fontsize(font_size_2)
        legend_rows = 1 + extra_legend_rows
        pyplot.legend(handles=legend_elements, prop={'size': font_size_1}, loc='upper center', bbox_to_anchor=(0.5, 1 + 0.025 * font_size_1 * legend_rows),
                      ncol=num_classes + extra_legend_columns)
        pyplot.grid(False)
        pyplot.tick_params(axis='x', which='major', bottom=True, length=2, width=0.5, direction='out', color='black', left=True)
        pyplot.tick_params(axis='y', which='major', bottom=True, length=3, width=1, direction='out', color='black', left=True)
        pyplot.tight_layout()
        logging.info(f'Writing to {os.path.join(to_dir, f"classification_overview_{self.task.output_layer_name()}.*")} ...')
        for out_path in out_paths:
            serializer.save_current_pyplot_figure(out_path)
        pyplot.close()

    def vertebrae_to_plot(self):
        vertebra_dict = self.as_dict()
        vertebrae = list(reversed(VERTEBRAE))  # ordered from bottom to top
        present_vertebrae = set(v for p, v in vertebra_dict.keys())

        first_idx = min(vertebrae.index(v) for v in present_vertebrae)
        last_idx = max(vertebrae.index(v) for v in present_vertebrae)
        stop_idx = last_idx + 1
        return vertebrae[first_idx:stop_idx]

    @staticmethod
    def upper_left_right_triangle(color, patient_idx, vertebra_idx, w1, label=None):
        return matplotlib.patches.Polygon(
            xy=[
                (patient_idx - w1 / 2, vertebra_idx + w1 / 2),
                (patient_idx + w1 / 2, vertebra_idx + w1 / 2),
                (patient_idx - w1 / 2, vertebra_idx - w1 / 2),
            ],
            color=color,
            label=label,
            edgecolor=None,
            fill=True,
            linestyle='None',
            linewidth=0,
            joinstyle='miter',
        )

    @staticmethod
    def lower_right_right_triangle(color, patient_idx, vertebra_idx, w1, label=None):
        return matplotlib.patches.Polygon(
            xy=[
                (patient_idx + w1 / 2, vertebra_idx - w1 / 2),
                (patient_idx + w1 / 2, vertebra_idx + w1 / 2),
                (patient_idx - w1 / 2, vertebra_idx - w1 / 2),
            ],
            color=color,
            label=label,
            edgecolor=None,
            fill=True,
            linestyle='None',
            linewidth=0,
            joinstyle='miter',
        )


class ListDifficultVertebrae(ModelAnalyzer):
    max_tasks_per_model = 20

    class InvalidTaskIndex(IndexError):
        pass

    IGNORE_EXCEPTIONS = (InvalidTaskIndex,)

    @classmethod
    def set_max_tasks_per_model(cls, max_tasks_per_model):
        cls.max_tasks_per_model = max_tasks_per_model

    def __init__(self, evaluator: FNetParameterEvaluator, task_idx=0, likely_difficulties=None, skip_existing=None):
        super().__init__(evaluator=evaluator, model_level_caching=True, analysis_needs_xs=False, skip_existing=skip_existing)
        self.task_idx = task_idx
        if likely_difficulties is None:
            likely_difficulties = {}
        self.likely_difficulties = likely_difficulties

    def name(self):
        return type(self).__name__ + f' (task {self.task_idx})'

    def to_subdir(self):
        return ''

    def model_level_cache_key(self, model_path: Optional[str] = None):
        # the last zero is for backwards compatibility with old cache files
        return super().model_level_cache_key(model_path), self.task_idx

    def directory_level_cache_key(self, model_dir):
        return super().directory_level_cache_key(model_dir) + (type(self.evaluator), self.coordinate_csv_path, self.task_idx)

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        tasks = tasks_of_any_model(self.model_files_being_analyzed, self.evaluator, only_tasks=self.ONLY_TASKS)
        if len(tasks) > self.max_tasks_per_model:
            raise RuntimeError(f'Please increase the max_tasks_per_model to at least {len(tasks)}.')
        try:
            task = tasks[self.task_idx]
        except IndexError:
            raise self.InvalidTaskIndex
        if not isinstance(task, VertebraClassificationTask):
            raise self.InvalidTaskIndex
        if not isinstance(task, VertebraClassificationTask):
            raise self.InvalidTaskIndex
        results = self.flatten_results(results)
        results = self.group_by_vertebra(results)
        results = self.filter_only_vertebrae_that_are_annotated_with_a_single_class(results, task)
        results = self.sort_results(results, task)
        if self.include_difficulty_column():
            self.add_difficulty_column(results)
        self.make_human_readable(results, task=task)
        super().after_multiple_models(results)
        difficult_vertebrae_table = DifficultVertebraTable(df=pandas.DataFrame(results, columns=self.columns()),
                                                           task=task,
                                                           model_dir=self.model_dir,
                                                           skip_existing=self.skip_existing)
        # self.print_results_table(task, difficult_vertebrae_table)
        return difficult_vertebrae_table

    def after_model_directory(self, results: DifficultVertebraTable):
        self.serializer().create_directory_if_not_exists(self.to_dir())
        base_name = os.path.join(self.to_dir(), f'difficult_vertebrae_{results.task.output_layer_name()}')
        self.serializer().save_dataframe(base_name + '.csv', results.df)
        self.serializer().save_pkl(base_name + '.pkl', results)

        md_string = ListDifficultVertebrae.table_description(results.task)
        md_string += '\n'
        md_string += results.only_mistakes().to_markdown(index=False)

        self.serializer().save_text(base_name + '.md', text=md_string)

        results.plot_classification_arrays(to_dir=os.path.join(self.to_dir()), serializer=self.serializer())
        return super().after_model_directory([results])

    # noinspection PyAttributeOutsideInit
    def before_model(self, model_path: str, ):
        super().before_model(model_path)
        self.table = []
        self.dummy_loss_model = dl_backend.b().empty_model(loss=self.task().loss_function())

    def task(self):
        try:
            return self.tasks[self.task_idx]
        except IndexError:
            raise self.InvalidTaskIndex

    def after_model(self, results: SingleModelEvaluationResult) -> SingleModelEvaluationResult:
        return super().after_model(self.table)

    @results_cache.cache(ignore=['self'], verbose=0)
    def _cached_model_analysis(self, model_path, _cache_key):
        self.before_model(model_path)
        results = self.analyze_test_and_validation_datasets()
        return self.after_model(results)

    def analyze_batch(self, batch, y_preds, names):
        _, ys, _ = batch
        record_batch_losses = RecordBatchLosses()
        self.dummy_loss_model.evaluate(x=y_preds[self.task_idx],
                                       y=ys[self.task_idx],
                                       verbose=0,
                                       callbacks=[record_batch_losses],
                                       batch_size=1,
                                       steps=len(names),
                                       return_dict=True)
        metrics = record_batch_losses.metrics
        assert len(metrics) == len(names)
        # (output,), (ground_truth,) = next(val_gen)
        for sample_idx in range(len(names)):
            output = y_preds[self.task_idx][sample_idx]
            ground_truth = ys[self.task_idx][sample_idx]
            patient, vertebra = split_name(names[sample_idx])
            self.table.append([
                metrics[sample_idx]['loss'],
                output,
                ground_truth,
                patient,
                vertebra,
            ])

    @staticmethod
    def group_by_vertebra(table: List[list]) -> List[list]:
        vertebrae = {}
        for row in table:
            vertebrae.setdefault((row[3], row[4]), []).append(row[:3])
        for v in vertebrae:
            vertebrae[v] = [numpy.mean([row[c] for row in vertebrae[v]], axis=0) for c in range(3)] + [numpy.array(len(vertebrae[v]))]
        table = [
            [*v, *vertebrae[v]]
            for v in vertebrae
            # for v in sorted(vertebrae, key=lambda v: vertebrae[v][1], reverse=True)
        ]
        return table

    @staticmethod
    def make_human_readable(table: List[list], task: VertebraClassificationTask):
        for row in table:
            row[4] = task.class_name_of_label(row[4])
            class_probabilities = task.class_probabilities(row[3])
            sorted_outputs = numpy.argsort(class_probabilities)[::-1]
            row[3:4] = [f'{class_probabilities[class_idx]:7.2%} {marker}{task.class_names()[class_idx]}{marker}'
                        for class_idx in sorted_outputs[:2]
                        for is_gt in [class_idx == task.class_names().index(row[4])]
                        for marker in ['*' if is_gt else '']]

    @staticmethod
    def table_description(task: VertebraClassificationTask):
        n = task.num_classes()
        general_description = f'''
        # Difficult vertebrae

        This table was created from multiple neural networks that were trained on subsets of the training data in a cross-validation setting.
        Each of these neural networks was then applied to the remaining subset of the training data and outputs a probability-like score for each of {n} classes ({task.class_names()}).
        The following table lists the "worst mistakes" of the model (cases where it assigns a very low score to the correct class).
        "Best guess" refers to the highest score that was assigned to one of the classes (this was used to sort the table).
        "ground truth" is the class of the vertebra according to the annotations.
        The table was filtered to only show rows with a "best guess" for the wrong class.
        '''
        # loss_description = '''
        # "loss" is the cross-entropy loss and was used to sort the table.
        # Lower is better and log({n}) = {log(n):.4g} would be achieved by outputting "{1 / n :2.2%}" for each of the four classes (random guessing).
        # In particular, a loss of above log({n}) means that the model assigned a score lower than {1 / n :2.2%} to the correct class.
        # A very large loss (top of the table) can be interpreted as the model being very confident of its wrong decision.
        # '''
        averaging_description = '''
        To get more reliable results the outputs and losses of multiple models were averaged to a single row per vertebra where the number of averaged models is listed in "#results".
        '''
        return re.sub(r'\n\s+', '\n', (general_description +
                                       # loss_description +
                                       averaging_description))

    def print_results_table(self, difficult_vertebrae_table: DifficultVertebraTable):
        print(self.markdown_results_table(difficult_vertebrae_table))

    def markdown_results_table(self, difficult_vertebrae_table: DifficultVertebraTable):
        return (
            '\n' + self.table_description(difficult_vertebrae_table.task) +
            '\n' + lib.util.my_tabulate(difficult_vertebrae_table.df, headers=self.columns()) +
            '\n'
        )

    def columns(self):
        result = ['study', 'vertebra', 'loss', 'best guess', 'second guess', 'ground truth', '#results']
        if self.include_difficulty_column():
            result.append('likely_difficulties')
        return result

    def include_difficulty_column(self):
        return len(self.likely_difficulties) > 0

    @staticmethod
    def flatten_results(results):
        return [r for result in results for r in result]

    @staticmethod
    def filter_only_vertebrae_that_are_annotated_with_a_single_class(results, task: VertebraClassificationTask):
        return [row for row in results if task.label_has_class(row[4])]

    @staticmethod
    def sort_results(results, task: VertebraClassificationTask):
        return sorted(results, key=lambda row: task.class_probabilities(row[3])[task.class_idx_of_label(row[4])], reverse=False)

    def add_difficulty_column(self, results):
        for row in results:
            vertebra = row[1]
            study = row[0]
            if study in self.likely_difficulties and vertebra in self.likely_difficulties[study]:
                row.append(self.likely_difficulties[study][vertebra])
            else:
                row.append('')


class DifficultVertebraePlotter(ModelAnalyzer):
    SKIP_TASKS = [
        BinaryOutputCombination,
        GenantScoreRegression,
    ]
    try:
        from trauma_tasks import CombinedFreshUnstableClassification, FreshnessClassification, StabilityClassification
    except ImportError:
        CombinedFreshUnstableClassification = FreshnessClassification = StabilityClassification = None
    else:
        SKIP_TASKS += [CombinedFreshUnstableClassification, FreshnessClassification, StabilityClassification, ]
    SKIP_CLASSES = [
        'whole_dataset'
    ]

    def __init__(self,
                 evaluator: FNetParameterEvaluator,
                 to_subdir: str,
                 difficult_vertebrae_table: Optional[pandas.DataFrame],
                 skip_existing=None,
                 save_as_nifti=False,
                 add_output_info_for_task_idx: int = None,
                 binary_threshold_or_method: Union[str, float] = None, ):
        super().__init__(evaluator, analysis_needs_xs=True, binary_threshold_or_method=binary_threshold_or_method)
        if skip_existing is None:
            skip_existing = self.SKIP_EXISTING_BY_DEFAULT
        self.task_idx = add_output_info_for_task_idx
        self.skip_existing = skip_existing
        self.save_as_nifti = save_as_nifti
        self.difficult_vertebrae_table = difficult_vertebrae_table
        self._to_subdir = to_subdir

    def to_subdir(self):
        return self._to_subdir

    def name(self):
        n = type(self).__name__
        if self.task_idx is not None:
            n += f' (task {self.task_idx})'
        return n

    def skip_task(self, task=None):
        if task is None:
            task = self.task()
        return any([isinstance(task, t) for t in self.SKIP_TASKS])

    def task(self):
        try:
            return self.tasks[self.task_idx]
        except IndexError:
            raise ListDifficultVertebrae.InvalidTaskIndex

    def analyze_batch(self, batch, y_preds, names):
        if self.difficult_vertebrae_table is not None:
            only_names = [str((row['study'], row['vertebra'])) for _, row in self.difficult_vertebrae_table.iterrows()]
        else:
            only_names = None

        names_tuples = [eval(name) for name in names]

        def filepath_for_name(to_dir: str, name: str):
            patient_id, vertebra = eval(name)
            if self.difficult_vertebrae_table is not None:
                row = self.row(patient_id, vertebra)
                rank = self.difficult_vertebrae_table.index[
                           (self.difficult_vertebrae_table['study'] == patient_id) & (self.difficult_vertebrae_table['vertebra'] == vertebra)].item() + 1
                pred = row['best guess'].replace(' ', '_').replace('*', '')
                y = row['ground truth']
                filename = f'#{rank:03d}_{patient_id}_{vertebra}_true={y}_pred={pred}'
            else:
                if self.task_idx is not None:
                    task = self.task()
                    y_pred = y_preds[self.tasks.output_layer_names().index(task.output_layer_name())]
                    y_pred = y_pred[names_tuples.index((patient_id, vertebra))]
                    class_idx = self.class_idx_for_model_output_using_precomputed_threshold(y_pred, task)
                    class_name = task.class_names()[class_idx]
                    if class_name in self.SKIP_CLASSES:
                        return DontSave.DONT_SAVE
                    prob = task.class_probabilities(y_pred)[class_idx]
                    class_name = class_name.replace(' ', '_')
                    filename = f'{class_name}_{prob:.2%}_{patient_id}_{vertebra}_{input_size_mm_string}'
                else:
                    filename = f'{patient_id}_{vertebra}_{input_size_mm_string}'
            return os.path.join(to_dir, filename)

        if self.task_idx is not None:
            if self.skip_task():
                return
            logging.info('Task: ' + self.task().output_layer_name())
            to_base_dir = os.path.join(self.to_dir(), self.task().output_layer_name())
        else:
            to_base_dir = self.to_dir()

        xs = batch[0]
        crop_to: Tuple[Z, Y, X] = self.evaluator.input_shape_px(self.config)
        assert len(self.evaluator.patch_requests()) == len(xs)
        for input_idx in range(len(xs)):
            xs[input_idx] = self.crop_center(xs[input_idx], axes=(1, 2, 3), crop_to=crop_to)
            input_size_mm_string = "_".join(str(s) for s in self.evaluator.patch_requests()[input_idx].size_mm)
            plot_patches_in_batch(xs[input_idx],
                                  names,
                                  only_names=only_names,
                                  to_dir=to_base_dir,
                                  generate_histograms=False,
                                  generate_axial_images=False,
                                  save_as_nifti=self.save_as_nifti,
                                  spacing=self.evaluator.patch_requests()[input_idx].spacing,
                                  skip_existing=self.skip_existing,
                                  filepath_creator=filepath_for_name)

    @staticmethod
    def crop_center(img, axes, crop_to):
        crop_to = {a: c for a, c in zip(axes, crop_to)}
        return img[tuple(slice(sh // 2 - crop_to[axis] // 2, sh // 2 - crop_to[axis] // 2 + crop_to[axis]) if axis in crop_to else slice(None)
                         for axis, sh in zip(itertools.count(), img.shape))]

    @cachetools.cached(cachetools.LRUCache(maxsize=3000), key=lambda self, patient_id, vertebra: (id(self.difficult_vertebrae_table), patient_id, vertebra))
    def row(self, patient_id: str, vertebra: str):
        for _, row in self.difficult_vertebrae_table.iterrows():
            if row['study'] == patient_id and row['vertebra'] == vertebra:
                break
        else:
            raise ValueError
        return row
