import functools
import json
import re
from typing import Tuple, Callable, Optional, Any, Union, Set

import SimpleITK
import matplotlib.patches
from matplotlib import pyplot
from tensorflow.keras import Model

from hiwi import ImageList
from lib.custom_layers import ShapedGaussianNoise, ShapedMultiplicativeGaussianNoise
from lib.mean_curve import plot_roc_curve, plot_pr_curve
from lib.my_logger import logging
from lib.parameter_search import mean_confidence_interval_size
from lib.util import beta_conf_interval, beta_stats, X, Y, Z, LogicError, round_to_digits
from load_data import MADER_SPACING, VERTEBRAE
from model.fnet.evaluate import FNetParameterEvaluator
from model.hnet.eval_utils import outputs_plot, empty_array_plot
from tasks import VertebraTask
from trauma_tasks import IsFracturedClassification
from lib.random_rotate import RandomRotation3D

import os
from math import nan, sqrt, inf
from typing import Dict, List

from tensorflow_addons.layers import InstanceNormalization
import tensorflow.keras
import numpy
import sklearn.metrics
import sklearn.svm

Vertebra = str
PatientId = str

IGNORE_VERTEBRAE_CLOSE_TO_BORDER = 5

IN_PERCENT = True
to_str_2: Callable[[float], str]
to_str_3: Callable[[float], str]
if IN_PERCENT:
    to_str_2 = lambda x: str(round_to_digits(x * 100, 2)) + '%'
    to_str_3 = lambda x: str(round_to_digits(x * 100, 3)) + '%'
else:
    to_str_3 = to_str_2 = str

# assume_beta_distribution = ['ROC-AUC',
#                             'Best thresholds',
#                             'Specificity',
#                             'Sensitivity',
#                             'Average precision',
#                             'F1',
#                             'Deformity Sensitivity',
#                             'Deformity Specificity', ]
assume_beta_distribution = []

CURVES = [
    (functools.partial(plot_roc_curve,
                       plot_individual=False,
                       num_thresholds=1000,
                       write_area=''), 'roc'),
    (functools.partial(plot_roc_curve,
                       plot_individual=False,
                       num_thresholds=1000,
                       plot_ci=False,
                       write_area='mean'), 'roc-no-CI'),
    (functools.partial(plot_pr_curve,
                       plot_individual=False,
                       num_thresholds=20,
                       write_area=''), 'pr'),
    # (functools.partial(plot_output_histogram, normalize=True), 'output_histogram_normalized'),
    # (functools.partial(plot_output_histogram, normalize=False), 'output_histogram'),
]

custom_layers = {
    klass.__name__: klass
    # If your model uses any custom layers, add them here
    for klass in [ShapedGaussianNoise, ShapedMultiplicativeGaussianNoise, InstanceNormalization, RandomRotation3D]
}


def draw_central_slices(volume, save_path_prefix, cmap='gray', suffix='', spacing_correction=True, axes=('coronal', 'axial', 'sagittal')):
    mader_spacing = numpy.ones(MADER_SPACING[::-1])
    while len(mader_spacing.shape) < len(volume.shape):
        mader_spacing = mader_spacing[..., numpy.newaxis]
    if spacing_correction:
        volume_spacing_1_1_1 = numpy.kron(volume, mader_spacing)
    else:
        volume_spacing_1_1_1 = volume
    if volume_spacing_1_1_1.shape[-1] == 1:
        volume_spacing_1_1_1 = numpy.squeeze(volume_spacing_1_1_1, axis=-1)
    if 'axial' in axes:
        pyplot.imsave(axial_slice_path(save_path_prefix, suffix), volume_spacing_1_1_1[volume_spacing_1_1_1.shape[0] // 2, :, :],
                      cmap=cmap)
    if 'coronal' in axes:
        pyplot.imsave(coronal_slice_path(save_path_prefix, suffix), volume_spacing_1_1_1[:, volume_spacing_1_1_1.shape[1] // 2, :],
                      cmap=cmap)
    if 'sagittal' in axes:
        pyplot.imsave(sagittal_slice_path(save_path_prefix, suffix), volume_spacing_1_1_1[:, :, volume_spacing_1_1_1.shape[2] // 2],
                      cmap=cmap)


def sagittal_slice_path(save_path_prefix, suffix):
    return slice_path('sagittal', save_path_prefix, suffix)


def coronal_slice_path(save_path_prefix, suffix):
    return slice_path('coronal', save_path_prefix, suffix)


def axial_slice_path(save_path_prefix, suffix):
    return slice_path('axial', save_path_prefix, suffix)


def slice_path(direction, save_path_prefix, suffix):
    return save_path_prefix + '_' + direction + suffix + '.png'


class NumpyEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/a/47626762
    """

    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def classify_fractures(fracture_model_name: str,
                       test_list: ImageList,
                       task: VertebraTask,
                       coordinates: Optional[Dict[Tuple[PatientId, Vertebra], Tuple[X, Y, Z]]] = None,
                       verbose: float = 1,
                       ignore=None,
                       results_path=None,
                       threshold: float = 0.5,
                       exclude_center_6=False, ):
    if ignore is None:
        ignore = []
    use_ground_truth_coordinates = coordinates is None
    if use_ground_truth_coordinates:
        logging.info('Using ground truth coordinates')
    patient_metrics: Dict[str, float]
    exclusions = 0
    inclusions = 0
    all_outputs = {}
    all_vertebra_fracture_outputs = {}
    all_vertebra_outputs = {}
    vertebra_metadata = {}
    task_name = task.output_layer_name()
    logging.info(f'Label mode: {task_name}')

    if verbose >= 0.5:
        print(f'Evaluating {fracture_model_name} ...')

    y_true_dict: Dict[int, numpy.ndarray] = {}
    y_pred_dict: Dict[Tuple[int, str], numpy.ndarray] = {}
    if use_ground_truth_coordinates:
        coordinates = coords_from_iml(test_list)
    # patient_ids = []
    y_true = []
    y_pred = []
    for image in test_list:
        if exclude_center_6 and image['patient_id'].startswith('6'):
            continue
        p = int(re.escape(image['patient_id']))
        if p in ignore:
            print(f'Image {p} excluded.')
            exclusions += 1
            continue
        inclusions += 1
        tool_outputs = image['tool_outputs'][fracture_model_name]
        image_label = task.binarized_image_label(image).item()
        assert task_name in tool_outputs
        found_vertebrae: Set[str] = {v for p, v in coordinates}
        for v in found_vertebrae:
            if (p, v) not in coordinates:
                continue  # missing vertebra, probably not in the image or not found
            if v in ['T1', 'T2', 'T3', 'L5'] or v.startswith('C') or v.startswith('S'):
                if verbose >= 2:
                    assert not use_ground_truth_coordinates
                    print(f'Skipping {v} of {p}, because the fracture model was not trained on this kind of vertebrae')
                continue
            assert p, v not in all_vertebra_outputs
            all_vertebra_outputs[p, v] = {k: tool_outputs[k][tool_outputs['names'].index(str((str(p), v)))] for k in tool_outputs}
            y_pred_dict[p, v] = numpy.array(all_vertebra_outputs[p, v][task_name])
            assert p, v not in all_vertebra_fracture_outputs
            all_vertebra_fracture_outputs[p, v] = y_pred_dict[p, v]
        for v in image.parts:
            try:
                vertebra_metadata[p, v] = dict(vertebra_info(image, v))
            except KeyError:
                vertebra_metadata[p, v] = None
        y_true_dict[p] = image_label
        y_true.append(y_true_dict[p])
        y_pred_patient_level = task.patient_level_aggregation(numpy.stack(y_pred_dict[p2, v] for p2, v in y_pred_dict if p2 == p))
        y_pred.append(task.binarized_label(y_pred_patient_level).item())
        all_outputs[p] = {'y_true': y_true[-1], 'y_pred': y_pred[-1]}
        if verbose >= 3:
            print(f'{p}: y_true = {y_true[-1]}, y_pred = {y_pred[-1]}')

    patient_metrics = metrics_dict(threshold, y_pred, y_true)
    print(f'{exclusions} of {exclusions + inclusions} images excluded')
    print(f'{inclusions} of {exclusions + inclusions} images included')
    print('  patient level')

    json_patient_outputs = {**dict(sorted(all_outputs.items()))}
    json_vertebra_fractures_outputs = {**dict(sorted(all_vertebra_fracture_outputs.items()))}
    json_vertebra_outputs = {**dict(sorted(all_vertebra_outputs.items()))}

    for metric in patient_metrics:
        if verbose >= 1:
            metric_summary(metric, [patient_metrics[metric]])

    all_results: Dict[str, Any] = {
        'comment': {'patients': 'Full pipeline outputs and labels',
                    'vertebral_fractures': 'Fracture output per vertebra level',
                    'vertebrae': ['All outputs per vertebra level in order:',
                                  'Fracture output (osteoporotic fracture = 1, other = 0)',
                                  'Deformity output three values for wedge, biconcave, crush, respectively',
                                  'Genant score estimation',
                                  # 'Differential Diagnosis (Spondylosis = [0, 1], other = [1, 0])',
                                  # 'Differential Diagnosis Category (Normal = [1, 0, 0], Deformity = [0, 1, 0]), Osteoporotic Fracture = [0, 0, 1])',
                                  ],
                    'patient_metrics': 'Evaluation results on patient level',
                    'vertebra_info': 'Some metadata about the vertebra.', },
        'patients': json_patient_outputs,
        'vertebral_fractures': nest_dict(json_vertebra_fractures_outputs),
        'vertebrae': nest_dict(json_vertebra_outputs),
        'patient_metrics': patient_metrics,
        'vertebra_info': nest_dict(vertebra_metadata),
    }
    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        logging.info(f'Outputting fracture evaluation to {os.path.join(results_path, "summary")}')
        with open(os.path.join(results_path,
                               'fracture_outputs.json'), 'w') as f:
            json.dump(all_results, f, cls=NumpyEncoder)
        json_results_path = os.path.join(results_path, f'evaluation_results_{task_name}.json')

        for plot_curve, name in CURVES:
            plot_curve(y_trues=[[all_outputs[p]['y_true']
                                 for p in sorted(list(all_outputs.keys()))]],
                       y_preds=[[all_outputs[p]['y_pred']
                                 for p in sorted(list(all_outputs.keys()))]])
            sens = all_results['patient_metrics']['Specificity']
            spec = all_results['patient_metrics']['Sensitivity']
            pyplot.scatter(y=[sens],
                           x=[1 - spec],
                           label=f'Operating point, threshold = {threshold:.3f}\nSensitivity = {sens:.3f}, Specificity = {spec:.3f}',
                           marker='x')
            pyplot.tight_layout()
            pyplot.savefig(os.path.join(results_path,
                                        f'{name}.svg'), transparent=True)
            pyplot.savefig(os.path.join(results_path,
                                        f'{name}.png'), transparent=False)
            pyplot.close()

        if os.path.isfile(json_results_path):
            with open(json_results_path) as f:
                results = json.load(f)

            outputs_plot(results=results, outputs=all_outputs)
            pyplot.tight_layout()
            pyplot.savefig(os.path.join(results_path,
                                        'pipeline_outputs.svg'), transparent=True)
            pyplot.savefig(os.path.join(results_path,
                                        'pipeline_outputs.png'), transparent=True)
            pyplot.close()

            outputs_plot(results=results, outputs=all_outputs, format_for_mlmi=True)
            pyplot.tight_layout()
            pyplot.savefig(os.path.join(results_path,
                                        'mlmi_pipeline_outputs.svg'), transparent=True)
            pyplot.savefig(os.path.join(results_path,
                                        'mlmi_pipeline_outputs.png'), transparent=True)
            pyplot.close()
    tensorflow.keras.backend.clear_session()
    return all_results


def fracture_classification_array_plot(outputs,
                                       axis=None,
                                       format_for_mlmi=False,
                                       exclude_center_6=False,
                                       exclude_normal=False,
                                       excluded_vertebrae: List[str] = None):
    if excluded_vertebrae is None:
        excluded_vertebrae = []
    if format_for_mlmi and axis is not None:
        raise ValueError
    print('Plotting Squares...')
    if format_for_mlmi:
        ordered_vertebrae = (
                ['T' + str(i) for i in range(4, 13)]
                + ['L' + str(i) for i in range(1, 5)]
        )
    else:
        ordered_vertebrae = VERTEBRAE
    patients = sorted(outputs['patients'].keys())
    # if format_for_mlmi:
    #     fig, axes = pyplot.subplots(2, figsize=(7.5, 3), dpi=202)
    #     plots = [(patients[s], ax) for s, ax in zip([slice(None, len(patients) // 2 + 1), slice(len(patients) // 2 + 1, None)], axes)]
    #     for ps, ax in plots:
    #         empty_array_plot(ax, results=None, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=ps, exclude_center_6=exclude_center_6)
    # else:
    ax, patients = empty_array_plot(axis, results=None, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=patients,
                                    exclude_center_6=exclude_center_6)
    plots = [(patients, ax)]

    w1 = 10 / 18

    cases = {
        'contains_fn': [],
        'contains_fp': [],
        'contains_tp': [],
        'contains_missing_fn': [],
    }

    for ps, ax in plots:
        for patient_idx, p in enumerate(ps):
            threshold = outputs['patient_metrics']['Used Threshold']
            for vertebra_idx, v in enumerate(ordered_vertebrae):
                assert vertebra_idx == ordered_vertebrae.index(v)
                assert patient_idx == ps.index(p), (patient_idx, ps.index(p))

                try:
                    y_true = float(outputs['vertebra_info'][p][v]['Differential Diagnosis Category'] == 'Osteoporotic Fracture')
                except (KeyError, TypeError):  # if there is no ground truth
                    y_true = None
                try:
                    y_pred = outputs['vertebral_fractures'][p][v]
                except (KeyError, TypeError):  # if there is no prediction
                    y_pred = None
                try:
                    is_normal = outputs['vertebra_info'][p][v]['Differential Diagnosis Category'] == 'Normal'
                except (KeyError, TypeError):  # if there is no prediction
                    is_normal = False

                if exclude_normal and is_normal:
                    color = 'white'
                elif y_pred is None and y_true is None:  # vertebra not there
                    color = 'white'
                elif y_pred is None and y_true == 1:
                    color = 'red'
                    cases['contains_missing_fn'].append(p)
                    cases['contains_fn'].append(p)
                elif y_pred is None and y_true == 0:
                    color = 'white'
                elif y_true is None and y_pred < threshold:
                    color = 'white'
                elif y_true is None and y_pred >= threshold:
                    color = 'darkorange'
                    cases['contains_fp'].append(p)
                elif v in excluded_vertebrae:
                    color = 'lightgrey'
                elif y_true == 1 and y_pred >= threshold:  # TP
                    color = 'blue'
                    cases['contains_tp'].append(p)
                elif y_true == 0 and y_pred >= threshold:  # FP
                    color = 'darkorange'
                    cases['contains_fp'].append(p)
                elif y_true == 1 and y_pred < threshold:  # FN
                    color = 'red'
                    cases['contains_fn'].append(p)
                elif y_true == 0 and y_pred < threshold:  # TN
                    color = 'palegreen'
                else:
                    raise LogicError

                if color != 'white':
                    rect = matplotlib.patches.Rectangle((patient_idx - w1 / 2, vertebra_idx - w1 / 2), w1, w1, color=color)
                    ax.add_patch(rect)

    counts = {
        'all_correct': 0,
        'correct_despite_fn': 0,
        'correct_despite_fp': 0,
        'wrong_because_fp': 0,
        'wrong_because_fn': 0,
        'correct_despite_fn_and_fp': 0,
    }
    for p in patients:
        if p not in cases['contains_fp'] and p not in cases['contains_fn']:
            counts['all_correct'] += 1
        if p in cases['contains_fp'] and p in cases['contains_tp'] and p not in cases['contains_fn']:
            counts['correct_despite_fp'] += 1
        if p in cases['contains_fn'] and p in cases['contains_tp'] and p not in cases['contains_fp']:
            counts['correct_despite_fn'] += 1
        if p in cases['contains_fn'] and p in cases['contains_fp']:
            counts['correct_despite_fn_and_fp'] += 1
        if p in cases['contains_fp'] and p not in cases['contains_tp'] and p not in cases['contains_fn']:
            counts['wrong_because_fp'] += 1
        if p in cases['contains_fn'] and p not in cases['contains_tp'] and p not in cases['contains_fp']:
            counts['wrong_because_fn'] += 1
    counts['sum'] = sum(counts.values())
    print(counts)

    ax.set_title('green: TN - red: FN - blue: TP - orange: FP')


def load_trained_model_for_evaluation(model_path) -> Tuple[Model, dict]:
    with open(model_path.replace('.h5', '.json')) as json_file:
        model_config = json.load(json_file)
    model = tensorflow.keras.models.load_model(os.path.abspath(model_path),
                                               custom_objects=custom_layers,
                                               compile=False)
    return model, model_config


def input_shape_with_defaults(model):
    input_shape = list(model.input_shape[1:4])
    if input_shape[0] is None:
        input_shape[0] = 20
    if input_shape[1] is None:
        input_shape[1] = 50
    if input_shape[2] is None:
        input_shape[2] = 40
    return input_shape


def nest_dict(d):
    result: Dict[Any, Union[Dict, Any]] = {}
    for k, v in d.items():
        if not hasattr(k, '__getitem__'):
            result[k] = v
        else:
            nested = result
            for key_part in k[:-1]:
                nested = nested.setdefault(key_part, {})
            nested[k[-1]] = v
    return result


def metric_summary(metric, values, verbose=1):
    print(f'    {metric}')
    if any(isinstance(value, str) for value in values):
        if verbose:
            print('      Values:', values)
        return nan, nan, (nan, nan)
    elif metric in assume_beta_distribution:
        stats = beta_stats(values)
        mean = stats[0]
        var = stats[1]
        ci = beta_conf_interval(values, conf=0.95)
    else:
        mean = sum(values) / len(values)
        conf = mean_confidence_interval_size(values, 0.95)
        ci = mean - conf, mean + conf
        var = numpy.var(values)
    if verbose:
        print('      Values:', values)
        print('      Mean:', mean)
        print('      Std:', sqrt(var))
        print('      ' + to_str_3(mean) + ' (95\\% CI: ' + to_str_3(ci[0]) + '--' + to_str_3(ci[1]) + ')')
        print('      ' + to_str_3(mean) + ' σ = ' + to_str_2(sqrt(var)))
        print('      ' + f'{mean:.3f} ({ci[0]:.3f}--{ci[1]:.3f})')
    return mean, sqrt(var), ci


def index_to_point_in_image_path(image_path, index):
    reader = SimpleITK.SimpleITK.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    metadata_image: SimpleITK.SimpleITK.Image = SimpleITK.GetImageFromArray(numpy.empty(tuple(1 for _ in index)))
    metadata_image.SetDirection(reader.GetDirection())
    metadata_image.SetSpacing(reader.GetSpacing())
    metadata_image.SetOrigin(reader.GetOrigin())
    return metadata_image.TransformContinuousIndexToPhysicalPoint(index)


def metrics_dict(threshold, y_pred, y_true):
    y_pred = numpy.array(y_pred)
    y_true = numpy.array(y_true)
    try:
        auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    except ValueError as e:
        if 'Only one class' in str(e):
            auc = nan
        else:
            raise
    ap_score = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_pred)
    recommended_threshold = choose_threshold_using_svm(y_pred, y_true)
    metrics = {
        'ROC-AUC': float(auc),
        'Average precision': float(ap_score),
        'Recommended Threshold': recommended_threshold,
    }
    if threshold is not None:
        tp = numpy.sum((y_pred >= threshold) * y_true)
        fp = numpy.sum((y_pred >= threshold) * (1 - y_true))
        tn = numpy.sum((y_pred < threshold) * (1 - y_true))
        fn = numpy.sum((y_pred < threshold) * y_true)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        confusion = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred >= threshold)
        metrics = {
            **metrics,
            'Sensitivity': float(sensitivity),
            'Specificity': float(specificity),
            'F1': 2 * float(precision * recall / (precision + recall)),
            'TN': float(confusion[0, 0]),
            'FP': float(confusion[0, 1]),
            'FN': float(confusion[1, 0]),
            'TP': float(confusion[1, 1]),
            'Used Threshold': float(threshold),
        }
    return metrics


def choose_threshold_to_maximize_sens_spec_product(y_pred, y_true):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_pred)
    recommended_threshold = thresholds[numpy.argmax((1 - fpr) * tpr)]
    return recommended_threshold.item()


def choose_threshold_using_svm(y_pred, y_true, C=1):
    if numpy.count_nonzero(y_true) == 0:
        return 1.
    if numpy.count_nonzero(numpy.logical_not(y_true)) == 0:
        return 0.
    sample_weights = numpy.zeros_like(y_true, dtype='float32')
    sample_weights[y_true.astype('bool')] = 0.5 / numpy.count_nonzero(y_true)
    sample_weights[numpy.logical_not(y_true)] = 0.5 / numpy.count_nonzero(numpy.logical_not(y_true))
    sample_weights *= sample_weights.size
    assert numpy.count_nonzero(sample_weights) == sample_weights.size
    svm = sklearn.svm.LinearSVC(C=C)
    svm.fit(y_pred[..., numpy.newaxis], y_true, sample_weight=sample_weights)
    t = -svm.intercept_.item() / svm.coef_.item()
    assert 0 <= t <= 1
    return t


def choose_threshold_using_adjusted_crossentropy(y_pred, y_true):
    sample_weights = numpy.zeros_like(y_true, dtype='float32')
    sample_weights[y_true.astype('bool')] = 0.75 / numpy.count_nonzero(y_true)
    sample_weights[numpy.logical_not(y_true)] = 0.25 / numpy.count_nonzero(numpy.logical_not(y_true))
    assert numpy.count_nonzero(sample_weights) == sample_weights.size
    recommended_threshold = None
    best_result = inf
    for threshold_candidate in numpy.arange(0, 1, 0.001):
        p = y_pred.copy()  # adjusted_y_pred
        t = threshold_candidate
        if t == 0:
            continue
        p[y_pred <= t] = y_pred[y_pred <= t] * 0.5 / t
        p[y_pred > t] = (-t * y_pred[y_pred > t] + y_pred[y_pred > t] / 2 + t - 0.5) / (t - 1) + y_pred[y_pred > t]
        assert numpy.isclose(t * 0.5 / t, (-t * t + t / 2 + t - 0.5) / (t - 1) + t)
        y_pred_adjusted = p
        assert y_pred_adjusted.min() >= 0
        assert y_pred_adjusted.max() <= 1
        result_this_threshold = sklearn.metrics.log_loss(y_true, y_pred_adjusted, sample_weight=sample_weights)
        if result_this_threshold < best_result:
            best_result = result_this_threshold
            recommended_threshold = threshold_candidate
    assert recommended_threshold is not None
    return recommended_threshold.item()


def osteoporotic_label(image):
    for vertebra in image.parts.values():
        if 'Differential Diagnosis Category' in vertebra:
            if vertebra['Differential Diagnosis Category'] == 'Osteoporotic Fracture':
                return 1
    return 0


def is_fractured(image):
    for vertebra_name in image.parts:
        metadata = image.parts[vertebra_name]
        if 'Differential Diagnosis Category' in metadata:
            ddc = str(metadata['Differential Diagnosis Category'])
            if ddc == "Osteoporotic Fracture":
                return 1
        try:
            v = IsFracturedClassification.load_trauma_vertebra(image, vertebra_name)
        except ValueError:
            continue
        if v.body_is_fractured():
            return 1
    return 0


def gs_label(image):
    raise NotImplementedError('Deprecated')


def vertebra_info(image, vertebra_name):
    for v, data in image.parts.items():
        if v == vertebra_name:
            return data
    raise KeyError


def coords_from_iml(iml: ImageList):
    results = {}
    for image in iml:
        vertebrae = image.parts.items()
        for v, vertebra in vertebrae:
            if vertebra.position is None:
                continue
            p = image['patient_id']
            p = int(p)
            results[p, v] = index_to_point_in_image_path(str(image.path), vertebra.position)
    return results


class NotIgnoringBorderVertebrae:
    def __enter__(self):
        global IGNORE_VERTEBRAE_CLOSE_TO_BORDER
        self.ignore_before = IGNORE_VERTEBRAE_CLOSE_TO_BORDER
        IGNORE_VERTEBRAE_CLOSE_TO_BORDER = 0
        print(f'IGNORE_VERTEBRAE_CLOSE_TO_BORDER set to {IGNORE_VERTEBRAE_CLOSE_TO_BORDER}.')

    def __exit__(self, exc_type, exc_val, exc_tb):
        global IGNORE_VERTEBRAE_CLOSE_TO_BORDER
        IGNORE_VERTEBRAE_CLOSE_TO_BORDER = self.ignore_before
        print(f'IGNORE_VERTEBRAE_CLOSE_TO_BORDER reset to {IGNORE_VERTEBRAE_CLOSE_TO_BORDER}.')
