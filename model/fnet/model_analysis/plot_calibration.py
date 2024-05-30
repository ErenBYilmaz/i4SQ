import os
from math import inf
from typing import List, Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import label_binarize
from sklearn.utils import column_or_1d, check_consistent_length

from lib.my_logger import logging
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, SingleModelEvaluationResult
from tasks import VertebraClassificationTask


def calibration_curve(y_true, y_prob, normalize=False, n_bins=5,
                      strategy='uniform', weighted=False):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         f"Provided labels {labels}.")
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if weighted:
        pos_indices = numpy.where(y_true)[0]
        neg_indices = numpy.where(1 - y_true)[0]
        balanced_indices = numpy.concatenate([numpy.repeat(pos_indices, repeats=max(len(neg_indices), 1)),
                                              numpy.repeat(neg_indices, repeats=max(len(pos_indices), 1))])
        y_true = y_true[balanced_indices]
        y_prob = y_prob[balanced_indices]
        y_prob *= numpy.random.normal(size=y_prob.shape, loc=1, scale=0.000001)

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_min_pred = numpy.full(len(bins), inf)
    np.minimum.at(bin_min_pred, binids, y_prob)
    bin_max_pred = numpy.full(len(bins), -inf)
    np.maximum.at(bin_max_pred, binids, y_prob)
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
    bin_min_pred = bin_min_pred[nonzero]
    bin_max_pred = bin_max_pred[nonzero]
    bin_total = bin_total[nonzero]

    return prob_true, prob_pred, bin_min_pred, bin_max_pred, bin_total


def plot_calibration_curve(y_true, probas_list, clf_names=None, n_bins=10,
                           title='Calibration plots (Reliability Curves)',
                           ax=None, figsize=None, cmap='nipy_spectral',
                           title_fontsize: Optional[str] = "large",
                           text_fontsize: Optional[str] = "medium",
                           strategy='quantile',
                           plot_type='line',
                           class_name='positive',
                           weighted=False):
    y_true = np.asarray(y_true)
    if not isinstance(probas_list, list):
        raise ValueError('`probas_list` does not contain a list.')

    classes = np.unique(y_true)
    if len(classes) > 2:
        raise ValueError('plot_calibration_curve only '
                         'works for binary classification')

    if clf_names is None:
        clf_names = ['Classifier {}'.format(x + 1)
                     for x in range(len(probas_list))]

    if len(clf_names) != len(probas_list):
        raise ValueError('Length {} of `clf_names` does not match length {} of'
                         ' `probas_list`'.format(len(clf_names),
                                                 len(probas_list)))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if weighted:
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated if true class distribution is uniform")
    else:
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (for this class distribution)")

    for i, probas in enumerate(probas_list):
        probas = np.asarray(probas)
        if probas.ndim > 2:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))
        if probas.ndim == 2:
            probas = probas[:, 1]

        if probas.shape != y_true.shape:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))

        # probas = (probas - probas.min()) / (probas.max() - probas.min())

        if y_true.size == 0:
            raise RuntimeError
        assert probas.size > 0

        fraction_of_positives, mean_predicted_value, min_predicted_value, max_predicted_value, bin_sizes = \
            calibration_curve(y_true, probas, n_bins=n_bins, strategy=strategy, weighted=weighted)

        color = plt.cm.get_cmap(cmap)(float(i) / len(probas_list))

        if plot_type == 'line':
            ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                    label=clf_names[i], color=color)
        elif plot_type == 'bar_and_line':
            pyplot.bar(x=min_predicted_value, align='edge',
                       width=max_predicted_value - min_predicted_value,
                       height=fraction_of_positives, )
            ax.plot(mean_predicted_value, fraction_of_positives, 'x-',
                    label=clf_names[i], color=color)
        else:
            raise NotImplementedError

    ax.set_title(title, fontsize=title_fontsize)

    if weighted:
        ax.set_ylabel(f'Weighted fraction of "{class_name}"', fontsize=text_fontsize)
    else:
        ax.set_ylabel(f'Fraction of "{class_name}"', fontsize=text_fontsize)
    if strategy == 'quantile':
        bin_size = round(len(y_true) / n_bins)
        ax.set_xlabel(f'Model confidence in "{class_name}" for bins of size {bin_size:.1f}', fontsize=text_fontsize)
    else:
        bin_width = 1 / n_bins
        ax.set_xlabel(f'Model confidence in "{class_name}" for bins of width {bin_width:.2g}', fontsize=text_fontsize)

    ax.set_ylim([0, 1.05])
    ax.legend()

    return ax


class CalibrationCurvePlotter(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator, skip_existing=None, model_level_plotting=True):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, skip_existing=skip_existing)
        self.model_level_caching = self.skip_existing
        self.model_level_plotting = model_level_plotting

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()
        _, ys, sample_weights = batch
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
        self.serializer().create_directory_if_not_exists(to_dir)
        result = {}
        for task, y_pred, y_true in zip(self.tasks, y_preds, ys):
            if not isinstance(task, VertebraClassificationTask):
                continue
            task: VertebraClassificationTask
            for class_idx, class_name in enumerate(task.class_names()):
                if len(task.class_names()) == 2 and class_idx == 0:
                    continue
                true_prob: numpy.ndarray = task.class_probabilities(y_true)[..., class_idx]
                pred_prob: numpy.ndarray = task.class_probabilities(y_pred)[..., class_idx]
                valid_indices = (true_prob == 1) | (true_prob == 0)
                true_prob = true_prob[valid_indices]
                pred_prob = pred_prob[valid_indices]
                class_name = class_name.replace(' ', '_')
                out_path_prefix = os.path.join(to_dir, f'{task.output_layer_name()}_{class_name}')

                clf_name = os.path.basename(self.model_path)
                if self.model_level_plotting:
                    if numpy.unique(pred_prob).size >= 2:
                        self.plot_curves(true_prob, pred_prob, class_name, out_path_prefix, clf_name=clf_name)
                result[task.output_layer_name(), class_idx] = {
                    'task': task,
                    'true_prob': true_prob,
                    'pred_prob': pred_prob,
                    'class_name': class_name,
                    'out_path_prefix': out_path_prefix
                }
        return result

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        results: List[Dict[Tuple[str, int], Dict[str, Any]]]  # see result of analyze_batch
        to_dir = os.path.join(self.to_dir(), 'summary')
        self.serializer().create_directory_if_not_exists(to_dir)
        all_plots: Dict[Tuple[str, int], Tuple[VertebraClassificationTask, str]] = {
            k: (v['task'], v['class_name'])
            for r in results for k, v in r.items()
        }
        for task_name, class_idx in all_plots:
            task, class_name = all_plots[task_name, class_idx]
            assert isinstance(task, VertebraClassificationTask)

            true_probs = [r[task_name, class_idx]['true_prob'] for r in results]
            pred_probs = [r[task_name, class_idx]['pred_prob'] for r in results]
            true_prob: numpy.ndarray = numpy.concatenate(true_probs)
            pred_prob: numpy.ndarray = numpy.concatenate(pred_probs)
            assert len(true_prob.shape) == 1
            assert len(pred_prob.shape) == 1

            out_path_prefix = os.path.join(to_dir, f'{task.output_layer_name()}_{class_name}')
            self.plot_curves(true_prob, pred_prob, class_name, out_path_prefix, clf_name='multiple models combined')

    def plot_curves(self, true_prob, pred_prob, class_name, out_path_prefix, clf_name: str):
        weighting = [
            True,
            False,
        ]
        for weighted in weighting:
            base_name = out_path_prefix
            if weighted:
                base_name += '_weighted'
            out_paths = [
                base_name + '.png',
                base_name + '.svg',
            ]
            if self.skip_existing and all(self.serializer().isfile(out_path) for out_path in out_paths):
                continue
            logging.info('Writing to ' + str(base_name + '.*'))
            pyplot.figure(figsize=(20, 8), dpi=202)
            plot_calibration_curve(
                true_prob,
                [pred_prob],
                clf_names=[clf_name],
                n_bins=20,
                ax=pyplot.gca(),
                title_fontsize=None,
                text_fontsize=None,
                class_name=class_name,
                strategy='quantile',
                plot_type='bar_and_line',
                weighted=weighted,
            )
            pyplot.tight_layout()
            for out_path in out_paths:
                self.serializer().save_current_pyplot_figure(out_path)
            pyplot.close()

    def to_subdir(self):
        return 'calibration_curves'
