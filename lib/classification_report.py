from math import nan, sqrt

from numpy.lib.nanfunctions import _replace_nan

from lib.parameter_search import mean_confidence_interval_size
from lib.util import beta_stats, beta_conf_interval
from matplotlib import pyplot as plt
import numpy
import sklearn
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, multilabel_confusion_matrix, ConfusionMatrixDisplay


def metric_summary(values, assume_beta_distribution=False, metric_name=None, verbose=0, print_number=str, return_dict=True, return_values=False):
    if isinstance(values, str):
        raise ValueError
    values = numpy.array(values)
    if assume_beta_distribution is None:
        assume_beta_distribution = numpy.issubdtype(values.dtype, numpy.number) and numpy.all(0 <= values) and numpy.all(values <= 1)
    if verbose:
        print(f'    {metric_name}')
    if values.size == 0 or any(isinstance(value, str) for value in values):
        if verbose:
            print('      Values:', values)
        if not return_dict:
            if return_values:
                return nan, nan, (nan, nan), values
            else:
                return nan, nan, (nan, nan)
        else:
            result = {
                'mean': nan,
                'std': nan,
                'ci': nan,
            }
            if return_values:
                result['values'] = values
            return result
    if assume_beta_distribution is None:
        assume_beta_distribution = numpy.all(0 <= values) and numpy.all(values <= 1)
    if assume_beta_distribution:
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
        if return_values:
            print('      Values:', values)
        print('      Mean:', mean)
        print('      Std:', sqrt(var))
        # print('      ' + print_number(mean) + ' (95\\% CI: ' + print_number(ci[0]) + '--' + print_number(ci[1]) + ')')
        print('      ' + f'{mean:.3f} ({ci[0]:.3f}--{ci[1]:.3f})')
    if not return_dict:
        if return_values:
            return mean, sqrt(var), ci, values
        else:
            return mean, sqrt(var), ci
    else:
        result = {
            'mean': mean,
            'std': sqrt(var),
            'ci': ci,
        }
        if return_values:
            result['values'] = values
        return result


class Unguessable(ValueError):
    pass


def custom_report(y_true, y_pred, target_names,
                  sample_weight=None,
                  zero_division="warn",
                  plot_confusion=None,
                  task='guess'):
    if task == 'guess':
        if numpy.array_equal(y_true, y_true.astype(int)):  # only zeros and ones
            task = 'classification'
        elif numpy.issubdtype(y_true.dtype, numpy.inexact) and numpy.issubdtype(y_pred.dtype, numpy.inexact):  # both arrays contain floats
            task = 'regression'
        else:
            raise Unguessable
    orig_target_names = target_names
    if task == 'classification':
        if not numpy.array_equal(y_true, y_true.astype(bool)):  # only zeros and ones in y_true
            y_true = numpy.round(y_true)
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(y_true)
            y_true = enc.transform(y_true).toarray()
            y_pred = numpy.round(y_pred)
            y_pred = numpy.maximum(y_pred, numpy.min(enc.categories_))
            y_pred = numpy.minimum(y_pred, numpy.max(enc.categories_))
            y_pred = enc.transform(numpy.round(y_pred)).toarray()
            if len(target_names) == 1:
                target_names = [(target_names[0] + '_' + str(label))
                                for label in enc.categories_[0]]

    if y_pred.shape != y_true.shape:
        raise ValueError

    if len(y_pred.shape) == 1:
        y_pred = y_pred[:, numpy.newaxis]
        y_true = y_true[:, numpy.newaxis]

    if task == 'classification':
        if y_pred.shape[1] == 1:  # only one class -> negative and positive class
            y_pred = numpy.concatenate([1 - y_pred, y_pred], axis=1)
            y_true = numpy.concatenate([1 - y_true, y_true], axis=1)
        # get hard classifications
        y_pred_hard = numpy.zeros_like(y_pred)
        class_axis = 1
        number_labels = numpy.argmax(y_pred, axis=class_axis)
        y_pred_hard[numpy.arange(number_labels.size), number_labels] = 1  # see https://stackoverflow.com/a/29831596
        # y_pred_hard[numpy.argmax(y_pred, axis=class_axis)] = 1
        report = classification_report(y_true=y_true,
                                       y_pred=y_pred_hard,
                                       target_names=target_names,
                                       sample_weight=sample_weight,
                                       output_dict=True,
                                       zero_division=zero_division).copy()
        # compute aps and roc for each class
        single_roc_aucs: numpy.ndarray = roc_auc_score(y_true=y_true, y_score=y_pred, average=None)
        assert single_roc_aucs.shape == (len(target_names),)
        single_ap_scores: numpy.ndarray = average_precision_score(y_true=y_true, y_score=y_pred, average=None)
        assert single_ap_scores.shape == (len(target_names),)
        ovr_confusion_matrices: numpy.ndarray = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred_hard)
        assert ovr_confusion_matrices.shape == (len(target_names), 2, 2)
        confusion_matrix: numpy.ndarray = sklearn.metrics.confusion_matrix(y_true=numpy.argmax(y_true, axis=1), y_pred=numpy.argmax(y_pred_hard, axis=1))
        assert confusion_matrix.shape == (len(target_names), len(target_names))
        if plot_confusion:
            plot_confusion_matrix_with_class_names(confusion_matrix, target_names, plot_confusion)
        for label_idx, label in enumerate(target_names):
            assert label in report
            report[label]['average_precision_score'] = single_ap_scores[label_idx]
            report[label]['roc_auc_score'] = single_roc_aucs[label_idx]
            report[label]['tp'] = ovr_confusion_matrices[label_idx, 1, 1]
            report[label]['fp'] = ovr_confusion_matrices[label_idx, 0, 1]
            report[label]['tn'] = ovr_confusion_matrices[label_idx, 0, 0]
            report[label]['fn'] = ovr_confusion_matrices[label_idx, 1, 0]
            for j in range(len(target_names)):
                report[label][f'predicted_to_be_{target_names[j]}'] = confusion_matrix[label_idx, j]
            report[label]['y_pred_mean'] = numpy.mean(y_pred[:, label_idx])
            report[label]['y_pred_median'] = numpy.median(y_pred[:, label_idx])
            report[label]['y_pred_std'] = numpy.std(y_pred[:, label_idx])
        for avg_method in ['micro', 'macro', 'weighted', 'samples']:
            try:
                report[f'{avg_method} avg']['roc_auc_score'] = roc_auc_score(y_true, y_pred, average=avg_method)
            except ValueError as e:
                if 'Only one class present in y_true' in str(e):
                    report[f'{avg_method} avg']['roc_auc_score'] = nan
                else:
                    raise
            report[f'{avg_method} avg']['average_precision_score'] = average_precision_score(y_true, y_pred, average=avg_method)
    elif task == 'regression':
        s = 0.4
        report = {}
        for label_idx, label in enumerate(target_names):
            report[label] = {}
            report[label]['mse'] = numpy.square(y_true[:, label_idx] - y_pred[:, label_idx]).mean()
            report[label]['rmse'] = numpy.sqrt(numpy.square(y_true[:, label_idx] - y_pred[:, label_idx]).mean())
            report[label]['mae'] = (y_true[:, label_idx] - y_pred[:, label_idx]).mean()
            if plot_confusion:
                plt.figure()
                plt.title(label)
                plt.scatter(y_pred[:, label_idx], y_true[:, label_idx], s=s)
                plt.gca().set(ylabel="True value",
                              xlabel="Predicted value")
                plt.savefig(plot_confusion[:-4] + '_' + label + '_' + plot_confusion[-4:])
                plt.clf()
                plt.close()
        if plot_confusion:
            plt.figure()
            plt.title(', '.join(target_names))
            plt.scatter(y_pred, y_true, s=s)
            plt.gca().set(ylabel="True value",
                          xlabel="Predicted value")
            plt.savefig(plot_confusion)
            plt.clf()
            plt.close()
        report['avg'] = {}
        for metric in report[target_names[0]]:
            report['avg'][metric] = numpy.mean([report[label][metric] for label in target_names])
    else:
        raise NotImplementedError
    return report


class NaNArray(numpy.ndarray):
    def min(self, *args, **kwargs):
        arg, _ = _replace_nan(self, numpy.inf)
        arg = arg.view(numpy.ndarray)
        assert not isinstance(arg, NaNArray)
        return numpy.min(arg, *args, **kwargs)

    def max(self, *args, **kwargs):
        arg, _ = _replace_nan(self, -numpy.inf)
        arg = arg.view(numpy.ndarray)
        assert not isinstance(arg, NaNArray)
        return numpy.max(arg, *args, **kwargs)


def plot_confusion_matrix_with_class_names(sklearn_confusion_matrix, class_names, to_file=None, normalize=False, float_format=None, remove_axis_labels=False):
    matrix = sklearn_confusion_matrix
    if float_format is None:
        float_format = normalize
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    cmd = ConfusionMatrixDisplay(confusion_matrix=matrix if float_format else matrix.astype(int),
                                 display_labels=class_names)
    cmd.confusion_matrix = cmd.confusion_matrix.view(NaNArray)
    from matplotlib import pyplot
    pyplot.rcParams.update({'font.size': 16})
    pyplot.rcParams.update({'xtick.labelsize': 16})
    pyplot.rcParams.update({'ytick.labelsize': 16})
    # pyplot.rcParams.update({'axes.labelsize': 12})
    pyplot.rcParams.update({'legend.fontsize': 16})
    values_format = '.2g' if float_format else 'd'
    plot_height = 300 + 70 * len(sklearn_confusion_matrix)
    plot_width = 640 / 480 * plot_height
    dpi = 100
    pyplot.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
    cmd.plot(xticks_rotation=90, cmap='Blues',
             values_format=values_format,
             colorbar=False,
             ax=pyplot.gca())
    plt.grid(False)
    if remove_axis_labels:
        pyplot.gca().set(ylabel="",
                         xlabel="")
    plt.tight_layout()
    if to_file is not None:
        plt.savefig(to_file)
        plt.close()


def aggregate_values(obj, return_values=False):
    ...  # TODO generalize to arbitrary dicts/lists


def aggregate_classification_reports(reports, return_values=False):
    aggregate_results = {}
    if len(reports) == 0:
        raise ValueError
    for target_name in reports[0]:
        for metric_name in reports[0][target_name]:
            if target_name not in aggregate_results:
                aggregate_results[target_name] = {}
            aggregate_results[target_name][metric_name] = metric_summary(values=[r[target_name][metric_name]
                                                                                 for r in reports],
                                                                         metric_name=target_name,
                                                                         return_dict=True,
                                                                         return_values=return_values)
    return aggregate_results
