from typing import List
from lib.util import beta_stats, beta_conf_interval

import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import roc_curve, auc, precision_recall_curve



def plot_curve(y_trues: List[List[float]], y_preds: List[List[float]],
               num_thresholds=1000,
               curve=roc_curve,
               plot_individual=False,
               curve_name='curve',
               area_name='area',
               write_area='mean',
               monotonic=0,
               title=None,
               xlabel='False Positive Rate',
               ylabel='True Positive Rate',
               flip_xy=False,
               plot_ci=True):
    if len(y_trues) != len(y_preds):
        raise ValueError
    yss = []
    areas = []
    mean_xs = numpy.linspace(numpy.nextafter(0, -1), numpy.nextafter(1, 2), num_thresholds)

    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        if len(y_true) != len(y_pred):
            raise ValueError
        # Compute ROC curve and area the curve
        xs, ys, _ = curve(y_true, y_pred)
        if flip_xy:
            xs, ys = ys, xs
        # sort xs
        xs, ys = zip(*sorted(zip(xs, ys)))
        if monotonic > 0:
            ys = numpy.maximum.accumulate(ys)
        elif monotonic < 0:
            ys = numpy.minimum.accumulate(ys)
        yss.append(numpy.interp(mean_xs, xs, ys))
        area = auc(mean_xs, yss[-1])
        areas.append(area)
        if plot_individual:
            plt.plot(xs, ys, lw=2, alpha=0.3, )

    yss = numpy.array(yss)  # shape (models, xs)
    mean_ys = numpy.zeros(shape=(yss.shape[1],))

    lower_conf_ys = numpy.zeros(shape=(yss.shape[1],))
    upper_conf_ys = numpy.zeros(shape=(yss.shape[1],))
    for idx in range(yss.shape[1]):
        mean_ys[idx], _ = beta_stats(yss[:, idx])
        lower_conf_ys[idx], upper_conf_ys[idx] = beta_conf_interval(yss[:, idx])
    if monotonic > 0:
        # the lower and upper bound can not go down
        lower_conf_ys = numpy.maximum.accumulate(lower_conf_ys)
        upper_conf_ys = numpy.minimum.accumulate(upper_conf_ys[::-1])[::-1]
    elif monotonic < 0:
        # the lower and upper bound can not go up
        lower_conf_ys = numpy.maximum.accumulate(lower_conf_ys[::-1])[::-1]
        upper_conf_ys = numpy.minimum.accumulate(upper_conf_ys)
    mean_area, _ = beta_stats(areas)
    lower_conf_area, upper_conf_area = beta_conf_interval(areas)
    if write_area == 'full':
        area_string = f' ({area_name} = {mean_area:0.3f}, 95% CI: {lower_conf_area:0.3f}-{upper_conf_area:0.3f})'
    elif write_area == 'mean':
        area_string = f' ({area_name} = {mean_area:0.3f})'
    else:
        area_string = ''
    plt.plot(mean_xs, mean_ys, color='b',
             label=f'{curve_name}{area_string}',
             lw=2, alpha=.8)

    if plot_ci:
        plt.fill_between(mean_xs, lower_conf_ys, upper_conf_ys, color='grey', alpha=.2,
                         label=r'95% CI')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    # plt.show()


def plot_roc_curve(y_trues: List[List[float]], y_preds: List[List[float]],
                   num_thresholds=1000,
                   curve=roc_curve,
                   plot_individual=False,
                   curve_name='ROC curve',
                   write_area='mean',
                   area_name='AUC',
                   monotonic=1,
                   title=None,
                   xlabel='False Positive Rate',
                   ylabel='True Positive Rate',
                   flip_xy=False,
                   plot_ci=True):
    return plot_curve(y_trues, y_preds,
                      num_thresholds,
                      curve,
                      plot_individual,
                      curve_name,
                      area_name,
                      write_area,
                      monotonic,
                      title,
                      xlabel,
                      ylabel,
                      flip_xy,
                      plot_ci=plot_ci)


def plot_pr_curve(y_trues: List[List[float]], y_preds: List[List[float]],
                  num_thresholds=20,
                  curve=precision_recall_curve,
                  plot_individual=False,
                  curve_name='PR curve',
                  write_area='mean',
                  area_name='AP',
                  monotonic=0,
                  title=None,
                  xlabel='Recall',
                  ylabel='Precision',
                  flip_xy=True):
    return plot_curve(y_trues, y_preds,
                      num_thresholds,
                      curve,
                      plot_individual,
                      curve_name,
                      area_name,
                      write_area,
                      monotonic,
                      title,
                      xlabel,
                      ylabel,
                      flip_xy)

