import matplotlib.pyplot as plt
import numpy as np

from colorama import Fore
from typing import List, Optional
from scipy.spatial import distance


class Evaluation:
    """Provides basic error calculations around a true and a predicted
    position.

    Using this object in a format string like `'{}'.format(eval)` shows
    the results in a way appropriate for human inspection.

    Can be used in conjunction with :class:`AvgEvaluation` to compute
    average metrics.
    """
    def __init__(self, true_pos: np.ndarray, pred_pos: np.ndarray,
                 localized: bool, spacing: Optional[np.ndarray] = None):
        #: Whether the target object has been correctly localized given a
        #: certain criterion.
        self.localized = localized

        #: Error in image coordinates expressed as Euclidean distance.
        self.error = distance.euclidean(true_pos, pred_pos)

        #: Optional: Error in in mm expressed as Euclidean distance.
        self.error_mm = None

        if spacing is not None:
            self.error_mm = distance.euclidean(true_pos * spacing,
                                               pred_pos * spacing)

    def __format__(self, spec):
        return 'localized = {loc}, error = {error}'.format(
            loc=format_localized(self.localized),
            error=format_error(self.error, self.error_mm))


class AvgEvaluation:
    """Computes certain statistics over a set of :class:`Evaluation`.

    Can be used as argument to a format string, similar to :class:`Evaluation`.
    """

    def __init__(self, evaluations: List[Evaluation]):
        errors = [e.error for e in evaluations]
        errors_mm = [e.error_mm for e in evaluations if e.error_mm is not None]

        #: Individual evaluations this average is based on.
        self.evaluations = evaluations

        #: Total number of properly localized evaluations.
        self.localized = sum(1 for e in evaluations if e.localized)

        #: Average error.
        self.error = np.mean(errors)

        #: Optional: Average error in mm.
        self.error_mm = None

        if errors_mm:
            self.error_mm = np.mean(errors_mm)

    def __format__(self, spec):
        return ('localized = {loc_per:5.1f}% ({loc_abs:{digits}d}/'
                '{total:{digits}d}), error = {error}').format(
            digits=digits(len(self.evaluations)),
            loc_per=self.localized / len(self.evaluations) * 100,
            loc_abs=self.localized, total=len(self.evaluations),
            error=format_error(self.error, self.error_mm))

    def plot_localizations(self, ax=None, max_error=-1, max_error_mm=-1):
        """Plots the localization rate as a function of the error threshold.

        :param ax: An optional `Axes` to plot to. If `None`, the current active
                   `Axes` retrieved via `gca()` is used.
        :param max_error: If `None` no pixel error is plotted, else it is
                          maximum error threshold with -1 being open ended.
        :param max_error_mm: If `None` no mm error is plotted, else it is
                             maximum error threshold with -1 being open ended.
        """
        if ax is None:
            ax = plt.gca()

        max_x = [0]

        def compute_graph(errors, max_error):
            if max_error == -1:
                max_error = int(np.ceil(errors.max()))

            max_x[0] = max(max_error, max_x[0])

            xs = np.arange(max_error + 1)
            ys = np.empty_like(xs)

            for i, t in enumerate(xs):
                ys[i] = np.sum(errors <= t) / len(errors) * 100

            return xs, ys

        title = 'Localization rate with Euclidean distance as error measure'

        if max_error is not None:
            errors = np.array([e.error for e in self.evaluations])
            if len(errors):
                ax.plot(*compute_graph(errors, max_error), 'b-', label='px',
                        marker='.')
                title += ', #images = {}'.format(len(errors))

        if max_error_mm is not None:
            errors_mm = np.array([e.error_mm for e in self.evaluations
                                  if e.error_mm is not None])
            if len(errors_mm):
                ax.plot(*compute_graph(errors_mm, max_error_mm), 'r-',
                        label='mm', marker='.')
                title += ', #images_mm = {}'.format(len(errors_mm))

        ax.set_title(title)
        ax.set_xlabel('Error threshold $t$ (px/mm)')
        ax.set_ylabel('Amount images with $e \\leq t$ (%)')
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_x[0])
        ax.grid(True)
        ax.legend(loc='lower right')


def format_error(error, error_mm=None, unit='px'):
    """Creates a nicely formatted error string with optional error in mm."""
    errors = ['{:5.1f}{}'.format(error, unit)]

    if error_mm is not None:
        errors.insert(0, '{:5.1f}mm'.format(error_mm))

    return ' / '.join(errors)


def format_localized(localized: bool) -> str:
    """Creates a nicely colored cross or checkmark for `localized`."""
    return (Fore.GREEN + '✓' if localized else Fore.RED + '✗') + Fore.RESET


def digits(number: int) -> int:
    """Counts the digits in the given `number`."""
    if number > 0:
        return int(np.log10(number)) + 1
    elif number == 0:
        return 1
    else:
        return int(np.log10(-number)) + 2
