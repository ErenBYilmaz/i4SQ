import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from natsort import natsorted
from numbers import Real
from pathlib import Path
from typing import Union


class Dashboard:
    """Allows to log and visualize time-series data.

    For now, this is a very simple and intuitive way to store data during an
    optimization and visualize it later on automatically.
    """

    def __init__(self):
        #: Mapping from name to lit of scalar values.
        self.scalars = defaultdict(list)

        #: Mapping from name to list of vectors.
        self.vectors = defaultdict(list)

    def scalar(self, name: str, value: Real) -> None:
        """Appends the scalar `value` to the time-series list identified
        by `name`.

        Args:
            name: The unique identifier for the series of values.
            value: The real scalar value.
        """
        self.scalars[name].append(value)

    def vector(self, name: str, vector) -> None:
        """Appends the vector `value` to the time-series list identified by
        `name`.

        Args:
            name: The unique identifier for the series of values.
            value: The real scalar value.
        """
        vector = np.reshape(vector, (len(vector,)))
        self.vectors[name].append(vector)

    def show(self) -> None:
        """Shows the current Dashboard in a new `Figure`."""
        self.plot(plt.figure())
        plt.show()

    def save(self, path: Union[str, Path]) -> None:
        """Plots and saves the current :class:`Dashboard`.

        Args:
            path: Where to save the plot to.
        """
        fig = plt.figure()
        self.plot(fig)
        plt.savefig(str(path), dpi=300)
        plt.close(fig)

    def plot(self, fig=None) -> None:
        """Plots the current data.

        Args:
            fig: An optional `Figure` to plot to, if `None` the current figure
                is used.
        """
        if fig is None:
            fig = plt.gcf()

        n_series = len(self.scalars) + len(self.vectors)

        cols = np.ceil(np.sqrt(n_series))
        rows = np.ceil(np.sqrt(n_series))

        plots = list(self.scalars.items()) + list(self.vectors.items())

        for i, (name, series) in enumerate(natsorted(plots,
                                                     key=lambda p: p[0])):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(name)
            ax.set_xlabel('t')

            y = np.array(series)
            x = range(len(series))

            if y.ndim == 1:
                ax.plot(x, y, '-')
            elif y.ndim == 2:
                for j in range(y.shape[1]):
                    ax.plot(x, y[:, j], '-')
            else:
                raise RuntimeError('Unable to handle %iD data' % y.ndim)
