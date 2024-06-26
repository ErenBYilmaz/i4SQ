import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Union

from .utils import guess_image_shape


__all__ = (
    'Landmark',
    'landmarks',
    'heatmap',
    'loc_rate',
    'distinct_colors'
)


class Landmark(str, Enum):
    """Different point landmark types."""

    TRUE_POS = 'true_pos'
    PRED_POS = 'pred_pos'
    CANDIDATES = 'candidates'


def landmarks(image: np.ndarray, landmarks: Dict[str, Dict[str, np.ndarray]],
              spacing: Optional[np.ndarray] = None,
              title: Optional[str] = None,
              views: List[str] = ['xy', 'yz', 'xz'], invert_axes: str = '',
              labeled = True):
    """Plots different types of landmarks for 2D and 3D images.

    In case of a 3D image, three planes are shown which are generated by
    averaging the slices of the given `true_pos` (or `pred_pos` if not given).

    Args:
        image: The image we want to plot landmarks on.
        landmarks: A mapping of landmark name to different type of landmarks,
            i.e., `true_pos`, `pred_pos` and `candidates`. You can also just
            give pass the position, which is than assumed to be the `true_pos`.
        spacing: An optional spacing, if given, everything is plotted in mm.
        title: An optional title of the figure.
        views: A list of slice views to generate for 3D images.
        invert_axes: Whether to invert certain axes.
    """
    assert all(len(v) == 2 and set(v).issubset('xyz') for v in views)
    assert set(invert_axes).issubset('xyz')

    if labeled:
        colors = distinct_colors(len(landmarks))
        border_colors = None
    else:
        colors = [{
            Landmark.TRUE_POS: 'black',
            Landmark.PRED_POS: 'red',
        } for _ in range(len(landmarks))]
        border_colors = [{
            Landmark.PRED_POS: 'black',
        } for _ in range(len(landmarks))]
    landmarks = dict(landmarks)

    for name, landmark in landmarks.items():
        if isinstance(landmark, np.ndarray):
            landmarks[name] = {Landmark.TRUE_POS: landmark}

    if spacing is None:
        spacing = np.ones(image.ndim, dtype=int)
        unit = 'px'
    else:
        unit = 'mm'

    n_dims, n_channels = guess_image_shape(image)

    # if 3D image, we only use one channel, as the modality is not clear
    if n_dims == 3 and n_channels is not None:
        image = image[:, :, :, 0]

    if n_dims == 2:
        _landmarks_2d(plt.gca(), image, landmarks, spacing, colors, unit, border_colors=border_colors)
    else:
        for i, view in enumerate(views):
            view = [2 + ord('x') - ord(a) for a in view]
            ignored = np.setdiff1d(range(3), view)[0]

            image_2d = np.zeros([image.shape[a] for a in view[::-1]])
            landmarks_2d = dict()
            spacing_2d = spacing[::-1][view]

            for name, landmark in landmarks.items():
                position = next(landmark[lm] for lm in Landmark
                                if lm in landmark)

                image_slice = image[tuple([slice(None) for _ in range(ignored)]
                                          + [int(position[::-1][ignored])])]
                if view == sorted(view):
                    image_slice = image_slice.T
                image_2d += image_slice

                landmark_2d = landmarks_2d[name] = {}
                for lm in Landmark:
                    if lm in landmark:
                        position = landmark[lm]
                        if lm == Landmark.CANDIDATES:
                            position = np.transpose(position)
                        landmark_2d[lm] = position[::-1][view]
                        if lm == Landmark.CANDIDATES:
                            landmark_2d[lm] = np.transpose(landmark_2d[lm])

            image_2d /= len(landmarks)

            ax = plt.subplot(1, len(views), 1 + i)

            _landmarks_2d(ax, image_2d, landmarks_2d, spacing_2d, colors,
                          unit if i == 0 else None, draw_line=labeled, border_colors=border_colors)

            ax.set_xlabel(f'{views[i][0]} / {unit}')
            ax.set_ylabel(f'{views[i][1]} / {unit}')

            if views[i][0] in invert_axes:
                ax.invert_xaxis()
            if views[i][1] in invert_axes:
                ax.invert_yaxis()

    if title is not None:
        plt.suptitle(title)


def heatmap(hm: np.ndarray, true_pos: Optional[np.ndarray] = None,
            pred_pos: Optional[np.ndarray] = None,
            spacing: Optional[np.ndarray] = None,
            title: Optional[str] = None):
    """Plots the given heatmap and additional meta information.

    All coordinates are assumed to be in X Y order.

    Args:
        hm: The heatmap we want to plot.
        true_pos: An optional true position.
        pred_pos: One or multiple predicted positions (optional).
        spacing: An optional spacing to show everything in mm.
        title: An optional title to apply.
    """
    assert hm.ndim == 2

    extent = None

    if spacing is not None:
        extent = [0, hm.shape[1] * spacing[0], hm.shape[0] * spacing[1], 0]
    else:
        spacing = np.ones(2)

    plt.imshow(hm, extent=extent)
    plt.colorbar()
    plt.autoscale(False)
    plt.ylabel('px' if extent is None else 'mm')
    plt.xlabel('px' if extent is None else 'mm')

    if title is not None:
        plt.title(title)

    if true_pos is not None:
        plt.plot(*(true_pos * spacing), 'r+')

    if pred_pos is not None:
        pred_pos = pred_pos * spacing

        if pred_pos.ndim == 2:
            for i, pos in enumerate(pred_pos):
                plt.annotate(f'{i + 1}', xy=pos, xytext=[1.5, 0],
                             textcoords='offset points', fontsize='small')

            pred_pos = pred_pos.T

        plt.plot(*pred_pos, 'g.')


def loc_rate(true_pos: np.ndarray, pred_pos: np.ndarray,
             max_dist: Union[int, float] = 20,
             thresh_dist: Optional[np.ndarray] = None,
             unit: str = 'px', title: Optional[str] = None) -> None:
    """Plots the localization rate with respect to different localization
    thresholds.

    Args:
        true_pos: An array of true positions.
        pred_pos: An array of corresponding predicted positions.
        max_dist: Compute rates for up to this distance.
        thresh_dist: Distance between true and prediction position when it is
            considered correct (<=).
        unit: The unit of the positions and the distance.
        title: An optional title to apply.
    """
    assert max_dist > 0
    assert thresh_dist is None or (0 < thresh_dist < max_dist)

    true_pos = np.asarray(true_pos)
    pred_pos = np.asarray(pred_pos)

    x = np.arange(int(np.ceil(max_dist)) + 1)
    y = np.zeros_like(x)

    dists = np.sqrt(np.sum((true_pos - pred_pos)**2, axis=1))

    for dist in dists:
        y[int(np.ceil(dist)):] += 1

    y = y / len(true_pos) * 100

    y_interp = interp1d(x, y, kind='slinear')
    x2 = np.linspace(x[0], x[-1], int(max_dist * 5))
    y2 = y_interp(x2)

    plt.plot(x2, y2, 'b-')
    plt.plot(x, y, 'b.')
    plt.xlim(0, x[-1])
    plt.ylim(0, 100)
    plt.xlabel(unit)
    plt.ylabel('Localization rate / %')
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(axis='y')

    if title is not None:
        plt.title(title)

    if thresh_dist is not None:
        x = np.asarray(thresh_dist)
        x.shape = (x.size,)

        y = [(dists <= d).sum() / len(true_pos) * 100 for d in x]

        plt.plot(x, y, 'r.')

        for p in zip(x, y):
            plt.annotate(f'{p[1]:.2f}%', xy=p, xytext=(0, 10),
                         textcoords='offset points', fontsize='small',
                         ha='center')

    ax = plt.gca().twinx()
    ax.set_yticks(np.arange(0, 101, 10) / 100 * len(true_pos))
    ax.set_ylim(0, len(true_pos), emit=False)
    ax.set_ylabel('Number of localizations')


def nbest_loc_rate(true_pos: np.ndarray, pred_pos: np.ndarray,
                   thresh_dist: float = 5., unit: str = 'px',
                   title: Optional[str] = None) -> None:
    """Plots the localization rate for a fixed threshold considering one of the
    n-best predictions (varying n).

    Args:
        true_pos: A list of true positions.
        pred_pos: A list of predictions positions, for each corresponding true
            position there should be more than 1. Essentially a 3d array.
        thresh_dist: Distance between true and prediction position when it is
            considered correct (<=).
        unit: The unit of the positions and the distance.
        title: An optional title to apply.
    """
    n = max(len(p) for p in pred_pos)

    rates = np.zeros((2, n))

    for tp, pps in zip(true_pos, pred_pos):
        localized = np.sqrt(np.sum((tp - pps)**2, axis=1)) <= thresh_dist
        rates[0, :len(localized)] += localized
        for i in range(n):
            rates[1, i] += localized[:i + 1].any()

    rates = rates / len(true_pos) * 100

    plt.xlim(0, n + 1)
    plt.ylim(0, 100)
    plt.xlabel('n-th local maximum')
    plt.ylabel('Localization rate / %')
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(np.arange(1, n + 1))
    plt.grid(axis='y')
    plt.bar(range(1, n + 1), rates[0], linewidth=0, width=0.6, color='b')
    plt.plot(range(1, n + 1), rates[1], 'r-')
    plt.plot(range(1, n + 1), rates[1], 'r.')

    if title is not None:
        plt.title(title)

    ax = plt.gca().twinx()
    ax.set_yticks(np.arange(0, 101, 10) / 100 * len(true_pos))
    ax.set_ylim(0, len(true_pos), emit=False)
    ax.set_ylabel('Number of localizations')


def save(path: Union[str, Path]) -> None:
    """Saves the current figure and closes it."""
    plt.savefig(str(path), dpi=150)
    plt.close()


def _landmarks_2d(ax, image, landmarks, spacing, colors, unit, draw_line: bool=True, border_colors=None):
    """Helper function to perform simple 2D rendering."""
    extent = (-.5 * spacing[0], (image.shape[1] + .5) * spacing[0],
              (image.shape[0] - .5) * spacing[1], -.5 * spacing[0])

    ax.imshow(image, cmap=plt.cm.gray if image.ndim == 2 else None,
              extent=extent, aspect=1)

    labels = {Landmark.TRUE_POS: 'True pos.', Landmark.PRED_POS: 'Pred. pos.',
              Landmark.CANDIDATES: 'Candidate pos.'}
    style = {Landmark.TRUE_POS: {'marker': '+', 'ms': 10, 'ls': 'None',
                                 'zorder': 1},
             Landmark.PRED_POS: {'marker': '*', 'ms': 3, 'ls': 'None',
                                 'zorder': 4},
             Landmark.CANDIDATES: {'marker': '.', 'ms': 1, 'ls': 'None',
                                   'zorder': 3}}

    for i, (name, landmark) in enumerate(landmarks.items()):
        for lm_type in Landmark:
            if lm_type in landmark:
                position = landmark[lm_type] * spacing
                if lm_type == Landmark.CANDIDATES:
                    position = np.transpose(position)
                if isinstance(colors[i], dict):
                    color = colors[i][lm_type]
                else:
                    color = colors[i]
                if border_colors is not None and isinstance(border_colors[i], dict) and lm_type in border_colors[i]:
                    border_color = border_colors[i][lm_type]
                    tmp_style = style[lm_type].copy()
                    tmp_style['ms'] *= 2
                    ax.plot(*position,
                            color=border_color,
                            **tmp_style)
                ax.plot(*position,
                        color=color,
                        **style[lm_type])

        if draw_line and Landmark.TRUE_POS in landmark and Landmark.PRED_POS in landmark:
            line = (np.array([landmark[Landmark.TRUE_POS],
                              landmark[Landmark.PRED_POS]]) * spacing).T
            ax.plot(*line, 'w-', zorder=2)

    legends = []

    for lm_type in Landmark:
        if any(lm_type in landmark for landmark in landmarks.values()):
            legends.append(ax.plot(0, 0, color='k', label=labels[lm_type],
                                   **style[lm_type]))

    ax.legend(loc='best', fontsize='xx-small', numpoints=1, markerscale=0.7)
    for legend in legends:
        legend[0].remove()

    if unit is not None:
        ax.set_xlabel(unit)
        ax.set_ylabel(unit)


def distinct_colors(count: int) -> List[Tuple]:
    """Generates `count` distinct colors to be used, e.g., in `matplotlib`
    renderings.
    """
    colors = [plt.cm.gist_rainbow(i / max(count - 1, 1))
              for i in range(count)]

    rng = np.random.RandomState(42)
    rng.shuffle(colors)

    return colors
