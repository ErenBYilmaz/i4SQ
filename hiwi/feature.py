import itertools
import numpy as np
import sys

from typing import List, Optional, Tuple, Union


class LocalMaxLocator:
    """State-ful `find_local_max` with refinement support.

    Args:
        max_peaks: Number of local maxima to extract.
        min_distance: The minimal distance between peaks.
        use_mm: Wheather min_distance is assumed to be in mm or pixel.
        refine: Whether to perform polynomial interpolation to refine the
            peak positions.
    """

    def __init__(self, max_peaks: int = 15,
                 min_distance: Union[float, np.ndarray] = 10.,
                 use_mm: bool = False, refine: bool = False) -> None:
        assert max_peaks > 0

        self.max_peaks = max_peaks
        """Number of local maxima to locate."""
        self.min_distance = min_distance
        """Minimal distance between peaks."""
        self.use_mm = use_mm
        """Whether to treat min_distance as mm or pixel."""
        self.refine = refine
        """Whether to refine the found maxima."""

    def locate(self, arr: np.ndarray, spacing: Optional[np.ndarray] = None,
               downsampled: Optional[np.ndarray] = None,
               target_shape: Optional[np.ndarray] = None) -> np.ndarray:
        """Locates local maxima.

        Also converts the coordinates back to the original, not downsampled
        space, if necessary.

        Args:
            arr: The array to look for maxima in.
            spacing: An optional spacing for this heatmap.
            downsampled: An optional downsampling factor of the heatmap.
            target_shape: Optional shape of the target image (before an
                optional downsampling happened), to limit the upsampled
                positions.

        Return: A 2D array containing the positions of the found local maxima.
        """
        assert not self.use_mm or spacing is not None, \
            'spacing must be given since min_distance is assumed to be mm'

        if downsampled is None:
            downsampled = np.ones(arr.ndim)

        min_distance = self.min_distance
        if self.use_mm:
            min_distance = min_distance / spacing

        min_distance = min_distance / downsampled
        min_distance = np.ceil(min_distance).astype(int)

        idx = find_local_max(arr, self.max_peaks, min_distance)
        idx.shape = (len(idx), arr.ndim)

        pos = idx.astype(float)

        if self.refine:
            for i in range(len(idx)):
                new_p = refine_local_peak(arr, idx[i])
                if new_p is not None:
                    pos[i] = new_p

        pos = (pos + 0.5) * downsampled - 0.5

        if target_shape is not None:
            min_pos = [-0.5] * arr.ndim
            max_pos = np.asarray(target_shape) - .5
            pos = np.minimum(np.maximum(pos, min_pos), max_pos)

        return pos


def find_local_max(arr: np.ndarray, max_peaks: int = sys.maxsize,
                   min_distance: Union[int, np.ndarray] = 1,
                   ret_vals: bool = False, lazy: bool = False):
    """Uses non-maximum suppression to find local maxima.

    Args:
        arr: The array to look for maxima in.
        max_peaks: Maximum number of peaks to extract.
        min_distance: Minimum distance between peaks in pixel.
        ret_vals: Whether to return the values in addition to the positions.
        lazy: Whether to return an iterator that lazily looks for local maxima.

    Returns:
        Either just the positions in a 2D array or a tuple consisting of
        the positions and the corresponding values if `ret_vals` is `True`.
    """
    assert max_peaks > 0
    assert (np.array(min_distance) >= 0).all()

    min_val = arr.min()
    arr_mut = np.copy(arr)
    min_distance = np.ones(arr.ndim, dtype=int) * min_distance

    def generator():
        for i in range(max_peaks):
            pos = np.unravel_index(np.argmax(arr_mut), arr.shape)
            value = arr_mut[tuple(pos)]

            if value == min_val:
                return

            yield (pos, value) if ret_vals else pos

            block = tuple([slice(max(p - m, 0), min(p + m + 1, s))
                           for p, m, s in zip(pos, min_distance, arr.shape)])
            arr_mut[block] = min_val

    if lazy:
        return generator()

    indices = []
    values = []

    for ret in generator():
        if ret_vals:
            values.append(ret[1])
            ret = ret[0]

        indices.append(ret)

    ret = np.array(indices)

    if ret_vals:
        ret = ret, np.array(values)

    return ret


def find_local_max_recursively(arr: np.ndarray,
                               levels: List[Tuple[int, ]],
                               ret_vals: bool = False):
    """Uses non-maximum suppression recursively to find local maxima.

    For example, this allows to coarsely find strong local maxima and then
    look for close neighboring maxima around the first strong local maxima.

    Args:
        arr: The array to look for maxima in.
        levels: A description of the search levels in the form of a list of
            `(max_peaks, min_distance, (optional) cutout_radius)`. The first
            two values are the same as used in `find_local_max`. The optional
            third one defines the radious around the found maximum to consider
            when looking for new local maxima in the next level. If it is not
            given, the `min_distance` is used as `cutout_radius`.
        ret_vals: Whether to return the values in addition to the positions.

    Returns:
        Either just the positions in a 2D array or a tuple consisting of
        the positions and the corresponding values if `ret_vals` is `True`.

    Examples:
        First finding 5 local maxima with a minimal distance of 25 and
        then, for each of those 5 local maxima, finding 10 local maxima with a
        minimal distance of 1,2,3 inside a radius of [30, 25, 40] around the
        original local maximum.

        ```python
        find_local_max_recursively(arr, [(5,          25, [30, 25, 40]),
                                         (10,  [1, 2, 3]  )])
        ```
    """
    def indexify(ix):
        return (np.ones(len(arr.shape), dtype=np.int64) * ix).astype(np.int64)

    positions = []
    min_value = arr.min()
    tracing_arr = np.copy(arr)

    def apply_level(origin, sub_arr, level_idx):
        level = levels[level_idx]

        max_peaks = level[0]
        min_distance = indexify(level[1])
        cutout_radius = indexify(level[1] if len(level) != 3 else level[2])

        for position in find_local_max(sub_arr, max_peaks, min_distance,
                                       lazy=True):
            position = origin + position
            index = tuple(position)

            positions.append(position)

            tracing_arr[index] = min_value

            if level_idx < len(levels) - 1:
                start = np.maximum(position - cutout_radius, 0)
                end = np.minimum(position + cutout_radius, tracing_arr.shape)
                region = tuple(slice(*x) for x in zip(start, end))

                sub_arr = tracing_arr[region]

                apply_level(start, sub_arr, level_idx + 1)

    apply_level(indexify(0), tracing_arr, 0)

    ret = np.array(positions)

    if ret_vals:
        ret = (ret, arr[tuple(ret.T)])

    return ret


def refine_local_peak(space: np.ndarray, position, neighborhood: int = 1):
    """Fits a second degree polynomial using key points around the given
    `position` in order to find the maximum of the polynomial, which is used
    to refine the given position.

    In case of a 3D volume, the refinement is calculated by fitting a 3d
    polynomial and performing a grid search instead of using a closed form
    solution as in the 2D case.

    :param space: The image/volume the position is in.
    :param position: The position in the `space` we want to refine given its
                     surrounding.
    :param neighborhood: The step size in both directions of each axis to draw
                         samples from.
    :return: Either a new refined position or `None` in case no maximum could
             be found.
    """
    assert space.ndim in (2, 3)

    position = np.asarray(position)

    if (position < 0).any() or (position >= space.shape).any():
        return None

    if space.ndim == 2:
        return _refine_local_peak_2d(space, position, neighborhood)
    elif space.ndim == 3:
        return _refine_local_peak_3d(space, position, neighborhood)


def _refine_local_peak_2d(space: np.ndarray, position: np.ndarray,
                          neighborhood: int):
    offsets = np.arange(-neighborhood, neighborhood + 1)

    X = []
    f = []

    for d in itertools.product(offsets, offsets):
        p = position + d
        if (p < 0).any() or (p >= space.shape).any():
            continue

        v = space[tuple(p)]
        if not np.isfinite(v):
            continue

        x, y = d
        X.append([x*x, y*y, x*y, x, y, 1])
        f.append(space[tuple(p)])

    # we might be underspecified at the borders
    if len(f) < 6:
        return None

    X = np.array(X, dtype=float)
    f = np.array(f)

    # estimate the polynomial coefficients
    try:
        a = np.linalg.inv(X.T @ X) @ X.T @ f
    except np.linalg.LinAlgError:
        return None

    base = 4 * a[0] * a[1] - a[2]*a[2]

    delta = [-(2 * a[1] * a[3] - a[2] * a[4]) / base,
             -(2 * a[0] * a[4] - a[2] * a[3]) / base]
    new_pos = position + delta

    # f = lambda x, y: a[0] * x ** 2 + a[1] * y ** 2 + a[2] * x * y
    # + a[3] * x + a[4] * y + a[5]
    fxx = 2 * a[0]
    fyy = 2 * a[1]
    fxy = a[4]
    det_hesse = fxx * fyy - fxy * fxy

    minimum = fxx > 0 and det_hesse > 0

    if det_hesse == 0 or minimum or (np.abs(delta) >= neighborhood).any():
        return None

    return new_pos


# FIXME: Implement generalized version using H
# https://en.wikipedia.org/wiki/Second_partial_derivative_test


def _refine_local_peak_3d(space: np.ndarray, position: np.ndarray,
                          neighborhood: int):
    offsets = list(range(-neighborhood, neighborhood + 1))

    # P(x,y,z) = a0 + a1*x + a2*y + a3*z + a4*x*y + a5*x*z + a6*y*z + a7*x^2
    # + a8*y^2 + a9*z^2

    X = []
    f = []

    for d in itertools.product(offsets, offsets, offsets):
        p = position + d

        if (p < 0).any() or (p >= space.shape).any():
            continue

        x, y, z = d

        X.append([1, x, y, z, x * y, x * z, y * z, x * x, y * y, z * z])
        f.append(space[tuple(p)])

    # we might be underspecified at the borders
    if len(f) < 10:
        return None

    X = np.array(X, dtype=float)
    f = np.array(f, dtype=float)

    # estimate the polynomial coefficients
    try:
        a = np.linalg.inv(X.T @ X) @ X.T @ f
    except np.linalg.LinAlgError:
        return None

    base = 2 * (a[4] * a[4] * a[9] - a[4] * a[5] * a[6] + a[5] * a[5] * a[8]
                + a[6] * a[6] * a[7] - 4 * a[7] * a[8] * a[9])

    x = a[1] * a[6] * a[6] - 4 * a[1] * a[8] * a[9] + 2 * a[2] * a[4] * a[9] \
        - a[2] * a[5] * a[6] - a[3] * a[4] * a[6] + 2 * a[3] * a[5] * a[8]
    y = 2 * a[1] * a[4] * a[9] - a[1] * a[5] * a[6] + a[2] * a[5] * a[5] \
        - 4 * a[2] * a[7] * a[9] - a[3] * a[4] * a[5] + 2 * a[3] * a[6] * a[7]
    z = -a[1] * a[4] * a[6] + 2 * a[1] * a[5] * a[8] - a[2] * a[4] * a[5] \
        + 2 * a[2] * a[6] * a[7] + a[3] * a[4] * a[4] - 4 * a[3] * a[7] * a[8]

    x = -x / base
    y = -y / base
    z = -z / base

    h1 = 2 * a[7]
    h2 = 4 * a[7] * a[8] - a[4] * a[4]
    h3 = 8 * a[7] * a[8] * a[9] + 2 * a[4] * a[5] * a[6] - 2 * a[5] * a[5] \
        * a[8] - 2 * a[4] * a[4] * a[9] - 2 * a[6] * a[6] * a[7]

    delta = np.array([x, y, z])

    if h1 >= 0 or h2 <= 0 or h3 >= 0 or (np.abs(delta) >= neighborhood).any():
        return None

    # def value(x, y, z):
    #     return a[0] + a[1] * x + a[2] * y + a[3] * z + a[4] * x * y \
    #         + a[5] * x * z + a[6] * y * z + a[7] * x * x + a[8] * y * y \
    #         + a[9] * z * z

    return position + delta
