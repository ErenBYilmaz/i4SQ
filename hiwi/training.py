import math
import numpy as np
import itertools
import time
import warnings

from numpy.random import RandomState
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from typing import Any, Generator, List, Optional, Tuple, Union

from .utils import guess_image_shape


def batchize(data: List[Any], batch_size: int,
             max_epochs: Optional[int] = None,
             max_iterations: Optional[int] = None, shuffle: bool = True,
             random_state: Union[None, int, RandomState] = 42) \
        -> Generator[Tuple[int, List[Any]], None, None]:
    """Creates a `generator` that yields batches of the given `data` which
    one can use to perform e.g. stochastic gradient descent training.

    It's ensured that a sample is contained only once in a set of batches that
    belong to the same epoch as long as `len(data)` is evenly dividable
    by `batch_size`. If it is not, it is ensured the samples have an even
    frequncy with increasing number of epochs.

    Args:
        data: An indexable vector of data to create batches from.
        batch_size: The size of the batches. If it's larger than the number of
            items contained in `data`, it's set to the length of `data`.
        max_epochs: Number of maximal epochs, i.e., number of passes through
            the whole dataset.
        max_iterations: Number of maximal batches to generate.
        shuffle: Whether to shuffle the data after each epoch.
        random_state: `RandomState` used to perform the shuffling, if `None`,
            a random seed is used.

    Returns:
        A `generator` that yields a tuple `(epoch, batch)` containing the
        current epoch (i.e., number of passes through the whole `data`) and
        the `batch` itself.
    """
    assert batch_size > 0
    assert max_epochs is None or max_epochs > 0
    assert max_iterations is None or max_iterations > 0

    if batch_size > len(data):
        warnings.warn('Requested batch_size={} is larger than data length {},'
                      'falling back to the data length'.format(batch_size,
                                                               len(data)))
        batch_size = len(data)

    if random_state is None or type(random_state) == int:
        random_state = RandomState(random_state)

    total_size = len(data)
    indices = np.arange(total_size)

    processed = 0
    iteration = 0
    start = 0

    while True:
        if start == 0 and shuffle:
            random_state.shuffle(indices)

        end = start + batch_size

        if end <= total_size:
            selected = indices[start:end]
            start = 0 if end == total_size else end
        else:
            if shuffle:
                random_state.shuffle(indices[:start])

            end = end - total_size
            selected = np.hstack((indices[start:], indices[:end]))
            start = end

            if shuffle:
                random_state.shuffle(indices[start:])

        assert len(selected) == batch_size

        epoch = processed // len(data)
        batch = [data[i] for i in selected]

        yield epoch, batch

        processed += batch_size
        iteration += 1

        if max_epochs is not None and processed // len(data) >= max_epochs:
            return

        if max_iterations is not None and iteration >= max_iterations:
            return


def place_gaussian(arr: np.ndarray, origin: np.ndarray,
                   radius: np.ndarray, scale: float = 1.,
                   border_value: float = 0.05,
                   fix_peak: bool = False) -> int:
    """Places a Gaussian at the given `origin`, distributing the values within
    given `radius` around.

    Values are scaled such that 1 is placed at the `origin` and `border_value`
    at the border within the valid `origin +- radius` area, respecting the
    boundaries of the `arr`. Everything with a value below `border_value`
    (i.e., in the corner of the `radius` box for instance) is ignored.

    Args:
        arr: The array where we want to place the Gaussian.
        origin: The mean of the Gaussian (there a 1 is placed).
        radius: We extend the Gaussian from the `origin` within this radius,
            such that a 0.05 is placed at the max `radius`.
        scale: A scaling factor to apply to the values.
        border_value: The value used at the border to scale the Gaussian
            accordingly.
        fix_peak: If the origin is not an integer, the pixel value for the
            corresponding index might be lass than `scale`. To prevent this,
            set this to `True`.

    Returns:
        The number of touched pixels/voxels.
    """
    unit = np.ones(arr.ndim)
    origin = unit * origin
    radius = unit * radius

    cov = np.diag(-2 * np.log(border_value) / radius**2)

    ranges = [np.arange(max(0, math.floor(o - r)),
                        min(math.ceil(o + r + 1), s))
              for o, r, s in zip(origin, radius, arr.shape)]

    touched = 0

    for position in itertools.product(*ranges):
        delta = origin - position
        value = np.exp(-0.5 * delta @ cov @ delta.T)

        if value < border_value:
            continue

        arr[position] = value * scale
        touched += 1

    if fix_peak:
        origin_idx = np.round(origin).astype(int)
        if (origin_idx >= 0).all() and (origin_idx < arr.shape).all():
            arr[tuple(origin_idx)] = scale

    return touched


def transform_elastically(image: np.ndarray, alpha: float, sigma: float,
                          order: int = 3, multiple: Optional[bool] = None,
                          positions: Optional[np.ndarray] = None,
                          seed: Optional[int] = 42):
    """Elastically transforms multiple or single images.

    This method is described in "Best Practices for Convolutional Neural
    Networks Applied to Visual Document Analysis" by Simard, Steinkraus and
    Platt. It has been adapted to support (multi channel) 2D and 3D images.

    Args:
        image: A single image or a list of images.
        alpha: Scaling factor to control the intensity of the deformations.
        sigma: Smoothing of the random field, very large numbers yield to no
            random field, while very low numbers yield a very random field.
        order: Spline order for interpolation, use 0 when transforming e.g.
            label maps.
        multiple: Whether multiple images are supplied, if `None`, it is
            assumed that multiple images are supplied if `image` is of type
            `list` or `tuple`.
        positions: An optional set of positions to transform as well (W.r.t.
            the deformation field). If this is given, the function returns a
            tuple containing the image and the transformed positions,
            respectively.
        seed: Used to initialize the PRNG, if `None`, the current time is used
            as seed value.

    Examples:
        ```python
        img2 = transform_elastically(img,
                                     alpha=np.array(x.shape) * 3,
                                     sigma=np.array(x.shape) * 0.07)
        ```

    References:
        - [Simard2003](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Simard.pdf) # noqa
        - [GitHub](https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a) #noqa
        - [Kaggle](https://www.kaggle.com/pscion/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation-0878921a) #noqa
    """
    if multiple is None:
        multiple = isinstance(image, (list, tuple))

    if seed is None:
        seed = int(time.time())

    images = image if multiple else [image]
    rng = np.random.RandomState(seed)
    n_dims, _ = guess_image_shape(images[0])
    shape = images[0].shape[:n_dims]

    alpha = np.ones(n_dims) * alpha
    sigma = np.ones(n_dims) * sigma

    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')

    coordinates = []
    for grid, a, s in zip(grids, alpha, sigma):
        displacement = rng.rand(*shape) * 2 - 1
        delta = gaussian_filter(displacement, s) * a
        coordinates.append(np.reshape(grid + delta, (-1,)))

    # convert optionally supplied positions
    trans_positions = None
    if positions is not None:
        trans_positions = np.empty(positions.shape)
        coords = np.array(coordinates).T

        for pos_idx, pos in enumerate(positions):
            dists = np.sum(np.square(coords - pos), axis=1)
            threshold = np.maximum(dists.min(), 2.)

            matching = dists <= threshold
            indices = np.where(matching)[0]
            weights = np.sqrt(threshold) * 1.5 - np.sqrt(dists[matching])
            weights /= weights.sum()

            # use the closest points and apply the distances as weights
            new_pos = np.transpose(np.unravel_index(indices, shape))
            new_pos = new_pos * weights[:, np.newaxis]
            new_pos = np.sum(new_pos, axis=0)

            trans_positions[pos_idx] = new_pos

    def tx(i, o):
        return map_coordinates(i, coordinates, output=o.reshape(o.size),
                               order=order, mode='nearest')

    trans_images = []

    for image in images:
        trans_image = np.empty_like(image)

        n_channels = None if image.ndim == n_dims else image.shape[-1]

        if n_channels is None:
            tx(image, trans_image)
        else:
            for c in range(n_channels):
                sel = (slice(None),) * n_dims + (c,)
                tx(image[sel], trans_image[sel])

        trans_images.append(trans_image)

    if not multiple:
        trans_images = trans_images[0]

    if trans_positions is not None:
        return trans_images, trans_positions

    return trans_images


class PatchExtractor:
    """Provides means to extract patches from arrays and reconstructing arrays
    from patches.
    """

    def __init__(self, patch_size: np.ndarray, strategy: str = 'pad',
                 stride: Optional[np.ndarray] = None) -> None:
        """Sets the `PatchExtractor` up.

        Args:
            patch_size: Size of the patches.
            strategy: How to handle cases where the array size doesn't match
                the stride and patch_size combination. 'pad' pads the input
                array with edge values. With 'overlap' the last two patches
                will be overlapped.
            stride: The step width. If `None`, the `patch_size` is used.
        """
        assert strategy in ('pad', 'overlap')

        #: Size of the patches.
        self.patch_size = np.asarray(patch_size, dtype=int)

        #: Spacing between patches to extract.
        self.stride = self.patch_size if stride is None else \
            np.asarray(stride, dtype=int)

        assert (self.patch_size > 0).all()
        assert (self.stride > 0).all()
        assert self.patch_size.size == self.stride.size

        self.strategy = strategy

    def extract(self, arr: np.ndarray, ret_origin: bool = False) \
            -> List[np.ndarray]:
        """Extracts the patches from the given array."""
        assert (arr.ndim == self.patch_size.size
                or arr.ndim == self.patch_size.size + 1)

        arr, slices = self._prepare(arr=arr)

        patches = []

        for slice_ in slices:
            patches.append(([s.start for s in slice_], arr[slice_])
                           if ret_origin else arr[slice_])

        return patches

    def reconstruct(self, shape: np.ndarray, patches: np.ndarray,
                    fill_value: Union[int, float] = 0.) -> np.ndarray:
        """Reconstructs the original array given a list of patches.

        Multiple written areas (overlaps) are averaged.

        Args:
            shape: The size of the array to reconstruct.
            patches: The patches to reconstruct the array from.
            fill_value: Not reconstructed pixels are filled with this value.

        Returns:
            The newly reconstructed array.
        """
        dtype = type(patches[0][0][0][0] if len(shape) == 3 else
                     patches[0][0][0])
        arr, slices = self._prepare(shape=shape, dtype=dtype)

        overlap = (self.stride < self.patch_size).any() or \
            self.strategy == 'overlap'
        space = (self.stride > self.patch_size).any()

        if overlap:
            norm = np.zeros_like(arr)
        if space:
            trace = np.ones(arr.shape, dtype=bool)

        for slice_, patch in zip(slices, patches):
            if overlap:
                arr[slice_] += patch
                norm[slice_] += 1
            else:
                arr[slice_] = patch

            if space:
                trace[slice_] = False

        if space:
            arr[trace] = fill_value

        if overlap:
            arr = arr / np.maximum(norm, 1)
            if np.issubdtype(dtype, np.floating):
                arr = arr.astype(dtype)

        return arr[tuple([slice(s) for s in shape])]

    def _prepare(self, shape=None, arr=None, dtype=np.float64):
        if arr is not None:
            shape = arr.shape[:self.patch_size.size]

        n_patches = np.maximum(np.ceil((shape - self.patch_size)
                                       / self.stride), 0).astype(int) + 1

        if self.strategy == 'pad':
            new_shape = (n_patches - 1) * self.stride + self.patch_size
        elif self.strategy == 'overlap':
            new_shape = np.maximum(shape, self.patch_size)

        if arr is not None:
            padding = [(0, ns - os) for os, ns in zip(shape, new_shape)]
            if len(padding) == arr.ndim - 1:
                padding.append((0, 0))
            arr = np.pad(arr, padding, 'edge')
        elif (self.stride < self.patch_size).any() or \
                self.strategy == 'overlap':
            arr = np.full(new_shape, 0., dtype=np.float64)
        else:
            arr = np.empty(new_shape, dtype=dtype)

        slices = []

        for patch_idx in itertools.product(*[np.arange(n) for n in n_patches]):
            one_patch_slices = []
            for i, p, s, shape in zip(patch_idx, self.patch_size, self.stride,
                                      new_shape):
                start = min(i * s, shape - p)
                one_patch_slices.append(slice(start, start + p))

            slices.append(tuple(s for s in one_patch_slices))

        return arr, slices
