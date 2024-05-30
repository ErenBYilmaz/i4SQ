"""
rfl
~~~

An object localizer consisting of an ensemble of decision tree regressors,
each combined with a random feature extractor based on a BRIEF-like pattern.

:copyright: 2016-2018 by Alexander Oliver Mader.
:license: MIT, see LICENSE for more details.
"""


import gzip
import itertools
import logging
import math
import numpy as np
import pickle
import time
import warnings

from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from multiprocessing import Pool as ProcessPool
from multiprocessing.pool import ThreadPool
from operator import methodcaller
from pathlib import Path
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max
from sklearn.tree import DecisionTreeRegressor
from threading import Lock
from typing import IO, Iterable, List, Optional, Tuple, Union

from ._features import FeatureExtractor
from ._utils import atleast_3d_index, atleast_3d_size, \
    is_image, is_index, is_size

try:
    import pyopencl as cl
    from pyopencl import cltypes

    # silence cache hit/miss warnings
    logging.getLogger('pyopencl').setLevel(logging.WARNING)
    logging.getLogger('pytools.persistent_dict').setLevel(logging.WARNING)
except ImportError:
    pass


__title__ = 'rfl'
__version__ = '3.0'
__author__ = 'Alexander Oliver Mader'
__copyright__ = 'Copyright 2016-2018 Alexander Oliver Mader'
__license__ = 'MIT'

__all__ = (
    'RandomForestLocalizer',
    'create_gaussian_samples',
    'init_opencl'
    'FeatureExtractor',
    'ScoreRegressor',
    'OCL',
)


log = logging.getLogger(__name__)


#: An OpenCL compute context you can pass to several methods to accelerate
#: the performance. Use :func:`init_opencl` to create such a context.
OCL = namedtuple('OCL', ['ctx', 'queue', 'kernel', 'program', 'local_size'])


def create_gaussian_samples(origin: 'array-like', radius: 'array-like',
                            size: 'array-like', border_value: float=0.05) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Creates sample positions and scores assuming a Gaussian distribution.

    Places a Gaussian at the given `origin`, distributing the values within
    given `radius` around.
    Values are scaled such that 1 is placed at the `origin` and `border_value`
    at the border within the valid `origin +- radius` area, respecting the
    boundaries of the `arr`. Everything with a value below `border_value` (i.e.,
    in the corner of the `radius` box for instance) is ignored.

    Args:
        origin: The mean of the Gaussian (there a 1 is placed).
        radius: We extend the Gaussian from the `origin` within this radius,
            such that `border_value` is placed at the max `radius`.
        size: Ensure that the sampled positions are within proper bounds.
        border_value: The value used at the border to scale the Gaussian
            accordingly.

    Returns:
        `(positions, values)`, where both elements are arrays.
    """
    unit = np.ones(len(size))
    origin = unit * origin

    # ensure that we have at least one sample
    radius = np.maximum(1, unit * radius)

    cov = np.diag(-2 * np.log(border_value) / radius**2)

    ranges = [np.arange(max(0, math.floor(o - r)), min(math.ceil(o + r + 1), s))
              for o, r, s in zip(origin, radius, size)]

    xs, ys = [], []

    for position in itertools.product(*ranges):
        delta = origin - position
        value = np.exp(-0.5 * delta @ cov @ delta.T)

        if value < border_value:
            continue

        xs.append(position)
        ys.append(value)

    return (np.array(xs, dtype=np.int32), np.array(ys, np.float32))


def init_opencl() -> Optional[OCL]:
    """Tries to create a compute context using OpenCL."""
    try:
        import pyopencl as cl
    except ImportError:
        warnings.warn('Install PyOpenCL to make use of GPU acceleration')
        return None

    try:
        ctx = cl.create_some_context(interactive=False)
    except cl.Error as e:
        log.warn('Failed to create OpenCL context: %s', e)
        return None

    queue = cl.CommandQueue(ctx)

    local_size = ctx.devices[0].max_work_group_size

    source = (Path(__file__).parent / 'kernel.cl').read_text()
    prg = cl.Program(ctx, source).build()

    # cache the kernel because each access creates a new one
    kernel = prg.predict

    preferred_multiple = kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        ctx.devices[0])

    if local_size % preferred_multiple != 0:
        log.warn('The work group size %s is not a multiple '
                 'of %i, performance may be reduced', local_size,
                 preferred_multiple)

    log.info('Created OpenCL compute context: %s', ctx.devices)

    return OCL(ctx, queue, kernel, prg, local_size)


def load(path_or_fp: Union[str, IO]) -> 'RandomForestLocalizer':
    """Loads an :class:`RandomForestLocalizer` from the given file.

    :param path_or_fp: Either a path to a model or a file-like object
                       containing a model.
    :return: An instance of :class:`RandomForestLocalizer`.
    """
    with gzip.open(path_or_fp, 'rb') as fp:
        model = pickle.load(fp)

        if type(model) != RandomForestLocalizer:
            raise EnvironmentError("not a RandomForestLocalizer model")

        return model


class RandomForestLocalizer:
    """Main entry point for all operations using our RF localizer approach.

    It represents an ensemble of decision tree regressors,
    :class:`ScoreRegressor`, each using a
    :class:`FeatureExtractor` supplied with a random mask to generate a
    certain number of features per pixel. The regressors are trained using the
    features extracted from a Gaussian centered around the landmark. The
    trained trees are used to generate predictions and the bad ones are added
    as negative samples to boost the performance.

    During testing, the predictions over all trees are averaged and a
    Gaussian kernel is used to smooth the final prediction a bit.

    Args:
        n_trees: Number of Decision Tree Regressors to use.
        patch_size: Size of the local patch from where to extract the features,
            encoded as (W,H[,D]).
        n_features: Number of differences to compute inside the patch.
        frac_origin_features: The fraction of features that start from origin
            rather than a random starting point.
        n_dims: Image dimensionality, it is automatically derived from
            `patch_size`, if possible.
        n_channels: Whether we work with multi-channel images or not.
        patch_origin: If None, the origin is located at the center of the
            patch.
        max_depth: Maximum number of levels a tree might have.
        pre_smoothing: The sigma of the Gaussian kernel used to smooth each
            image prior to extracting features. Can be a list of sigmas, one
            for each axis.
        oob_value: The feature value used when an offset is pointing outside
            the image (out-of-bounds).
        n_negative_fraction: Amount of negative samples w.r.t. the amount
            of positive samples.
        batch_size: Amount of images to extract negative samples prior
            to re-training.
        bootstrapping_runs: How often an image is used to extract negative
            samples using a new tree in each run.
        seed: Used to initialize the RNG, use None for a random seed.
    """

    def __init__(self, n_trees: int=32, patch_size: Iterable[int]=(50, 50),
                 n_features: int=128, frac_origin_features: float=1.0,
                 n_dims: Optional[int]=None, n_channels: Optional[int]=None,
                 patch_origin: Optional[Iterable[int]]=None,
                 max_depth: Optional[int]=None,
                 pre_smoothing: float=2.15,
                 oob_value: float=1.0e10,
                 n_negative_fraction: float=1.0,
                 batch_size: int=10,
                 bootstrapping_runs: int=2,
                 seed=1337):
        assert n_trees > 0
        assert n_features > 0
        assert 0 <= frac_origin_features <= 1
        assert n_channels is None or n_channels > 0
        assert n_dims is None or 2 <= n_dims <= 3
        assert max_depth is None or max_depth > 0
        assert n_negative_fraction > 0
        assert batch_size > 0
        assert bootstrapping_runs > 0

        # we internally work with z,y,x instead of the x,y,z used as API
        patch_size = np.array(patch_size[::-1], dtype=np.int32)
        assert is_size(patch_size)

        patch_origin = patch_size // 2 if patch_origin is None else \
            np.array(patch_origin[::-1], dtype=np.int32)
        assert patch_origin is None or is_index(patch_origin, patch_size)

        if hasattr(pre_smoothing, '__iter__'):
            pre_smoothing = np.array(pre_smoothing)[::-1]
        assert np.greater_equal(pre_smoothing, 0).all()

        if n_dims is None:
            assert 2 <= len(patch_size) <= 3
            n_dims = len(patch_size)
        else:
            assert n_dims == len(patch_size)

        trees = []
        for i in range(n_trees):
            local_seed = None if seed is None else (i + 1) * seed

            feature_extractor = FeatureExtractor(
                atleast_3d_size(patch_size), atleast_3d_index(patch_origin),
                n_channels or 1, n_features, frac_origin_features, oob_value,
                np.random.RandomState(local_seed))

            patch_regressor = ScoreRegressor(
                '{}/{}'.format(i + 1, n_trees), feature_extractor,
                max_depth, np.random.RandomState(local_seed),
                len(patch_size),
                n_negative_fraction, batch_size, bootstrapping_runs)

            trees.append(patch_regressor)

        #: List of decision tree regressors in this ensemble.
        self.trees = trees

        #: Whether this ensemble operates on 2D or 3D images.
        self.n_dims = n_dims

        #: Number of channels in each image.
        self.n_channels = n_channels or 1

        #: The amount of sigma to apply each image prior to extracting
        #: features.
        self.pre_smoothing = pre_smoothing

    def train(self, images: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
              ocl: Optional[OCL]=None) -> None:
        """Train the RF localizer given a list of images with reference point
        annotations.

        :param images: A list of tuples `(image, x, y)`, one for each
                       training image, with positions (`x`) and
                       scores (`y`) used for training, in addition to the
                       automatically derived negative samples.
        :param ocl: Optional: An OpenCL compute context.
        """
        log.info('Going to train the RFL consisting of %i trees',
                 len(self.trees))

        def preprocess_sample(sample):
            image, xs, ys = sample

            image = self._prepare_image(image)
            xs = [atleast_3d_index(x) for x in xs]

            return (image, xs, ys)

        images = list(map(preprocess_sample, images))

        if ocl is None:
            with ProcessPool() as pool:
                self.trees = pool.map(methodcaller('train', images),
                                      self.trees)
        else:
            with ThreadPool() as pool:
                list(pool.map(methodcaller('train', images, ocl, Lock()),
                              self.trees))

        log.info('Finished training the RFL')

    def test(self, image: np.ndarray, ocl: Optional[OCL]=None) -> np.ndarray:
        """Applies the trained localizer to the given image.

        :param image: The image to test the ensemble on.
        :param ocl: Optional: An OpenCL compute context.
        :return: The predicted scores with the same size as the image.
        """
        log.info('Going to test RFL model in parallel on an '
                 'image of size %s', image.shape[::-1])

        image = self._prepare_image(image)

        start = time.time()

        if ocl is None:
            values = np.zeros(image.shape[:3], np.float32)

            with ProcessPool() as pool:
                for local_values in pool.imap(methodcaller('test', image),
                                              self.trees):
                    values += local_values
        else:
            values = np.empty(image.shape[:3], np.float32)

            image_cl = cl.Buffer(ocl.ctx, cl.mem_flags.READ_ONLY |
                                 cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
            values_cl = cl.Buffer(ocl.ctx, cl.mem_flags.WRITE_ONLY,
                                  values.nbytes)

            cl.enqueue_fill_buffer(ocl.queue, values_cl, b'\x00', 0,
                                   values.nbytes)

            # call the actual kernel for each tree
            kernel_events = [tree.test_ocl(ocl, image_cl, values_cl,
                                           image.shape, ocl.kernel)
                             for tree in self.trees]

            cl.enqueue_copy(ocl.queue, values, values_cl,
                            wait_for=kernel_events)

        stop = time.time()
        log.debug('Actual testing took %.3fs', stop - start)

        values /= len(self.trees)

        if self.n_dims == 2:
            values = values[0]

        return values

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Helper method to create a smoothed 3D image with self.n_channels."""
        assert np.less(0, image.shape).all()
        assert ((image.ndim == self.n_dims and self.n_channels == 1) or
                (image.ndim - 1 == self.n_dims and
                 image.shape[-1] == self.n_channels)), 'invalid image'

        image = image.astype(np.float32)

        new_shape = image.shape

        # missing channels axis
        if image.ndim == self.n_dims and self.n_channels == 1:
            new_shape = new_shape + (1,)

        assert 3 <= len(new_shape) <= 4

        # add z-axis
        if len(new_shape) == 3:
            new_shape = (1,) + new_shape

        if new_shape != image.shape:
            image = image.reshape(new_shape)

        new_image = np.empty_like(image)

        for c in range(image.shape[3]):
            new_image[:, :, :, c] = gaussian_filter(image[:, :, :, c],
                                                    self.pre_smoothing)

        return new_image

    def save(self, path_or_fp: Union[str, IO]) -> None:
        """Saves the current instance to a given file.

        :param path_or_fp: Either a path or a file-like object to store the
                           instance in.
        """
        with gzip.open(path_or_fp, mode='wb') as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_masks(self, fig: Figure=None, image: np.ndarray=None,
                   position: np.ndarray=None) -> Figure:
        """Plots the feature mask and the feature importance of all trees
        optionally overlayed on a reference image.

        **Beware**: Works only in the 2D case and if an image is given, you
        must also give a position.

        :param figure: Optional: A :class:`Figure` to plot to. If `None`, a
                       backend-less one is created and returned.
        :param image: Optional: A reference image.
        :param position: Optional: Where to overlay the mask in the image.
        :return: A :class:`Figure` one can save or plot.
        """
        assert self.n_dims == 2
        assert (image is None and position is None) or \
               (is_image(image, 2) and is_index(position, image.shape))

        if fig is None:
            fig = Figure()

        xy_tiles = np.ceil(np.sqrt(len(self.trees)))

        for i, tree in enumerate(self.trees):
            ax = fig.add_subplot(xy_tiles, xy_tiles, i + 1)

            patch_size = tree.feature_extractor.patch_size
            origin = tree.feature_extractor.origin

            if image is not None:
                start = position - origin[1:]
                end = start + patch_size[1:]

                ax.imshow(image[start[0]:end[0], start[1]:end[1]],
                          cmap=plt.cm.gray)
                ax.axis('off')

            ax.set_xlim(0, patch_size[2])
            ax.set_ylim(patch_size[1], 0)
            ax.set_aspect('equal')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            for j in range(len(tree.feature_extractor.offsets1)):
                position1 = (origin + tree.feature_extractor.offsets1[j])
                position2 = (origin + tree.feature_extractor.offsets2[j])

                vector = np.transpose([position1[2:0:-1], position2[2:0:-1]])

                ax.plot(*vector, '-', color='k', lw=0.2)
                ax.plot(*vector, '.', color='r', ms=0.5)

        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.set_size_inches(patch_size[2] / fig.dpi * xy_tiles,
                            patch_size[1] / fig.dpi * xy_tiles)

        return fig


class ScoreRegressor:
    """A decision tree regressor that uses a :class:`FeatureExtractor`
    to compute features and predict a score for them.
    """

    def __init__(self, index, feature_extractor, max_depth, random_state,
                 n_dims, n_negative_fraction, batch_size, bootstrapping_runs):
        self._id = index
        self._n_dims = n_dims
        self._max_depth = max_depth
        self._random_state = random_state
        self._n_negative_fraction = n_negative_fraction
        self._batch_size = batch_size
        self._bootstrapping_runs = bootstrapping_runs

        #: The :class:`StarFeatureExtractor` used for this tree.
        self.feature_extractor = feature_extractor

        self.children_left = None
        self.children_right = None
        self.split_features = None
        self.split_thresholds = None
        self.leaf_values = None

    def train(self, images: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
              ocl: Optional[OCL]=None,
              kernel_lock: Optional[Lock]=None) -> 'ScoreRegressor':
        """Trains the :class:`ScoreRegressor` given a list of training images.

        For easy multi-process parallalization, this method returns the
        :class:`ScoreRegressor` instance.

        :param images: A list of tuples `(image, x, y)`, one for each
                       training image, with positions `x` and
                       scores `y` used for training, in addition to the
                       automatically derived negative samples.
        :param ocl: Optional: An OpenCL compute context.
        :return: Itself.
        """
        tree_regressor = DecisionTreeRegressor(random_state=self._random_state,
                                               max_depth=self._max_depth)

        # this method is run in a threaded environment and the kernel
        # invocation is not thread safe, thus we must create a new instance
        if ocl is not None:
            kernel = ocl.program.predict
            assert kernel_lock is not None

        # we grow this list of features over time, only the first `n_samples`
        # entries are valid ones
        n_positive_features = sum(len(x) for image, y, x in images)
        n_negative_features = int(np.ceil(n_positive_features *
                                          self._n_negative_fraction))
        features = np.zeros((n_positive_features + n_negative_features,
                             self.feature_extractor.n_features),
                            dtype=np.float32)
        values = np.zeros(len(features), dtype=np.float32)

        n_samples = 0

        training_data = []

        # we start by adding all the given samples and training a first tree
        for image, xs, ys in images:
            # FIXME: good candidate for a sparse array
            image_shape = image.shape[:3]
            sampled = np.zeros(image_shape, dtype=bool)

            for x, y in zip(xs, ys):
                if (x < 0).any() or (x >= image.shape[:3]).any():
                    warnings.warn('Given training sample position {} is '
                                  'outside the valid shape {}'.format(
                                      x, image.shape[:3]))
                    continue

                values[n_samples] = y
                self.feature_extractor.extract(
                    image, x, out=features[n_samples, :])

                n_samples += 1

                sampled[tuple(x)] = True

            training_data.append((image_shape, image, sampled))

        def retrain_tree():
            log.debug('Regressor %s: Training decision tree using %i '
                      'samples', self._id, n_samples)

            tree_regressor.fit(features[:n_samples, :], values[:n_samples])

            tree = tree_regressor.tree_
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.split_features = tree.feature.astype(np.int32)
            self.split_thresholds = tree.threshold.astype(np.float32)
            self.leaf_values = tree.value[:, 0, 0].astype(np.float32)

        retrain_tree()

        # number of additional features to add per image per boostrapping run
        additional_features = int(n_negative_features / len(images) /
                                  self._bootstrapping_runs)

        for bootstrapping_run in range(self._bootstrapping_runs):
            for b in range(0, len(images), self._batch_size):
                log.debug('Regressor %s: Discriminative selection %i/%i on '
                          'images %i-%i', self._id, bootstrapping_run + 1,
                          self._bootstrapping_runs, b,
                          min(b + self._batch_size, len(images)))

                batch_images = training_data[b:b + self._batch_size]

                for image_shape, image_data, sampled in batch_images:
                    if ocl is None:
                        image_predictions = self.test(image_data)
                    else:
                        image_predictions = np.empty(image_shape, np.float32)

                        image_cl = cl.Buffer(ocl.ctx, cl.mem_flags.READ_ONLY |
                                             cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=image_data)
                        values_cl = cl.Buffer(ocl.ctx, cl.mem_flags.WRITE_ONLY,
                                              image_predictions.nbytes)

                        cl.enqueue_fill_buffer(ocl.queue, values_cl, b'\x00',
                                               0, image_predictions.nbytes)

                        kernel_event = self.test_ocl(ocl, image_cl, values_cl,
                                                     image_data.shape, kernel,
                                                     kernel_lock)

                        cl.enqueue_copy(ocl.queue, image_predictions,
                                        values_cl, wait_for=[kernel_event])

                    # use NMS to find only the important bad candidates
                    image_predictions[sampled] = 0

                    local_maxima = peak_local_max(image_predictions,
                                                  min_distance=5,
                                                  exclude_border=False,
                                                  indices=False)

                    image_predictions *= local_maxima
                    image_predictions *= np.logical_not(sampled)

                    remaining = additional_features

                    for index in reversed(np.argsort(image_predictions.flat)):
                        position = np.unravel_index(index, image_shape)

                        sampled[position] = True

                        self.feature_extractor.extract(
                            image_data, np.array(position, dtype=np.int32),
                            out=features[n_samples, :])

                        n_samples += 1
                        remaining -= 1

                        if not remaining:
                            break

                retrain_tree()

        log.debug('Regressor %s: Finished training.', self._id)

        self.tree_regressor = tree_regressor

        return self

    def test(self, image: np.ndarray) -> np.ndarray:
        """Applies the regressor to the features generated for all pixels
        in the given image.

        :param image: The image to compute the features for.
        :return: Regression values for the extracted features.
        """
        return self.feature_extractor.predict(
            image, self.children_left, self.children_right,
            self.split_features, self.split_thresholds, self.leaf_values)

    def test_ocl(self, ocl: OCL, image_cl: 'cl.Buffer', out_cl: 'cl.Buffer',
                 image_size: Tuple[int, ...], kernel: 'cl.Kernel',
                 kernel_lock: Optional[Lock]=None):
        """The OpenCL version of the normal :meth:`test` method."""
        def create_buffer(array):
            return cl.Buffer(ocl.ctx, cl.mem_flags.READ_ONLY |
                             cl.mem_flags.COPY_HOST_PTR, hostbuf=array)

        image_size_cl = cltypes.make_int4(*image_size)
        offsets1_cl = create_buffer(self.feature_extractor.offsets1)
        offsets2_cl = create_buffer(self.feature_extractor.offsets2)
        children_left_cl = create_buffer(self.children_left)
        children_right_cl = create_buffer(self.children_right)
        split_features_cl = create_buffer(self.split_features)
        split_thresholds_cl = create_buffer(self.split_thresholds)
        leaf_values_cl = create_buffer(self.leaf_values)

        global_size = int(np.ceil(np.prod(image_size[:3]) / ocl.local_size) *
                          ocl.local_size)

        try:
            if kernel_lock is not None:
                kernel_lock.acquire()

            return kernel(ocl.queue, (global_size,), (ocl.local_size,),
                          image_cl, image_size_cl, offsets1_cl, offsets2_cl,
                          children_left_cl, children_right_cl,
                          split_features_cl, split_thresholds_cl,
                          leaf_values_cl, self.feature_extractor.oob_value,
                          out_cl)
        finally:
            if kernel_lock is not None:
                kernel_lock.release()
