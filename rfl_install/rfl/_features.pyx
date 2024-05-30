# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np

cimport cython
cimport numpy as np

from sklearn.tree import DecisionTreeRegressor
from typing import Optional

from ._utils import is_image, is_size, is_index


class FeatureExtractor:
    """Patch-based feature extractor computing intensity difference between
    two pixels.

    **Beware:** This operates on 3D data only for simplicity, so make sure to
    transform your 2D case into a 3D one.

    Args:
        patch_size: Size of the patch to sample the offsets from.
        patch_origin: Position of the origin inside the patch.
        n_channels: Number of channels to support, mostly it's just 1.
        n_offset_paits: Number of offsets to use to compute the features.
        frac_origin_pairs: Fraction of features to compute against the origin value.
        oob_value: Feature value used in case one or both positions are outside the field
            of view.
        random_state: A random state to use for sampling.
    """

    def __init__(self, patch_size: np.ndarray, patch_origin: np.ndarray, n_channels: int,
                 n_pairs: int, frac_origin_pairs: float,
                 oob_value: float, random_state: np.random.RandomState):
        assert is_size(patch_size, 3)
        assert is_index(patch_origin, patch_size)
        assert n_channels > 0
        assert n_pairs > 0 and n_pairs % 2 == 0
        assert n_pairs + 1 <= np.prod(patch_size)
        assert 0 <= frac_origin_pairs <= 1

        sampling_patch_size = np.maximum(patch_origin,
                                         patch_size - patch_origin) * 2

        def sample_position():
            while True:
                position = random_state.normal(patch_origin,
                                               sampling_patch_size / 5.).round()

                # ensure that the position is inside the patch
                if np.any(position < 0) or np.any(position >= patch_size):
                    continue

                return position

        n_origin_pairs = round(frac_origin_pairs * n_pairs)

        n_sampled_pairs = 0
        offsets1 = np.empty((n_pairs, 3), np.int32)
        offsets2 = np.empty((n_pairs, 3), np.int32)

        # we first generate the center-based features
        while n_sampled_pairs < n_origin_pairs:
            position = sample_position()

            # don't use the origin, since this is our reference point and
            # provides no information
            if np.array_equal(patch_origin, position):
                continue

            offsets1[n_sampled_pairs] = patch_origin
            offsets2[n_sampled_pairs] = position
            n_sampled_pairs += 1

        # and than we generate the random features
        while n_sampled_pairs < n_pairs:
            offsets1[n_sampled_pairs] = sample_position()
            offsets2[n_sampled_pairs] = sample_position()
            n_sampled_pairs += 1

        def add_channels(offsets):
            offsets = offsets - patch_origin
            offsets = np.tile(offsets, (n_channels, 1))
            channels = np.repeat(np.arange(n_channels), n_pairs).reshape(offsets.shape[0], 1)
            return np.hstack((offsets, channels)).astype(np.int32)

        #: Vectors from the origin to the first pixel.
        self.offsets1 = add_channels(offsets1)

        #: Vectors from the origin to the second pixel.
        self.offsets2 = add_channels(offsets2)

        #: Number of feature components.
        self.n_features = n_pairs * n_channels

        #: The feature used when an offses is out-of-bounds.
        self.oob_value = np.float32(oob_value)

        #: Position of the origin inside the mask.
        self.origin = patch_origin

        #: Size of the used patch to generate the offsets.
        self.patch_size = patch_size

        assert self.offsets1.flags.c_contiguous
        assert self.offsets2.flags.c_contiguous
        assert self.origin.flags.c_contiguous
        assert self.patch_size.flags.c_contiguous

    def extract(self, image: np.ndarray, position: np.ndarray,
                out: Optional[np.ndarray]=None):
        """Extracts a feature vector at the given position in the given
        image.

        :param image: The image to extract the feature vector in.
        :param position: Where to place the origin of the mask.
        :param out: Optional: A pre-allocated array to store the features in.
        :return: The feature vector.
        """
        assert is_image(image, 4)
        assert image.flags.c_contiguous
        # FIXME: Apparently Cython doesn't support shape when image is typed
        #assert is_index(position, image.shape)
        assert out is None or len(out) == self.n_features

        if out is None:
            out = np.empty(self.n_features, dtype=np.float32)

        position = position.astype(np.int32)

        assert position.flags.c_contiguous
        assert out.flags.c_contiguous

        extract_features(image, position, self.offsets1, self.offsets2,
                         self.oob_value, out)

        return out

    def predict(self, image: np.ndarray,
                children_left: np.ndarray,
                children_right: np.ndarray,
                split_features: np.ndarray,
                split_thresholds: np.ndarray,
                leaf_values: np.ndarray,
                out: np.ndarray=None):
        """Predicts the regression values for the features extracted for each
        pixel in the supplied image.

        :param image: An image for which to generate the predictions.
        :param out: Optional: A pre-allocated array to store the results in.
        :return: The array containing the predictions.
        """
        assert is_image(image, 4)
        assert image.flags.c_contiguous
        assert children_left.flags.c_contiguous
        assert children_right.flags.c_contiguous
        assert split_features.flags.c_contiguous
        assert split_thresholds.flags.c_contiguous
        assert leaf_values.flags.c_contiguous
        #assert out is None or image.shape[:3] == out.shape

        if out is None:
            out = np.empty((image.shape[0], image.shape[1], image.shape[2]), np.float32)

        assert out.flags.c_contiguous

        predict(image, self.offsets1, self.offsets2, children_left,
                children_right, split_features, split_thresholds, leaf_values,
                self.oob_value, out)

        return out


# extracts all feature values for a given position
cdef inline extract_features(np.float32_t[:, :, :, ::1] image,
                             np.int32_t[::1] position,
                             np.int32_t[:, ::1] offsets1,
                             np.int32_t[:, ::1] offsets2,
                             np.float32_t oob_value,
                             np.float32_t[::1] out):
    cdef np.int32_t z = position[0]
    cdef np.int32_t y = position[1]
    cdef np.int32_t x = position[2]

    cdef np.int32_t i

    for i in range(offsets1.shape[0]):
        out[i] = extract_feature(image, z, y, x, offsets1, offsets2, i,
                                 oob_value)


# an optimized tree evaluation to compute only the necessary feature
# values at each node
cdef inline predict(np.float32_t[:, :, :, ::1] image,
                    np.int32_t[:, ::1] offsets1,
                    np.int32_t[:, ::1] offsets2,
                    np.int32_t[::1] children_left,
                    np.int32_t[::1] children_right,
                    np.int32_t[::1] split_features,
                    np.float32_t[::1] split_thresholds,
                    np.float32_t[::1] leaf_values,
                    np.float32_t oob_value,
                    np.float32_t[:, :, ::1] out):
    # define types for _ALL_ running variables
    cdef np.int32_t z, y, x, node, feature, z1, y1, x1
    cdef np.float32_t value, value1

    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                # iterate all nodes starting from the root
                node = 0

                while children_left[node] != -1:
                    feature = split_features[node]

                    value = extract_feature(image, z, y, x, offsets1, offsets2,
                                            feature, oob_value)

                    if value <= split_thresholds[node]:
                        node = children_left[node]
                    else:
                        node = children_right[node]

                out[z, y, x] = leaf_values[node]


# helper method to perfom the intensity difference computation given the
# reference point position and value, as well as the offsets
cdef inline extract_feature(np.float32_t[:, :, :, ::1] image,
                            np.int32_t z, np.int32_t y, np.int32_t x,
                            np.int32_t[:, ::1] offsets1,
                            np.int32_t[:, ::1] offsets2,
                            np.int32_t feature,
                            np.float32_t oob_value):
    cdef np.int32_t x1 = x + offsets1[feature, 2]
    if 0 > x1 or x1 >= image.shape[2]: return oob_value
    cdef np.int32_t y1 = y + offsets1[feature, 1]
    if 0 > y1 or y1 >= image.shape[1]: return oob_value
    cdef np.int32_t z1 = z + offsets1[feature, 0]
    if 0 > z1 or z1 >= image.shape[0]: return oob_value

    cdef np.int32_t x2 = x + offsets2[feature, 2]
    if 0 > x2 or x2 >= image.shape[2]: return oob_value
    cdef np.int32_t y2 = y + offsets2[feature, 1]
    if 0 > y2 or y2 >= image.shape[1]: return oob_value
    cdef np.int32_t z2 = z + offsets2[feature, 0]
    if 0 > z2 or z2 >= image.shape[0]: return oob_value

    cdef np.int32_t c1 = offsets1[feature, 3]
    cdef np.int32_t c2 = offsets2[feature, 3]

    return image[z2, y2, x2, c2] - image[z1, y1, x1, c1]
