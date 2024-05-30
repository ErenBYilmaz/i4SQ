import numpy as np

from typing import Optional, Union


def is_image(image: np.ndarray, n_dims: Optional[int] = None):
    """Tests whether the given `array` is a valid 2D/3D image."""
    valid_dims = (2, 3) if n_dims is None else (n_dims,)
    return image.ndim in valid_dims and np.all(np.less(0, image.shape))


def is_size(size: np.ndarray, n_axes: Optional[int] = None):
    """Tests whether the given `size` is a valid 2D/3D image size."""
    valid_axes = (2, 3) if n_axes is None else (n_axes,)
    return size.ndim == 1 and np.all(0 < size) and len(size) in valid_axes


def is_index(index: np.ndarray,
             n_axes_or_size: Optional[Union[int, np.ndarray]] = None,
             allow_negative: bool = False):
    """Tests whether the given `index` is a valid position in a 2D/3D image."""
    if index.ndim != 1 or (not allow_negative and np.any(0 > index)):
        return False

    if n_axes_or_size is None:
        return len(index) in (2, 3)
    elif type(n_axes_or_size) == int:
        return len(index) == n_axes_or_size
    else:
        return np.all(index < n_axes_or_size)


def atleast_3d_index(index: np.ndarray):
    """Converts the given index into a 3D index, if it isn't."""
    assert is_index(index, allow_negative=True)

    if index.shape[0] == 2:
        return np.insert(index, 0, 0)

    return index


def atleast_3d_size(size: np.ndarray):
    """Converts the given size into a 3D size, if it isn't."""
    assert is_size(size)

    if size.shape[0] == 2:
        return np.insert(size, 0, 1)

    return size


def atleast_3d_image(image: np.ndarray):
    """Converts the given image into a 3D image, if it isn't."""
    assert is_image(image)

    if image.ndim == 2:
        return np.expand_dims(image, axis=0)

    return image
