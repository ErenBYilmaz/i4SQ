import numpy
import numpy as np
from skimage.exposure import equalize_adapthist


def clahe(img: numpy.ndarray, kernel_size=None, clip_limit=0.01, num_bins=128) -> numpy.ndarray:
    data = equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=num_bins)
    result = data - np.amin(data)
    result = result / np.amax(result)
    assert numpy.all(result >= 0) and numpy.all(result <= 1)
    return result