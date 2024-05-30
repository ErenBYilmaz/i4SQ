import random
from typing import Tuple

import numpy


def random_crop(img: numpy.ndarray, desired_shape: Tuple[int, ...]):
    assert len(img.shape) == len(desired_shape)
    assert (s >= d for s, d in zip(img.shape, desired_shape))
    start_coords = tuple(random.randint(0, s - d) for s, d in zip(img.shape, desired_shape))
    # return img[c:c+s for c, s in zip(start_coords, desired_shape)] # syntax error but equivalent
    return img.__getitem__(tuple(slice(c, c + s) for c, s in zip(start_coords, desired_shape)))


def main():
    x = numpy.array([[1, 1, 2, 3], [7, 4, 5, 6]])
    print(x)
    for _ in range(20):
        print(random_crop(x, (2, 2)))


if __name__ == '__main__':
    main()
