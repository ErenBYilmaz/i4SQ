import random
from typing import Tuple

import numpy
import numpy as np

last_pos = None

PAD_MODE_NO_PADDING = 0
PAD_MODE_IF_NECESSARY = 1
PAD_MODE_ALWAYS = 2


def random_crop(img: numpy.ndarray, desired_shape: Tuple[int, ...], pad_mode=PAD_MODE_NO_PADDING):
    assert len(img.shape) == len(desired_shape)
    if pad_mode == PAD_MODE_NO_PADDING:
        assert all(s >= d for s, d in zip(img.shape, desired_shape))
        pos = tuple(random.randint(0, s - d) for s, d in zip(img.shape, desired_shape))
    elif pad_mode == PAD_MODE_ALWAYS:
        pos = tuple(random.randint(-d, s) for s, d in zip(img.shape, desired_shape))
    elif pad_mode == PAD_MODE_IF_NECESSARY:
        pos = tuple(random.randint(0, s - d) if s >= d else random.randint(s - d, 0)
                    for s, d in zip(img.shape, desired_shape))
    else:
        raise NotImplementedError
    # return img[c:c+s for c, s in zip(pos, desired_shape)] # syntax error but equivalent
    global last_pos
    last_pos = pos
    if pad_mode in [PAD_MODE_ALWAYS, PAD_MODE_IF_NECESSARY]:
        return crop_with_pad(img, desired_shape=desired_shape, pos=pos)
    else:
        return crop(img, desired_shape=desired_shape, pos=pos)


def crop(img: numpy.ndarray, desired_shape: Tuple[int, ...], pos: Tuple[int, ...]):
    return img.__getitem__(tuple(slice(c, c + s) for c, s in zip(pos, desired_shape)))


def fill_crop(img, pos, crop):
    """
    Fills `crop` with values from `img` at `pos`,
    while accounting for the crop being off the edge of `img`.
    *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
    """
    img_shape, pos, crop_shape = np.array(img.shape), np.array(pos), np.array(crop.shape),
    end = pos + crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    # if any(pos != np.clip(pos, a_min=0, a_max=img_shape)):
    #     print('pos', (pos - np.clip(pos, a_min=0, a_max=img_shape)) / crop_shape)
    # if any(end != np.clip(end, a_min=0, a_max=img_shape)):
    #     print('end', (end - np.clip(end, a_min=0, a_max=img_shape)) / crop_shape)
    pos = np.clip(pos, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]


def crop_with_pad(img: numpy.ndarray, desired_shape: Tuple[int, ...], pos: Tuple[int, ...], pad_value=0, dtype='float32'):
    if numpy.isscalar(pad_value):
        result = np.full(shape=desired_shape, fill_value=pad_value, dtype=dtype)
    else:
        result = np.empty(shape=desired_shape, dtype=dtype)
        result[:] = pad_value
    fill_crop(img, pos=pos, crop=result)
    assert tuple(result.shape) == tuple(desired_shape)
    return result


def main():
    x = numpy.array([[8, 1, 2, 3], [7, 4, 5, 6]])
    print(x)
    print()
    for _ in range(20):
        print(random_crop(x, desired_shape=(4, 3), pad_mode=PAD_MODE_IF_NECESSARY))


if __name__ == '__main__':
    main()
