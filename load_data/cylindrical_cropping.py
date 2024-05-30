import numpy


class LogicError(Exception):
    pass


AVAILABLE_AXES = [0, 1, 2]


def is_point_in_ellipse(x, y, h, k, r_x, r_y, allow_border):
    """formula from https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse"""
    if allow_border:
        return (x - h) * (x - h) / r_x * r_x + (y - k) * (y - k) / r_y * r_y <= 1
    else:
        return (x - h) * (x - h) / r_x * r_x + (y - k) * (y - k) / r_y * r_y < 1


def point_not_in_ellipse(x, y, h, k, r_x, r_y, allow_border):
    """formula from https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse"""
    if allow_border:
        return (x - h) * (x - h) / (r_x * r_x) + (y - k) * (y - k) / (r_y * r_y) > 1
    else:
        return (x - h) * (x - h) / (r_x * r_x) + (y - k) * (y - k) / (r_y * r_y) >= 1


def coordinate_to_point(x, y):
    # noinspection PyRedundantParentheses
    return (x + 0.5, y + 0.5)


def pad_cylindrical(a, cylinder_axis, pad_value):
    # one of the first three axes must be the cylinder axis and the other two must be the other dimensions of the cylinder
    # this means it is always a 3D cylinder regardless of the rank of a
    if len(a.shape) < 3:
        raise ValueError
    if cylinder_axis not in AVAILABLE_AXES:
        raise ValueError

    if cylinder_axis == 0:
        x, y = numpy.meshgrid(numpy.arange(a.shape[1]),
                              numpy.arange(a.shape[2]),
                              indexing='ij')
        condition = point_not_in_ellipse(x,
                                         y,
                                         h=a.shape[1] / 2 - 0.5,
                                         k=a.shape[2] / 2 - 0.5,
                                         r_x=a.shape[1] / 2,
                                         r_y=a.shape[2] / 2,
                                         allow_border=True)
        a[:, condition] = pad_value
    elif cylinder_axis == 1:
        x, z, y = numpy.meshgrid(numpy.arange(a.shape[0]),
                                 numpy.arange(a.shape[1]),
                                 numpy.arange(a.shape[2]),
                                 indexing='ij')
        condition = point_not_in_ellipse(x,
                                         y,
                                         h=a.shape[0] / 2 - 0.5,
                                         k=a.shape[2] / 2 - 0.5,
                                         r_x=a.shape[0] / 2,
                                         r_y=a.shape[2] / 2,
                                         allow_border=True)
        a[condition] = pad_value
    elif cylinder_axis == 2:
        x, y = numpy.meshgrid(numpy.arange(a.shape[0]),
                              numpy.arange(a.shape[1]),
                              indexing='ij')
        condition = point_not_in_ellipse(x,
                                         y,
                                         h=a.shape[0] / 2 - 0.5,
                                         k=a.shape[1] / 2 - 0.5,
                                         r_x=a.shape[0] / 2,
                                         r_y=a.shape[1] / 2,
                                         allow_border=True)
        a[condition, :] = pad_value
    else:
        raise LogicError


def main():
    for axis in [0, 1, 2]:
        print(f'axis: {axis}')
        x = numpy.ones(shape=(5, 4, 3, 2))
        x[:, :, :, 1] = 2
        pad_cylindrical(x, cylinder_axis=axis, pad_value=[0, -1])
        for idx in range(5):
            print(x[idx, :, :, 0])
        for idx in range(5):
            print(x[idx, :, :, 1])


if __name__ == '__main__':
    main()
