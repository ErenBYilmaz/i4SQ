class MismatchError(ValueError):
    pass


def custom_zip(l1, l2, key):
    """
    return elements from the beginning of the lists but skip all where the keys do not match
    if not matching, first skip elements from l1, then from l2
    """
    if len(l1) == 0 or len(l2) == 0:
        return

    def _check_match():
        nonlocal idx1, idx2
        while idx1 < len(l1) and idx2 < len(l2):
            if key(l1[idx1]) == key(l2[idx2]):
                yield l1[idx1], l2[idx2]
                idx1 += 1
                idx2 += 1
            else:
                break

    idx1 = idx2 = 0
    while True:
        for m in _check_match():
            yield m
        while idx1 < len(l1) and key(l1[idx1]) == key(l1[max(idx1 - 1, 0)]):
            idx1 += 1
        for m in _check_match():
            yield m
        while idx2 < len(l2) and key(l2[idx2]) == key(l2[max(idx2 - 1, 0)]):
            idx2 += 1
        if idx1 >= len(l1) or idx2 >= len(l2):
            break
        if key(l1[max(idx1 - 1, 0)]) != key(l1[idx1]) != key(l2[idx2]) != key(l2[max(idx2 - 1, 0)]):
            return

def test():
    x = [3, 4, 0, 10, 20, 30, 500]
    y = [8, 11, 12, 13, 21, 22, 23, 31, 32, 33, 200, 500]
    print(x)
    print(y)
    print(list(custom_zip(x, y, lambda x: x // 10)))
