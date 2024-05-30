def genant_score_from_deformity(deformity) -> float:
    if deformity < 0.2:
        return 0
    elif deformity < 0.25:
        return 1
    elif deformity < 0.4:
        return 2
    else:
        return 3


def genant_score_from_deformities(deformities) -> float:
    """
    :param deformities: wedge, biconcave and crush as ratios (0.15 for 15% for example)
    :return:
    """
    if len(deformities) != 3:
        raise ValueError(f'This method was intended to be used with wegde, biconcave and crush deformities (in arbitrary order), '
                         f'but {len(deformities)} values were provided.')
    if any(deformity < -1 or deformity > 1 for deformity in deformities):
        print(f'WARNING: Unusually large deformity values. Expected are values between -1 and 1, but got {deformities}')
    return max(genant_score_from_deformity(deformity) for deformity in deformities)
