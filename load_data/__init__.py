from typing import Tuple, List

import numpy

X = Y = Z = float
OutputList = List[numpy.ndarray]
SampleWeightList = List[numpy.ndarray]
InputList = List[numpy.ndarray]
Batch = Tuple[InputList, OutputList, SampleWeightList]
MADER_SPACING: Tuple[X, Y, Z] = (3, 1, 1)  # mm per voxel
YILMAZ_INPUT_SIZE: Tuple[X, Y, Z] = (51, 60, 54)  # mm per verbetra patch
VERTEBRAE = (
        ['C' + str(i + 1) for i in range(7)]
        + ['T' + str(i + 1) for i in range(12)]
        + ['L' + str(i + 1) for i in range(6)]
)
