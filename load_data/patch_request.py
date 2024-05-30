from typing import Tuple

from lib.util import EBC, Z, Y, X


class PatchRequest(EBC):
    def __init__(self, size_mm: Tuple[X, Y, Z], spacing: Tuple[X, Y, Z], offset=0):
        """
        This class resembles an input to the fNet, a 3D cuboid centered on a vertebra
        :param size_mm:
        The cuboids size in mm
        :param spacing:
        The size of a single voxel in the array representing the cuboid
        :param offset:
        Offset=0 is just the vertebra itself. offset=1 and offset=-1 are the adjacent vertebrae.
        This can be used to feed adjacent vertebrae as additional input patches into the NN
        """
        if isinstance(spacing, list):
            spacing = tuple(spacing)
        if isinstance(size_mm, list):
            size_mm = tuple(size_mm)
        self.spacing = spacing
        self.offset = offset
        self.size_mm = size_mm

    @staticmethod
    def from_input_size_px_and_mm(input_size_px: Tuple[X, Y, Z],
                                  input_size_mm: Tuple[X, Y, Z],
                                  offset=0):
        return PatchRequest(
            size_mm=input_size_mm,
            spacing=(input_size_mm[0] / input_size_px[0],
                     input_size_mm[1] / input_size_px[1],
                     input_size_mm[2] / input_size_px[2]),
            offset=offset
        )

    @staticmethod
    def from_input_size_px_and_spacing(input_size_px: Tuple[X, Y, Z],
                                       spacing: Tuple[X, Y, Z],
                                       offset=0):
        return PatchRequest(
            size_mm=(input_size_px[0] * spacing[0],
                     input_size_px[1] * spacing[1],
                     input_size_px[2] * spacing[2]),
            spacing=spacing,
            offset=offset
        )

    def size_px(self) -> Tuple[Z, Y, X]:
        # noinspection PyTypeChecker
        return tuple(round(s / sp) for s, sp in zip(self.size_mm, self.spacing))[::-1]

    def set_size_px_by_changing_size_mm(self, size_px: Tuple[Z, Y, X]):
        self.size_mm = tuple(round(sp * s_px) for sp, s_px in zip(self.spacing, size_px[::-1]))
