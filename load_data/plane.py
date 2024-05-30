from lib.util import EBE


class Plane(EBE):
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2

    def variable_coordinates(self):
        return {
            Plane.AXIAL: 'xy',
            Plane.CORONAL: 'xz',
            Plane.SAGITTAL: 'yz',
        }[self]

    @staticmethod
    def from_axis(axis):
        return {
            'x': Plane.SAGITTAL,
            'y': Plane.CORONAL,
            'z': Plane.AXIAL,
        }[axis]

    def fixed_coordinate(self):
        return {
            Plane.AXIAL: 'z',
            Plane.CORONAL: 'y',
            Plane.SAGITTAL: 'x',
        }[self]

    @staticmethod
    def from_fixed_coordinate(c: str) -> 'Plane':
        return {
            'x': Plane.SAGITTAL,
            'y': Plane.CORONAL,
            'z': Plane.AXIAL,
        }[c]

    def axis_xyz(self):
        return {
            Plane.AXIAL: 2,
            Plane.CORONAL: 1,
            Plane.SAGITTAL: 0,
        }[self]

    def axis_zyx(self):
        return {
            Plane.AXIAL: 0,
            Plane.CORONAL: 1,
            Plane.SAGITTAL: 2,
        }[self]

    def x_2d_to_3d(self):
        """
        Returns the 3D coordinate axis corresponding to the 2D x-coordinate axis in a slice of the direction of the plane
        """
        return self.variable_coordinates()[0]

    def y_2d_to_3d(self):
        """
        Returns the 3D coordinate axis corresponding to the 2D y-coordinate axis in a slice of the direction of the plane
        """
        return self.variable_coordinates()[1]
