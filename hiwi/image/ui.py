import numpy as np
import vtk

from . import ImageList
from .renderer import ImageRenderer
from ..utils import guess_image_shape


class ImageListVisualizer:
    """An interactive GUI to visualize an :class:`ImageList`, which works for
    2D and 3D.

    Uses :class:`ImageRenderer` to show the image as well as the annotations.
    """
    def __init__(self, images: ImageList) -> None:
        super().__init__()

        self.images = images
        self.image_idx = 0
        self.image_renderer = ImageRenderer()

    def update_renderer(self) -> None:
        image = self.images[self.image_idx]

        try:
            image_data = image.data
        except Exception:
            image_data = None

        self.image_renderer.clear_annotations()

        if image_data is None:
            spacing = 1
        else:
            n_dims, _ = guess_image_shape(image_data)
            spacing = np.ones(n_dims) if image.spacing is None else \
                image.spacing

        if image_data is not None:
            self.image_renderer.set_image(image.data, spacing[::-1])
        self.image_renderer.add_text_2d('{}/{}: {}'.format(
            self.image_idx + 1, len(self.images), image.name), [10, 10])

        def annotate_object(obj, name=None):
            if obj.position is not None:
                position = (obj.position * spacing)[::-1]
                self.image_renderer.add_point(position)

                if name is not None:
                    self.image_renderer.add_caption(name, position)

            for part, sub_obj in obj.parts.items():
                annotate_object(sub_obj, part)

        for obj in image.objects:
            annotate_object(obj)

    def show(self) -> None:
        window = vtk.vtkRenderWindow()
        window.AddRenderer(self.image_renderer.renderer)

        self.update_renderer()

        def handle_key_press(interactor, event):
            key = interactor.GetKeySym()

            if key == 'Left' and self.image_idx > 0:
                self.image_idx -= 1
                self.update_renderer()
                window.Render()
            if key == 'Right' and self.image_idx < len(self.images) - 1:
                self.image_idx += 1
                self.update_renderer()
                window.Render()

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.AddObserver('KeyPressEvent', handle_key_press, 1.0)
        interactor.Initialize()
        interactor.Start()
