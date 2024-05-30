import numpy as np
import vtk

from pathlib import Path
from typing import Optional, Union
from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray

from ..utils import guess_image_shape


class ImageRenderer:
    """2D/3D image visualization with basic annotation capabilities.

    Provides dimensional-agnostic methods to create VTK-driven visualizations.
    The axes must be ordered as [Z,] Y, X, as well as all points. They are
    converted appropriately.
    """

    def __init__(self):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)

        #: The :class:`vtkRenderer` that is used to place the actors.
        self.renderer: vtk.vtkRenderer = renderer

        self._image_actor = None
        self._annotation_actors = []

    def set_image(self, image: np.ndarray,
                  spacing: Optional[np.ndarray] = None) -> None:
        """Updates the currently shown image.

        Args:
            image: The image to show.
            spacing: An optional spacing to interpolate the image properly.
        """
        if self._image_actor is not None:
            self.renderer.RemoveActor(self._image_actor)

        spacing = np.ones(3) if spacing is None else \
            self._make_position(spacing)

        importer = vtkImageImportFromArray()
        importer.SetArray(image)
        importer.SetDataSpacing(spacing)

        n_dims, n_channels = guess_image_shape(image)

        is_3d = n_dims == 3

        if is_3d:
            self.renderer.AutomaticLightCreationOn()

            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetBlendModeToMaximumIntensity()
            mapper.AutoAdjustSampleDistancesOn()
            mapper.SetInputConnection(importer.GetOutputPort())

            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(image.min(), 0, 0, 0)
            color_tf.AddRGBPoint(np.max(image), 1, 1, 1)

            opacity_tf = vtk.vtkPiecewiseFunction()
            opacity_tf.AddPoint(image.min(), 0)
            opacity_tf.AddPoint(np.median(image), 1.)
            opacity_tf.AddPoint(image.max(), .0)

            prop = vtk.vtkVolumeProperty()
            prop.SetInterpolationTypeToLinear()
            prop.SetColor(color_tf)
            prop.SetScalarOpacity(opacity_tf)
            prop.ShadeOff()

            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)

            view_up = (0, 0, 0)
        else:
            self.renderer.AutomaticLightCreationOff()

            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(0.0, 0.0, 0.0)
            plane.SetPoint1(image.shape[1] * spacing[0], 0, 0)
            plane.SetPoint2(0, image.shape[0] * spacing[1], 0)

            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(image.min(), 0, 0, 0)
            color_tf.AddRGBPoint(image.max(), 1, 1, 1)

            colormap = vtk.vtkImageMapToColors()
            colormap.SetInputConnection(importer.GetOutputPort())
            colormap.SetOutputFormatToLuminance()
            colormap.SetLookupTable(color_tf)

            texture = vtk.vtkTexture()
            texture.SetInputConnection(colormap.GetOutputPort())

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(texture)

            view_up = (0, -1, 0)

        center = self._make_position(np.array(image.shape[:n_dims])) \
            * spacing / 2

        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(*view_up)
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + [0, 0, -100]))

        self.renderer.AddActor(actor)
        self._image_actor = actor

        bounds = self._make_position(np.array(image.shape[:n_dims])) * spacing

        axes_actor = vtk.vtkCubeAxesActor2D()
        axes_actor.SetCamera(camera)
        axes_actor.SetBounds(0, bounds[0], 0, bounds[1], 0, bounds[2])
        axes_actor.SetCornerOffset(0)
        axes_actor.SetFlyMode(axes_actor.VTK_FLY_OUTER_EDGES if is_3d else
                              axes_actor.VTK_FLY_NONE)
        axes_actor.SetFontFactor(0.5)
        if not is_3d:
            axes_actor.ZAxisVisibilityOff()

        self.renderer.AddActor(axes_actor)
        self._annotation_actors.append(axes_actor)

        self.renderer.ResetCamera()

    def add_point(self, position: np.ndarray, color=(1, 0, 0)) -> None:
        """Adds a point to the scene in world coordinates.

        Args:
            position: The center of the point.
            color: The color of the point.
        """
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(self._make_position(position))
        sphere.SetRadius(1)
        sphere.SetThetaResolution(100)
        sphere.SetPhiResolution(100)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        self._annotation_actors.append(actor)

    def add_line(self, position_a: np.ndarray, position_b: np.ndarray,
                 color=(1, 0, 0)) -> None:
        """Adds a line to the scene in world coordinates.

        Args:
            position_a: First point of the line.
            position_b: Second point of the line.
            color: The color of the line.
        """
        line = vtk.vtkLineSource()
        line.SetPoint1(self._make_position(position_a))
        line.SetPoint2(self._make_position(position_b))

        tube = vtk.vtkTubeFilter()
        tube.SetRadius(1)
        tube.SetNumberOfSides(50)
        tube.SetInputConnection(line.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        self._annotation_actors.append(actor)

    def add_caption(self, text: str, position: np.ndarray,
                    color=(1, 1, 1)) -> None:
        """Adds a caption to the scene w.r.t. a position in world coordinates.

        Args:
            text: The annotation text itself.
            position: Where the caption should point to.
            color: The color of the text.
        """
        caption = vtk.vtkCaptionActor2D()
        caption.SetAttachmentPoint(self._make_position(position))
        caption.SetCaption(text)
        caption.BorderOff()

        caption.GetTextActor().SetTextScaleModeToNone()
        caption.GetCaptionTextProperty().BoldOff()
        caption.GetCaptionTextProperty().ItalicOff()
        caption.GetCaptionTextProperty().SetColor(*color)
        caption.GetCaptionTextProperty().SetFontSize(10)
        caption.GetCaptionTextProperty().Modified()

        self.renderer.AddActor(caption)
        self.renderer.ResetCamera()

        self._annotation_actors.append(caption)

    def add_text_2d(self, text: str, position: np.ndarray) -> None:
        """Adds a 2D text on top of the rendered scene.

        Args:
            text: The text to show.
            position: Where to position the text treating the upper left corner
                as origin. Relative values (< 1) refer to a fraction of the
                window size.
        """
        text_actor = vtk.vtkTextActor()
        text_actor.SetTextScaleModeToViewport()
        text_actor.SetInput(text)

        text_property = text_actor.GetTextProperty()
        text_property.SetFontSize(8)
        text_property.SetColor(1, 1, 1)
        text_property.SetJustification(vtk.VTK_TEXT_LEFT)
        text_property.SetVerticalJustification(vtk.VTK_TEXT_TOP)

        position = np.asarray(position)

        def update_position(window_size):
            new_position = np.where(position < 1, position * window_size,
                                    position)
            new_position[1] = window_size[1] - new_position[1]

            text_actor.SetDisplayPosition(*new_position)

        def handle_modified(window, event):
            update_position(window.GetSize())

        window = self.renderer.GetVTKWindow()
        window.AddObserver('ModifiedEvent', handle_modified, 1.0)

        update_position(window.GetSize())

        self.renderer.AddActor(text_actor)
        self._annotation_actors.append(text_actor)

    def clear_annotations(self):
        """Removes all annotations."""
        for actor in self._annotation_actors:
            self.renderer.RemoveActor(actor)

        self._annotation_actors = []

    def save(self, path: Union[str, Path]) -> None:
        """Saves the rendered image to `path`.

        Args:
            path: Path to an image location.
        """
        self.renderer.ResetCamera()

        graphics_factory = vtk.vtkGraphicsFactory()
        graphics_factory.SetOffScreenOnlyMode(1)
        graphics_factory.SetUseMesaClasses(1)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(self.renderer)
        window.SetOffScreenRendering(1)
        window.Render()

        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(window)
        window_to_image.SetMagnification(3)
        window_to_image.SetInputBufferTypeToRGBA()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()

        writers = {'.png': vtk.vtkPNGWriter,
                   '.jpg': vtk.vtkJPEGWriter,
                   '.jpeg': vtk.vtkJPEGWriter}

        writer = writers[Path(path).suffix]()
        writer.SetFileName(str(path))
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()

    def show(self):
        """Shows the image and all annotations in an interactive window.

        This function blocks the thread until the window is closed.
        """
        self.renderer.ResetCamera()

        window = vtk.vtkRenderWindow()
        window.AddRenderer(self.renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera())

        interactor.Initialize()
        interactor.Start()

    def _make_position(self, position: np.ndarray) -> np.ndarray:
        """Helper method to create a 3D position."""
        assert position.ndim == 1 and position.size in (2, 3)

        position = position[::-1]

        if len(position) == 2:
            position = np.hstack((position, 0))

        return position
