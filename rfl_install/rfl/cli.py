import click
import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
import rfl
import sys

from hiwi import Image, Evaluation, AvgEvaluation, save_image, load_image
from hiwi import guess_image_shape
from hiwi.cli import IMAGE_LIST
from pathlib import Path


__all__ = (
    'main'
)


log = logging.getLogger(__name__)


class AxisValues(click.ParamType):
    name = 'x[,y[,z]]'

    def __init__(self, element_type, singleton=True):
        self._element_type = element_type
        self._singleton = singleton

        if not singleton:
            self.name = 'x,y[,z]'

    def convert(self, value, param, ctx):
        if value is None:
            return None

        try:
            if self._singleton and type(value) == self._element_type:
                return value

            value = tuple(map(self._element_type, value.split(',')))
            if self._singleton:
                assert 1 <= len(value) <= 3

                if len(value) == 1:
                    return value[0]
            else:
                assert 2 <= len(value) <= 3
            return value
        except Exception:
            self.fail('{} is ill-formed'.format(value), param, ctx)


class ModelType(click.ParamType):
    name = 'model'

    def convert(self, value, param, ctx):
        try:
            return rfl.load(value)
        except Exception as e:
            self.fail('{} is not a valid RFL model ({})'.format(value, e),
                      param, ctx)


AXIS_INTS = AxisValues(int)
AXIS_FLOATS = AxisValues(float)
MODEL = ModelType()


# read the default values from the constructor so we don't have to repeat
# the default values in the CLI
fullargspec = inspect.getfullargspec(rfl.RandomForestLocalizer.__init__)
defaults = dict(zip(reversed(fullargspec.args),
                    reversed(fullargspec.defaults)))


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(rfl.__version__, '-V', '--version', prog_name='rfl')
def main():
    """Random Forest Localizer (RFL)"""
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(name)-12s '
                               '%(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        stream=sys.stderr)


@main.command(short_help='Trains a new RFL model and saves it')
@click.option('-t', '--trees', type=int,
              help='Number of decision tree regressors/masks to use',
              default=defaults['n_trees'], show_default=True)
@click.option('-d', '--max-depth', type=int,
              help='Maximal depth of a decision tree, infinite if none given',
              default=defaults['max_depth'], show_default=True)
@click.option('-f', '--features', type=int,
              help='Number of features computed for a patch',
              default=defaults['n_features'], show_default=True)
@click.option('-F', '--from-origin', type=float,
              help='Fraction of feature vectors to start from the origin',
              default=defaults['frac_origin_features'], show_default=True)
@click.option('-s', '--patch-size', type=AXIS_INTS,
              help='Size of the local patch to sample from encoded as X,Y[,Z]',
              default=','.join(map(str, defaults['patch_size'])),
              show_default=True)
@click.option('-o', '--patch-origin', type=AXIS_INTS,
              help='Origin inside the patch, if none given it defaults to the '
                   'patch\'s center')
@click.option('-g', '--pre-smoothing', type=AXIS_FLOATS,
              help='Sigma of the Gaussian kernel used to presmooth the images',
              default=defaults['pre_smoothing'], show_default=True)
@click.option('-p', '--positive-radius', type=AXIS_FLOATS,
              help='Radius around the landmark to consider as positive',
              default=5, show_default=True)
@click.option('-n', '--part', type=str,
              help='Use the position of a specific part')
@click.option('--seed', type=int, default=defaults['seed'], show_default=True,
              help='Seed to initialize the RNG for reproducability')
@click.option('--gpu/--no-gpu', default=True, show_default=True,
              help='Whether to utilize OpenCL\'s compute capabilities')
@click.argument('image_list', type=IMAGE_LIST)
@click.argument('output_model', type=click.Path(dir_okay=False))
def train(trees, max_depth, features, from_origin, patch_size, patch_origin,
          pre_smoothing, positive_radius, seed, gpu, part, image_list,
          output_model):
    """Trains a new RFL model given a list of training images."""

    training_samples = []

    n_channels = None
    n_dims = None

    with click.progressbar([(image, obj) for image in image_list
                            for obj in image.objects],
                           label='Generating positive samples') as bar:
        for image, obj in bar:
            if part is not None and part not in obj.parts:
                continue

            # fill in missing image information, since they are fixed now
            if n_channels is None:
                n_dims, n_channels = guess_image_shape(image.data)

                if n_channels is None:
                    n_channels = 1

                positive_radius = (np.ones(n_dims) * positive_radius)[::-1]

            assert ((image.data.ndim == n_dims and n_channels == 1) or
                    (image.data.ndim - 1 == n_dims and
                     image.data.shape[-1] == n_channels)), 'inhomogeneous data'

            true_position = (obj.position if part is None
                             else obj.parts[part].position)[::-1]

            positions, values = rfl.create_gaussian_samples(
                true_position, positive_radius, image.data.shape[:n_dims])

            if len(positions) == 0:
                log.warn('%s: Unable to generate positive samples',
                         image.path)

            training_samples.append((image.data, positions, values))

    if len(training_samples) == 0:
        log.error('Found no matching training sample')
        return

    model = rfl.RandomForestLocalizer(
        n_trees=trees, max_depth=max_depth, n_features=features,
        n_dims=n_dims, n_channels=n_channels,
        frac_origin_features=from_origin, patch_size=patch_size,
        patch_origin=patch_origin, pre_smoothing=pre_smoothing, seed=seed)

    ocl = None
    if gpu:
        ocl = rfl.init_opencl()

        if ocl is None:
            log.warn('Unable to initialize OpenCL compute context')

    model.train(training_samples, ocl)

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    model.save(output_model)


@main.command(short_help='Applies a RFL model')
@click.option('-o', '--output', type=click.Path(file_okay=False),
              help='Optional directory where to store the prob. map(s)')
@click.option('--gpu/--no-gpu', default=True, show_default=True,
              help='Whether to utilize OpenCL\'s compute capabilities')
@click.argument('model', type=MODEL)
@click.argument('image', type=click.Path(exists=True), nargs=-1)
def test(output, gpu, model, image):
    """Applies a RFL model and finds the best matching location."""
    ocl = rfl.init_opencl() if gpu else None

    for image_path in image:
        image, meta = load_image(image_path, meta=True)

        # FIXME: Hacky way to extract the name without image extensions
        image_name = Image(path=image_path).name

        scores = model.test(image, ocl)

        max_idx = np.argmax(scores.flat)
        max_position = np.unravel_index(max_idx, scores.shape)

        if output is not None:
            save_image(scores, Path(output) / (image_name + '.nii.gz'),
                       meta=meta)

        print('{}: ({})'.format(image_path,
                                ', '.join(map(str, reversed(max_position)))))


@main.command(short_help='Evaluate the performance of a model')
@click.option('-o', '--output', type=click.Path(file_okay=False),
              help='Optional path where to store the prob. maps')
@click.option('-n', '--part', type=str,
              help='Use the position of a specific part')
@click.option('-t', '--error-threshold', type=float,
              help='Threshold of the Euclidean distance to determine a '
                   'correct localization')
@click.option('--gpu/--no-gpu', default=True, show_default=True,
              help='Whether to utilize OpenCL\'s compute capabilities')
@click.argument('model', type=MODEL)
@click.argument('image_list', type=IMAGE_LIST)
def evaluate(output, part, gpu, error_threshold, model, image_list):
    """Applies a RFL model to a list of images and computes common erroc
    metrics."""

    ocl = None
    if gpu:
        ocl = rfl.init_opencl()

        if ocl is None:
            log.warn('Unable to initialize OpenCL compute context')

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    evaluations = []

    with click.progressbar(image_list, label='Testing images') as bar:
        for image in bar:
            positions = []

            for obj in image.objects:
                if part is None:
                    positions.append(obj.position)
                elif part in obj.parts:
                    positions.append(obj.parts[part].position)

            if not positions:
                continue

            prob_map = model.test(image.data)

            if output is not None:
                save_image(prob_map, output / (image.name + '.nii.gz'),
                           image.spacing[::-1])

            pred_pos = np.unravel_index(prob_map.argmax(),
                                        prob_map.shape)[::-1]

            positions = [obj.position if part is None
                         else obj.parts[part].position
                         for obj in image.objects]

            distances = np.sum((np.asarray(positions) - pred_pos)**2, axis=1)
            true_pos = positions[distances.argsort()[0]]

            evaluation = Evaluation(true_pos, pred_pos, False,
                                    spacing=image.spacing)
            evaluation.localized = evaluation.error_mm < 10

            evaluations.append(evaluation)

            print('{}: {}'.format(image.path, evaluation))

    print('Average: {}'.format(AvgEvaluation(evaluations)))


@main.command(short_help='Visualizes important information of a model')
@click.option('-o', '--output', type=click.Path(dir_okay=False),
              help='Image path to store the masks in.')
@click.argument('model', type=MODEL)
@click.argument('images', required=False, type=IMAGE_LIST)
def show(output, model, images):
    """Applies a RFL model and finds the best matching location.
    If IMAGES is given, the first image in this image list is used to show the
    mask w.r.t. the reference point.
    """
    fig = plt.figure()

    if images is not None:
        model.plot_masks(fig, images[0].image_data,
                         images[0].object_instances[0].reference_point[::-1])
    else:
        model.plot_masks(fig)

    if output is not None:
        plt.savefig(output, bbox_inches='tight', pad_inches=0, frameon=False)

    plt.show()
