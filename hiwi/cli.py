import click
import configparser
import fnmatch
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

from collections import defaultdict, OrderedDict
from natsort import natsorted
from pathlib import Path
from tabulate import tabulate

from .image import ImageList, load_image
from .watchdog import Watchdog


class ImageListType(click.ParamType):
    """A Click parameter type that is a path point to a `hiwi.ImageList`."""

    name = 'image-list'

    def convert(self, value, param, ctx):
        try:
            images = ImageList.load(value)
        except Exception as e:
            self.fail('{} is not a valid image-list ({})'.format(value, e),
                      param, ctx)

        if len(images) == 0:
            self.fail(f'{value} lists no images')

        return images


class NumbersType(click.ParamType):
    name = 'x[,y[,z]]'

    def __init__(self, element_type):
        self._element_type = element_type

    def convert(self, value, param, ctx):
        if value is None:
            return None

        try:
            value = np.array([self._element_type(v) for v in value.split(',')])
            assert 1 <= len(value) <= 3
            return value
        except Exception:
            self.fail('{} is not valid'.format(value), param, ctx)


FLOATS = NumbersType(float)
IMAGE_LIST = ImageListType()


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def main():
    """Collection of useful utility functions."""
    pass


@main.group()
def images():
    """Image list related operations."""
    pass


@images.command()
@click.option('-s', '--search', help='Optional search string to filter images')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def show(search, path):
    """Shows an image or an image list."""
    try:
        images = ImageList.load(path)
    except Exception:
        images = None

    if search is not None:
        pattern = '*' + search + '*'
        images_new = [image for image in images
                      if fnmatch.fnmatch(image.name, pattern)]
        click.echo(f'{len(images_new)} of {len(images)} images match your '
                   f'search pattern "{pattern}"')
        images = images_new

    if images is not None:
        # vtk is required to make the visualization, which is not a hard
        # dependency, so we defer the loading
        from .image.ui import ImageListVisualizer

        visualizer = ImageListVisualizer(images)
        visualizer.show()
    else:
        image = load_image(path)

        if not (image.ndim == 2 or (image.ndim == 3 and
                                    image.shape[-1] in (1, 3))):
            print('Only 2D images can be shown!')
            return

        cmap = None
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            cmap = plt.cm.gray

        plt.imshow(image, cmap=cmap)
        plt.show()


@images.command()
@click.option('--scale-spacing', type=FLOATS,
              help='New spacing for the images, either for one or all '
                   'dimensions')
@click.option('--scale-factor', type=FLOATS,
              help='Scaling factor applied to each image, either for one or '
                   'all dimensions')
@click.option('--scale-size', type=FLOATS,
              help='New size of the images, i.e., one final size for all. Use '
                   '0 as placeholder to automatically deduce an axis\' size.')
@click.option('-n', '--normalization',
              type=click.Choice(('standard-score', 'abs-standard-score')),
              help='Whether to normalize the image data')
@click.option('-d', '--images-dir', default='images/',
              help='Directory where to store the images, relative to the '
                   'output image list')
@click.option('-r', '--use-references', type=bool, default=True,
              show_default=True, help='Whether to store the meta data next '
              'to the image or in the list')
@click.argument('input_image_list', type=IMAGE_LIST)
@click.argument('output_image_list', type=click.Path(dir_okay=False))
def transform(scale_spacing, scale_factor, scale_size, normalization,
              images_dir, use_references, input_image_list, output_image_list):
    """Transforms the image and meta data of the given image list.

    \b
    Examples:
      1. Convert all images to a constant isotropic spacing:
         $ hiwi images transform --scale-spacing 1,1,1 images.iml out.iml
      2. Convert all images to a constant height and variable width:
         $ hiwi images transform --scale-size 0,500 images.iml out.iml
    """
    assert scale_spacing is None or scale_factor is None, \
        'you may not specify spacing and scaling together'

    images_dir = Path(output_image_list).parent / images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    with click.progressbar(input_image_list,
                           label='Transforming images') as bar:
        new_image_list = ImageList()

        for image in bar:
            scaling = None
            if scale_spacing is not None:
                scaling = image.spacing / scale_spacing
            elif scale_factor is not None:
                scaling = scale_factor
            elif scale_size is not None:
                size = image.data.shape[::-1]

                if len(scale_size) != len(size):
                    raise click.BadParameter('Must specify size for all axis',
                                             param_hint='scale-size')

                if 1 < (scale_size > 0).sum() < len(size):
                    raise click.BadParameter('At most one axis can be fixed '
                                             'if remaining ones should be '
                                             'calculated',
                                             param_hint='scale-size')

                if (scale_size > 0).sum() == 1:
                    axis = np.where(scale_size > 0)[0][0]
                    scaling = scale_size[axis] / size[axis]
                else:
                    assert (scale_size > 0).all()
                    scaling = scale_size / size

            new_image = image.transformed(images_dir / f'{image.name}.nii.gz',
                                          scaling=scaling,
                                          normalization=normalization)
            new_image_list.append(new_image)

        new_image_list.dump(output_image_list, use_references=use_references)


@images.command()
@click.argument('image_list', type=IMAGE_LIST)
def stats(image_list):
    """Show statistical information about the image list."""
    print('Total images:', len(image_list))

    sizes = []
    spacings = []
    parts = defaultdict(lambda: 0)

    for image in image_list:
        sizes.append(image.data.shape[::-1])

        if image.spacing is not None:
            spacings.append(image.spacing)

        for obj in image.objects:
            for name, part in obj.parts.items():
                if part.position is not None:
                    parts[name] += 1

    def print_stats(data):
        table = [('Min', *np.min(data, axis=0)),
                 ('Avg', *np.mean(data, axis=0)),
                 ('Max', *np.max(data, axis=0)),
                 ('Std', *np.std(data, axis=0))]

        lines = tabulate(table, headers=('', 'x', 'y', 'z')).split('\n')
        lines = '  ' + '\n  '.join(lines)

        print(lines)

    print('\nImage size:')
    print_stats(sizes)

    if spacings:
        print('\nImage spacing:')
        print_stats(spacings)

    print('\nParts with positions:')
    print('\n'.join('  - {:30s}: {}'.format(k, parts[k]) for k
                    in natsorted(parts.keys())))


@images.command()
@click.option('-m', '--mode', type=click.Choice(('simple', 'kfold')),
              default='simple', help='How to split the list in sub lists')
@click.option('-o', '--output', type=click.Path(file_okay=False),
              help='Directory where to store the resulting sub lists')
@click.option('-t', '--training', type=float, default=0.5, show_default=True,
              help='Fraction of training images when --mode simple')
@click.option('-p', '--filter-part', multiple=True,
              help='Use only images that have a matching part annotation')
@click.option('--shuffle/--no-shuffle', default=True, show_default=True,
              help='Whether to shuffle the image list before splitting')
@click.option('-k', '--folds', type=int, default=5, show_default=True,
              help='Number of folds to use when --mode kfold')
@click.option('-g', '--grouped', type=str,
              help='Ensure same group is not present in test and training by '
                   'looking at the group-identifying parameter of the image, '
                   'e.g., patient_id')
@click.option('-s', '--seed', type=int, default=42, show_default=True,
              help='Fix the RNG to a specific seed for reproducability')
@click.option('--references/--no-references', default=True,
              show_default=True, help='Whether to just link to the image '
              'meta files rather than including everything in the lists')
@click.argument('image_list', type=click.Path(exists=True, dir_okay=False))
def split(mode, filter_part, training, output, folds, grouped, shuffle, seed,
          references, image_list):
    """Split image list in sub lists.

    The resulting image lists are placed next to the original image list,
    except one specifies --output to change the output dir.

    \b
    Examples:
      1. A simple 50/50 split:
        $ hiwi images split images.iml
      2. 10-fold cross validation split:
        $ hiwi images split --mode kfold --folds 10 images.iml
      3. 70% training images instead of 50%:
        $ hiwi images split --training 0.7 images.iml
      4. Ensure patients are not in training and test:
        $ hiwi images split --grouped patient_id images.iml
    """
    assert 0 < training < 1

    image_list_path = Path(image_list)
    image_list = ImageList.load(image_list)

    if filter_part:
        new_image_list = ImageList()

        for image in image_list:
            if all(all(p in obj.parts for p in filter_part)
                   for obj in image.objects):
                new_image_list.append(image)

        image_list = new_image_list

    print('Splitting a total of {} images...'.format(len(image_list)))

    rng = np.random.RandomState(seed)

    if shuffle:
        rng.shuffle(image_list)

    if grouped is not None:
        groups = OrderedDict()

        for image in image_list:
            identifier = image[grouped]

            if identifier not in groups:
                groups[identifier] = [image]
            else:
                groups[identifier].append(image)

        image_list = list(groups.values())

    sub_lists = []

    if mode == 'simple':
        split = int(np.ceil(len(image_list) * training))

        sub_lists.append(('training', image_list[:split]))
        sub_lists.append(('test', image_list[split:]))

    elif mode == 'kfold':
        for i in range(folds):
            start = int(np.ceil(len(image_list) / folds * i))
            end = int(np.ceil(len(image_list) / folds * (i + 1)))

            sub_lists.append(('fold{}_training'.format(i + 1),
                              image_list[:start] + image_list[end:]))
            sub_lists.append(('fold{}_test'.format(i + 1),
                              image_list[start:end]))

    output = image_list_path.parent if output is None else Path(output)

    for name, sub_list in sub_lists:
        if grouped is not None:
            sub_list = ImageList(sum(sub_list, []))

        sub_list_path = output / (name + '.iml')

        print('{}: {} images'.format(sub_list_path, len(sub_list)))
        sub_list.dump(sub_list_path, use_references=references)


@main.command()
@click.option('-p', '--poll', type=int, default=30, show_default=True,
              help='Poll process status every --poll seconds')
@click.option('-r', '--report', type=int,
              help='Send a report of watched processes after --report seconds')
@click.argument('pattern', type=str, nargs=-1)
def watchdog(poll, report, pattern):
    """Looks for newly created and dying processes.

    Use Python regular expressions as patterns to filter the list of processes.
    """
    config_path = Path(os.path.expanduser('~/.hiwi.conf'))

    if not config_path.exists():
        print(f'The file {config_path} must have the following contents:')
        print('[watchdog]\nhost = ...\nport = ...\nuser = ...\npassword = ...'
              '\nto = ...')
        exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)
    config = config['watchdog']

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(name)-12s '
                               '%(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        stream=sys.stderr)

    watchdog = Watchdog(host=config['host'].strip(), port=int(config['port']),
                        user=config['user'].strip() or None,
                        password=config['password'].strip() or None,
                        from_=config['from'].strip() or None,
                        starttls=(config['starttls'] or '').lower() in
                        ('yes', 'true', '1'),
                        to=config['to'].strip(), patterns=pattern,
                        interval=poll, report_every=report)
    watchdog.run()
