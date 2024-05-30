import os
import random
from typing import Tuple, Dict, Optional

import numpy
import pandas
import plotly.express

from lib.image_processing_tool import TrainableImageProcessingTool

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils.vis_utils import plot_model
import skimage.transform
import wandb

from data import Z, Y
from lib.custom_layers import RandomBlur3D
from lib.my_logger import logging
from lib.util import EBC
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, split_name
from tasks import VertebraClassificationTask, BinaryOutputCombination, BinaryClassificationTask


class NoLayersFoundError(ValueError):
    pass


def intermediate_outputs_model(base_model, layer_filter=lambda layer: True, flatten=False, only_last_n_layers=None):
    if flatten:
        extra_layer = lambda x: Flatten(name=f'flattened_{x.name}')(x.output)
    else:
        extra_layer = lambda x: x.output
    base_new_outputs_on_layers = [layer for layer in base_model.layers if layer_filter(layer)]
    if only_last_n_layers is not None:
        base_new_outputs_on_layers = base_new_outputs_on_layers[-only_last_n_layers:]
    new_outputs = [extra_layer(layer) for layer in base_new_outputs_on_layers]
    if len(new_outputs) == 0:
        raise NoLayersFoundError('No layers found')
    return Model(inputs=base_model.inputs, outputs=new_outputs)


class OnlyHiddenConvolutionalAndDenseLayers(EBC):
    def __init__(self, base_model: Model, max_output_size=None, force_include_input_layer=False, allow_sub_models=False):
        self.allow_sub_models = allow_sub_models
        self.force_include_input_layer = force_include_input_layer
        self.base_model = base_model
        self.max_output_size = max_output_size
        self._filter_cache: Dict[int, bool] = {}

    def __call__(self, layer) -> bool:
        if self.force_include_input_layer and isinstance(layer, InputLayer):
            return True
        if self.max_output_size is not None and numpy.prod(layer.output_shape[1:]) > self.max_output_size:
            return False
        if layer.name in self.base_model.output_names:
            return False
        if isinstance(layer, Conv):
            if not isinstance(layer, RandomBlur3D):
                return True
        if isinstance(layer, Dense):
            return True
        if isinstance(layer, Model) and self.allow_sub_models:
            return True
        return False


def img_to_unicode(image):
    height = image.shape[0]
    width = image.shape[1]
    image = (image - image.min())
    image = image / image.max()
    # print(image.max(), image.min())
    chars = " \u2591\u2592\u2593\u2588"
    # chars = " .:-=+*#%@"
    result = ''
    rough_aspect_ratio = (1, 1)
    # rough_aspect_ratio = (1, 2)
    # rough_aspect_ratio = (3, 5)
    image = image * len(chars) - 1
    image = numpy.round(image).astype(int)
    for h in range(height):
        row = ''
        for w in range(width):
            row += chars[image[w, h].item()] * rough_aspect_ratio[1]
        result += (row + '\n') * rough_aspect_ratio[0]
    return result


class TSNEPlotter(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator, only_last_n_layers=None, also_plot_at_input_layer=False, skip_existing=None, exclude_binary_tasks=False):
        super().__init__(evaluator=evaluator, analysis_needs_xs=True, skip_existing=skip_existing,
                         binary_threshold_or_method=0.5)
        self.also_plot_at_input_layer = also_plot_at_input_layer
        self.model_level_caching = self.skip_existing
        self.log_with_plotly = True
        self.log_to_wandb = False
        self.only_last_n_layers = only_last_n_layers
        self.exclude_binary_tasks = exclude_binary_tasks

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)

    def analyze_batch(self, batch, y_preds, names):
        self.check_if_analyzing_dataset()
        xs, ys, sample_weights = batch
        to_dir = os.path.join(self.to_dir(), os.path.basename(self.model_path))
        model = self.load_trained_model_for_evaluation()
        if not isinstance(getattr(model, 'model', None), Model):
            raise RuntimeError('Model does not contain a keras model')
        else:
            model = model.model
        try:
            model = intermediate_outputs_model(model,
                                               layer_filter=OnlyHiddenConvolutionalAndDenseLayers(model,
                                                                                                  max_output_size=80 * 1024,
                                                                                                  force_include_input_layer=self.also_plot_at_input_layer,
                                                                                                  allow_sub_models=False),
                                               only_last_n_layers=self.only_last_n_layers,
                                               flatten=True)
        except NoLayersFoundError:
            logging.warning(f'No layers found TSNE-analysis of {self.model_path}')
            return
        model_path = os.path.join(to_dir, 'intermediate_outputs_model.png')
        if not self.serializer().isfile(model_path):
            self.serializer().create_directory_if_not_exists(to_dir)
            plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, to_file=model_path)

        enumerated_intermediate_outputs = self.enumerated_intermediate_outputs(model, xs)

        perplexities = self.use_perplexities()
        true_class_namess = self.get_class_names(self.tasks, ys)
        pred_class_namess = self.get_class_names(self.tasks, y_preds)

        data = {
            'name': names,
            'vertebra_name': [split_name(name)[1] for name in names],
            'center': [split_name(name)[0][0] for name in names],
        }
        for task in self.tasks:
            data[f'{task.output_layer_name()}_true'] = true_class_namess[task.output_layer_name()]
            data[f'{task.output_layer_name()}_pred'] = pred_class_namess[task.output_layer_name()]
        requests = self.evaluator.patch_requests(adjust_for_rotation=False)

        # optimizer = img2unicode.BestGammaOptimizer()
        # renderer = img2unicode.GammaRenderer(default_optimizer=optimizer)

        for input_idx, volumes in enumerate(xs):
            patch_request = requests[input_idx]
            assert len(volumes.shape) == 5, 'Currently TSNEPlotter only supports 3D volumes of shape (b, z, y, x, c)'
            batch_size = volumes.shape[0]
            image_width_x = volumes.shape[3]
            shape_before: Tuple[Z, Y] = (volumes.shape[1], volumes.shape[2])
            spacing = patch_request.spacing
            resized = skimage.transform.resize(volumes[:, :, :, image_width_x // 2][..., ::-1],
                                               (batch_size, round(shape_before[0] * spacing[2]), round(shape_before[1] * spacing[1])))
            captions = []
            for sample_idx in range(len(resized)):
                c = {k: v[sample_idx] for k, v in data.items()}
                c.update({'spacing': patch_request.spacing, 'offset': patch_request.offset, 'size_mm': patch_request.size_mm})
                captions.append(str(c))
            if self.log_to_wandb:
                data[f'sagittal_slice_{model.input_names[input_idx]}'] = [
                    wandb.Image(x, caption=caption)
                    for caption, x in zip(captions, resized)
                ]
            elif self.log_with_plotly:
                data['img'] = ['Not Implemented' for _x in resized]
                # data['img'] = [img_to_unicode(x[:, :, 0]) for x in resized]
                # data['img'] = [renderer.render_numpy(x[:, :, 0], file=s)[0] for x in resized]

        for output_idx, intermediate_output in enumerated_intermediate_outputs:
            out_paths = [
                os.path.join(to_dir, f'{model.output_names[output_idx]}', f'{task.output_layer_name()}_{perplexity}' + '.html')
                for task in self.tasks
                for perplexity in perplexities
            ]
            if all(self.serializer().isfile(out_path) for out_path in out_paths) and not self.log_to_wandb:
                continue
            for perplexity in perplexities:
                out_paths = [
                    os.path.join(to_dir, f'{model.output_names[output_idx]}', f'{task.output_layer_name()}_{perplexity}' + '.html')
                    for task in self.tasks
                ]
                if all(self.serializer().isfile(out_path) for out_path in out_paths) and not self.log_to_wandb:
                    continue
                logging.info('Computing TSNE embeddings...')
                # https://github.com/DmitryUlyanov/Multicore-TSNE
                embeddings = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    init='random',
                ).fit_transform(intermediate_output)
                data[f'tsne_x_p{perplexity}_{model.output_names[output_idx]}'] = embeddings[:, 0]
                data[f'tsne_y_p{perplexity}_{model.output_names[output_idx]}'] = embeddings[:, 1]

                if self.log_with_plotly:
                    logging.info('Logging with plotly...')
                    for task_idx, task in enumerate(self.tasks):
                        if not isinstance(task, VertebraClassificationTask):
                            continue
                        if self.exclude_binary_tasks and isinstance(task, BinaryClassificationTask):
                            continue
                        if isinstance(task, BinaryOutputCombination):
                            continue
                        out_path = os.path.join(to_dir, f'{model.output_names[output_idx]}', f'{task.output_layer_name()}_{perplexity}' + '.html')
                        if self.skip_existing and self.serializer().isfile(out_path):
                            continue
                        self.serializer().create_directory_if_not_exists(out_path)

                        assert len(embeddings) == len(names)
                        true_class_names = true_class_namess[task.output_layer_name()]
                        pred_class_names = pred_class_namess[task.output_layer_name()]
                        outputs_correct = [
                            true_class_names[sample_idx] == pred_class_names[sample_idx]
                            for sample_idx in range(len(embeddings))
                        ]
                        hover_data = {}
                        for output_name, true_class, pred_class in zip(self.tasks.output_layer_names(), true_class_namess.values(), pred_class_namess.values()):
                            hover_data[f'{output_name}_true'] = true_class
                            hover_data[f'{output_name}_pred'] = pred_class
                        # patches_path = os.path.relpath('img/generated/patches', start=out_path)
                        hover_data['img'] = data['img']
                        #     [
                        #     f'<image x="0" y="0" width="50" height="50" xlink:href="file://{patches_path}/{patient_id}_{vertebra}_0_db2k_{requests[0].size_mm}.png"></image>'
                        #     for name in names
                        #     for patient_id, vertebra in [split_name(name)]
                        # ]
                        fig = plotly.express.scatter(
                            embeddings, x=0, y=1,
                            color=true_class_names,
                            symbol=outputs_correct,
                            symbol_map={False: 'circle', True: 'cross'},
                            hover_name=names,
                            hover_data=hover_data,
                            labels={'color': 'Class', 'symbol': 'Correct'}
                        )

                        logging.info(f'Writing to {out_path}')
                        fig.write_html(out_path)
        print()

        if self.log_to_wandb:
            df = pandas.DataFrame(data)
            wandb_table = wandb.Table(dataframe=df)
            run = ...  # TODO
            key = os.path.join(self.to_dir(short=True), os.path.basename(self.model_path)).replace('/', '__').replace('\\', '__')
            logging.info(f'Logging to wandb: {key}')
            run.log({key: wandb_table})

    def use_perplexities(self):
        perplexities = [5, ]
        if self.RANDOM_SUBDIRECTORY_ORDER:
            if len(perplexities) > 1:
                logging.info('Processing perplexities in a random order')
                random.shuffle(perplexities)
        return perplexities

    def enumerated_intermediate_outputs(self, model: TrainableImageProcessingTool, xs):
        intermediate_outputs = self.get_intermediate_outputs_as_list(model, xs)
        enumerated_intermediate_outputs = list(reversed(list(enumerate(intermediate_outputs))))
        if self.RANDOM_SUBDIRECTORY_ORDER:
            logging.info('Processing output layers in a random order')
            numpy.random.shuffle(enumerated_intermediate_outputs)
        return enumerated_intermediate_outputs

    @staticmethod
    def get_intermediate_outputs_as_list(model: TrainableImageProcessingTool, xs: numpy.ndarray):
        intermediate_outputs = model.predict(xs, batch_size=16)
        if isinstance(intermediate_outputs, numpy.ndarray):
            # only a single layer
            intermediate_outputs = [intermediate_outputs]
        return intermediate_outputs

    def to_subdir(self):
        return 'tsne'

    def get_class_names(self, tasks, y_trues_or_preds):
        assert len(y_trues_or_preds) == len(tasks)
        true_class_namess = {
            task.output_layer_name(): [
                self.class_name_for_model_output_using_precomputed_threshold(y_true[sample_idx], task)
                for sample_idx in range(len(y_true))
            ] for y_true, task in zip(y_trues_or_preds, tasks)
        }
        return true_class_namess

    def model_level_cache_key(self, model_path: Optional[str] = None):
        return super().model_level_cache_key(model_path) + (self.also_plot_at_input_layer, self.exclude_binary_tasks, self.only_last_n_layers)

    def directory_level_cache_key(self, model_dir: str):
        return super().directory_level_cache_key(model_dir) + (self.also_plot_at_input_layer, self.exclude_binary_tasks, self.only_last_n_layers)
