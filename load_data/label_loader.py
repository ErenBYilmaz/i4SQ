import functools

import numpy

import hiwi
from lib.util import EBC
from tasks import VertebraTask
from copy import copy


class LabelLoader(EBC):
    def load(self, task: VertebraTask, img: hiwi.Image, vertebra):
        raise NotImplementedError('Abstract method')

    def modified_task(self, task: VertebraTask):
        new_task = copy(task)
        new_task.y_true_from_hiwi_image_and_vertebra = functools.partial(self.load, task)
        return new_task


class GroundTruthLoader(LabelLoader):
    def load(self, task: VertebraTask, img: hiwi.Image, vertebra):
        return task.y_true_from_hiwi_image_and_vertebra(img, vertebra)


class PseudoLabelsLoader(LabelLoader):
    def load(self, task: VertebraTask, img: hiwi.Image, vertebra):
        try:
            return task.y_true_from_hiwi_image_and_vertebra(img, vertebra)
        except task.UnlabeledVertebra:
            if 'tool_outputs' not in img.parts[vertebra]:
                raise
            pseudo_labels = img.parts[vertebra]['tool_outputs']
            return self.load_pseudo_label(pseudo_labels, task.output_layer_name())

    def load_pseudo_label(self, pseudo_labels, output_layer_name):
        raise NotImplementedError('Abstract method')


class MeanPseudoLabelsLoader(PseudoLabelsLoader):
    def load_pseudo_label(self, pseudo_labels, output_layer_name):
        return numpy.mean([labels[output_layer_name] for labels in pseudo_labels.values()], axis=0)
